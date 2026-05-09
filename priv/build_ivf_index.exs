#!/usr/bin/env elixir
# Build an IVF (Inverted File Index) for KNN specialist.
# Clusters borderline reference vectors using k-means, then saves the index.
#
# Usage:
#   mix run --no-start priv/build_ivf_index.exs resources/references.json.gz priv/model_params.bin priv/ivf_index.bin [num_refs] [num_clusters]

{:ok, _} = Application.ensure_all_started(:exla)
Nx.default_backend(EXLA.Backend)
Nx.Defn.default_options(compiler: EXLA)

args = System.argv()
input_path = Enum.at(args, 0)
model1_path = Enum.at(args, 1)
output_path = Enum.at(args, 2)
target_total = String.to_integer(Enum.at(args, 3, "20000"))
num_clusters = String.to_integer(Enum.at(args, 4, "32"))

IO.puts("Config: #{target_total} refs, #{num_clusters} clusters")

IO.puts("Loading Model 1 from #{model1_path}...")
model1_params = File.read!(model1_path) |> Nx.deserialize()

IO.puts("Reading #{input_path}...")
data =
  input_path
  |> File.read!()
  |> :zlib.gunzip()
  |> Jason.decode!()

IO.puts("#{length(data)} entries total")

IO.puts("Building tensors...")
{vectors_list, labels_list} =
  data
  |> Enum.map(fn entry ->
    v = entry["vector"]
    [amount, _inst, amount_vs_avg, _hour, _dow, _min_since, _km_last, km_home,
     tx_count, is_online, card_present, unknown_merchant, mcc_risk, _merchant_avg] = v

    vector = v ++ [
      amount * unknown_merchant,
      amount_vs_avg * is_online,
      amount * mcc_risk,
      km_home * unknown_merchant,
      is_online * unknown_merchant,
      amount_vs_avg * unknown_merchant,
      tx_count * amount,
      (1.0 - card_present) * km_home
    ]

    label = if entry["label"] == "fraud", do: 1.0, else: 0.0
    {vector, label}
  end)
  |> Enum.unzip()

data = nil
:erlang.garbage_collect()

vectors = Nx.tensor(vectors_list, type: :f32)
labels = Nx.tensor(labels_list, type: :f32)
vectors_list = nil
labels_list = nil
:erlang.garbage_collect()

IO.puts("Vectors: #{inspect(Nx.shape(vectors))}")

# Score all with Model 1
IO.puts("Scoring with Model 1...")
scores =
  vectors
  |> Nx.dot(model1_params["dense_0"]["kernel"])
  |> Nx.add(model1_params["dense_0"]["bias"])
  |> Nx.max(0)
  |> Nx.dot(model1_params["dense_1"]["kernel"])
  |> Nx.add(model1_params["dense_1"]["bias"])
  |> Nx.max(0)
  |> Nx.dot(model1_params["dense_2"]["kernel"])
  |> Nx.add(model1_params["dense_2"]["bias"])
  |> Nx.sigmoid()
  |> Nx.flatten()

# Select borderline: Model 1 score between 0.15 and 0.95
borderline_mask = Nx.logical_and(Nx.greater_equal(scores, 0.15), Nx.less_equal(scores, 0.95))
borderline_count = borderline_mask |> Nx.sum() |> Nx.to_number() |> round()
IO.puts("Found #{borderline_count} borderline cases")

indices = Nx.argsort(borderline_mask, direction: :desc) |> Nx.slice([0], [borderline_count])
border_vectors = Nx.take(vectors, indices)
border_labels = Nx.take(labels, indices)

fraud_count = border_labels |> Nx.sum() |> Nx.to_number() |> round()
legit_count = borderline_count - fraud_count
IO.puts("Borderline: #{fraud_count} fraud, #{legit_count} legit")

# Stratified subsampling
{border_vectors, border_labels, total, fraud_count, legit_count} =
  if borderline_count > target_total do
    labels_bin = Nx.backend_copy(border_labels, Nx.BinaryBackend) |> Nx.to_flat_list()

    fraud_idx =
      labels_bin
      |> Enum.with_index()
      |> Enum.filter(fn {l, _} -> l == 1.0 end)
      |> Enum.map(fn {_, i} -> i end)

    legit_idx =
      labels_bin
      |> Enum.with_index()
      |> Enum.filter(fn {l, _} -> l == 0.0 end)
      |> Enum.map(fn {_, i} -> i end)

    fraud_ratio = fraud_count / borderline_count
    target_fraud = round(target_total * fraud_ratio) |> max(1) |> min(length(fraud_idx))
    target_legit = (target_total - target_fraud) |> max(1) |> min(length(legit_idx))

    :rand.seed(:exsss, {42, 42, 42})
    sampled_fraud = fraud_idx |> Enum.shuffle() |> Enum.take(target_fraud)
    sampled_legit = legit_idx |> Enum.shuffle() |> Enum.take(target_legit)
    sampled = (sampled_fraud ++ sampled_legit) |> Enum.shuffle()

    sample_idx = Nx.tensor(sampled, type: :s64)
    new_vectors = Nx.take(border_vectors, sample_idx)
    new_labels = Nx.take(border_labels, sample_idx)
    new_count = length(sampled)
    new_fraud = new_labels |> Nx.sum() |> Nx.to_number() |> round()
    new_legit = new_count - new_fraud

    IO.puts("Stratified subsample: #{new_count} (#{new_fraud} fraud / #{new_legit} legit)")
    {new_vectors, new_labels, new_count, new_fraud, new_legit}
  else
    {border_vectors, border_labels, borderline_count, fraud_count, legit_count}
  end

_ = {fraud_count, legit_count}
IO.puts("Building IVF index with #{num_clusters} clusters from #{total} vectors...")

# Copy to BinaryBackend before switching default
vectors_bin = Nx.backend_copy(border_vectors, Nx.BinaryBackend)
labels_bin = Nx.backend_copy(border_labels, Nx.BinaryBackend)

# K-means clustering (on BinaryBackend to avoid EXLA issues in eager loop)
Nx.default_backend(Nx.BinaryBackend)

# Initialize centroids: random sample of vectors
:rand.seed(:exsss, {123, 456, 789})
init_idx = Enum.take_random(0..(total - 1), num_clusters) |> Enum.sort()
centroids = Nx.take(vectors_bin, Nx.tensor(init_idx, type: :s64))

# Run k-means for 20 iterations
kmeans_iterations = 20

centroids =
  Enum.reduce(1..kmeans_iterations, centroids, fn iter, centroids ->
    # Assign each vector to nearest centroid
    # dist = v^2 - 2*v·c + c^2
    v_sq = Nx.sum(Nx.multiply(vectors_bin, vectors_bin), axes: [1], keep_axes: true)
    c_sq = Nx.sum(Nx.multiply(centroids, centroids), axes: [1])
    dot = Nx.dot(vectors_bin, Nx.transpose(centroids))
    dists = Nx.add(Nx.subtract(v_sq, Nx.multiply(2, dot)), c_sq)
    assignments = Nx.argmin(dists, axis: 1)

    # Update centroids
    new_centroids =
      for c <- 0..(num_clusters - 1) do
        mask = Nx.equal(assignments, c)
        count = Nx.sum(mask) |> Nx.to_number()

        if count > 0 do
          mask_f = Nx.reshape(Nx.as_type(mask, :f32), {:auto, 1})
          sum = Nx.sum(Nx.multiply(mask_f, vectors_bin), axes: [0])
          Nx.divide(sum, count)
        else
          Nx.slice(centroids, [c, 0], [1, 22]) |> Nx.squeeze()
        end
      end

    new_centroids = Nx.stack(new_centroids)

    if rem(iter, 5) == 0 do
      assigns_list = assignments |> Nx.to_flat_list()
      sizes = Enum.frequencies(Enum.map(assigns_list, &round/1))
      min_size = sizes |> Map.values() |> Enum.min()
      max_size = sizes |> Map.values() |> Enum.max()
      IO.puts("  K-means iter #{iter}: cluster sizes min=#{min_size} max=#{max_size}")
    end

    new_centroids
  end)

# Final assignment
v_sq = Nx.sum(Nx.multiply(vectors_bin, vectors_bin), axes: [1], keep_axes: true)
c_sq = Nx.sum(Nx.multiply(centroids, centroids), axes: [1])
dot = Nx.dot(vectors_bin, Nx.transpose(centroids))
dists = Nx.add(Nx.subtract(v_sq, Nx.multiply(2, dot)), c_sq)
assignments = Nx.argmin(dists, axis: 1) |> Nx.to_flat_list()

# Build padded cluster arrays
vectors_flat = Nx.backend_copy(vectors_bin, Nx.BinaryBackend) |> Nx.to_batched(1) |> Enum.map(&Nx.squeeze/1)
labels_flat = Nx.backend_copy(labels_bin, Nx.BinaryBackend) |> Nx.to_flat_list()

clusters = Enum.zip([assignments, vectors_flat, labels_flat])
  |> Enum.group_by(fn {a, _, _} -> round(a) end, fn {_, v, l} -> {v, l} end)

cluster_sizes = for c <- 0..(num_clusters - 1), do: length(Map.get(clusters, c, []))
max_cluster_size = Enum.max(cluster_sizes)
IO.puts("Cluster sizes: min=#{Enum.min(cluster_sizes)} max=#{max_cluster_size} avg=#{div(total, num_clusters)}")

# Build padded tensors
cluster_vectors_list =
  for c <- 0..(num_clusters - 1) do
    members = Map.get(clusters, c, [])
    vecs = Enum.map(members, fn {v, _l} -> v end)
    # Pad with zeros to max_cluster_size
    padding = List.duplicate(Nx.broadcast(0.0, {22}), max_cluster_size - length(vecs))
    Nx.stack(vecs ++ padding)
  end

cluster_labels_list =
  for c <- 0..(num_clusters - 1) do
    members = Map.get(clusters, c, [])
    labs = Enum.map(members, fn {_v, l} -> l end)
    # Pad with -1.0 (invalid marker, but mask will exclude)
    padding = List.duplicate(-1.0, max_cluster_size - length(labs))
    Nx.tensor(labs ++ padding, type: :f32)
  end

centroids = Nx.backend_copy(centroids, Nx.BinaryBackend)
cluster_vectors = Nx.stack(cluster_vectors_list)  # {num_clusters, max_cs, 22}
cluster_labels = Nx.stack(cluster_labels_list)     # {num_clusters, max_cs}
cluster_sizes_t = Nx.tensor(cluster_sizes, type: :s64)

index = %{
  "centroids" => centroids,
  "cluster_vectors" => cluster_vectors,
  "cluster_labels" => cluster_labels,
  "cluster_sizes" => cluster_sizes_t
}

IO.puts("Index shapes:")
IO.puts("  centroids: #{inspect(Nx.shape(centroids))}")
IO.puts("  cluster_vectors: #{inspect(Nx.shape(cluster_vectors))}")
IO.puts("  cluster_labels: #{inspect(Nx.shape(cluster_labels))}")
IO.puts("  cluster_sizes: #{inspect(Nx.shape(cluster_sizes_t))}")

IO.puts("Saving IVF index to #{output_path}...")
File.write!(output_path, Nx.serialize(index))
info = File.stat!(output_path)
IO.puts("Saved IVF index (#{div(info.size, 1024)} KB)")
