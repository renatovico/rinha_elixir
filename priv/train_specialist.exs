#!/usr/bin/env elixir
# Extract borderline cases from Model 1 for KNN specialist lookup.
# Saves vectors + labels as a serialized Nx tensor pair.
#
# Usage:
#   mix run --no-start priv/train_specialist.exs resources/references.json.gz priv/model_params.bin priv/knn_specialist.bin

{:ok, _} = Application.ensure_all_started(:exla)
Nx.default_backend(EXLA.Backend)
Nx.Defn.default_options(compiler: EXLA)

args = System.argv()
input_path = Enum.at(args, 0)
model1_path = Enum.at(args, 1)
output_path = Enum.at(args, 2)
target_total = String.to_integer(Enum.at(args, 3, "10000"))

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

# Stratified subsampling while preserving fraud/legit ratio.

{border_vectors, border_labels, borderline_count, fraud_count, legit_count} =
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

# Copy to binary backend for serialization
border_vectors = Nx.backend_copy(border_vectors, Nx.BinaryBackend)
border_labels = Nx.backend_copy(border_labels, Nx.BinaryBackend)

IO.puts("Saving #{borderline_count} reference vectors to #{output_path}...")
File.write!(output_path, Nx.serialize({border_vectors, border_labels}))

info = File.stat!(output_path)
IO.puts("Saved KNN specialist data (#{div(info.size, 1024)} KB)")
