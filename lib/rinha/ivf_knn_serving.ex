defmodule Rinha.IvfKnnServing do
  @moduledoc """
  IVF (Inverted File Index) KNN specialist.
  Pre-clusters reference vectors with k-means. At query time, finds the nearest
  centroid in Elixir, then brute-forces KNN within that cluster via EXLA JIT.

  All clusters are padded to the same max size so EXLA compiles a single program.
  Invalid (padded) entries are masked out with huge distances.
  """
  import Nx.Defn

  @k 5

  def build(index) do
    centroids = Nx.backend_copy(index["centroids"], Nx.BinaryBackend)
    cluster_vectors = index["cluster_vectors"]  # {nc, max_cs, 22} — already padded
    cluster_labels = index["cluster_labels"]    # {nc, max_cs}
    cluster_sizes = index["cluster_sizes"]
    num_clusters = Nx.axis_size(centroids, 0)
    max_cs = Nx.axis_size(cluster_vectors, 1)

    # One serving per cluster — refs captured in closure.
    # All clusters padded to same max_cs so EXLA compiles once and caches.
    cluster_servings =
      for c <- 0..(num_clusters - 1) do
        vecs = Nx.slice(cluster_vectors, [c, 0, 0], [1, max_cs, 22])
               |> Nx.squeeze(axes: [0])
               |> Nx.backend_copy(Nx.BinaryBackend)
        labs = Nx.slice(cluster_labels, [c, 0], [1, max_cs])
               |> Nx.squeeze(axes: [0])
               |> Nx.backend_copy(Nx.BinaryBackend)
        size = Nx.slice(cluster_sizes, [c], [1])
               |> Nx.squeeze()
               |> Nx.backend_copy(Nx.BinaryBackend)
        refs = %{"vectors" => vecs, "labels" => labs, "size" => size}

        Nx.Serving.jit(fn batch -> knn_predict_masked(refs, batch) end, compiler: EXLA)
        |> Nx.Serving.client_preprocessing(fn vector_list ->
          tensor = Nx.tensor(vector_list, type: :f32, backend: Nx.BinaryBackend) |> Nx.reshape({22})
          {Nx.Batch.stack([tensor]), :ok}
        end)
        |> Nx.Serving.client_postprocessing(fn {output, _metadata}, :ok ->
          output |> Nx.squeeze() |> Nx.to_number()
        end)
      end

    # Pre-compute centroid squared norms for fast lookup
    c_sq = Nx.sum(Nx.multiply(centroids, centroids), axes: [1])

    %{centroids: centroids, c_sq: c_sq, cluster_servings: cluster_servings, num_clusters: num_clusters}
  end

  def score(ivf, vector_list) do
    # 1. Find nearest cluster centroid (pre-computed norms)
    query = Nx.tensor(vector_list, type: :f32, backend: Nx.BinaryBackend) |> Nx.reshape({1, 22})

    q_sq = Nx.sum(Nx.multiply(query, query), axes: [1], keep_axes: true)
    dot = Nx.dot(query, Nx.transpose(ivf.centroids))
    dists = Nx.add(Nx.subtract(q_sq, Nx.multiply(2, dot)), ivf.c_sq) |> Nx.squeeze()
    nearest = Nx.argmin(dists) |> Nx.to_number()

    # 2. Brute-force KNN in nearest cluster
    serving = Enum.at(ivf.cluster_servings, nearest)
    Nx.Serving.run(serving, vector_list)
  end

  defn knn_predict_masked(refs, query) do
    ref_vectors = refs["vectors"]  # {max_cs, 22}
    ref_labels = refs["labels"]    # {max_cs}
    cluster_size = refs["size"]    # scalar
    max_cs = Nx.axis_size(ref_labels, 0)

    # Euclidean distances
    q_sq = Nx.sum(query * query, axes: [1], keep_axes: true)
    r_sq = Nx.sum(ref_vectors * ref_vectors, axes: [1])
    dot = Nx.dot(query, Nx.transpose(ref_vectors))
    distances = q_sq - 2 * dot + r_sq

    # Mask padded entries with huge distance
    valid_mask = Nx.less(Nx.iota({1, max_cs}), cluster_size)
    big = Nx.tensor(1.0e18, type: :f32)
    masked_distances = Nx.select(valid_mask, distances, big)

    # Find k nearest
    {_top_vals, top_idx} = Nx.top_k(Nx.negate(masked_distances), k: @k)

    # Gather labels via one-hot
    flat_idx = Nx.reshape(top_idx, {:auto})
    one_hot = Nx.equal(Nx.reshape(flat_idx, {:auto, 1}), Nx.iota({1, max_cs}))
    gathered = Nx.dot(Nx.as_type(one_hot, :f32), ref_labels)
    nearest_labels = Nx.reshape(gathered, {:auto, @k})

    Nx.sum(nearest_labels, axes: [1]) / @k
  end
end
