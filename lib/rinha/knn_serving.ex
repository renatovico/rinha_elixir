defmodule Rinha.KnnServing do
  @moduledoc """
  Brute-force KNN specialist using EXLA JIT.
  Computes euclidean distances via matrix ops and returns majority vote of k=5 neighbors.
  """
  import Nx.Defn

  @k 5

  def build(ref_vectors, ref_labels) do
    ref_vectors = Nx.backend_copy(ref_vectors, Nx.BinaryBackend)
    ref_labels = Nx.backend_copy(ref_labels, Nx.BinaryBackend)
    refs = %{"vectors" => ref_vectors, "labels" => ref_labels}

    Nx.Serving.jit(fn batch -> knn_predict(refs, batch) end, compiler: EXLA)
    |> Nx.Serving.client_preprocessing(fn vector_list ->
      tensor = Nx.tensor(vector_list, type: :f32, backend: Nx.BinaryBackend) |> Nx.reshape({22})
      {Nx.Batch.stack([tensor]), :ok}
    end)
    |> Nx.Serving.client_postprocessing(fn {output, _metadata}, :ok ->
      output |> Nx.squeeze() |> Nx.to_number()
    end)
  end

  defn knn_predict(refs, query) do
    ref_vectors = refs["vectors"]
    ref_labels = refs["labels"]
    n = Nx.axis_size(ref_labels, 0)

    # query: {batch, 22}, ref_vectors: {n, 22}
    # Euclidean distance: ||q - r||^2 = ||q||^2 - 2*q·r^T + ||r||^2
    q_sq = Nx.sum(query * query, axes: [1], keep_axes: true)
    r_sq = Nx.sum(ref_vectors * ref_vectors, axes: [1])
    dot = Nx.dot(query, Nx.transpose(ref_vectors))
    distances = q_sq - 2 * dot + r_sq

    # Find k nearest: get the k smallest distances using top-k via negation
    neg_distances = Nx.negate(distances)
    {_top_vals, top_idx} = Nx.top_k(neg_distances, k: @k)

    # Gather labels without Nx.take: one-hot index then dot with labels
    # top_idx: {batch, k}, ref_labels: {n}
    flat_idx = Nx.reshape(top_idx, {:auto})
    one_hot = Nx.equal(Nx.reshape(flat_idx, {:auto, 1}), Nx.iota({1, n}))
    gathered = Nx.dot(Nx.as_type(one_hot, :f32), ref_labels)
    nearest_labels = Nx.reshape(gathered, {:auto, @k})

    # Fraud ratio per batch entry
    Nx.sum(nearest_labels, axes: [1]) / @k
  end
end
