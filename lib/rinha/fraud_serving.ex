defmodule Rinha.FraudServing do
  @moduledoc """
  Nx.Serving for fraud prediction using trained neural networks.
  Supports both the primary model and the specialist model.
  """
  import Nx.Defn

  def build(params) do
    params = to_binary_backend(params)

    Nx.Serving.jit(fn batch -> predict(params, batch) end, compiler: EXLA)
    |> Nx.Serving.client_preprocessing(fn vector_list ->
      tensor = Nx.tensor(vector_list, type: :f32, backend: Nx.BinaryBackend) |> Nx.reshape({22})
      {Nx.Batch.stack([tensor]), :ok}
    end)
    |> Nx.Serving.client_postprocessing(fn {output, _metadata}, :ok ->
      output |> Nx.squeeze() |> Nx.to_number()
    end)
  end

  def build_specialist(params) do
    params = to_binary_backend(params)

    Nx.Serving.jit(fn input -> predict_specialist(params, input) end, compiler: EXLA)
    |> Nx.Serving.client_preprocessing(fn vector_list ->
      tensor = Nx.tensor(vector_list, type: :f32, backend: Nx.BinaryBackend) |> Nx.reshape({1, 22})
      {Nx.Batch.stack([tensor]), :ok}
    end)
    |> Nx.Serving.client_postprocessing(fn {output, _metadata}, :ok ->
      output |> Nx.squeeze() |> Nx.to_number()
    end)
  end

  defp to_binary_backend(%Nx.Tensor{} = t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  defp to_binary_backend(%{} = map) do
    Map.new(map, fn {k, v} -> {k, to_binary_backend(v)} end)
  end

  defp to_binary_backend(other), do: other

  # Forward pass for primary model (128→64→1)
  defn predict(params, input) do
    input
    |> dense(params["dense_0"])
    |> Nx.max(0)
    |> dense(params["dense_1"])
    |> Nx.max(0)
    |> dense(params["dense_2"])
    |> Nx.sigmoid()
  end

  # Forward pass for specialist model (64→32→16→1)
  defnp predict_specialist(params, input) do
    input
    |> dense(params["dense_0"])
    |> Nx.max(0)
    |> dense(params["dense_1"])
    |> Nx.max(0)
    |> dense(params["dense_2"])
    |> Nx.max(0)
    |> dense(params["dense_3"])
    |> Nx.sigmoid()
  end

  defnp dense(input, layer_params) do
    Nx.dot(input, layer_params["kernel"]) |> Nx.add(layer_params["bias"])
  end
end
