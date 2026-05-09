defmodule Rinha.Model do
  @moduledoc """
  Axon models for fraud detection.
  Model 1: broad classifier. Model 2: specialist for borderline cases.
  """

  def build do
    Axon.input("data", shape: {nil, 22})
    |> Axon.dense(128, activation: :relu, name: "dense_0")
    |> Axon.dropout(rate: 0.2)
    |> Axon.dense(64, activation: :relu, name: "dense_1")
    |> Axon.dense(1, activation: :sigmoid, name: "dense_2")
  end

  def build_specialist do
    Axon.input("data", shape: {nil, 22})
    |> Axon.dense(64, activation: :relu, name: "dense_0")
    |> Axon.dropout(rate: 0.1)
    |> Axon.dense(32, activation: :relu, name: "dense_1")
    |> Axon.dropout(rate: 0.1)
    |> Axon.dense(16, activation: :relu, name: "dense_2")
    |> Axon.dense(1, activation: :sigmoid, name: "dense_3")
  end
end
