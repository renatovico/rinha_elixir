defmodule Rinha.FraudModel do
  @moduledoc """
  Defines the Axon neural network for fraud detection.
  Shared between training (preprocess) and inference (runtime).
  """

  def model do
    Axon.input("input", shape: {nil, 14})
    |> Axon.dense(128, activation: :relu)
    |> Axon.dense(64, activation: :relu)
    |> Axon.dense(32, activation: :relu)
    |> Axon.dense(1, activation: :sigmoid)
  end
end
