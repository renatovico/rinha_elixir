defmodule Rinha.FraudScorer do
  @moduledoc """
  Cascade fraud scoring: NN (fast) then KNN specialist for borderline cases.
  """

  @responses %{
    0 => ~s({"approved":true,"fraud_score":0.0}),
    1 => ~s({"approved":true,"fraud_score":0.2}),
    2 => ~s({"approved":true,"fraud_score":0.4}),
    3 => ~s({"approved":false,"fraud_score":0.6}),
    4 => ~s({"approved":false,"fraud_score":0.8}),
    5 => ~s({"approved":false,"fraud_score":1.0})
  }

  def score(payload) do
    t0 = System.monotonic_time(:microsecond)

    vector = Rinha.VectorTransformer.transform(payload)
    t1 = System.monotonic_time(:microsecond)

    nn_serving = :persistent_term.get(:nn_serving)
    fraud_prob = Nx.Serving.run(nn_serving, vector)
    t2 = System.monotonic_time(:microsecond)

    {result, knn_us} =
      cond do
        # High confidence fraud
        fraud_prob >= 0.80 -> {Map.fetch!(@responses, 5), 0}
        # Borderline - ask KNN specialist
        fraud_prob >= 0.50 ->
          tk0 = System.monotonic_time(:microsecond)
          knn_serving = :persistent_term.get(:knn_serving)
          knn_fraud_ratio = Nx.Serving.run(knn_serving, vector)
          tk1 = System.monotonic_time(:microsecond)
          r =
            if knn_fraud_ratio >= 0.5 do
              Map.fetch!(@responses, 5)
            else
              if knn_fraud_ratio >= 0.3 do
                Map.fetch!(@responses, 3)
              else
                Map.fetch!(@responses, 2)
              end
            end
          {r, tk1 - tk0}
        fraud_prob >= 0.30 -> {Map.fetch!(@responses, 2), 0}
        fraud_prob >= 0.12 -> {Map.fetch!(@responses, 1), 0}
        true -> {Map.fetch!(@responses, 0), 0}
      end

    t3 = System.monotonic_time(:microsecond)
    total = t3 - t0
    transform_us = t1 - t0
    nn_us = t2 - t1

    # Sample 1 in 100 requests to avoid log flooding
    counter = :atomics.add_get(:persistent_term.get(:prof_counter), 1, 1)
    if rem(counter, 100) == 0 do
      require Logger
      Logger.info("[PROF] total=#{total}us transform=#{transform_us}us nn=#{nn_us}us knn=#{knn_us}us prob=#{Float.round(fraud_prob, 3)} borderline=#{knn_us > 0}")
    end

    result
  end
end
