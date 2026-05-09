# Profile the fraud scoring pipeline locally
# Usage: mix run priv/profile.exs
# The app starts automatically and creates the serving processes.

alias Rinha.VectorTransformer

IO.puts("=== Serving processes already started by application ===")
IO.puts("  Checking NNServing: #{inspect(Process.whereis(Rinha.NNServing))}")
IO.puts("  Checking KNNServing: #{inspect(Process.whereis(Rinha.KNNServing))}")

# Sample payloads that will trigger different code paths
payloads = [
  # Likely legit (low score)
  %{"amount" => 50.0, "merchant" => %{"mcc" => "5411", "name" => "Grocery", "lat" => -23.5, "long" => -46.6},
    "card" => %{"first_activity" => "2020-01-01T00:00:00Z"}, "transaction" => %{"time" => "2024-06-15T14:00:00Z"},
    "customer" => %{"zip_code" => "01001"}},
  # Likely fraud (high score)
  %{"amount" => 9500.0, "merchant" => %{"mcc" => "7995", "name" => "Casino Online", "lat" => 55.0, "long" => 37.0},
    "card" => %{"first_activity" => "2024-06-14T00:00:00Z"}, "transaction" => %{"time" => "2024-06-15T03:30:00Z"},
    "customer" => %{"zip_code" => "99999"}},
  # Borderline (will trigger KNN)
  %{"amount" => 800.0, "merchant" => %{"mcc" => "5812", "name" => "Restaurant", "lat" => -23.55, "long" => -46.65},
    "card" => %{"first_activity" => "2023-01-01T00:00:00Z"}, "transaction" => %{"time" => "2024-06-15T22:00:00Z"},
    "customer" => %{"zip_code" => "04001"}}
]

IO.puts("\n=== Warmup (JIT compilation) ===")
for p <- payloads do
  vector = VectorTransformer.transform(p)
  result = Nx.Serving.batched_run(Rinha.NNServing, vector)
  IO.puts("  warmup nn prob=#{inspect(result)} vector=#{inspect(Enum.take(vector, 5))}")
  knn_result = Nx.Serving.batched_run(Rinha.KNNServing, vector)
  IO.puts("  warmup knn=#{inspect(knn_result)}")
end
IO.puts("Warmup done.")

IO.puts("\n=== Manual timing (single request, each path) ===")
for {p, label} <- Enum.zip(payloads, ["legit", "fraud", "borderline"]) do
  t0 = System.monotonic_time(:microsecond)
  vector = VectorTransformer.transform(p)
  t1 = System.monotonic_time(:microsecond)
  prob = Nx.Serving.batched_run(Rinha.NNServing, vector)
  t2 = System.monotonic_time(:microsecond)
  knn_result = Nx.Serving.batched_run(Rinha.KNNServing, vector)
  t3 = System.monotonic_time(:microsecond)

  IO.puts("  [#{label}] transform=#{t1-t0}us  nn=#{t2-t1}us  knn=#{t3-t2}us  total=#{t3-t0}us  prob=#{Float.round(prob, 4)}  knn=#{Float.round(knn_result, 4)}")
end

IO.puts("\n=== Concurrent load simulation (100 total requests) ===")
t_start = System.monotonic_time(:millisecond)

tasks = for i <- 1..100 do
  Task.async(fn ->
    p = Enum.at(payloads, rem(i, 3))
    t0 = System.monotonic_time(:microsecond)
    vector = VectorTransformer.transform(p)
    prob = Nx.Serving.batched_run(Rinha.NNServing, vector)
    used_knn = prob >= 0.55 and prob < 0.85
    if used_knn do
      Nx.Serving.batched_run(Rinha.KNNServing, vector)
    end
    t1 = System.monotonic_time(:microsecond)
    {t1 - t0, used_knn}
  end)
end

results = Task.await_many(tasks, 60_000)
t_end = System.monotonic_time(:millisecond)

durations = Enum.map(results, &elem(&1, 0))
knn_count = Enum.count(results, &elem(&1, 1))
sorted = Enum.sort(durations)

IO.puts("  Total wall time: #{t_end - t_start}ms for 100 requests")
IO.puts("  KNN requests: #{knn_count}/100")
IO.puts("  Latencies (us): min=#{List.first(sorted)} median=#{Enum.at(sorted, 49)} p95=#{Enum.at(sorted, 94)} p99=#{Enum.at(sorted, 98)} max=#{List.last(sorted)}")
IO.puts("  Mean: #{Float.round(Enum.sum(durations) / 100, 0)}us")

IO.puts("\n=== Done ===")
