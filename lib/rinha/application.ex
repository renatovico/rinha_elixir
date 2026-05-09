defmodule Rinha.Application do
  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    path = Application.get_env(:rinha, :shared_data_path, "/app/references.bin")

    bandit_opts =
      case System.get_env("SOCKET_PATH") do
        nil ->
          port = Application.get_env(:rinha, :port, 4000)
          [plug: Rinha.Router, port: port]

        socket_path ->
          [plug: Rinha.Router, port: 0, ip: {:local, socket_path}]
      end

    Logger.info("Loading model params from #{path}...")
    params = File.read!(path) |> Nx.deserialize()

    knn_path = Application.get_env(:rinha, :knn_data_path, "/app/knn_specialist.bin")
    Logger.info("Loading KNN specialist from #{knn_path}...")
    {ref_vectors, ref_labels} = File.read!(knn_path) |> Nx.deserialize()

    :persistent_term.put(:prof_counter, :atomics.new(1, signed: false))

    # NN: inline serving (no process, no queue)
    Logger.info("Building NN serving (inline)...")
    nn_serving = Rinha.FraudServing.build(params)
    warmup_vector = List.duplicate(0.0, 22)
    Logger.info("Warming up NN...")
    Nx.Serving.run(nn_serving, warmup_vector)
    :persistent_term.put(:nn_serving, nn_serving)

    # KNN: brute-force inline serving
    Logger.info("Building KNN serving (inline)...")
    knn_serving = Rinha.KnnServing.build(ref_vectors, ref_labels)
    Logger.info("Warming up KNN...")
    Nx.Serving.run(knn_serving, warmup_vector)
    :persistent_term.put(:knn_serving, knn_serving)
    :persistent_term.put(:rinha_ready, true)

    # Signal readiness via file (for Docker healthcheck)
    ready_file = System.get_env("READY_FILE", "/tmp/ready")
    File.write!(ready_file, "ok")
    Logger.info("Servings warmed up, ready!")

    children = [{Bandit, bandit_opts}]

    opts = [strategy: :one_for_one, name: Rinha.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
