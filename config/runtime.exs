import Config

config :rinha,
  port: String.to_integer(System.get_env("PORT") || "4000"),
  shared_data_path:
    System.get_env("SHARED_DATA_PATH") ||
      Path.join(File.cwd!(), "priv/model_params.bin"),
  knn_data_path:
    System.get_env("KNN_DATA_PATH") ||
      Path.join(File.cwd!(), "priv/knn_specialist.bin")
