import Config

config :rinha,
  port: 4000,
  shared_data_path: "/app/references.bin"

# EXLA for JIT-compiled neural net inference in all environments
config :nx, default_backend: EXLA.Backend
config :nx, default_defn_options: [compiler: EXLA]
