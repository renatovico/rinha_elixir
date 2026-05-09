#!/usr/bin/env elixir
# Train a neural network from references.json.gz and save only the params.
#
# Output: Nx.serialize(params) — just the learned weights.
# The model architecture lives in Rinha.Model.build/0.
#
# Usage:
#   mix run --no-start priv/preprocess.exs <input.json.gz> <output.bin>

{:ok, _} = Application.ensure_all_started(:exla)
Nx.default_backend(EXLA.Backend)
Nx.Defn.default_options(compiler: EXLA)

[input_path, output_path] = System.argv()

IO.puts("Reading #{input_path}...")

data =
  input_path
  |> File.read!()
  |> :zlib.gunzip()
  |> Jason.decode!()

n = length(data)
IO.puts("#{n} entries total")

{fraud, legit} = Enum.split_with(data, fn e -> e["label"] == "fraud" end)
IO.puts("Original: #{length(fraud)} fraud, #{length(legit)} legit")

# Oversample fraud to balance classes (use all data)
fraud_oversampled =
  Stream.cycle(fraud)
  |> Enum.take(length(legit))

data = Enum.shuffle(fraud_oversampled ++ legit)

n = length(data)
IO.puts("Building tensors from #{n} entries (balanced)...")

vectors =
  data
  |> Enum.map(fn entry ->
    v = entry["vector"]
    # v is [amount, installments, amount_vs_avg, hour, dow, min_since, km_last, km_home,
    #       tx_count, is_online, card_present, unknown_merchant, mcc_risk, merchant_avg]
    [amount, _inst, amount_vs_avg, _hour, _dow, _min_since, _km_last, km_home,
     tx_count, is_online, card_present, unknown_merchant, mcc_risk, _merchant_avg] = v

    v ++ [
      amount * unknown_merchant,
      amount_vs_avg * is_online,
      amount * mcc_risk,
      km_home * unknown_merchant,
      is_online * unknown_merchant,
      amount_vs_avg * unknown_merchant,
      tx_count * amount,
      (1.0 - card_present) * km_home
    ]
  end)
  |> Nx.tensor(type: :f32)

labels =
  data
  |> Enum.map(fn entry -> if entry["label"] == "fraud", do: 1.0, else: 0.0 end)
  |> Nx.tensor(type: :f32)
  |> Nx.reshape({:auto, 1})

data = nil
:erlang.garbage_collect()

IO.puts("Vectors: #{inspect(Nx.shape(vectors))}, Labels: #{inspect(Nx.shape(labels))}")

# Build model
model = Rinha.Model.build()

# Batched training data
batch_size = 1024

train_data =
  Stream.zip(
    Nx.to_batched(vectors, batch_size),
    Nx.to_batched(labels, batch_size)
  )
  |> Stream.map(fn {x, y} -> {%{"data" => x}, y} end)
  |> Enum.to_list()

IO.puts("Training: #{length(train_data)} batches/epoch, batch_size=#{batch_size}")

# Focal loss: focuses training on hard-to-classify samples
# gamma=2 down-weights easy examples, alpha=0.6 slightly favors fraud detection
focal_loss = fn y_pred, y_true ->
  eps = 1.0e-7
  gamma = 2.0
  alpha = 0.6
  y_pred = Nx.clip(y_pred, eps, 1.0 - eps)

  # p_t = p if y=1, (1-p) if y=0
  p_t = Nx.add(Nx.multiply(y_true, y_pred), Nx.multiply(Nx.subtract(1.0, y_true), Nx.subtract(1.0, y_pred)))
  # alpha_t = alpha if y=1, (1-alpha) if y=0
  alpha_t = Nx.add(Nx.multiply(y_true, alpha), Nx.multiply(Nx.subtract(1.0, y_true), 1.0 - alpha))

  loss = Nx.negate(Nx.multiply(alpha_t, Nx.multiply(Nx.pow(Nx.subtract(1.0, p_t), gamma), Nx.log(p_t))))
  Nx.mean(loss)
end

loop =
  Axon.Loop.trainer(model, focal_loss, :adam)
  |> Axon.Loop.metric(:accuracy)

trained_state =
  Axon.Loop.run(loop, train_data, Axon.ModelState.empty(), epochs: 30)

# Extract and save params only
params = trained_state.data

IO.puts("Saving params to #{output_path}...")
File.write!(output_path, Nx.serialize(params))

info = File.stat!(output_path)
IO.puts("Saved model params (#{div(info.size, 1024)} KB)")
