defmodule Rinha.VectorTransformer do
  @moduledoc """
  Transforms a fraud-score request payload into a 22-dimensional normalized vector
  following the rules in DETECTION_RULES.md.
  Features 0-13: raw features, 14-21: interaction features.
  """

  @mcc_risk %{
    "5411" => 0.15,
    "5812" => 0.30,
    "5912" => 0.20,
    "5944" => 0.45,
    "7801" => 0.80,
    "7802" => 0.75,
    "7995" => 0.85,
    "4511" => 0.35,
    "5311" => 0.25,
    "5999" => 0.50
  }

  @max_amount 10_000.0
  @max_installments 12.0
  @amount_vs_avg_ratio 10.0
  @max_minutes 1_440.0
  @max_km 1_000.0
  @max_tx_count_24h 20.0
  @max_merchant_avg_amount 10_000.0

  def transform(payload) do
    transaction = payload["transaction"]
    customer = payload["customer"]
    merchant = payload["merchant"]
    terminal = payload["terminal"]
    last_tx = payload["last_transaction"]

    amount = transaction["amount"] || 0.0
    installments = transaction["installments"] || 0
    requested_at = transaction["requested_at"]
    avg_amount = customer["avg_amount"] || 1.0
    tx_count_24h = customer["tx_count_24h"] || 0
    known_merchants = customer["known_merchants"] || []
    merchant_id = merchant["id"]
    mcc = merchant["mcc"]
    merchant_avg = merchant["avg_amount"] || 0.0
    is_online = terminal["is_online"]
    card_present = terminal["card_present"]
    km_from_home = terminal["km_from_home"] || 0.0

    {hour, day_of_week} = parse_datetime(requested_at)

    {minutes_since, km_from_last} = last_transaction_dims(last_tx, requested_at)

    amount_norm = clamp(amount / @max_amount)
    installments_norm = clamp(installments / @max_installments)
    amount_vs_avg = clamp(amount / avg_amount / @amount_vs_avg_ratio)
    km_home_norm = clamp(km_from_home / @max_km)
    tx_count_norm = clamp(tx_count_24h / @max_tx_count_24h)
    is_online_f = bool_to_float(is_online)
    card_present_f = bool_to_float(card_present)
    unknown_merchant = if(merchant_id in known_merchants, do: 1.0, else: 0.0)
    mcc_risk_val = Map.get(@mcc_risk, mcc, 0.5)
    merchant_avg_norm = clamp(merchant_avg / @max_merchant_avg_amount)

    [
      # 0-13: raw features
      amount_norm,
      installments_norm,
      amount_vs_avg,
      hour / 23.0,
      day_of_week / 6.0,
      minutes_since,
      km_from_last,
      km_home_norm,
      tx_count_norm,
      is_online_f,
      card_present_f,
      unknown_merchant,
      mcc_risk_val,
      merchant_avg_norm,
      # 14-21: interaction features
      amount_norm * unknown_merchant,
      amount_vs_avg * is_online_f,
      amount_norm * mcc_risk_val,
      km_home_norm * unknown_merchant,
      is_online_f * unknown_merchant,
      amount_vs_avg * unknown_merchant,
      tx_count_norm * amount_norm,
      (1.0 - card_present_f) * km_home_norm
    ]
  end

  defp clamp(x) when x < 0.0, do: 0.0
  defp clamp(x) when x > 1.0, do: 1.0
  defp clamp(x), do: x + 0.0

  defp bool_to_float(true), do: 1.0
  defp bool_to_float(_), do: 0.0

  defp last_transaction_dims(nil, _requested_at), do: {-1.0, -1.0}

  defp last_transaction_dims(last_tx, requested_at) do
    last_ts = last_tx["timestamp"]
    km = last_tx["km_from_current"] || 0.0

    minutes =
      case {parse_iso8601(requested_at), parse_iso8601(last_ts)} do
        {{:ok, req_dt}, {:ok, last_dt}} ->
          DateTime.diff(req_dt, last_dt, :second) / 60.0

        _ ->
          0.0
      end

    {clamp(minutes / @max_minutes), clamp(km / @max_km)}
  end

  defp parse_datetime(nil), do: {0.0, 0.0}

  defp parse_datetime(iso_string) do
    case parse_iso8601(iso_string) do
      {:ok, dt} ->
        hour = dt.hour + 0.0
        # Date.day_of_week returns 1=Monday..7=Sunday, we need 0=Monday..6=Sunday
        dow = Date.day_of_week(dt) - 1 + 0.0
        {hour, dow}

      _ ->
        {0.0, 0.0}
    end
  end

  defp parse_iso8601(str) when is_binary(str) do
    case DateTime.from_iso8601(str) do
      {:ok, dt, _offset} -> {:ok, dt}
      _ -> :error
    end
  end

  defp parse_iso8601(_), do: :error
end
