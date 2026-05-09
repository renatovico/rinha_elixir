defmodule Rinha.VectorTransformerTest do
  use ExUnit.Case, async: true

  alias Rinha.VectorTransformer

  # Example 1 from DETECTION_RULES.md: legitimate transaction
  @legit_payload %{
    "id" => "tx-1329056812",
    "transaction" => %{
      "amount" => 41.12,
      "installments" => 2,
      "requested_at" => "2026-03-11T18:45:53Z"
    },
    "customer" => %{
      "avg_amount" => 82.24,
      "tx_count_24h" => 3,
      "known_merchants" => ["MERC-003", "MERC-016"]
    },
    "merchant" => %{
      "id" => "MERC-016",
      "mcc" => "5411",
      "avg_amount" => 60.25
    },
    "terminal" => %{
      "is_online" => false,
      "card_present" => true,
      "km_from_home" => 29.23
    },
    "last_transaction" => nil
  }

  @legit_expected [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]

  # Example 2 from DETECTION_RULES.md: fraudulent transaction
  @fraud_payload %{
    "id" => "tx-3330991687",
    "transaction" => %{
      "amount" => 9505.97,
      "installments" => 10,
      "requested_at" => "2026-03-14T05:15:12Z"
    },
    "customer" => %{
      "avg_amount" => 81.28,
      "tx_count_24h" => 20,
      "known_merchants" => ["MERC-008", "MERC-007", "MERC-005"]
    },
    "merchant" => %{
      "id" => "MERC-068",
      "mcc" => "7802",
      "avg_amount" => 54.86
    },
    "terminal" => %{
      "is_online" => false,
      "card_present" => true,
      "km_from_home" => 952.27
    },
    "last_transaction" => nil
  }

  @fraud_expected [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]

  describe "transform/1 - Example 1 (legit, last_transaction=nil)" do
    test "produces correct 14-dimensional vector" do
      result = VectorTransformer.transform(@legit_payload)
      assert length(result) == 14
      assert_vectors_close(result, @legit_expected, 0.001)
    end

    test "dim 0: amount normalized" do
      [dim0 | _] = VectorTransformer.transform(@legit_payload)
      # 41.12 / 10_000 = 0.004112
      assert_in_delta dim0, 0.0041, 0.001
    end

    test "dim 1: installments normalized" do
      result = VectorTransformer.transform(@legit_payload)
      # 2 / 12 = 0.1667
      assert_in_delta Enum.at(result, 1), 0.1667, 0.001
    end

    test "dim 2: amount_vs_avg normalized" do
      result = VectorTransformer.transform(@legit_payload)
      # (41.12 / 82.24) / 10 = 0.05
      assert_in_delta Enum.at(result, 2), 0.05, 0.001
    end

    test "dim 3: hour_of_day normalized" do
      result = VectorTransformer.transform(@legit_payload)
      # 18 / 23 = 0.7826
      assert_in_delta Enum.at(result, 3), 0.7826, 0.001
    end

    test "dim 4: day_of_week normalized" do
      result = VectorTransformer.transform(@legit_payload)
      # 2026-03-11 is Wednesday, day_of_week=3 (1-indexed), so (3-1)/6 = 0.3333
      assert_in_delta Enum.at(result, 4), 0.3333, 0.001
    end

    test "dims 5-6: last_transaction nil → sentinel -1" do
      result = VectorTransformer.transform(@legit_payload)
      assert Enum.at(result, 5) == -1.0
      assert Enum.at(result, 6) == -1.0
    end

    test "dim 7: km_from_home normalized" do
      result = VectorTransformer.transform(@legit_payload)
      # 29.23 / 1000 = 0.02923
      assert_in_delta Enum.at(result, 7), 0.0292, 0.001
    end

    test "dim 8: tx_count_24h normalized" do
      result = VectorTransformer.transform(@legit_payload)
      # 3 / 20 = 0.15
      assert_in_delta Enum.at(result, 8), 0.15, 0.001
    end

    test "dim 9: is_online = false → 0.0" do
      result = VectorTransformer.transform(@legit_payload)
      assert Enum.at(result, 9) == 0.0
    end

    test "dim 10: card_present = true → 1.0" do
      result = VectorTransformer.transform(@legit_payload)
      assert Enum.at(result, 10) == 1.0
    end

    test "dim 11: known merchant → 0.0" do
      result = VectorTransformer.transform(@legit_payload)
      # MERC-016 is in known_merchants
      assert Enum.at(result, 11) == 0.0
    end

    test "dim 12: mcc_risk for 5411 → 0.15" do
      result = VectorTransformer.transform(@legit_payload)
      assert_in_delta Enum.at(result, 12), 0.15, 0.001
    end

    test "dim 13: merchant_avg_amount normalized" do
      result = VectorTransformer.transform(@legit_payload)
      # 60.25 / 10_000 = 0.006025
      assert_in_delta Enum.at(result, 13), 0.006, 0.001
    end
  end

  describe "transform/1 - Example 2 (fraud, last_transaction=nil)" do
    test "produces correct 14-dimensional vector" do
      result = VectorTransformer.transform(@fraud_payload)
      assert length(result) == 14
      assert_vectors_close(result, @fraud_expected, 0.001)
    end

    test "dim 0: high amount clamped to <= 1.0" do
      result = VectorTransformer.transform(@fraud_payload)
      # 9505.97 / 10_000 = 0.950597
      assert_in_delta Enum.at(result, 0), 0.9506, 0.001
    end

    test "dim 2: amount_vs_avg clamped to 1.0" do
      result = VectorTransformer.transform(@fraud_payload)
      # (9505.97 / 81.28) / 10 = 11.69 → clamped to 1.0
      assert Enum.at(result, 2) == 1.0
    end

    test "dim 8: tx_count_24h at max clamped to 1.0" do
      result = VectorTransformer.transform(@fraud_payload)
      # 20 / 20 = 1.0
      assert Enum.at(result, 8) == 1.0
    end

    test "dim 11: unknown merchant → 1.0" do
      result = VectorTransformer.transform(@fraud_payload)
      # MERC-068 is NOT in known_merchants
      assert Enum.at(result, 11) == 1.0
    end

    test "dim 12: mcc_risk for 7802 → 0.75" do
      result = VectorTransformer.transform(@fraud_payload)
      assert_in_delta Enum.at(result, 12), 0.75, 0.001
    end
  end

  describe "transform/1 - with last_transaction" do
    @payload_with_last %{
      "id" => "tx-3576980410",
      "transaction" => %{
        "amount" => 384.88,
        "installments" => 3,
        "requested_at" => "2026-03-11T20:23:35Z"
      },
      "customer" => %{
        "avg_amount" => 769.76,
        "tx_count_24h" => 3,
        "known_merchants" => ["MERC-009", "MERC-009", "MERC-001", "MERC-001"]
      },
      "merchant" => %{
        "id" => "MERC-001",
        "mcc" => "5912",
        "avg_amount" => 298.95
      },
      "terminal" => %{
        "is_online" => false,
        "card_present" => true,
        "km_from_home" => 13.7090520965
      },
      "last_transaction" => %{
        "timestamp" => "2026-03-11T14:58:35Z",
        "km_from_current" => 18.8626479774
      }
    }

    test "dims 5-6: computed from last_transaction" do
      result = VectorTransformer.transform(@payload_with_last)
      # minutes_since: diff between 20:23:35 and 14:58:35 = 325 minutes
      # 325 / 1440 = 0.2257
      assert_in_delta Enum.at(result, 5), 0.2257, 0.001

      # km_from_last: 18.86 / 1000 = 0.01887
      assert_in_delta Enum.at(result, 6), 0.0189, 0.001
    end

    test "dim 11: known merchant in list → 0.0" do
      result = VectorTransformer.transform(@payload_with_last)
      # MERC-001 is in known_merchants
      assert Enum.at(result, 11) == 0.0
    end

    test "dim 12: mcc 5912 → 0.20" do
      result = VectorTransformer.transform(@payload_with_last)
      assert_in_delta Enum.at(result, 12), 0.20, 0.001
    end
  end

  describe "transform/1 - edge cases" do
    test "unknown MCC defaults to 0.5" do
      payload = %{
        "transaction" => %{"amount" => 100.0, "installments" => 1, "requested_at" => "2026-01-01T12:00:00Z"},
        "customer" => %{"avg_amount" => 100.0, "tx_count_24h" => 1, "known_merchants" => []},
        "merchant" => %{"id" => "MERC-001", "mcc" => "9999", "avg_amount" => 50.0},
        "terminal" => %{"is_online" => true, "card_present" => false, "km_from_home" => 0.0},
        "last_transaction" => nil
      }

      result = VectorTransformer.transform(payload)
      assert Enum.at(result, 12) == 0.5
    end

    test "very high amount gets clamped to 1.0" do
      payload = %{
        "transaction" => %{"amount" => 999_999.0, "installments" => 1, "requested_at" => "2026-01-01T12:00:00Z"},
        "customer" => %{"avg_amount" => 100.0, "tx_count_24h" => 0, "known_merchants" => []},
        "merchant" => %{"id" => "M1", "mcc" => "5411", "avg_amount" => 0.0},
        "terminal" => %{"is_online" => false, "card_present" => true, "km_from_home" => 0.0},
        "last_transaction" => nil
      }

      result = VectorTransformer.transform(payload)
      assert Enum.at(result, 0) == 1.0
    end

    test "output is always a list of 14 floats" do
      payload = %{
        "transaction" => %{"amount" => 0.0, "installments" => 0, "requested_at" => "2026-06-15T00:00:00Z"},
        "customer" => %{"avg_amount" => 1.0, "tx_count_24h" => 0, "known_merchants" => []},
        "merchant" => %{"id" => "M1", "mcc" => "5411", "avg_amount" => 0.0},
        "terminal" => %{"is_online" => false, "card_present" => false, "km_from_home" => 0.0},
        "last_transaction" => nil
      }

      result = VectorTransformer.transform(payload)
      assert length(result) == 14
      assert Enum.all?(result, &is_float/1)
    end

    test "is_online = true → 1.0" do
      payload = %{
        "transaction" => %{"amount" => 100.0, "installments" => 1, "requested_at" => "2026-01-01T12:00:00Z"},
        "customer" => %{"avg_amount" => 100.0, "tx_count_24h" => 1, "known_merchants" => []},
        "merchant" => %{"id" => "M1", "mcc" => "5411", "avg_amount" => 50.0},
        "terminal" => %{"is_online" => true, "card_present" => false, "km_from_home" => 0.0},
        "last_transaction" => nil
      }

      result = VectorTransformer.transform(payload)
      assert Enum.at(result, 9) == 1.0
      assert Enum.at(result, 10) == 0.0
    end
  end

  # Helper to compare two vectors element-wise with tolerance
  defp assert_vectors_close(actual, expected, tolerance) do
    Enum.zip(actual, expected)
    |> Enum.with_index()
    |> Enum.each(fn {{a, e}, i} ->
      assert_in_delta a, e, tolerance,
        "dim #{i}: expected #{e}, got #{a}"
    end)
  end
end
