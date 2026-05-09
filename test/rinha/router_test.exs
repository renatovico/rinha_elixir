defmodule Rinha.RouterTest do
  use ExUnit.Case, async: true
  import Plug.Test
  import Plug.Conn

  alias Rinha.Router

  @opts Router.init([])

  describe "GET /ready" do
    test "returns 503 when not ready" do
      # Ensure :rinha_ready is not set (or false)
      try do
        :persistent_term.put(:rinha_ready, false)
      rescue
        _ -> :ok
      end

      conn = conn(:get, "/ready") |> Router.call(@opts)
      assert conn.status == 503
    end

    test "returns 200 when ready" do
      :persistent_term.put(:rinha_ready, true)

      conn = conn(:get, "/ready") |> Router.call(@opts)
      assert conn.status == 200
    end
  end

  describe "POST /fraud-score" do
    test "returns 400 for invalid JSON" do
      conn =
        conn(:post, "/fraud-score", "not json")
        |> put_req_header("content-type", "application/json")
        |> Router.call(@opts)

      assert conn.status == 400
      body = Jason.decode!(conn.resp_body)
      assert body["error"] == "invalid json"
    end
  end

  describe "unknown routes" do
    test "returns 404" do
      conn = conn(:get, "/unknown") |> Router.call(@opts)
      assert conn.status == 404
    end

    test "POST to unknown path returns 404" do
      conn = conn(:post, "/anything") |> Router.call(@opts)
      assert conn.status == 404
    end
  end
end
