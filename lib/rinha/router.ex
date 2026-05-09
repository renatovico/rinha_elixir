defmodule Rinha.Router do
  use Plug.Router

  plug :match
  plug :dispatch

  get "/ready" do
    if :persistent_term.get(:rinha_ready, false) do
      send_resp(conn, 200, "OK")
    else
      send_resp(conn, 503, "NOT READY")
    end
  end

  post "/fraud-score" do
    unless :persistent_term.get(:rinha_ready, false) do
      send_resp(conn, 503, ~s({"error":"warming up"}))
    else
      {:ok, body, conn} = Plug.Conn.read_body(conn)

      case Jason.decode(body) do
        {:ok, payload} ->
          response = Rinha.FraudScorer.score(payload)

          conn
          |> put_resp_content_type("application/json")
          |> send_resp(200, response)

        {:error, _} ->
          send_resp(conn, 400, ~s({"error":"invalid json"}))
      end
    end
  end

  match _ do
    send_resp(conn, 404, "")
  end
end
