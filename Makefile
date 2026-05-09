.PHONY: help deps compile test preprocess run smoke load docker-build docker-up docker-down docker-test clean docker-run
.DEFAULT_GOAL := help

REFS_GZ   ?= $(HOME)/dev/rinha-de-backend-2026/resources/references.json.gz
REFS_BIN  := priv/references.bin
IMAGE     := renatovico/rinha-elixir:latest
BASE_URL  ?= http://localhost:4000

# ── Help ─────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Dev ──────────────────────────────────────────────

deps: ## Fetch dependencies
	mix deps.get

compile: deps ## Compile the project
	mix compile

test: compile ## Run ExUnit tests
	mix test

preprocess: compile ## Generate references.bin from .json.gz
	@if [ ! -f "$(REFS_GZ)" ]; then \
		echo "Error: $(REFS_GZ) not found. Set REFS_GZ=path/to/references.json.gz"; \
		exit 1; \
	fi
	mix run --no-start priv/preprocess.exs $(REFS_GZ) $(REFS_BIN)

run: compile $(REFS_BIN) ## Start app in iex
	iex -S mix

run-bg: compile $(REFS_BIN) ## Start app in background
	mix run --no-halt &

$(REFS_BIN):
	$(MAKE) preprocess

# ── k6 Tests ────────────────────────────────────────

smoke: ## k6 smoke test (local)
	k6 run -e BASE_URL=$(BASE_URL) test/k6/smoke.js

load: ## k6 load test (local)
	k6 run -e BASE_URL=$(BASE_URL) test/k6/test.js

smoke-docker: ## k6 smoke test (docker)
	k6 run test/k6/smoke.js

load-docker: ## k6 load test (docker)
	k6 run test/k6/test.js

# ── Docker ───────────────────────────────────────────

resources/references.json.gz:
	@mkdir -p resources
	@if [ ! -f "$(REFS_GZ)" ]; then \
		echo "Error: $(REFS_GZ) not found. Set REFS_GZ=path/to/references.json.gz"; \
		exit 1; \
	fi
	cp $(REFS_GZ) resources/references.json.gz

docker-build: resources/references.json.gz ## Build Docker image
	docker build -t $(IMAGE) .

docker-up: resources/references.json.gz ## Build & start Docker Compose
	docker compose up -d --build

docker-down: ## Stop Docker Compose
	docker compose down

docker-test: docker-up ## Smoke test in Docker
	@echo "Waiting for services..."
	@sleep 5
	k6 run test/k6/smoke.js

docker-load: docker-up ## Load test in Docker
	@echo "Waiting for services..."
	@sleep 5
	k6 run test/k6/test.js

docker-stats: ## Show container stats
	docker stats --no-stream

docker-run: docker-down docker-up ## Full cycle: rebuild, start, smoke + load test, teardown
	@echo "Waiting for services..."
	@sleep 5
	k6 run test/k6/smoke.js
	k6 run test/k6/test.js
	docker compose down

# ── Cleanup ──────────────────────────────────────────

clean: ## Remove build artifacts
	rm -f $(REFS_BIN)
	rm -rf _build deps
	rm -f resources/references.json.gz
