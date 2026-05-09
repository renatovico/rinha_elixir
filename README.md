# Rinha de Backend 2026 - Fraud Detection

Real-time fraud detection API built with Elixir, using a cascade ML approach for fast and accurate classification.

## Architecture

- **2 API instances** behind HAProxy load balancer
- **Cascade scoring**: Neural Network (fast path) → KNN specialist (borderline cases)
- **EXLA/XLA** compiled ML inference on CPU
- **Bandit** HTTP server over Unix sockets

## How It Works

1. **Neural Network** scores every transaction (~300-600µs). High confidence results (>0.80 fraud or <0.12) are returned immediately.
2. **KNN Specialist** is invoked only for borderline cases (NN score 0.50-0.80), comparing against 10K reference vectors from known borderline transactions (~600-1000µs extra).
3. Combined scoring achieves low false negatives (65 FN) while keeping latency under control (p99 ~34ms).

## Stack

- Elixir 1.19 / OTP 28
- EXLA (XLA CPU backend) for ML inference
- Bandit HTTP server
- HAProxy load balancer
- Docker Compose

## Running

```bash
docker compose build
docker compose up -d
```

## Testing

```bash
k6 run test/k6/test.js
```

## Score

- **Final Score: 2146.65**
- p99: 34.25ms
- FN: 65 | FP: 255
- Detection rate: 97.63%

