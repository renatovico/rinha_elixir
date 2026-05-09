########################################
# Stage 1: Build Elixir release
########################################
FROM elixir:1.19.5-otp-28 AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV MIX_ENV=prod
ENV XLA_TARGET=cpu
ENV EXLA_TARGET=host

WORKDIR /app

# Install hex + rebar
RUN mix local.hex --force && mix local.rebar --force

# Copy dependency files first for caching
COPY mix.exs mix.lock ./
RUN mix deps.get --only prod

# Compile deps first (heavy: EXLA C++) so editing lib/ doesn't bust this layer
COPY config config
RUN mix deps.compile

# Copy source code and compile only app
COPY lib lib
COPY priv priv
RUN mix compile

# Build release
RUN mix release

########################################
# Stage 2: Runtime image
########################################
FROM debian:trixie-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends libstdc++6 openssl ca-certificates locales && \
    sed -i '/^# *en_US.UTF-8 /s/^# *//' /etc/locale.gen && \
    locale-gen && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MIX_ENV=prod \
    XLA_TARGET=cpu \
    EXLA_TARGET=host \
    ELIXIR_ERL_OPTIONS="+fnu +MBas aobf +zdbbl 32768 +sbwt none +P 50000" \
    ERL_CRASH_DUMP_SECONDS=0

WORKDIR /app

COPY --from=builder /app/_build/prod/rel/rinha ./
COPY --from=builder /app/priv/model_params.bin /app/references.bin
COPY --from=builder /app/priv/knn_specialist.bin /app/knn_specialist.bin

# Script to optionally copy data to shared volume
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PORT=4000
ENV SHARED_DATA_PATH=/app/references.bin
ENV KNN_DATA_PATH=/app/knn_specialist.bin
ENV RELEASE_DISTRIBUTION=none

EXPOSE 4000

CMD ["/app/entrypoint.sh"]
