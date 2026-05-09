#!/bin/sh
set -e

# Remove stale socket file if it exists
if [ -n "$SOCKET_PATH" ] && [ -e "$SOCKET_PATH" ]; then
  rm -f "$SOCKET_PATH"
fi

# Remove stale ready file
rm -f /tmp/ready

exec /app/bin/rinha start
