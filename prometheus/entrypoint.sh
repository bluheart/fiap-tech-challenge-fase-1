#!/bin/sh
set -e

# Create prometheus directory if it doesn't exist
mkdir -p /prometheus
chown nobody:nobody /prometheus

# Run prometheus as nobody user
exec nobody /bin/prometheus "$@"