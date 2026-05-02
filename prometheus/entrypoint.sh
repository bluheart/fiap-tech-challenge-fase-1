#!/bin/sh
set -e

# Run prometheus with all arguments passed to the script
exec /bin/prometheus "$@"