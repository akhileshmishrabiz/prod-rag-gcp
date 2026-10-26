#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
COMPOSE="$ROOT/local/compose.yaml"

docker compose -f "$COMPOSE" up --build --wait
python3 "$ROOT/local/scripts/seed.py"
python3 "$ROOT/local/scripts/smoke.py"
