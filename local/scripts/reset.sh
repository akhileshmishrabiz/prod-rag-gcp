#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
docker compose -f "$ROOT/local/compose.yaml" down --volumes --remove-orphans
printf 'local Compose containers, network, and named volumes removed\n'
