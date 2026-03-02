#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[wtc] project root: ${ROOT_DIR}"
find "${ROOT_DIR}" -maxdepth 2 -type d | sort
