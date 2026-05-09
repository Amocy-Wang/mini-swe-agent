#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="${1:-fea-unified:latest}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command not found. Please install Docker Desktop first." >&2
  exit 1
fi

echo "[build] building image: ${IMAGE_TAG}"
docker build \
  --pull \
  --tag "${IMAGE_TAG}" \
  --file "${SCRIPT_DIR}/Dockerfile" \
  "${SCRIPT_DIR}"

echo "[done] image built: ${IMAGE_TAG}"
echo "[hint] quick check: docker run --rm -it ${IMAGE_TAG} bash -lc 'python3 --version && node --version && java -version'"
