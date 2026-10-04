#!/bin/bash
# Build and push the GHCR carrier image (TensorFold wheel at the pinned commit + dependency lock + render script).
#   bash scripts/build-carrier.sh           # build only
#   PUSH=1 bash scripts/build-carrier.sh    # build + push (needs `docker login ghcr.io` with write:packages)
# The wheel is pure Python, so the carrier can be built on any machine with Docker and python3.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-ghcr.io/drowzeys/keys-mac-tensorfold-minimax-h3-mlx}"
TAG="${TAG:-1.0}"
TF_REPO="https://github.com/drowzeys/TensorFold.git"
TF_COMMIT="b8d1682ec315fa4b3649aa4ec9579670a3b89576"
cd "$HERE"
mkdir -p payload
python3 -m pip wheel --no-deps -q -w payload "tensorfold @ git+$TF_REPO@$TF_COMMIT"
cp requirements.lock h3_generate.py payload/
( cd payload && shasum -a 256 tensorfold-*.whl requirements.lock h3_generate.py > SHA256SUMS )
cat payload/SHA256SUMS
if [ "${PUSH:-0}" = 1 ]; then
  docker buildx build --platform linux/arm64,linux/amd64 -f Dockerfile.ghcr -t "$IMAGE:$TAG" -t "$IMAGE:latest" --push .
  docker buildx imagetools inspect "$IMAGE:$TAG" | grep -E '^Digest' || true
else
  docker build -f Dockerfile.ghcr -t "$IMAGE:$TAG" -t "$IMAGE:latest" .
fi
