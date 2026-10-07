#!/usr/bin/env bash
# =============================================================================
# One-shot: MiniMax H3 (video + audio) on TensorFold 0.6.5 with the H3 family, Apple Silicon
#
#   bash oneshot-setup.sh             # install engine, fetch weights + Turbo adapter, render a test clip
#   bash oneshot-setup.sh --verify    # check an existing install, no downloads, no render
#   bash oneshot-setup.sh --no-render # install and fetch only
#
# Engine payload order: this clone's ./payload -> GHCR carrier image -> git at the pinned commit.
# Installs into its own venv ($PREFIX, default ~/.local/opt/tensorfold-h3). Touches nothing else.
# Weights go to $H3_MODEL_DIR (default ~/h3-models/MiniMax-H3): the FL2VA partition is 144 GB.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-h3}"
H3_MODEL_DIR="${H3_MODEL_DIR:-$HOME/h3-models/MiniMax-H3}"
IMAGE="${IMAGE:-ghcr.io/drowzeys/keys-mac-tensorfold-minimax-h3-mlx:1.3}"
TF_REPO="https://github.com/drowzeys/TensorFold.git"
TF_COMMIT="99fa80a32f7e61d331092064576ada72fdd2d1a4"
REF_REPO="https://github.com/mrbizarro/minimax-h3-mlx.git"
REF_COMMIT="79190205258454b43e6c9e50e577de234222419c"
ADAPTER_NAME="lightx2v_v1.0_768p_ourlayout.safetensors"
ADAPTER_URL="https://github.com/mrbizarro/Phosphene/releases/download/weights-ltx25-v1/$ADAPTER_NAME"
ADAPTER_SHA256="d51d626fe0845da7e5845a47c323cf3f29086d44d24cb1a4b980882488746197"
MODE="${1:-}"

die() { echo "FATAL: $*" >&2; exit 1; }
ok()  { echo "  ✓ $*"; }
step(){ echo; echo "==> $*"; }

step "Preflight"
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die "Apple silicon macOS only"
CHIP=$(sysctl -n machdep.cpu.brand_string); RAM=$(( $(sysctl -n hw.memsize) / 1073741824 ))
ok "$CHIP, macOS $(sw_vers -productVersion), ${RAM} GB"
[ "$RAM" -ge 128 ] || die "needs 128 GB+ unified memory: the float32 adapter merge peaks at 103 GB (measured on 256 GB only)"
case "$CHIP" in *M5*) ok "M5: int8 kernels on the tensor units";;
  *) echo "  ! $CHIP is not M5: the int8 kernels need Metal 4 tensor operations; without them the engine runs bfloat16 and the README numbers do not apply";; esac
export PATH="/opt/homebrew/bin:$PATH"
command -v uv >/dev/null || die "uv required: brew install uv"
command -v python3.11 >/dev/null || die "python3.11 required: brew install python@3.11"
command -v ffmpeg >/dev/null || die "ffmpeg required: brew install ffmpeg"
command -v git >/dev/null || die "git required"

fetch_ghcr() {
  command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 || return 1
  echo "  payload from GHCR $IMAGE"
  docker pull -q "$IMAGE" >/dev/null || return 1
  docker run --rm -v "$HERE":/out "$IMAGE" cp -a /payload/. /out/payload/
}

step "TensorFold 0.6.5 + H3 family @ ${TF_COMMIT:0:8} (own venv at $PREFIX)"
# an install from an older pack lacks the re-voice pass in the sampler; reinstall it
HAS_ENGINE='from tensorfold.families.h3.sampler import revoice, default_gates'
if [ ! -x "$PREFIX/venv/bin/python" ] || ! "$PREFIX/venv/bin/python" -c "$HAS_ENGINE" 2>/dev/null; then
  [ "$MODE" = "--verify" ] && die "TensorFold with the H3 family is not installed at $PREFIX"
  mkdir -p "$HERE/payload" "$PREFIX"
  # a payload left by an older pack would fail its checksums against this pack's files; fetch a fresh one
  ( cd "$HERE/payload" 2>/dev/null && shasum -a 256 -c SHA256SUMS >/dev/null 2>&1 && cmp -s h3_generate.py "$HERE/h3_generate.py" ) \
    || { rm -f "$HERE"/payload/tensorfold-*.whl "$HERE"/payload/SHA256SUMS; fetch_ghcr || echo "  no carrier payload; installing from git"; }
  [ -x "$PREFIX/venv/bin/python" ] || uv venv -q -p "$(command -v python3.11)" "$PREFIX/venv"
  if ls "$HERE"/payload/tensorfold-*.whl >/dev/null 2>&1; then
    ( cd "$HERE/payload" && shasum -a 256 -c SHA256SUMS >/dev/null ) || die "carrier payload checksum mismatch"
    ok "carrier payload checksums verified"
    uv pip install -q --reinstall-package tensorfold -p "$PREFIX/venv/bin/python" -r "$HERE/requirements.lock" "$HERE"/payload/tensorfold-*.whl
  else
    uv pip install -q --reinstall-package tensorfold -p "$PREFIX/venv/bin/python" -r "$HERE/requirements.lock" "tensorfold @ git+$TF_REPO@$TF_COMMIT"
  fi
fi
cp "$HERE/h3_generate.py" "$PREFIX/h3_generate.py"
"$PREFIX/venv/bin/python" - <<'EOF' || die "the installed TensorFold has no H3 family"
import importlib.metadata as m
import mlx.core as mx
from tensorfold.families import families
from tensorfold.kernels.minimax.h3.v1 import mlp_int8
assert "minimax_h3" in families()
print(f"  ✓ tensorfold {m.version('tensorfold')} with the H3 family, mlx {mx.__version__}")
print("  ✓ int8 tensor-unit kernels available" if mlp_int8.available() else "  ! int8 kernels unavailable on this GPU: bfloat16 only")
EOF

step "Text encoder, audio decoder and MP4 writer: minimax-h3-mlx @ ${REF_COMMIT:0:8}"
if [ ! -d "$PREFIX/minimax-h3-mlx/.git" ]; then
  [ "$MODE" = "--verify" ] && die "minimax-h3-mlx is not at $PREFIX/minimax-h3-mlx"
  git clone -q "$REF_REPO" "$PREFIX/minimax-h3-mlx"
fi
git -C "$PREFIX/minimax-h3-mlx" fetch -q origin "$REF_COMMIT" 2>/dev/null || true
git -C "$PREFIX/minimax-h3-mlx" checkout -q "$REF_COMMIT" || die "cannot check out minimax-h3-mlx $REF_COMMIT"
PYTHONPATH="$PREFIX/minimax-h3-mlx" "$PREFIX/venv/bin/python" -c "import minimax_h3_mlx.text_encoder, minimax_h3_mlx.media" \
  || die "minimax-h3-mlx does not import"
ok "minimax-h3-mlx $(git -C "$PREFIX/minimax-h3-mlx" rev-parse --short HEAD)"

step "Weights: MiniMaxAI/MiniMax-H3 (FL2VA partition, 144 GB) -> $H3_MODEL_DIR"
if [ ! -f "$H3_MODEL_DIR/FL2VA/transformer/model.safetensors.index.json" ]; then
  [ "$MODE" = "--verify" ] && die "no H3 weights at $H3_MODEL_DIR"
  echo "  MiniMax H3 is under the MiniMax H3 Community License, which limits where it may be used. Read it first:"
  echo "  https://huggingface.co/MiniMaxAI/MiniMax-H3"
  "$PREFIX/venv/bin/hf" download MiniMaxAI/MiniMax-H3 --include "FL2VA/*" --include model_index.json \
    --include LICENSE --include README.md --local-dir "$H3_MODEL_DIR"
fi
ok "FL2VA at $H3_MODEL_DIR ($(du -sh "$H3_MODEL_DIR/FL2VA" | cut -f1))"

step "Turbo adapter (lightx2v v1.0, runner layout, 1.96 GB)"
mkdir -p "$PREFIX/adapters"
ADAPTER="$PREFIX/adapters/$ADAPTER_NAME"
if [ ! -f "$ADAPTER" ]; then
  [ "$MODE" = "--verify" ] && die "no Turbo adapter at $ADAPTER"
  curl -L --fail --retry 3 -C - -o "$ADAPTER.partial" "$ADAPTER_URL"
  echo "$ADAPTER_SHA256  $ADAPTER.partial" | shasum -a 256 -c - >/dev/null || die "Turbo adapter checksum mismatch"
  mv "$ADAPTER.partial" "$ADAPTER"
fi
ok "$ADAPTER_NAME"

if [ "$MODE" = "--verify" ] || [ "$MODE" = "--no-render" ]; then
  echo; echo "DONE. Render with: bash $HERE/scripts/generate.sh \"your prompt\" out.mp4"; exit 0
fi

step "Test render: 5 s, 864x480, standard settings (Turbo 5 passes, sound made again without the adapter)"
PREFIX="$PREFIX" H3_MODEL_DIR="$H3_MODEL_DIR" PROMPT_FILE="$HERE/prompts/black-mirror-scene.txt" SEED=2077 \
  bash "$HERE/scripts/generate.sh" "" "$HERE/outputs/test.mp4" 2>&1 | grep -E "tensorfold\] \{|rounded|rror|Trace" | sed 's/^/  /'
[ -s "$HERE/outputs/test.mp4" ] || die "the test render wrote no file"
echo
echo "DONE. $HERE/outputs/test.mp4"
echo "Render: bash $HERE/scripts/generate.sh \"your prompt\" out.mp4   (WIDTH/HEIGHT multiples of 32, FRAMES = 17n+5)"
echo "Image to video: FIRST_FRAME=photo.jpg bash $HERE/scripts/generate.sh \"what happens next\" out.mp4"
echo "Higher quality, slower: QUALITY=high bash $HERE/scripts/generate.sh \"your prompt\" out.mp4"
