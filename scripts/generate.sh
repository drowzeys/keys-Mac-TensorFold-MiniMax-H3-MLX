#!/bin/bash
# Render one clip with the fastest measured configuration.
#   bash scripts/generate.sh "a prompt" out.mp4
#   PROMPT_FILE=prompts/black-mirror-scene.txt bash scripts/generate.sh "" out.mp4
#   ADAPTER=none POINTS=21 bash scripts/generate.sh "a prompt" out.mp4     # 20 steps, no adapter
#   FIRST_FRAME=photo.jpg bash scripts/generate.sh "what happens next" out.mp4   # image to video
# Environment: PREFIX, H3_MODEL_DIR, ADAPTER (a file, or none), FIRST_FRAME (an image the clip starts from; it is
# stretched onto WIDTH x HEIGHT, so match its aspect ratio), WIDTH, HEIGHT, FRAMES (17n+5), SEED,
# POINTS (sigma points: one more than the forwards), EXTRA (extra flags for h3_generate.py).
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-h3}"
H3_MODEL_DIR="${H3_MODEL_DIR:-$HOME/h3-models/MiniMax-H3}"
ADAPTER="${ADAPTER:-$PREFIX/adapters/lightx2v_v1.0_768p_ourlayout.safetensors}"
PROMPT="${1:-}"; OUT="${2:-outputs/h3.mp4}"
mkdir -p "$(dirname "$OUT")"
export PATH="/opt/homebrew/bin:$PATH" PYTHONPATH="$PREFIX/minimax-h3-mlx"
ARGS=("$H3_MODEL_DIR" -o "$OUT" --width "${WIDTH:-864}" --height "${HEIGHT:-480}" --frames "${FRAMES:-124}"
      --seed "${SEED:-0}" --points "${POINTS:-4}" --int8-mlp --int8-qkv --int8-out)
[ "$ADAPTER" = none ] || ARGS+=(--lora "$ADAPTER")
[ -z "${FIRST_FRAME:-}" ] || ARGS+=(--first-frame "$FIRST_FRAME")
if [ -n "${PROMPT_FILE:-}" ]; then ARGS+=(--prompt-file "$PROMPT_FILE"); else ARGS+=(--prompt "$PROMPT"); fi
# shellcheck disable=SC2086
exec "$PREFIX/venv/bin/python" "$PREFIX/h3_generate.py" "${ARGS[@]}" ${EXTRA:-}
