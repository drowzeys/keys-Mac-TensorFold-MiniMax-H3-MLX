#!/bin/bash
# Render one clip with MiniMax H3 on TensorFold.
# Standard: the Turbo adapter makes the picture in 5 passes, then the model without the adapter makes the sound again
# against the finished picture (20 audio steps). QUALITY=high: 20 steps without the adapter, fast recipe. QUALITY=full: plain 20 steps.
#   bash scripts/generate.sh "a prompt" out.mp4
#   PROMPT_FILE=prompts/black-mirror-scene.txt bash scripts/generate.sh "" out.mp4
#   QUALITY=high bash scripts/generate.sh "a prompt" out.mp4               # 20 steps, no adapter, fast recipe (QUALITY=full for plain)
#   FIRST_FRAME=photo.jpg bash scripts/generate.sh "what happens next" out.mp4   # image to video
#   X2=1 X2_VAE=decoder.safetensors WIDTH=1344 HEIGHT=768 bash scripts/generate.sh "a prompt" out.mp4   # 2x decoder
# Environment: PREFIX, H3_MODEL_DIR, ADAPTER (a file, or none), FIRST_FRAME (an image the clip starts from; it is
# stretched onto the canvas, so match its aspect ratio), WIDTH, HEIGHT, FRAMES (17n+5), SEED,
# QUALITY (standard, high or full), POINTS (sigma points, one more than the passes: default 6 with the adapter, 21
# without), REVOICE (audio steps made again without the adapter after an adapter render: default 20, 0 keeps the
# adapter's own sound), AUDIO_EQ / AUDIO_BASS / AUDIO_BASS_TARGET (see below), AUDIO_SHIFT (experimental: the audio schedule's shift,
# 3 in the model; lower values add bass but roughen speech), EXTRA (extra flags for h3_generate.py).
# X2=1 makes WIDTH x HEIGHT the size of the finished clip: the model generates at half of each and the 2x decoder
# ($X2_VAE) doubles it, so both must be multiples of 64 (default 1344x768). CROP=WxH centre-crops the frames before the clip is written.
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-h3}"
H3_MODEL_DIR="${H3_MODEL_DIR:-$HOME/h3-models/MiniMax-H3}"
# QUALITY=high: 20 steps without the adapter, with the fast recipe (the first four and last two steps in full, a
# velocity cache and attention reuse in between: about twice as fast as plain). QUALITY=full: plain 20 steps.
case "${QUALITY:-standard}" in high) ADAPTER=none; RECIPE="--step-cache 0.05 --attention-every 2";; full) ADAPTER=none; RECIPE="";; *) RECIPE="";; esac
ADAPTER="${ADAPTER:-$PREFIX/adapters/lightx2v_v1.0_768p_ourlayout.safetensors}"
# sigma points are one more than the passes: 5 passes with the adapter, 20 without
if [ "$ADAPTER" = none ]; then POINTS="${POINTS:-21}"; REVOICE="${REVOICE:-0}"; else POINTS="${POINTS:-6}"; REVOICE="${REVOICE:-20}"; fi
X2_VAE="${X2_VAE:-$PREFIX/adapters/MiniMax-H3-X2-Detail-v1.safetensors}"
PROMPT="${1:-}"; OUT="${2:-outputs/h3.mp4}"
mkdir -p "$(dirname "$OUT")"
export PATH="/opt/homebrew/bin:$PATH" PYTHONPATH="$PREFIX/minimax-h3-mlx"
if [ "${X2:-0}" = 1 ]; then
  W="${WIDTH:-1344}"; H="${HEIGHT:-768}"
  [ $((W % 64)) = 0 ] && [ $((H % 64)) = 0 ] || { echo "with X2=1, WIDTH and HEIGHT must be multiples of 64, got ${W}x${H}" >&2; exit 2; }
  [ -f "$X2_VAE" ] || { echo "no 2x decoder at $X2_VAE (this pack does not download one: set X2_VAE to MiniMax-H3-X2-Detail-v1.safetensors; the Studio pack installs it)" >&2; exit 2; }
  W=$((W / 2)); H=$((H / 2))
else
  W="${WIDTH:-864}"; H="${HEIGHT:-480}"
fi
ARGS=("$H3_MODEL_DIR" -o "$OUT" --width "$W" --height "$H" --frames "${FRAMES:-124}"
      --seed "${SEED:-0}" --points "$POINTS" --int8-mlp --int8-qkv --int8-out)
[ "$REVOICE" = 0 ] || ARGS+=(--revoice "$REVOICE")
[ "${X2:-0}" != 1 ] || ARGS+=(--upscale-vae "$X2_VAE")
[ "$ADAPTER" = none ] || ARGS+=(--lora "$ADAPTER")
[ -z "${AUDIO_SHIFT:-}" ] || ARGS+=(--audio-shift "$AUDIO_SHIFT")
[ -z "${FIRST_FRAME:-}" ] || ARGS+=(--first-frame "$FIRST_FRAME")
if [ -n "${PROMPT_FILE:-}" ]; then ARGS+=(--prompt-file "$PROMPT_FILE"); else ARGS+=(--prompt "$PROMPT"); fi
[ -z "${CROP:-}" ] || ARGS+=(--crop "$CROP")
# shellcheck disable=SC2086
"$PREFIX/venv/bin/python" "$PREFIX/h3_generate.py" "${ARGS[@]}" $RECIPE ${EXTRA:-}
# Few-step audio varies a lot from take to take, and some takes come out thin (little 120-300 Hz). With an adapter,
# the uncompressed track the render leaves beside the clip is measured: when the bass band is under the level of a
# full take, a low shelf lifts it by what is missing (6 dB at most) and a limiter guards the peaks. A take that is
# already full is left exactly as rendered. The picture is never re-encoded. AUDIO_EQ=off skips this;
# AUDIO_BASS=<dB> forces a gain; AUDIO_BASS_TARGET sets the bass to mid-band energy ratio aimed for (default 0.5).
WAV="${OUT%.*}.wav"
if [ "$ADAPTER" != none ] && [ "${AUDIO_EQ:-on}" != off ] && [ -s "$WAV" ]; then
  GAIN="${AUDIO_BASS:-$("$PREFIX/venv/bin/python" - "$WAV" "${AUDIO_BASS_TARGET:-0.5}" <<'PY'
import sys, wave
import numpy as np
with wave.open(sys.argv[1]) as handle:
    rate = handle.getframerate()
    audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32)
    audio = audio.reshape(-1, handle.getnchannels()).mean(axis=1)
power = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2
freq = np.fft.rfftfreq(len(audio), 1 / rate)
bass = power[(freq >= 120) & (freq < 300)].sum()
mid = power[(freq >= 300) & (freq < 4000)].sum()
ratio = bass / max(mid, 1e-20)
gain = float(np.clip(10 * np.log10(float(sys.argv[2]) / max(ratio, 1e-6)), 0.0, 6.0))
print(f"{gain:.1f}")
PY
)}"
  if [ "$(printf '%.0f' "$(echo "$GAIN * 10" | bc)")" -ge 10 ]; then
    FILTER="bass=g=$GAIN:f=250:w=0.6,alimiter=limit=0.95:level=disabled"
    ffmpeg -v error -y -i "$OUT" -i "$WAV" -map 0:v -map 1:a -c:v copy -af "$FILTER" -c:a aac -b:a 256k "${OUT%.*}.eq.mp4"
    mv "${OUT%.*}.eq.mp4" "$OUT"
    echo "[tensorfold] thin take: bass lifted by $GAIN dB"
  else
    echo "[tensorfold] audio left as rendered (bass is full)"
  fi
fi
