# Agent one-shot — MiniMax H3 on TensorFold (MLX, Apple Silicon)

**Repo:** https://github.com/drowzeys/keys-Mac-TensorFold-MiniMax-H3-MLX  
**Carrier image:** `ghcr.io/drowzeys/keys-mac-tensorfold-minimax-h3-mlx:1.2` (not a runtime; TensorFold wheel + lock + render script)  
**Engine:** TensorFold 0.6.5 + H3 family @ `drowzeys/TensorFold` `218bfe24`, own venv, int8 tensor-unit kernels

```bash
git clone https://github.com/drowzeys/keys-Mac-TensorFold-MiniMax-H3-MLX.git
cd keys-Mac-TensorFold-MiniMax-H3-MLX
brew install python@3.11 uv ffmpeg
bash oneshot-setup.sh      # engine, minimax-h3-mlx @ 79190205, weights 144 GB, Turbo adapter 1.96 GB, test render
bash scripts/generate.sh "your prompt" out.mp4
FIRST_FRAME=photo.jpg WIDTH=1344 HEIGHT=768 bash scripts/generate.sh "what happens next" out.mp4   # image to video
```

Rules:

- This is not a server. `tensorfold serve` does not route H3; render with `scripts/generate.sh`.
- Frames must be `17n + 5` (56, 73, 90, 124, 192, 243, 362); width and height multiples of 32, at most 768x1344 pixels.
- Image to video: `FIRST_FRAME` is stretched onto the canvas, so match the image's aspect ratio. Only a first frame;
  no last frame, no reference images.
- The fast path is M5-only (Metal 4 tensor operations). Elsewhere drop the `--int8-*` flags; the README numbers
  do not apply.
- Standard = Turbo adapter, 5 passes, then the sound made again by the base model (`REVOICE=20`). Do not go back to
  the adapter's own sound (`REVOICE=0`) unless asked: it was judged poor by ear. `QUALITY=high` = 20 steps, no
  adapter, about three times slower. `POINTS` is passes plus one.
- Merge adapters with the tool (`--lora`), never by hand into bfloat16: that loses 15-50% of the adapter.
- Turbo (3 forwards) with int8 lands on a different composition from Turbo in bfloat16. For the closest match
  to the 20-step picture use `QUALITY=high`.
- 128 GB+ unified memory; measured on 256 GB only. The float32 adapter merge peaks at 103 GB.
- MiniMax H3 is under the MiniMax H3 Community License, with territory limits. Do not redistribute weights.
- The text encoder, audio decoder and MP4 writer are minimax-h3-mlx's, pinned by commit. Do not unpin.

**Credits:** keep CREDITS.md in sync. MiniMax (model), antirez (h3.c), RobZombAI (H3MLX), mrbizarro
(minimax-h3-mlx / Phosphene), Ash Hart (TensorFold), NVIDIA Research (Sol-Engine / Sol-Attn / Sol-H3), LightX2V
(Turbo adapter), FastVideo (FastH3), Apple MLX, @itxabdullaa (test prompt). Authors only — no credit hyperlinks.
