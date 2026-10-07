# keys-Mac-TensorFold-MiniMax-H3-MLX

**Thank you to everyone this stands on:** antirez (h3.c), RobZombAI (H3MLX), Ash Hart and the TensorFold
contributors, mrbizarro (minimax-h3-mlx / Phosphene), Apple's MLX team and every MLX contributor, the MiniMax team
(MiniMax H3), NVIDIA Research (Sol-Engine, Sol-Attn, Sol-H3), LightX2V (the Turbo adapter) and FastVideo (FastH3).
This pack is their work, ported, pinned and measured. See [CREDITS.md](CREDITS.md).

**1.3** · Mac Studio M5 Ultra 256 GB · [TensorFold](https://github.com/ashhart/TensorFold) **0.6.5** + an H3 family ·
**MiniMax H3** (FL2VA, bfloat16 weights) · int8 kernels on the M5 tensor units · lightx2v **Turbo** adapter

> **1.3 (2026-10-07): the 20-step mode is twice as fast.** `QUALITY=high` now uses a velocity cache and attention
> reuse after mlx-serve's fast recipe: a 10 second 1312x736 clip in 1,301 s against 2,763 s for plain 20 steps, with
> stills that hold up beside it. `QUALITY=full` keeps the plain 20 steps.
>
> **Update, 1.2 (2026-10-04): a new standard setting, and the best result this pack has made.** Video now takes
> **5 Turbo passes**, and the **sound is made again by the base model** against the finished picture. Judged by eye
> and ear by the pack's owner on speech and singing clips: 5 passes gave the best picture among 3, 4, 5 and 6, and
> the base-model sound fixed what was wrong with the adapter's (harsh, echo-like, thin). The full 20 steps are one
> switch away: `QUALITY=high`. The speed tables below were measured with the earlier 3-pass setting.

**A 5 second 864x480 clip with stereo audio in 95 seconds at the standard setting** (5 Turbo passes 50 s, sound made
again 21 s, decode 14 s, loading and text 10 s), or about 50 seconds with the earlier 3 passes and the adapter's own
sound (`POINTS=4 REVOICE=0`). At 768x448: 76 seconds standard, 40 earlier. One-shot install from a GHCR prebuilt carrier. Text to video and **image to video** from a
first frame.

| Mode | How | 5 s, 768x448 | 5 s, 864x480 | 15 s (362 frames), 672x384 |
|---|---|---:|---:|---:|
| **Standard** | Turbo adapter, 5 passes; then the base model denoises the sound again, 20 steps, against the finished picture | **76 s** | **95 s** | **219 s** |
| Fast | Turbo adapter, 3 passes, its own sound (`POINTS=4 REVOICE=0`) | 40-41 s | about 50 s | about 115 s |
| **High quality** (`QUALITY=high`) | base model, 20 steps, no adapter | not timed | 210 s | 606 s |

Standard at larger sizes, 8 second clips (from the Studio pack's runs, image step excluded): 864x480 164 s; 1344x768
698 s; with the 2x decoder, 2048x1152 289 s and 2560x1440 624 s.

The audio step recomputes only the audio rows (about 4% of the sequence) against the picture's stored attention keys
and values, so its 20 steps cost less than one full pass. Text-to-image scouting, a 2x decoder and 2K presets are in
the companion pack, [keys-Mac-TensorFold-Studio](https://github.com/drowzeys/keys-Mac-TensorFold-Studio).

## The base-model audio step

The Turbo adapter is what makes the picture fast, and it is also what spoils the sound. So the sound is made twice:
once with the picture, by the adapter, and thrown away; then again by the base model, which is the one that sounds
right.

1. After the Turbo passes, the adapter-merged model is unloaded and the released weights are loaded (4 s).
2. The finished picture, any first frame and the prompt go through the model once, held still, and every block's
   attention keys and values over them are stored.
3. The audio starts again from noise and is denoised in 20 steps. Each step runs only the audio rows (1,206 of 29,009
   on a 15 second clip) against the stored keys and values: about 1.3 s a step instead of 24 s.

It is on by default whenever the adapter is used. `REVOICE=0` keeps the adapter's own sound; `REVOICE=<steps>` sets the
number of audio steps (20 is the only value tried). Proposed upstream as a draft, ashhart/TensorFold#405.

![h3.c 20 steps / TensorFold int8 20 steps / TensorFold Turbo int8, same prompt](samples/contact_sheet.jpg)

Rows: antirez h3.c, 20 steps · TensorFold int8, 20 steps · TensorFold Turbo adapter + int8, 3 forwards. Frames 20,
48, 72, 102 of the same prompt. The engines draw different noise for the same seed, so compositions differ.
Clips: [`samples/`](samples/).

## Headline (measured 2026-10-03/04, Mac Studio M5 Ultra 256 GB, macOS 27.0.1)

Test prompt "BLACK MIRROR: FINAL REFLECTION" by @itxabdullaa ([`prompts/`](prompts/)), text-to-video, seed 2077,
124 frames at 24 fps with stereo audio, one render at a time.

### Same Mac, same canvas, same adapter: 768x448, Turbo adapter, 3 forwards

| Engine | Clip time | Denoise | Decode | Note |
|---|---:|---:|---:|---|
| **TensorFold H3 family (this pack)** | **40-41 s** | **21.6-22.3 s** | 8.2 s | wall time from process start, model loaded each run |
| Phosphene 4.17.4 (minimax-h3-mlx `7919020`), bfloat16 | 42-44 s | 31.5 s | 7.6-7.8 s | panel's per-job time; 44-47 s from submit |
| Phosphene, its 8-bit transformer | 43-45 s | 32.7-34.8 s | 7.6-8.1 s | |

Two runs each. The lead is the denoise: 7.1-7.4 s per forward against 10.5 s. Phosphene applies the adapter as a
bfloat16 side branch and TensorFold as int8 weights, so the pictures differ; this compares time, not quality.

### 864x480 against the 20-step engines

| Engine | Steps | Per step | Denoise | Clip time |
|---|---:|---:|---:|---:|
| antirez h3.c `8974cc0` | 20 | 7.85 s | 157 s | 202 s |
| H3MLX `db3cae1` (with local fixes to run on the released weights) | 20 | 7.1 s | 141 s | 153 s |
| TensorFold, bfloat16 | 20 | 13.9 s | 277 s | about 305 s |
| TensorFold, int8 kernels | 20 | 9.3 s | 190 s | 210 s |
| **TensorFold, Turbo adapter + int8 kernels** | **3** | 9.0 s | **27.6 s** | **about 50 s** |

- **Per step, h3.c and H3MLX are still faster than TensorFold** (7.85 and 7.1 s against 9.0-9.3 s). TensorFold's
  clip time comes from running the few-step adapter, which the C engines do not load.
- **Where the 50 s goes:** text encoding 3.1 s, load + adapter merge + int8 quantize 6.2 s, denoise 27.6 s,
  decode 13.3 s (video 11.1 s).
- A 15 second 864x480 clip takes 1,218 s on h3.c and 946 s on H3MLX at 20 steps. It has not been run here.

Raw numbers and method: [`bench/results/RESULTS.md`](bench/results/RESULTS.md).

### Shootout: 8 seconds, image to video, 1344x768, four implementations

Same first frame, prompt and seed (40905090), 192 frames with audio. Clips and method: [`shootout/`](shootout/).

| Entry | Generation time | Speed vs h3.c | Steps | Technology | Video / audio quality |
|---|---:|---:|---:|---|---|
| **keys TensorFold H3 + Turbo** | **357 s** | **5.65x** | 3 | MLX + int8 Metal kernels + lightx2v Turbo adapter | sharp, most motion / line spoken correctly |
| RobZombAI H3MLX | 1,675 s | 1.20x | 20 | C + Metal, second-order solver, int8 MLP | sharpest / line spoken correctly |
| antirez h3.c | 2,018 s | 1.00x (baseline) | 20 | C + Metal, int8 projections | crisp / line spoken correctly |
| keys TensorFold H3 | 2,106 s | 0.96x | 20 | MLX + int8 Metal kernels | softer / line spoken correctly |
| mrbizarro minimax-h3-mlx | 2,451 s | 0.82x | 20 | Python + MLX, pruned bf16 transformer | softer / line spoken correctly |

At 20 steps the C engines are faster and sharper; TensorFold leads only with the Turbo adapter. Quality was judged
from stills, an edge score and speech recognition, not by watching.

![shootout stills](shootout/contact_sheet_shootout.jpg)

## What makes it fast

| Piece | Effect at 864x480, 124 frames |
|---|---|
| Turbo adapter, 3 forwards instead of 20 | most of the clip time |
| int8 SwiGLU MLP on the M5 tensor units | 81 ms to 41 ms per block |
| int8 QKV with the q/k norm and RoPE inside the kernel | 46 ms to 22.5 ms per block |
| int8 attention output | 12 ms to 7 ms per block |
| AdaLN tables projected once per run, weights released | 24 GiB freed |
| Adapter merged in float32, quantized straight to int8 | keeps the adapter; a bfloat16 merge loses 15-50% of it |
| int8 kernels in the video decoder | decode 17.3 s to 10.7 s at 47 dB |

Attention itself (93 ms per block, over half of a forward) is MLX's scaled dot-product attention. An int8
flash-style attention kernel was built and measured at parity with it (95.5 ms against 95.8 ms), so it is not used.

## One-shot

```bash
git clone https://github.com/drowzeys/keys-Mac-TensorFold-MiniMax-H3-MLX.git
cd keys-Mac-TensorFold-MiniMax-H3-MLX
brew install python@3.11 uv ffmpeg
bash oneshot-setup.sh
```

`oneshot-setup.sh` does the following:

1. Gets TensorFold 0.6.5 with the H3 family (`drowzeys/TensorFold` at `99fa80a3`), from the **GHCR prebuilt
   carrier** when Docker is available (checksums verified), otherwise from git at the same commit.
2. Installs it with the exact dependency lock ([`requirements.lock`](requirements.lock): mlx 0.32.3, mlx-lm
   0.32.0, mlx-vlm 0.7.4, transformers 5.18.0, …) into its own venv at `~/.local/opt/tensorfold-h3`.
3. Clones minimax-h3-mlx at `79190205` beside it, for the text encoder, audio decoder and MP4 writer.
4. Downloads the MiniMax H3 `FL2VA` partition (**144 GB**) to `~/h3-models/MiniMax-H3` and the Turbo adapter
   (1.96 GB, checksum verified).
5. Renders a 5 second test clip to `outputs/test.mp4`.

Override `PREFIX` or `H3_MODEL_DIR` through the environment. `bash oneshot-setup.sh --verify` checks an existing
install; `--no-render` skips the test clip.

### Render

```bash
bash scripts/generate.sh "A hummingbird hovering over red flowers, soft wing hum" out.mp4
FIRST_FRAME=photo.jpg WIDTH=1344 HEIGHT=768 bash scripts/generate.sh "She turns to the camera and smiles" out.mp4
WIDTH=768 HEIGHT=448 SEED=7 bash scripts/generate.sh "..." out.mp4
PROMPT_FILE=prompts/black-mirror-scene.txt SEED=2077 bash scripts/generate.sh "" out.mp4
QUALITY=high bash scripts/generate.sh "..." out.mp4                    # 20 steps, no adapter: slower, calmer, softer
POINTS=4 REVOICE=0 bash scripts/generate.sh "..." out.mp4              # the earlier fast setting: 3 passes, adapter sound
```

`FRAMES` must be `17n + 5` (56, 73, 90, 124, 192, 243, 362); `WIDTH` and `HEIGHT` multiples of 32, at most 768x1344
pixels in total (the model's released limit).

### Image to video

`FIRST_FRAME=image` starts the clip from that image and the prompt describes what happens next. The image is
stretched onto `WIDTH` x `HEIGHT`, so pick a canvas with its aspect ratio (1344x768 for 16:9, 768x1344 for 9:16,
768x768 for square) or crop it first. MiniMax's own image-to-video prompts open with a line such as
`For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`
followed by the scene description; plain descriptions work too. Works in every mode. Last-frame and reference-image modes are not wired.

### GHCR prebuilt carrier

```bash
docker pull ghcr.io/drowzeys/keys-mac-tensorfold-minimax-h3-mlx:1.3
# index digest sha256:36b11fa3e0df8359859912d686829266af917e6ad9020ed9348b6807ef23444e (linux/arm64 + linux/amd64)
docker run --rm -v "$PWD":/out ghcr.io/drowzeys/keys-mac-tensorfold-minimax-h3-mlx:1.3 cp -a /payload/. /out/payload/
```

The carrier holds the TensorFold wheel, `requirements.lock`, `h3_generate.py` and `SHA256SUMS`. **It is not a Mac
runtime**: Metal does not run in a container, so `oneshot-setup.sh` installs the payload natively. Rebuild it with
`PUSH=1 bash scripts/build-carrier.sh`.

## Stack

| Piece | Value |
|---|---|
| Host | Mac Studio M5 Ultra, 256 GB, macOS 27.0.1 |
| Engine | TensorFold 0.6.5 (`609ca419`) + ten commits, `drowzeys/TensorFold` branch `studio` @ `99fa80a32f7e61d331092064576ada72fdd2d1a4` (Apache-2.0) |
| Model | `MiniMaxAI/MiniMax-H3`, `FL2VA` partition: 33B transformer (61.7 GiB bfloat16), Qwen3-VL text encoder, video and audio VAEs |
| Adapter | lightx2v MiniMax H3 Turbo v1.0, runner layout as published by Phosphene (sha256 `d51d626f…`) |
| Borrowed at run time | minimax-h3-mlx @ `79190205`: text encoder, audio decoder, MP4 writer |
| MLX | 0.32.3 |
| Peak memory | 103 GB during the adapter merge; 65 GB without an adapter |

## Notes

- **This is a development render path, not a product.** The transformer, sampler, adapters, kernels and video
  decoder are TensorFold's H3 family. The text encoder, audio decoder and MP4 writer are still minimax-h3-mlx's,
  run through `h3_generate.py`. There is no `tensorfold generate` command and no server; the model loads on every
  run.
- **Image to video is new in 1.1 and lightly tested.** A first frame was checked on one image at 1344x768 (56 frames):
  the clip opens on the image (33-34 dB against it after encoding) and follows the prompt. The image is encoded by
  minimax-h3-mlx's VAE encoder and vision tower, which this pack already installs. The speed tables above are text
  to video; a first frame adds one latent frame of rows and the vision tokens to the sequence.
- **Picture quality is not graded.** The transformer equals minimax-h3-mlx's on a real step in bfloat16, and the
  float32 video decode equals its decode. The int8 path does not: a 20-step int8 render keeps the bfloat16
  composition with different detail, and a 3-forward Turbo render in int8 lands on a different composition from
  Turbo in bfloat16. Stills were checked; motion and audio were not reviewed by ear or eye.
- **Sound.** With the Turbo adapter the sound was judged poor by ear: harsh on speech, worse on singing, with an
  echo-like quality; the model's two audio channels agree at 20 steps and disagree under the adapter. Making the
  sound again with the base model fixed it and is standard (`REVOICE=0` keeps the adapter's sound). The audio
  decoder's hard clip is lifted, an overshooting take is turned down as a whole, and a thin take gets a bass shelf
  of up to 6 dB (`AUDIO_EQ=off`). Lip-sync was judged by eye, not measured.
- **Turbo is a distilled model.** It gives sharp frames in 3 forwards and sometimes duplicates figures. The
  20-step render without it is closer to what the base model makes.
- **M5 only for these numbers.** The int8 kernels need Metal 4 tensor operations. Without them the family runs
  bfloat16, slower than h3.c.
- **128 GB+ of memory, measured on 256 GB only.**
- **FastH3** (`FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA`, `dense-datafree`) is read by the same loader
  (`ADAPTER=…/adapter_model.safetensors POINTS=5`). On the test prompt at 864x480 it took 76 s and looked busier
  than Turbo; it is published for 1344x768, which was not tested.
- **Licence.** MiniMax H3 is under the MiniMax H3 Community License, which excludes some territories. This pack
  ships no weights. Read the licence before downloading them.
- Proposed upstream as draft pull requests: the H3 family (ashhart/TensorFold#384) and the audio step with the 2x
  decoder (#405). Neither is part of a TensorFold release.

## Credits

Cite the original authors first: the MiniMax team (MiniMax H3), antirez (h3.c), RobZombAI (H3MLX), mrbizarro
(minimax-h3-mlx, Phosphene), Ash Hart and the TensorFold contributors, NVIDIA Research (Sol-Engine, Sol-Attn,
Sol-H3), LightX2V (Turbo adapter), FastVideo (FastH3), Apple MLX and its contributors, and @itxabdullaa for the
test prompt. Full list: [CREDITS.md](CREDITS.md). Pack: drowzeys / keys.
