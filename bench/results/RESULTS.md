# Results

Mac Studio M5 Ultra, 256 GB, 64-core GPU, macOS 27.0.1, MLX 0.32.3. Measured 2026-10-03 and 2026-10-04.
Prompt: `prompts/black-mirror-scene.txt` (by @itxabdullaa), text-to-video, seed 2077, 124 frames at 24 fps with
stereo audio unless noted. One render at a time. "Clip time" is wall clock to a finished MP4.

## Engines on the same Mac

### 864x480, 124 frames

| Engine | Version | Steps | Per step | Denoise | Clip time | Weights |
|---|---|---:|---:|---:|---:|---|
| antirez h3.c | `8974cc0` | 20 | 7.85 s | 157 s | 202 s | released bfloat16; int8 projections at run time |
| H3MLX | `db3cae1` + local fixes | 20 | 7.1 s | 141 s | 153 s | same |
| H3MLX, Frontier 12 | same | 8 | 7.6 s | 61 s | 85 s incl. 4K master | same; rendered before the audio-noise fix below |
| TensorFold H3 family | `b8d1682e` | 20 | 13.9 s | 277 s | about 305 s | bfloat16 |
| TensorFold, int8 MLP | | 20 | 11.7 s | 232 s | | |
| TensorFold, int8 MLP + QKV + attention out (unfused QKV) | | 20 | 9.7 s | 195 s | | |
| TensorFold, int8, fused QKV, int8 video decoder | | 20 | 9.3 s | 190 s | 210 s | |
| TensorFold, Turbo adapter, bfloat16 (adapter rounded) | | 3 | 12.6 s | 38 s | about 67 s | |
| TensorFold, Turbo adapter, int8, fused QKV, int8 video decoder | | 3 | 9.0 s | 27.6 s | about 50 s | adapter merged in float32 |
| TensorFold, FastH3 dense adapter, int8 | | 4 | 9.0 s | 38 s | about 76 s | |

Long runs read about 10% slower per forward than short ones (12.6 s over 3 forwards against 13.9 s over 20 in
bfloat16); compare like with like.

H3MLX local fixes: three shader kernels it dispatches were not registered; its copy of the sampler never filled the
audio latent with noise, which gave a constant tone and doubled figures in the video; an audio resample and
loudness filter at the mux was removed. With the audio noise restored its picture tracks h3.c's shot for shot.

### 768x448, 124 frames, Turbo adapter, 3 forwards (two runs each)

| Engine | Denoise | Video decode | Clip time |
|---|---:|---:|---:|
| TensorFold, int8 | 21.6 / 22.3 s | 6.1 s | 40 / 41 s from process start (text 3.1 s, load 6.4 s, denoise, decode 8.2 s) |
| Phosphene 4.17.4, bfloat16 transformer | 31.5 / 31.5 s | 7.8 / 7.6 s | 44.1 / 41.7 s per job; 47 / 44 s from submit |
| Phosphene, 8-bit transformer | 32.7 / 34.8 s | 7.6 / 8.1 s | 43.4 / 44.8 s per job; 43 / 47 s from submit |

Phosphene's other tiers on this Mac, bfloat16 transformer, no adapter: Standard 768x448, 8 forwards, 101 s; High
1024x576, 15 forwards, 373 s; High 15 second clip (three chained windows) 1,161 s.

### 15 second clips, 864x480, 362 frames, 20 steps

| Engine | Clip time |
|---|---:|
| antirez h3.c | 1,218 s |
| H3MLX (with the audio fix) | 946 s |
| Phosphene High, 1024x576, 16 points, three chained 5 s windows | 1,161 s |

TensorFold has not been run at 15 seconds.

## Inside a TensorFold forward (864x480, 15,918 rows)

| Stage, per block | bfloat16 | int8 |
|---|---:|---:|
| QKV + q/k norm + RoPE | 46.1 ms | 26.5 ms unfused, 22.5 ms fused |
| Attention | 92.8 ms | 95.5 ms in the int8 flash kernel (not used) |
| Attention output | 12.2 ms | 7.0 ms |
| SwiGLU MLP | 81.0 ms | 41.4 ms |

Kernel accuracy on block 0, real weights and real inputs: int8 MLP cosine 0.9994 against bfloat16 (relative error
0.035); fused QKV differs from the unfused int8 path by one bfloat16 rounding step.

## Adapter merge

On four projections of the Turbo adapter, the update is 0.9-1.7e-3 of the weight's norm. Rounding the merged
weight to bfloat16 keeps 52-85% of the update (projection onto it) with rounding noise 0.5-0.8 times its size.
Quantizing the float32 merge to int8 keeps the projection at 1.00.

## Video decoder (124 frames, 864x480: 7 clips x 15 tiles)

| Decoder | Time | Against the float32 decode |
|---|---:|---|
| minimax-h3-mlx, float32 | 17.3 s | reference |
| TensorFold, float32 | 17.2 s | identical (max difference 0) |
| TensorFold, int8 FFN + QKV + output | 10.7 s | 47.1 dB |

float16, bfloat16 and a batch of 32 tiles decode in the same time as float32 with 8.

## Attention sparsity (why there is no block-sparse attention)

32-row blocks, steps 0 / 9 / 19 of a 20-step run, five blocks, eight heads. Share of (query block, key block)
pairs needed to keep that share of each query block's attention:

| Kept | Per head | One mask for all heads |
|---|---|---|
| 95% | 22-68% | 51-95% |
| 99% | 44-83% | 81-99% |

## Image to video (1.1)

First frame from one 1344x768 image, 56 frames, seed 40905090, MiniMax-format prompt. First decoded frame against the
input image, and wall time, on the same Mac:

| Engine | Steps | First frame | Wall |
|---|---:|---:|---:|
| antirez h3.c | 4 | 33.0 dB | 94 s |
| H3MLX | 4 | 30.8 dB | 58 s |
| minimax-h3-mlx runner | 4 | 32.6 dB | 97 s |
| TensorFold, int8 | 4 | 32.9 dB | 76 s |
| TensorFold, Turbo adapter, int8 | 3 | 34.2 dB | 63 s |

All five clips open on the image and follow the prompt in stills. This was a smoke test, not a quality comparison.

## Install check

`oneshot-setup.sh` from a fresh prefix with the carrier payload: checksums verified, minimax-h3-mlx cloned at its pin,
Turbo adapter downloaded and verified, test render written in 54 s of wall time (text 5.5 s, load 6.4 s, denoise
27.3 s, decode 13.4 s). `--verify` passes on the result. The 144 GB weight download was already on disk.

## What was checked and what was not

- The transformer equals minimax-h3-mlx's on a real first step (zero difference with the same bfloat16 input
  rounding). The float32 video decode equals its decode on real latents.
- Pictures were compared from stills (three or four frames per clip). Motion and audio were not reviewed by eye
  or ear; audio was checked for varying loudness and normal low-frequency content.
- The 20-step int8 render keeps the bfloat16 composition (median 18 dB between the clips' frames). Turbo in int8
  lands on a different composition from Turbo in bfloat16 (12-13 dB).
- Timings from 2026-10-03 for the bfloat16 20-step and Turbo bfloat16 rows were taken while a kernel benchmark
  shared the GPU at times; the bfloat16 20-step row was re-run on a quiet GPU (277 s).
