# Credits

**Thank you.** Almost everything here is other people's work. Keys / drowzeys ported pieces into a TensorFold
family, wrote the int8 tensor-unit kernels with their designs as the guide, pinned it and measured it on a Mac
Studio M5 Ultra. Cite the authors first.

## The model: MiniMax

**MiniMax** released **MiniMax H3** (Hailuo) as open weights with code: the 33B joint video and audio diffusion
transformer, the Qwen3-VL text encoder setup, and the video and audio VAEs. Nothing in this pack exists without
that release. The weights are under the MiniMax H3 Community License; this pack ships none of them.

## The first native Mac engine: Salvatore Sanfilippo (antirez)

**antirez** wrote **h3.c**, MiniMax H3 in C and Metal. It is the reference for what is possible on Apple
Silicon and the cleanest picture we measured. The int8 kernels here follow h3.c's quantization scheme: per-row
activation scales, per-output-channel weight scales, one scale per 1,024 channels on wide inputs, 128x128x128
int8 tiles on the M5 tensor units, and the q/k norm and rotation inside the QKV kernel. No h3.c source is
copied; the design is his. (MIT.)

## H3MLX: RobZombAI

**RobZombAI** built **H3MLX** on top of h3.c: a second-order solver, int8 paths, 4K mastering and a large preset
library. Working through it on an M5 Ultra is what started this pack, and its Sol-style switches and solver
gave us measured points to compare against.

## The MLX port this family is adapted from: mrbizarro

**mrbizarro** wrote **minimax-h3-mlx**, the MLX port of H3 that runs inside **Phosphene**. TensorFold's H3
family is adapted from it: the transformer module tree, the packed sequence and rotary positions, the two sigma
schedules, the video decoder with its tiling and clip chunking, and the adapter layout handling. Its notes on
bfloat16 adapter merging, QKV row order and SwiGLU half order saved days. This pack still runs its text
encoder, audio decoder and MP4 writer directly, and downloads the runner-layout Turbo adapter asset that
Phosphene publishes. (Apache-2.0.)

## The engine: Ash Hart

**Ash Hart** (ashhart) and the TensorFold contributors — **TensorFold**: the family and kernel structure this
lives in, and the practice of reaching the M5 tensor units from `mx.fast.metal_kernel` that the int8 kernels
are built on (`kernels/qwen/dense/v1/lane_qmm.py`). This pack pins TensorFold 0.6.5 with one commit on top.
(Apache-2.0.)

## Sol-Engine, Sol-Attn and Sol-H3: NVIDIA Research

**NVIDIA Research, Efficient AI Team and Singapore Lab** (Jincheng Yu, Junsong Chen, Yitong Li, Haopeng Li,
Haocheng Xi, Song Han, Enze Xie and co-authors of the Sol-Engine and Sol-Attn papers). Sol-H3 showed which
optimizations matter for H3: few-step adapters, fused norm / RoPE / SwiGLU kernels, precomputed AdaLN, int8 QKV,
and dense attention on a single device. The cached AdaLN tables and the fused QKV kernel here follow that list.
Their runtime is CUDA; no Sol code runs here.

## Few-step adapters

- **LightX2V** (ModelTC) — the **MiniMax H3 Turbo** 4-step adapter (`lightx2v/Minimax-h3-Turbo`, Apache-2.0),
  the adapter behind every fast number in this pack.
- **larryvrh** — `MiniMax-H3-Turbo-Lora`, the adapter lineage minimax-h3-mlx's loader was written for.
- **FastVideo / Hao AI Lab @ UCSD** — **FastH3** 4-step adapters (`fastvideo-lora-v2`), which this family also
  reads.
- **TaoLiveAIGC** — the TaoMate-H3 3-step ladder, documented in Phosphene and supported by the sigma subset
  option.

## Platform and tools

- **Apple / ml-explore** — **MLX**, and the Metal 4 tensor operations on M5.
- **mlx-vlm** contributors — the Qwen3-VL language model the text encoder runs on.
- **Qwen team (Alibaba)** — Qwen3-VL.
- **Kijai**, **DeepBeepMeep**, **ddalcu** — the ComfyUI conversions, pruned checkpoints and MLX quants that the
  wider H3 ecosystem (and Phosphene's install) relies on.
- **FFmpeg**.

## The test prompt

"BLACK MIRROR: FINAL REFLECTION" is by **@itxabdullaa** on X (onlyprompts.ai X collection).

Pack: drowzeys / keys.
