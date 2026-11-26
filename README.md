# Ling-3.2-air → MLX 4-bit: build & verification kit

Everything used to turn [inclusionAI/Ling-3.2-air](https://huggingface.co/inclusionAI/Ling-3.2-air)
(124B-A5.5B hybrid-MoE VL checkpoint, FP8 block-128) into
[`teejer/Ling-3.2-air-mlx`](https://huggingface.co/teejer/Ling-3.2-air-mlx), a text-only 4-bit MLX
pack served by [mlx-serve](https://github.com/ddalcu/mlx-serve) via its built-in `bailing_hybrid`
plugin — plus the engine patch that unlocks the pack's MTP head for speculative decoding.

## Contents

| File | What it does |
|---|---|
| `convert_ling_to_mlx.py` | The converter. HF FP8 (e4m3 + 128×128 `weight_scale_inv`) → exact e4m3fn dequant (verified 0.0 max-diff vs torch) → mlx-lm layout (`mlp.switch_mlp.*` expert stacks, raw `kv_b_proj`, `[dim,4,1]` convs) → 4-bit g64 affine requant. Drops the vision tower, keeps the NextN head at `model.layers.42.*`, flattens the VL config to `model_type: bailing_hybrid` with `rope_interleave: true` and the per-layer SwiGLU clamp lists preserved. |
| `check_pack_python.py` | Independent validator: loads a pack through **mlx-lm's** `bailing_moe_v3` (absorbing `kv_b_proj` into `embed_q`/`unembed_out` on the fly) and greedily generates. Coherent text here + garbage under a Zig engine ⇒ engine bug, not pack bug. |
| `mtp-mlx-packs.patch` | mlx-serve patch (2 gates + 1 test): declares/loads the NextN MTP head for MLX packs, not just streaming EXL3 packs. Mirrors the `mimo_v2` pattern; headless packs fall back gracefully. Upstreamed as [mlx-serve PR #1316](https://github.com/ddalcu/mlx-serve/pull/1316). |

## Build

```bash
uv venv --python 3.12 .venv && .venv/bin/pip install mlx-lm   # mlx.core + quant primitives
.venv/bin/python convert_ling_to_mlx.py --hf-dir <local HF snapshot> --out Ling-3.2-air-MLX-4bit
```

~30 s wall on an M3 Ultra (lazy MLX graphs), output ~68 GB / 13 shards.
Layer-type rule (MLA iff `(layer+1) % layer_group_size == 0`) and the quant/keep-wide split mirror
mlx-lm's `bailing_moe_v3.quant_predicate` and the published `rapid-mlx` packs.

## Verification performed

- e4m3fn dequant exact vs torch (`safetensors` + `torch.float8_e4m3fn`) on sampled tensors incl. the
  ragged-576-row `kv_a_proj_with_mqa`: max diff 0.0 in f32.
- Quant roundtrip (dequant→quant→dequant) vs torch reference: max err ≈ 4–9% of weight range,
  correct std — the expected 4-bit g64 band.
- mlx-serve `bailing_hybrid` integration suite: **23/23** (greedy QA, thinking split, GLM-style tool
  calls + round-trip, 32k needle, prefix cache, streaming).
- With `mtp-mlx-packs.patch`: `MTP head ready (depth=1)`, drafts accepted **77.8%**, decode
  **11.1 → 8.5 ms/tok (+31%)**, suite still 23/23.

## Gotchas learned

- `mx.load` returns FP8 tensors as raw **uint8** (no float8 dtype in the MLX python build) — decode
  through an e4m3fn lookup table, never a dtype cast.
- The release omits `rope_interleave` (reference default true); the engine default is false — set it
  explicitly in the pack config.
- Pack dirs must be passed to mlx-serve as **absolute** paths (the plugin resolves relative dirs
  against its own cwd).
- HF blobs are extension-less; `mx.load` dispatches on suffix — read the symlinked snapshot names.

## Vision (VL pack + engine branch)

`convert_ling_to_mlx.py --vl` builds `Ling-3.2-air-MLX-4bit-VL` (~74 GB): the text pack plus the
bf16 `model.visual.*` tower and `linear_proj.*` readout kept wide, `vision_config` (incl. the
processor's pixel bounds), `mrope_section [8,12,12]`, and the vision token ids
(`<|image_pad|>` 157157 …). Served by the mlx-serve `ling-vl` engine branch, which adds native
`bailing_moe_v3_vl` support: the Qwen3-VL-shaped tower runs in the engine's vision encoder, rows
splice at `<|image_pad|>`, and the trunk ropes with the checkpoint's own M-RoPE — the
body-diagonal position table (`getRopeIndexBailing`) and the `apply_video_rope` axis layout
(`bailingSelector`). Torch-oracle tests cover the tower rows, the position table, and the
cos/sin goldens against the release modeling file.

Verified on the VL pack (mlx-serve `ling-vl`, M3 Ultra): tower init `27 blocks, 1152d/16 heads,
merge 2, 4608 → 2560`; single-image shape/color QA correct, with the position table compressing a
165-token image to one step exactly as the reference delta predicts; two-image difference QA correct (delta −58 for 2×30 tokens); text
regression coherent; decode ~98 tok/s (this branch predates the MTP merge — expect the +31%
once both land). Published as [`teejer/Ling-3.2-air-mlx-vl`](https://huggingface.co/teejer/Ling-3.2-air-mlx-vl)
(all 13 shards SHA-256-verified after upload; LFS dedupes against the text pack, ~138 MB actually
transferred). Engine branch upstreamed as [mlx-serve PR #1334](https://github.com/ddalcu/mlx-serve/pull/1334).
