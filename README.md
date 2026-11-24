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
| `mtp-mlx-packs.patch` | mlx-serve patch (2 gates + 1 test): declares/loads the NextN MTP head for MLX packs, not just streaming EXL3 packs. Mirrors the `mimo_v2` pattern; headless packs fall back gracefully. |

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
