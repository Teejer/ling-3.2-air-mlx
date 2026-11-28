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
| `check_pack_python.py` | Independent validator: loads a pack through **mlx-lm's** `bailing_moe_v3` (absorbing `kv_b_proj` into `embed_q`/`unembed_out` on the fly, solving per-tensor bits from the packed geometry so mixed 4/8-bit packs load too) and greedily generates. Coherent text here + garbage under a Zig engine ⇒ engine bug, not pack bug. |
| `mtp-mlx-packs.patch` | mlx-serve patch (2 gates + 1 test): declares/loads the NextN MTP head for MLX packs, not just streaming EXL3 packs. Mirrors the `mimo_v2` pattern; headless packs fall back gracefully. Upstreamed as [mlx-serve PR #1316](https://github.com/ddalcu/mlx-serve/pull/1316). |
| `profile_expert_coding.py` | Coding-expert profiler: runs the 4-bit pack under mlx-lm with every `BailingSparseMoE` router wrapped, tallies top-k expert activations over a code corpus vs a prose baseline, and ranks layers by `conc10 × spec_mass` → `expert_profile_coding.json`. |
| `move_hf_blobs_to_nvme.sh` | Houses the 120 GB HF source checkpoint on the external NVMe (`/Volumes/NVME/hf-cache/ling-3.2-air-blobs`, symlinks left in the hub cache); skips any blob another repo shares. |

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

## Mixed 4/8-bit (coding-expert promotion) + coding eval

`profile_expert_coding.py` profiles the 512-expert routers on code vs prose and finds real
specialization (`spec_mass ≈ conc10 ≈ 0.29`); the converter (`--expert8-profile
--expert8-top 12`) then quantizes the whole expert stack **8-bit** on the 12 layers with the
highest `conc10 × spec_mass` (11, 12, 17, 20, 23, 24, 27, 29, 30, 32, 35, 36) and 4-bit
elsewhere. mlx-serve solves bits per tensor from packed geometry (U32 words: 4-bit g64
gate_proj is (512,768,320), 8-bit is (512,768,640)), so mixed packs load unmodified — same
convention as `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`. Provenance lives in
`mixed-quant.json` next to the shards. Published:
[`teejer/Ling-3.2-air-mlx-mixed-4-8bit`](https://huggingface.co/teejer/Ling-3.2-air-mlx-mixed-4-8bit)
(91.3 GB) and [`…-mixed-4-8bit-vl`](https://huggingface.co/teejer/Ling-3.2-air-mlx-mixed-4-8bit-vl)
(92.2 GB), byte-exact verified.

`coding_eval/` measures the effect: 30 original problems with executable asserts (16 tier-1,
14 tier-2), greedy, sandboxed execution, `validate_problems.py` proves every tier-2 assert
against a reference solution first (bogus asserts caused three false model failures before
that guard existed). Same engine, same client, temperature 0:

| Pack | tier-1+2 | tier-1 | tok/q |
|---|---|---|---|
| Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit | 16/16 *(tier-1 only)* | 16/16 | 230 |
| **Ling-3.2-air-MLX-mixed-4-8bit** | 16/16 *(tier-1 only)* | 16/16 | 818 |
| Ling-3.2-air-MLX-4bit (base) | 13/16 *(tier-1 only)* | 13/16 | 728 |

The mixed pack fixes exactly the base pack's semantic edge-case failures (`case_convert`,
`csv_line`, `palindrome_sub`) — the promotion earned its +23 GB. Speed columns understate
both (same-host contention during evals); Qwen additionally keeps ~3.5× the token economy.
MTP speculative decode confirmed live on these packs with the NextN head (mlx-serve PR
#1316, cherry-picked to the eval build): `MTP head ready (depth=1)`, 85–94% per-draft
acceptance. Tier-2 numbers land with the next eval round; distillation next steps
(including using Claude/ChatGPT as curriculum authors) are in `LORA_DISTILL.md`.
