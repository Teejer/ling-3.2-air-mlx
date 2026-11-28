# Donor field for the trace bank

Who we ask to write reference solutions when the student cannot. Traces are
provenance-tagged; the bank is student-agnostic (only a *weights* derivative
inherits a teacher's attribution).

## Rights (checked 2026-11-28)

| Source | Local run | Outputs for training | Notes |
| --- | --- | --- | --- |
| GLM-5.3-Flash (any MLX quant) | yes | yes | open weights |
| Kimi K3.1-Pro / REAP prunes | yes | **yes** | Kimi K3.1-Pro License = MIT grant ("run, deploy, fine-tune, create derivative works") + two thresholds: MaaS business with >$20M/12mo revenue needs a separate agreement; products >100M MAU or >$20M/month must display "Kimi K3.1-Pro". §4 exempts internal use entirely. Our harvest and our published adapters are far under both. |
| Moonshot API (K3.1 Pro) | — | yes | same rights, priced: first-party Batch ≈ 60% of list |
| ChatGPT / Claude chat | — | **no** | ToS forbids training → curriculum-author role only |

So a *local* Kimi is legally the cleanest strong teacher we have: free, and the
license explicitly permits fine-tuning and derivatives.

## Weight-schema reality (why the first GLM pull would not load)

`:11234` is MLX-Serve.app 26.11.4 with a native `glm5_next` plugin. It loads the
**engine export layout**, not the HF-style MLX layout:

| layout | naming | experts | loads in 26.11.4 |
| --- | --- | --- | --- |
| engine | `language_model.model.*`, `attn_hc.base`, `forget_gate.*`, joined `conv1d`, `mtp.0.block.*` | stacked (≈3.1k keys) | **yes** |
| HF-MLX | `model.language_model.*`, `hc_attn_base`, `f_a_proj`, `q_conv1d`+`k_conv1d` | per-expert (≈114k keys) | no → `MissingWeight` |

`trunk.zig probePrefix` already tries `language_model.model` / `model.language_model`
/ `model`, so the failure is *not* the prefix — it is the per-element schema.

Known packs by layout:

- engine layout, ungated: `Jundot/GLM-5.3-Flash-oQ4e-mtp` (186.4 GB, MTP, imatrix-
  calibrated, updated 11-19), `TensorFold/GLM-5.3-Flash-MLX-oQ4-MTP` (185.7 GB, MTP),
  `pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit` (181.9 GB, no MTP)
- HF layout, refused: `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` (169 GB, the one we
  pulled first — dead weight until an alias loader exists), `mlx-community/GLM-5.3-Flash-4bit`,
  `orcarouter/GLM-5.3-Flash-MLX` (1.1 TB), `Solstice-AI/*-oQ4e-DFlash2` (mislabelled, HF layout)
- `ddalcu/GLM-5.3-Flash-MLX-Serve-iQ-MLX-5.1bpw` (the plugin's gate model) is 404: the
  plugin was built against packs the engine itself quantized.

Donor of record: **`Jundot/GLM-5.3-Flash-oQ4e-mtp`**.

## Kimi K3.1 as a donor — staged, decided 2026-11-28

`pipenetwork/Kimi-K3.1-Pro-REAP80-code-MLX-mxfp4-q8` — 350 GB, 95 shards,
model_type `kimi_k31`, 93 layers / hidden 7168 / 96 heads, `linear_attn_config`
KDA + gated MLA with DSA indexer, latent MoE, 1M ctx, ships its own remote code
(`kimi_k31.py` + tiktoken). REAP = **179 of 896 routed experts kept per layer**,
code-weighted calibration, top-16 of 179 routed. MLX-Serve 26.11.4 has **zero**
`kimi_k3*` support, so it cannot be served there today.

- **A. GLM first.** Run the 104-suite greedy donor pass, then `donor_value.py`:
  `CEILING` (donor-fail ∧ every local run fail) is the true size of the Kimi ask.
- **B. Validate the prune before spending Zig days.** Serve the REAP pack through
  the venv with `mlx_lm.server --trust-remote-code` (works today, no engine change)
  over the CEILING subset only. Two unknowns to settle: does a 20%-expert prune
  still clear problems GLM-5.3-Flash cannot, and real tok/s on this box — the pack's
  own README says each decoded token reads 70-85 GB, i.e. single-digit tok/s and an
  overnight full run, and it warns of real degradation outside code. RAM: 350 GB
  resident means Ling must be evicted; the two never co-reside.
- **C. `kimi_k31` plugin for the fork** (authorized 2026-11-28; lands on
  `Teejer/mlx-serve`, upstream PR discussed after). Reuse is high because the
  `kda_hybrid` base already owns most of the op surface:

  | K3.1 piece | status in the fork |
  | --- | --- |
  | KimiDeltaAttention (`f_a/f_b`, `g_a/g_b`, `b_proj`, `o_norm`, short conv) | reuse `kda_hybrid/kda.zig` (GLM's arms are the same shapes) |
  | DSA indexer (`wq_b`, `wk`, `k_norm`, `weights_proj`, index_topk 2048) | reuse `glm5_next` |
  | gated MLA (`q_a/q_b`, `kv_a_proj_with_mqa`, `g_proj` output gate) | reuse + the output-gate arm |
  | sparse MoE | reuse `blocks/moe.zig`; e2m1 MoE Metal kernels already exist |
  | SiTU activation (`hidden_act: situ`, beta / linear_beta) | **new**, small |
  | Attention Residuals (`attn_res_block_size`) | **new**, needs per-block residual cache |
  | latent MoE (`routed_expert_hidden_size`, expert down/up, `latent_moe_use_norm`) | **new**, wraps the MoE block |
  | REAP per-layer `expert_counts` + `expert_bank_split` (hi/lo width banks) | **new**, loader + block change |
  | MXFP4 experts as stored (MLX `mxfp4`) | verify against `sdk/quant` (has nvfp4/mxfp8, not mxfp4) |

  Plugin system is cheap to extend (`src/plugins/_template`, `plugins.generate(b)`
  auto-discovers a directory; `proof_test.zig`/`served_test.zig` are the harness).

  B gates the *urgency*, not the value: C is upstreamable on its own because
  `kimi_k31` packs exist in the wild and the engine refuses them all.

## Open

- Delete `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` (169 GB) or keep it as the corpus
  for an HF-layout alias loader — decide once the Jundot donor is serving.
