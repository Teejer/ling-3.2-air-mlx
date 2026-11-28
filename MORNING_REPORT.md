# Morning report — 2026-11-28 (overnight autonomous run)

## Scoreboard (py-server stack, identical conditions where noted)

| model | overall | held-out | train-pool | note |
|---|---|---|---|---|
| Qwen3.8-Flash | 47/49 | 10/12 | 37/37 | engine, old 49-suite |
| Ling-mixed | 45/49 | 8/12 | 37/37 | engine, old 49-suite |
| Ling-mixed control (py-server) | 54/66 | 8/12 | 46/54 | no adapter |
| **Ling-mixed-r1 (LoRA)** | **59/66** | **9/12** | 50/54 | **champion** |
| Ling-mixed-r2 (LoRA2) | 44/66 | 3/12 | 41/54 | negative result |
| Ling-mixed-r3 (LoRA3) | running | — | — | lands ~07:15 |

r1 vs control on identical serving: **+5 net** (gained `csv_line`,
`palindrome_sub`, `lru`, `coin_change`, `interval_overlap_count`,
`normalize_path`, `shortest_cycle_length`; lost `case_convert`,
`fraction_add`). Held-out 8→9. Adapter verified ACTIVE (train-seq loss
1.42→0.06 via load_adapters).

## What happened

1. **Suite 55→66**: Kimi batch 10/10 digested (8 auto-repaired from
   ChatHub's flattening + 2 hand-fixed provably-wrong asserts); ChatGPT
   share-link re-run +6. Validator 50/50. Holdout = 12 problems, excluded
   from all training data at harvest time.
2. **Phase-0 harvest**: 247 + 60 = 307 executor-verified Ling traces.
   `render_escaped_template` + `weighted_interval_schedule` got ZERO
   passing samples at temp 0.8 — untrainable by self-distillation, they
   are Phase-1 (teacher-trace) targets.
3. **Round 1 LoRA** (last-16 layers, rank 8, 250 iters): +5 net, held-out
   +1. LoRA does what the plan said: edge-discipline gains.
4. **Round 2 LoRA** (12 promoted layers, rank 16, per-expert): -15,
   held-out collapse. Capacity >> data. Lesson: scale DATA, not adapters.
5. **Round 3** (single variable: r1 recipe × 307 traces) running now.

## Gotchas killed tonight (all documented in LORA_DISTILL.md)

- mlx-lm yaml key must be `lora_parameters:` (not `lora:`) — root cause of
  the "rank ignored" mystery.
- `load_adapters` re-wraps from `adapter_config.json` — custom trainers
  must sync rank/scale there or the server 404s everything.
- `mlx_lm.server` registers model as `default_model` — run_eval has
  `--model-name` now.
- Fused packs save `embed_q/unembed_out`; the Zig engine wants
  `kv_b_proj` → `coding_eval/fix_fused_pack.py` (dry-tested on r1 pack,
  7 MLA tensors) + `fuse_engine.sh` are ready to put r1/r3 on the fast
  engine.

## For you

- **Nothing blocking.** Daily drip unchanged: paste seed prompt (avoid
  list refreshed to 66 ids, on archpro when it's back online — it went
  down ~23:00 and the retry is pending), share links for ChatGPT/Claude
  are enough now (no manual copy) — `coding_eval/share_grab.py <url> <name>.json`.
- Kimi: no correction message needed (both broken problems hand-fixed).
  ChatGPT: correction optional for `matrix_diagonal_difference` only
  (near-dup, likely skip).
- When r3 lands: if held-out ≥ r1, next lever is teacher traces (Kimi API
  key unlocks the two zero-trace problems + volume); if < r1, r1 stands
  and we push r1 onto the engine via fuse_engine.sh.
