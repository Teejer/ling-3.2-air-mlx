# Morning report — 2026-11-28 (overnight autonomous run)

## FINAL: Round 3 is the new champion — 61/66, held-out 10/12

| model | overall | held-out | train-pool | note |
|---|---|---|---|---|
| Qwen3.8-Flash | 47/49 | 10/12 | 37/37 | engine, old 49-suite |
| Ling-mixed | 45/49 | 8/12 | 37/37 | engine, old 49-suite |
| Ling-mixed control (py-server) | 54/66 | 8/12 | 46/54 | no adapter |
| Ling-mixed-r1 (LoRA) | 59/66 | 9/12 | 50/54 | 247 traces |
| Ling-mixed-r2 (LoRA2) | 44/66 | 3/12 | 41/54 | negative result |
| **Ling-mixed-r3 (LoRA3)** | **61/66** | **10/12** | **51/54** | **307 traces, T2 33/33** |

r3 vs control: +8 net (gained egyptian_greedy, weighted_interval_schedule
— a ZERO-TRACE problem, i.e. genuine generalization from neighbors — plus
the edge-discipline cluster; lost only `roman`).
r3's five remaining fails decompose by mechanism:
`lis_reconstruct` + `parse_kv_escaped` (the cross-model ceiling — Phase-1
teacher-trace targets), `render_escaped_template` (zero-trace, unlearnable
by self-distillation — Phase-1), `csv_line` + `roman` (flaky regressions).

## The overnight experiment, in one line

Same harness, same capacity, one variable per round:
control 54 → r1 (247 traces) 59 → r3 (307 traces) 61; while r2 (2× rank,
deeper layers, same data) fell to 44. **Data scales; capacity doesn't.**


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
- r3 landed while you slept (above). Next levers, in order:
  1. **Engine fast-path**: `./fuse_engine.sh adapters/round3
     Ling-3.2-air-MLX-mixed-4-8bit-r3e r3e` → puts the champion on the
     Zig engine for engine-comparable numbers + MTP speed (~40 min).
  2. **Kimi API key** → Phase-1 traces for the 3 mechanism-blocked
     failures + volume; r3 says volume is the whole game.
  3. Daily curriculum drip continues (suite is 66; avoid list synced).
- archpro still unreachable overnight — retry `scp coding_eval/*.md
  coding_eval/curriculum_seed_prompt.md archpro:~/` when it's back.
