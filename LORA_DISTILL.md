# LoRA distillation plan — making Ling-3.2-air code like (or better than) its betters

Status: plan (2026-11-27). Baseline that motivated it: mixed-4-8bit pack hits
16/16 tier-1, Qwen3.8-Flash-Next hits 16/16 tier-1, base 4-bit hits 13/16 —
tier-1 is saturated, so `coding_eval` gained 14 tier-2 problems
(`validate_problems.py` proves the asserts against reference solutions).

## What the 2026 literature says (this is load-bearing, not decoration)

1. **Imitating frontier API teachers can *hurt*.** [compete-then-collaborate]
   (arXiv 2607.08255, github.com/shawnkim678/compete-then-collaborate) ranked
   Claude, Codex, Grok, Gemini as teachers with an execution judge, then SFT'd
   an already-competent coder on their *verified* solutions: MBPP pass@1 went
   **76.7% → 72.7%** (7B) and 82.0% → 80.0% (32B). The same problems used as a
   **GRPO/RLVR reward environment** instead improved the student (5.9% → 8.8%
   on held-out competition). They release the task bank, verifier, and LoRA.
2. **The verifier is the curriculum.** (arXiv 2607.09709) Rejection-sampling
   *self*-distillation gated by a strict deterministic verifier compounds
   across rounds (clean-launch 8.8% → 42.2% in 3 rounds, fresh LoRA from base
   each round). Verifier **precision** is what matters: admitting 25% false
   positives costs −1.58pp; *missing* 75% of good candidates costs −0.03pp.
   Our assert-harness is exactly this kind of verifier.
3. **Small verified sets work.** s1K (github.com/simplescaling/s1): 1,000
   curated traces. MapCoder-Lite (EACL'26 findings): teacher trajectories beat
   self-generated ones for *algorithmic diversity*, and only end-to-end
   execution-passing trajectories should be kept.
4. **Fully open distillation data exists.** OpenThoughts (github.com/open-thoughts/open-thoughts):
   1.2M reasoning rows incl. 250k code, fully open, trained OpenThinker3-7B
   past R1-distill. distilabel (argilla-io/distilabel) is the standard
   pipeline framework (generation + feedback + rejection sampling) if we want
   one, though this kit already implements ~80% of the loop.

## Pipeline for this box (M3 Ultra, 512 GB, mlx-serve + mlx-lm)

**Phase 0 — self-distillation, zero teacher, zero ToS questions.**
Ling-mixed serves on :11250; for each problem sample k=8 at temp 0.8; the
harness gates every candidate; passing (prompt, solution) pairs become JSONL.
Anchor pool from open data (OpenThoughts3 code subset, s1K). LoRA SFT via
mlx-lm, fresh adapter per round, re-eval tiers 1+2 between rounds. This is the
literature's best-evidenced move for an already-competent student, and every
byte of data is ours.

**Phase 1 — open local teacher.** Qwen3.8-Flash-Next generates the same
problems ×variants (rename functions, permute constants, mutate tests, dedupe
by hash): it is at parity on tier-1, so it won't teach raw capability, but it
teaches *token economy* (230 vs 818 tok/q) and edge-case discipline. Measure
the other locals first (GLM-5.3-Flash, laya, MiniMax-H3 are already on the
NVMe): whichever beats Qwen on the 30-problem eval becomes the Phase-1
generator. Generation-based KD does not care about tokenizer mismatch — we
train on text.

**Phase 2 — Claude / ChatGPT (the "work accounts" question).**
Technically: yes, trivially — their outputs are just text; no logits needed.
Two real constraints and the way around them:
- **Terms**: both labs' consumer terms prohibit using outputs to train
  competing models; a *work* account adds corporate data governance on top.
  If the account is API/enterprise with training-permissive terms, run Phase 1
  with Claude as the generator and keep only execution-passing traces.
- **Evidence**: even ToS-cleared, *imitation* of frontier style is the
  degrading direction for a competent student (finding 1 above).
- **The ToS-clean *and* evidence-aligned use**: frontier models as **curriculum
  authors, not output donors** — ask Claude/ChatGPT for new problems + hidden
  tests (like compete-then-collaborate's task bank, which deliberately
  redistributes no teacher outputs), validate their tests are executable and
  unambiguous through `validate_problems.py`, then train the student on
  **Ling's own execution-passing solutions** to those harder problems. That is
  Phase 0 with an infinitely creative problem generator.

**Phase 3 — RLVR (parked).** The task bank + asserts is already a GRPO reward
environment. mlx-lm has no GRPO for this MoE; compete-then-collaborate's
TRL+PEFT path (their repo ships GB10/bleeding-edge patches) is the reference
implementation on CUDA. Only worth it if Phase 0–2 plateau.

## Which teacher is "even more capable"? (ranked for this student)

1. **Ling-3.2 flagship (inclusionAI, same family/tokenizer)** — the only
   teacher enabling *logit-level* KD (KL over shared vocab) and native chat
   style; "air" almost certainly descends from it, so the ceiling is what
   distillation is for. Won't fit in 512 GB for fast serving, but mlx-serve's
   in-tree expert-streamer (`kit.streamed`) can run it slowly for overnight
   trace generation. Availability to check: HF inclusionAI.
2. **Claude / ChatGPT (frontier)** — strongest coders available; use as
   curriculum authors (+ verifiers) per Phase 2; direct output distillation
   only under permissive account terms.
3. **Qwen3.8-Flash-Next** — parity, not more capable; the right teacher for
   efficiency/consistency, wrong one for a capability jump.
4. **Unranked locals (GLM-5.3-Flash, laya, MiniMax-H3)** — serve on :11250,
   run the 30-problem eval, promote whichever actually beats Qwen.

**Curriculum-author standings (first-pass validated rate, batches 1-1):**
Claude 9/10 then 10/10 with corrections — most reliable author.
Kimi K3 (via ChatHub) **8/10** — best *design* rate of any teacher (its
math/tier-3 TASK variant delivered tsp_exact_bitmask, matrix_chain, etc.),
but ChatHub flattens every `ref` (repaired by the DFS ladder, so score
post-repair) and its 2 misses were genuine assert errors, hand-fixed per
the established curation rule.
ChatGPT 4/10 → 6/10 — good designs, chronically sloppy expected values.
For trace *donors* (Phase 1), note Kimi K3's license is permissive below
the MaaS/attribution gates (verified 2026-11-27), unlike ChatGPT/Claude
consumer terms.

Honest ceiling: LoRA transfers style, consistency and edge-case discipline —
not pretraining knowledge. Expect tier-2 movement and token-economy gains,
not a frontier leap. The 12 coding-specialist expert layers identified by
`expert_profile_coding.json` are the natural LoRA target layers, which makes
the quant story and the training story point at the same tensors.

## Round-1 results (2026-11-27 night)

Data: 247 executor-verified self-harvest traces (mixed pack, temp 0.8, 42
problems, cap 6/problem), 12-problem held-out split excluded at harvest.
Train: rank 8 (config rank silently ignored), last-16 layers, attention +
shared experts + SwitchGLU per-expert, 250 iters = 1 epoch, ~12 min.
Eval: mlx_lm.server + adapter, full suite (66 problems at the time).

Scoreboard: r1 59/66; vs engine baseline on the shared 49-problem subset
45/49 = 45/49, but GAINED `eval_expr` + `shortest_cycle_length` (both
held-out) and LOST `case_convert` + `egyptian_greedy`. Held-out 8/12 ->
9/12. Adapter probe: load_adapters takes train-seq loss 1.4220 -> 0.0617
— adapter fully ACTIVE; the memorization is verbatim and transfer is
~nil, which is the expected small-n outcome. A py-server control run
(no adapter) is pending to split adapter effect from serving drift.

Round-2 levers, ranked: (1) more traces — round-2 harvest armed for the
12 new problems, and k-sampling more problems; (2) target the 12 promoted
coding layers via train_round2.py (snapshot-restore selective LoRA —
written, needs supervised smoke); (3) fewer iters or lower scale to trade
memorization for generalization; (4) rank 16 actually applied.

## Round-2 results (2026-11-28 ~05:15): NEGATIVE — targeting backfired

r2 (12 promoted layers, rank 16/scale 32, per-expert adapters, 307 traces):
**44/66, held-out 3/12** — 16 regressions vs r1, 5 held-out losses. Same
stack: control 54, r1 59, r2 44. Interpretation: adapter capacity
(3.3 GB trainable incl. per-expert LoRA on EARLY layers) wildly exceeded
the data (307 traces); the model memorized answer shapes and lost
general skills (roman, rle, rotate_matrix, histogram...). Round-1's
smaller footprint (last-16, rank 8, scale 20) was already at the
capacity/data edge; going deeper+heavier crossed it.

Round-3 plan (single-variable): r1's EXACT recipe (last 16 layers, rank 8,
scale 20, lr 5e-5) on the larger 307-trace set — tests whether data volume
alone buys held-out transfer. If yes, scale data (teacher traces for the
zero-trace problems) rather than adapter capacity.

## Immediate next steps

0. **mlx-lm training support: VERIFIED with one local patch.** mlx-lm
   0.32.2 ships `bailing_moe_v3` and trains LoRA on quantized layers
   natively, but its `sanitize()` hard-refuses quantized `kv_b_proj`
   because the MLA reparameterization (kv_b_proj -> embed_q/unembed_out)
   cannot split packed tensors. Patched site-packages to dequantize ONLY
   that tensor at load time (everything else trains QLoRA-style on the
   quantized pack, matching serving). Adapters must therefore target
   shared experts + attention projections that keep their serving names —
   NOT embed_q/unembed_out (the Zig engine serves kv_b_proj directly; and
    mlx-lm FUSE saves embed_q/unembed_out into the fused pack, so the
    engine rejects it with "MISSING WEIGHT: kv_b_proj" — fix is lossless
    pack rename surgery, kv_b_proj = concat(swapaxes(embed_q), unembed_out)
    (round 1 put no adapter on those tensors), or an engine-side loader
    alias. Serving-with-adapter fallback meanwhile: mlx_lm.server
    --adapter-path (reports model id inclusionAI/Ling-3.2-air).
    REVISION after round 1: the adapter manifest proves mlx-lm 0.32.2 DOES
    wrap SwitchGLU — per-expert LoRA [512, r, in] on the routed stacks.
    Round 2 should target the 12 promoted coding layers
    {11,12,17,20,23,24,27,29,30,32,35,36} via a custom layer-selection
    script (CLI --num-layers only takes the last N; round 1 covered
    layers 26-41). NOTE (resolved): the rank gotcha was MY yaml key — mlx-lm
    reads `lora_parameters:`, not `lora:`. Corollary: if a custom trainer
    forces different rank/scale at wrap time (train_round2 does: 16/32),
    EDIT adapters/*/adapter_config.json lora_parameters to match, or
    mlx_lm.server/load_adapters wraps with defaults and rejects the
    weights (server then 404s every request).

1. `--harvest` mode in run_eval.py: emit JSONL of passing traces
   (messages-formatted with Ling's chat template) for Phase 0 data.
2. mlx-lm LoRA support check for `bailing_moe_v3` (adapter targets: attention,
   shared experts, routed experts of the 12 promoted layers if supported).
3. Rank the local teachers on the 30-problem eval (serve, `--expect`, one
   run each).
4. Round 1: ~2–5k verified traces, LoRA SFT, re-eval tier-2.
