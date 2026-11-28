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
4. **GLM-5.3-Flash — MEASURED, it is the donor (2026-11-28).** Not "unranked"
   anymore: `Jundot/GLM-5.3-Flash-oQ4e-mtp` scores **100/104** on the suite
   (92/104 greedy with thinking off in 10.5 min, +7 of the misses with thinking,
   +1 at a 24k budget) against r3's 61/66 and the Qwen control's 47/49. It is the
   local teacher of record; details and the two-pass protocol are in
   `coding_eval/DONORS.md`. **GLM-5.4 does not exist** (0 hits on HF; newest from
   zai-org is GLM-5.3-Flash, 2026-10-15). The bigger non-Flash `zai-org/GLM-5.3`
   (78 layers, `glm_moe_dsa`, 756 GB BF16) has no MLX build — only AMD/Intel MXFP4
   at ~400 GB for other stacks — so a ~4-bit version would not fit resident next to
   the student either. The 4 problems the donor cannot solve are worth API cents,
   not another 200 GB. **laya stays excluded** (user: it cannot chat at all — pure
   decision model, so it can neither author nor solve nor donate traces). Note the
   HF cache spells MiniMax as MiniMax-M3-oQ3, not H3.

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

**Day-2 batches (2026-11-28, suite 66 → 95).** Kimi K3 **10/10** clean on the
math/tier-3 variant — one ChatHub glyph drop (`≡`) restored, no teacher error.
ChatGPT 9/10 plus `expand_counted_pattern` recovered by `repair_element.py`,
so 10/10 *after* repair; its `ref`s arrived correctly indented this time, the
only damage was half-escaped backslashes. Claude 8/10 clean, 1 recovered
(`decode_nested_repeat`, ref hand-restored from prompt semantics, passes all
its asserts), 1 dropped (`full_justify`) — 5 of its asserts contradict its own
prompt ("every line exactly width chars" but expects a 10-char line at width
16), and rewriting that many expected values is authoring, so forward-only
policy dropped it rather than spending a correction message. Net +29.

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

Round-3 result (same night): r1's recipe on 307 traces = **61/66,
held-out 10/12, tier-2 33/33** — new champion, +7 over control, +2 over
r1, and it gained a ZERO-TRACE problem (`weighted_interval_schedule`),
i.e. real transfer, not memorization. Verdict: scale data, not capacity.
Remaining fails are mechanism-blocked: 2 ceiling problems +
`render_escaped_template` (zero-trace) → Phase-1 teacher traces;
`csv_line`/`roman` flaky.

## Round-4 (started 2026-11-28 ~11:50): the donor's data, same capacity

Data is the only variable: **438 verified traces** (the 307 self-harvested that made
r3 the champion + **131 GLM-5.3-Flash donor traces** covering the 38 problems the bank
had never touched — every one of the 38 contributed at least one, 417 train / 21
valid, 90 problems). Capacity is byte-identical to r3 — last 16 layers, `self_attn` +
`shared_experts`, rank 8, scale 20, lr 5e-5, batch 1, seed 0, 660.5M trainable
(0.531%) — with iters scaled to the data (250 × 438/307 ≈ 350, still ~0.85 of one
epoch, the same fraction r3 saw). `coding_eval/lora_round4.yaml`.

The donor traces were harvested at temp 0.8 with `{"enable_thinking": false}` so they
look like what the student produces at inference (the student has no thinking phase),
and every trace was kept only because its extracted code passed that problem's tests
on this machine.

**Result: NEGATIVE.** Champion stays r3.

Scoreboard, all four rows measured on the **identical 104-id coverage** (greedy, same
system prompt, executed tests, `mlx_lm.server` + adapter — the 38-problem holes in the
older runs are now filled: `results/Ling-3.2-air-MLX-mixed-4-8bit-r3-104.json` is the
union of r3's two runs, `CONTROL-untrained-104` is the untrained pack):

| model | 104 suite | trainable 92 | held-out 12 |
|---|---|---|---|
| control — untrained pack | 81 | 73 | 8 |
| **r3 — 307 self-harvested traces** | **92** | **82** | **10** |
| r4 — r3's bank + 131 GLM donor traces | 88 | 79 | 9 |
| GLM-5.3-Flash donor (the teacher) | 100 | 89 | — |

So distillation is worth **+11 over the untrained pack** (81 → 92, and +2 on problems it
never saw), and r4 gave 4 of that back. r4 lost on *both* kinds of material: −2 on the
66 ids r3 was trained on (`egyptian_greedy`, `stable_frequency_sort`,
`weighted_interval_schedule` out; `csv_line` in) and −2 on the 38 new ones (4 gained, 6
lost) **despite** 131 traces aimed at exactly those problems. Deterministic re-runs at
temp 0 confirmed the 9 consistent failures are capability, not sampling.

The reason is visible in the baseline: the r3 champion already solved **31 of the 38**
problems the donor traces were harvested for. Gradient steps spent imitating a teacher on
material the student already pass buy nothing and cost distribution — plus a style tax,
mean completion 184 → 222 tokens as the student picks up GLM's idiom.

**The headroom, now that a control exists** (`donor_value.py` with full-coverage refs).
Of the 92 trainable problems, r3 fails 10, and they split three ways:

| bucket | n | ids | what to do |
|---|---|---|---|
| **A additive** — r3 fails, untrained fails, donor passes | 4 | `csv_line`, `decode_nested_repeat`, `expand_counted_pattern`, `shortest_threshold_slice` | harvest donor traces for these **only** |
| **B training regression** — r3 fails but the *untrained pack* passes | 5 | `directed_euler_walk`, `grid_keys_shortest_path`, `min_assignment_cost`, `roman`, `solve_linear_congruence` | anti-drift replay, see below |
| **C ceiling** — no local model solves it | 1 | `render_escaped_template` | the entire remaining paid ask (~$0.22) |

Bucket B is the uncomfortable finding: **r1–r3's +9 was paid for partly by breaking 5
problems the base model could already do.** That is the same failure as r2 and r4, just
milder — collateral from LoRA drift on problems the training set didn't cover. It also
suggests the cheapest fix in the project: for B, harvest the *untrained* pack's own
passing samples (temp 0.8, executor-verified) and add them as replay anchors. Those are
the student's own outputs, so no rights question, and the target is "stop drifting", not
"learn a new style".

**Round-5 design** (one variable at a time, after r3e): r3's bank of 307 + ~4 donors per
A problem + ~6 self-replay per B problem (≈350 traces, not 438), iters back to 250,
capacity frozen at the r1/r3 footprint. If that recovers B without losing the 82, drift
replay is real and becomes a standing part of the pipeline.

**Gotcha that cost 15 minutes:** `mlx_lm.lora` (and plain `mlx_lm.load`) died twice at
`mx.eval(model.parameters())` with `[METAL] Command buffer execution failed: Caused GPU
Timeout Error (kIOGPUCommandBufferCallbackErrorTimeout)` — after this box had served
186 GB of GLM. Not memory (119 GB free, 0 swap), not another GPU client (unloading Tess
did not help), not the disk (900 MB/s), not a broken pack: it is the **cold mmap
page-in path** — the GPU touching memory-mapped shard pages that are no longer in the
unified cache blows the command-buffer watchdog. Fix is one line and takes 10 s when
the cache is warm:

    cat Ling-3.2-air-MLX-mixed-4-8bit/*.safetensors > /dev/null

Warm the pack before any training run that follows a big serve/evict cycle; r1–r3 never
hit this because they ran right after an eval that had just read the same files.

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
