"""Build (and later harvest) the paid-teacher batch for problems no free donor solves.

`donor_value.py` says WHICH ids still need a stronger teacher (`kimi_ask`: the donor
failed them and so did every local run). This turns that list into a Moonshot-style
batch file, and turns the returned file into verified training traces.

The ask is deliberately narrow. Every problem that a local donor can solve is free and
already provenance-clean, so paying per output token only makes sense for the ceiling
set — and only at the cheapest reasoning effort that still clears a problem. Two
rounds, never more (forward-only policy):

  round 1   effort low,  n=1 per problem     -> cheap sweep of the whole ask-list
  round 2   effort max,  n=3 per problem     -> only ids round 1 failed to reproduce

Traces are kept only if the extracted code PASSES the problem's tests here, on this
machine, at ingest time. A teacher's claim that it solved something is not evidence;
executed asserts are. Held-out ids are excluded by default: their traces could never
be trained on, so asking about them costs money for nothing.

Usage:
  python kimi_batch_request.py --plan                        # newest donor-value json
  python kimi_batch_request.py --plan --ids a,b --effort max --n 3
  python kimi_batch_request.py --ingest batches/kimi-responses.jsonl
"""

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

CONTRACT = "coding-eval-ask-v1"
# The contract the teacher is held to. Same system prompt the student sees (parity is
# the point) plus the one thing eval needs: a single runnable block.
ASK_TAIL = (
    "\n\nSolve it in one ```python block containing only the completed function. "
    "Every given example must hold. Do not add tests, imports you do not use, or "
    "placeholder branches."
)
LICENSING = ("Kimi K3.1-Pro License grants use, fine-tuning and derivatives; the "
             "MaaS and UI-attribution clauses trigger only above $20M revenue / 100M "
             "MAU and exempt internal use. Outputs are ours to train on. A derivative "
             "of Kimi WEIGHTS would carry attribution; a student trained on these "
             "traces does not.")


def problem_index():
    from problems import PROBLEMS, HOLDOUT_IDS
    return {p["id"]: p for p in PROBLEMS}, set(HOLDOUT_IDS)


def newest_donor_value():
    hits = sorted(HERE.glob("results/donor-value-*.json"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def plan(args):
    probs, holdout = problem_index()
    if args.ids:
        ids = [i.strip() for i in args.ids.split(",") if i.strip()]
        src = "--ids"
    else:
        path = Path(args.ask_file) if args.ask_file else newest_donor_value()
        if not path:
            raise SystemExit("no donor-value-*.json yet — run donor_value.py first, "
                             "or pass --ids")
        ids = json.loads(Path(path).read_text())["kimi_ask"]
        src = str(path)

    dropped_hold = [i for i in ids if i in holdout]
    if dropped_hold and not args.keep_holdout:
        ids = [i for i in ids if i not in holdout]
    unknown = [i for i in ids if i not in probs]
    if unknown:
        raise SystemExit(f"ids not in the suite: {unknown}")
    if not ids:
        print("ask-list is empty: nothing needs a paid teacher")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = HERE / "batches" / f"kimi-ask-{stamp}.jsonl"
    side = out.with_suffix(".sidecar.json")
    in_tok = 0
    with out.open("w") as fh:
        for pid in ids:
            p = probs[pid]
            for n in range(args.n):
                cid = f"{pid}#r{n}"
                fh.write(json.dumps({
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": [
                            {"role": "system", "content": args.system},
                            {"role": "user", "content": p["prompt"] + ASK_TAIL},
                        ],
                        # n>1 needs heat or the samples are identical
                        "temperature": 0.0 if args.n == 1 else args.temperature,
                        "max_tokens": args.max_tokens,
                        "reasoning_effort": args.effort,
                    },
                }, ensure_ascii=False) + "\n")
                in_tok += len(p["prompt"]) // 4 + 60  # rough, for the cost sheet

    requests = len(ids) * args.n
    est_out = requests * args.est_out_tokens
    cost = in_tok / 1e6 * args.price_in + est_out / 1e6 * args.price_out
    side.write_text(json.dumps({
        "contract": CONTRACT, "created": stamp, "ask_source": src, "model": args.model,
        "reasoning_effort": args.effort, "samples_per_problem": args.n,
        "max_tokens": args.max_tokens, "ids": ids, "requests": requests,
        "dropped_holdout": dropped_hold, "licensing": LICENSING,
        "pricing_assumption": {"usd_per_m_in": args.price_in,
                               "usd_per_m_out": args.price_out,
                               "est_out_tokens_per_request": args.est_out_tokens,
                               "confirm_at_purchase": True},
        "est_usd": round(cost, 2), "budget_usd": args.budget,
        # filled in by --ingest
        "results": {},
    }, indent=2, ensure_ascii=False))

    print(f"ask source : {src}")
    print(f"problems   : {len(ids)}  x {args.n} @ effort={args.effort} "
          f"max_tokens={args.max_tokens}  -> {requests} requests")
    if dropped_hold:
        print(f"held out   : {len(dropped_hold)} not asked ({', '.join(dropped_hold)})")
    print(f"est cost   : ${cost:.2f} "
          f"(in {in_tok/1e6:.2f}M @ ${args.price_in}/M + out {est_out/1e6:.2f}M @ "
          f"${args.price_out}/M)  budget ${args.budget:.0f}")
    if args.budget and cost > args.budget:
        print(f"  OVER BUDGET by ${cost - args.budget:.2f} — cut the list or the effort")
    print(f"\n-> {out}\n-> {side}")
    return 0


def ingest(args):
    """Batch responses -> traces, keeping only code that passes the tests here."""
    from run_eval import extract_code, run_tests, SYSTEM
    probs, holdout = problem_index()
    src = Path(args.ingest)
    traces = src.with_name(f"traces-{src.stem}.jsonl")
    kept = 0
    tally = {}
    usage = {}
    seen = set()
    outcomes = {}
    with traces.open("w") as out:
        for line in src.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("custom_id", "?")
            pid = cid.split("#")[0]
            prob = probs.get(pid)
            if prob is None:
                print(f"  {cid}: id not in suite — skipped")
                continue
            if pid in holdout and not args.keep_holdout:
                outcomes[cid] = "holdout_not_trainable"
                continue
            body = (rec.get("response") or {}).get("body") or {}
            for u, v in (body.get("usage") or {}).items():
                usage[u] = usage.get(u, 0) + (v if isinstance(v, int) else 0)
            ch = (body.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            content = msg.get("content") or ""
            if not content.strip():
                content = msg.get("reasoning_content") or msg.get("reasoning") or ""
            code = extract_code(content)
            if not code:
                outcomes[cid] = "NO_CODE"
                continue
            with tempfile.TemporaryDirectory(dir="/private/tmp/opencode") as wd:
                status, detail = run_tests(wd, code, prob["tests"])
            if status != "PASS":
                outcomes[cid] = f"FAIL:{status}"
                continue
            norm = re.sub(r"\s+", " ", code).strip()
            if norm in seen:
                outcomes[cid] = "duplicate"
                continue
            seen.add(norm)
            kept += 1
            tally[pid] = tally.get(pid, 0) + 1
            outcomes[cid] = "PASS"
            out.write(json.dumps({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prob["prompt"]},
                    {"role": "assistant", "content": "```python\n" + code + "\n```"},
                ],
                "meta": {"id": pid, "tier": prob.get("tier", 1),
                         "model": rec.get("model") or body.get("model") or args.model,
                         "source": "kimi-batch", "custom_id": cid,
                         "provenance": {"teacher": args.model,
                                        "reasoning_effort": rec.get("reasoning_effort"),
                                        "batch_file": src.name,
                                        "verified_by": "executed unit tests",
                                        "license": "kimi-k3.1-pro"},
                         "thinking": bool(msg.get("reasoning_content")),
                         "thinking_chars": len(msg.get("reasoning_content") or "")},
            }, ensure_ascii=False) + "\n")
    side = src.with_name(f"{src.stem}.outcomes.json")
    side.write_text(json.dumps({"contract": CONTRACT, "per_request": outcomes,
                                "usage": usage, "traces": str(traces)}, indent=2))
    from collections import Counter
    c = Counter(v.split(":")[0] for v in outcomes.values())
    print(f"requests {len(outcomes)} -> {'  '.join(f'{k}={v}' for k, v in c.most_common())}")
    print(f"kept {kept} verified traces over {len(tally)} problems -> {traces}")
    if usage:
        print(f"token usage: {usage}")
    print(f"-> {side}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--ingest", default=None, help="batch response JSONL")
    ap.add_argument("--ask-file", default=None, help="donor-value-*.json (default newest)")
    ap.add_argument("--ids", default=None, help="explicit comma list, skips the ask-file")
    ap.add_argument("--model", default="kimi-k3.1-pro",
                    help="the served id on the Kimi API; kimi-k3.1-pro and kimi-k3 are "
                         "the two that support Batch (60%% of list)")
    ap.add_argument("--effort", default="low", choices=["low", "max"],
                    help="low first; max only for ids the cheap round missed")
    ap.add_argument("--n", type=int, default=1, help="samples per problem")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="only used when --n > 1")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--system", default=None, help="defaults to the student's own system prompt")
    ap.add_argument("--budget", type=float, default=40.0)
    ap.add_argument("--price-in", type=float, default=1.0)
    ap.add_argument("--price-out", type=float, default=6.0,
                    help="Moonshot first-party Batch, ~60%% of list; CONFIRM at purchase")
    ap.add_argument("--est-out-tokens", type=int, default=4000,
                    help="per request; max effort roughly doubles it")
    ap.add_argument("--keep-holdout", action="store_true")
    args = ap.parse_args()

    if args.system is None:
        from run_eval import SYSTEM
        args.system = SYSTEM
    if bool(args.ingest) == bool(args.plan):
        raise SystemExit("pass --plan or --ingest")
    return ingest(args) if args.ingest else plan(args)


if __name__ == "__main__":
    raise SystemExit(main())
