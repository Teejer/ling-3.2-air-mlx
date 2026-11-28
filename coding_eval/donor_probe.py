"""Probe how a reasoning donor answers the eval contract before scoring it.

Reports where the engine puts chain-of-thought (content vs reasoning_content),
how many tokens it spends getting there, and whether a runnable ```python block
survives in each field — using the same extraction run_eval.py applies. Worth
running before committing to an eval budget: a donor that thinks past
--max-tokens scores NO_CODE and looks far weaker than it is, and a donor that
only writes code inside its thinking needs --extra to change the contract.

Usage:
  python donor_probe.py [--base-url http://127.0.0.1:11234] [--expect GLM]
                        [--limit 3 | --only id,id] [--max-tokens 3072]
                        [--extra '{"reasoning_budget": 2048}']
"""

import argparse
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from problems import PROBLEMS  # noqa: E402
from run_eval import SYSTEM, extract_code, loaded_model, post, run_tests  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:11234")
    ap.add_argument("--expect", default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--extra", default=None)
    args = ap.parse_args()
    extra = json.loads(args.extra) if args.extra else {}

    model = loaded_model(args.base_url, args.expect)
    print(f"model: {model}\nextra: {extra or '{}'}\n")

    probs = PROBLEMS
    if args.only:
        want = set(args.only.split(","))
        probs = [p for p in probs if p["id"] in want]
    else:
        probs = probs[: args.limit]

    rows, raw = [], {}
    for prob in probs:
        t0 = time.time()
        resp = post(args.base_url, {
            "model": args.model_name or model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prob["prompt"]},
            ],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "stream": False,
            **extra,
        })
        secs = time.time() - t0
        ch = resp["choices"][0]
        msg = ch["message"]
        raw[prob["id"]] = resp
        content = msg.get("content") or ""
        think = msg.get("reasoning_content") or msg.get("reasoning") or ""
        finish = ch.get("finish_reason")
        usage = resp.get("usage") or {}
        code_in = [n for n, t in (("content", content), ("reasoning", think))
                   if extract_code(t)]
        status = "NO_CODE"
        if code_in:
            with tempfile.TemporaryDirectory(dir="/private/tmp/opencode") as wd:
                status, _ = run_tests(wd, extract_code(content) or extract_code(think),
                                      prob["tests"])
        rows.append(dict(id=prob["id"], finish=finish, status=status,
                         completion=usage.get("completion_tokens"),
                         reason_chars=len(think), content_chars=len(content),
                         code_in=code_in, secs=round(secs, 1),
                         extra_keys=sorted(set(msg) - {"role", "content"})))
        print(f"{prob['id']:<22} {status:<12} finish={finish:<10} "
              f"{str(usage.get('completion_tokens')):>5} tok  "
              f"think={len(think):>6}ch content={len(content):>5}ch "
              f"code_in={','.join(code_in) or '-':<15} {secs:5.1f}s "
              f"msg_keys={rows[-1]['extra_keys']}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path("/private/tmp/opencode") / f"donor_probe-{stamp}.json"
    out.write_text(json.dumps(dict(model=model, extra=extra,
                                   max_tokens=args.max_tokens,
                                   rows=rows, raw=raw), indent=2))
    n = len(rows)
    print(f"\npassed {sum(r['status'] == 'PASS' for r in rows)}/{n}  "
          f"NO_CODE {sum(r['status'] == 'NO_CODE' for r in rows)}/{n}  "
          f"truncated(finish=length) {sum(r['finish'] == 'length' for r in rows)}/{n}")
    print(f"raw responses -> {out}")
    trunc = [r for r in rows if r["finish"] == "length"]
    if trunc:
        print(f"\nNOTE: {len(trunc)}/{n} hit the token ceiling — raise --max-tokens "
              f"or cap thinking with --extra before the real run.")


if __name__ == "__main__":
    main()
