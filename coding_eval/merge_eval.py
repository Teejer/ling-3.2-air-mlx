"""Union several eval runs on the same suite into one best-of file.

Built for the donor's two-pass protocol: a fast pass with thinking off scores most
of the suite in minutes, and a thinking pass re-runs only what that missed. A
problem counts as solved if EITHER pass solved it — that is the donor's ceiling,
and a budget artifact (NO_CODE at finish=length) must never be recorded as an
ability result.

Keeps the winning attempt's row, so status/token counts stay attributable, and
tags every row with which files it appeared in.

Usage:
  python merge_eval.py --out results/donor-merged.json A.json B.json
  python merge_eval.py --out ... --label "GLM donor (nothink+think)" A.json B.json
"""

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="results JSONs, in run order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    runs = []
    for r in args.runs:
        p = Path(r) if Path(r).is_absolute() else HERE / r
        runs.append((p.name, json.loads(p.read_text())))

    per = {}
    order = []
    for name, d in runs:
        for row in d["results"]:
            pid = row["id"]
            prev = per.get(pid)
            if prev is None:
                order.append(pid)
                per[pid] = dict(row, runs=[name])
                continue
            per[pid]["runs"].append(name)
            # a later PASS wins; otherwise keep the row that got furthest
            if row.get("ok") and not prev.get("ok"):
                per[pid] = dict(row, runs=per[pid]["runs"])

    rows = [per[p] for p in order]
    passed = sum(1 for r in rows if r.get("ok"))
    out = {
        "model": args.label or f"merge({len(runs)})",
        "merged_from": [n for n, _ in runs],
        "temperature": 0.0,
        "max_tokens": "|".join(str(d.get("max_tokens")) for _, d in runs),
        "extra": {k: v for _, d in runs for k, v in (d.get("extra") or {}).items()},
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "passed": passed,
        "total": len(rows),
        "results": rows,
    }
    dest = Path(args.out) if Path(args.out).is_absolute() else HERE / args.out
    dest.write_text(json.dumps(out, indent=2))
    for name, d in runs:
        print(f"  {name:52s} {d.get('passed')}/{d.get('total')}")
    print(f"best-of union: {passed}/{len(rows)} -> {dest}")


if __name__ == "__main__":
    main()
