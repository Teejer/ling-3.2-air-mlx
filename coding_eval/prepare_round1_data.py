"""Prepare Round-1 LoRA dataset from harvest JSONL(s).

Dedupes by (problem id, normalized code), shuffles, splits 95/5, writes
mlx-lm-format train.jsonl/valid.jsonl. Held-out problems must not appear in
any harvest file — asserted, not assumed.

Usage: python prepare_round1_data.py OUT_DIR [harvest/*.jsonl ...]
(default input: newest coding_eval/harvest/*.jsonl)
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from problems import HOLDOUT_IDS  # noqa: E402

def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    srcs = [Path(a) for a in sys.argv[2:]]
    if not srcs:
        srcs = sorted((HERE / "harvest").glob("*.jsonl"))[-1:]
    seen, rows = set(), []
    for f in srcs:
        for line in f.open():
            r = json.loads(line)
            mid = r["meta"]["id"]
            assert mid not in HOLDOUT_IDS, f"held-out {mid} leaked into harvest!"
            code = r["messages"][2]["content"]
            key = (mid, re.sub(r"\s+", " ", code).strip())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"messages": r["messages"]})
    random.Random(13).shuffle(rows)
    n_valid = max(1, len(rows) // 20)
    (out / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:n_valid]))
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[n_valid:]))
    per = {}
    for f in srcs:
        for line in f.open():
            r = json.loads(line)
            per[r["meta"]["id"]] = per.get(r["meta"]["id"], 0) + 1
    print(f"sources: {[f.name for f in srcs]}")
    print(f"{len(rows)} unique traces -> {out} (train {len(rows)-n_valid} / valid {n_valid})")
    print(f"problems covered: {len(per)}")

if __name__ == "__main__":
    main()
