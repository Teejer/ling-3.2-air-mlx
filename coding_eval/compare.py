"""Summarize coding_eval/results/*.json into a comparison table."""
import json
from pathlib import Path


def rate(by_id, tiers=None):
    ids = {i: v for i, v in by_id.items() if tiers is None or i in tiers}
    if not ids:
        return None
    return sum(1 for v in ids.values() if all(v)) / len(ids)


rows = []
for p in sorted((Path(__file__).parent / "results").glob("*.json")):
    d = json.loads(p.read_text())
    by_id, tier_of = {}, {}
    for r in d["results"]:
        by_id.setdefault(r["id"], []).append(r.get("ok", False))
        tier_of[r["id"]] = r.get("tier", 1)
    t1 = {i for i, t in tier_of.items() if t == 1}
    t2 = {i for i, t in tier_of.items() if t == 2}
    toks = [r.get("completion_tokens") or 0 for r in d["results"]]
    secs = [r.get("seconds") or 0 for r in d["results"]]
    fails = sorted({r["id"] for r in d["results"] if not r.get("ok")})
    rows.append((d["model"], d["temperature"], rate(by_id), rate(by_id, t1),
                 rate(by_id, t2), len(by_id), sum(toks) / max(len(toks), 1),
                 sum(secs) / max(len(secs), 1), fails))

rows.sort(key=lambda r: -(r[2] or 0))
print(f"{'model':<52} {'temp':>4} {'all':>7} {'t1':>7} {'t2':>7} {'n':>3} {'tok/q':>6} {'s/q':>5}")
for m, t, a, s1, s2, n, tok, sec, fails in rows:
    f = lambda x: "  -  " if x is None else f"{x * 100:>5.1f}%"
    print(f"{m:<52} {t:>4} {f(a)} {f(s1)} {f(s2)} {n:>3} {tok:>6.0f} {sec:>5.1f}")
    if fails:
        print(f"{'':>52} fails: {', '.join(fails)}")
