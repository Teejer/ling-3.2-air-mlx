"""Round-1 scoreboard: fused adapter vs baselines, held-out split honest.

Usage: python compare_round1.py [results-dir-glob ...]
Defaults to the newest result file per model slug.
"""
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import HOLDOUT_IDS  # noqa: E402

LABELS = {
    "ddalcu_Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit": "Qwen3.8-Flash",
    "Ling-3.2-air-MLX-mixed-4-8bit": "Ling-mixed",
    "Ling-3.2-air-MLX-mixed-4-8bit-r1": "Ling-mixed-r1 (LoRA)",
    "Ling-3.2-air-MLX-4bit": "Ling-4bit",
}

def newest(slug):
    files = sorted(glob.glob(f"coding_eval/results/{slug}-*.json"))
    return files[-1] if files else None

def main():
    pats = sys.argv[1:] or []
    slugs = list(LABELS)
    data = {}
    for s in slugs:
        f = newest(s)
        if not f:
            continue
        d = json.load(open(f))
        data[LABELS[s]] = {p["id"]: (p["ok"], p["tier"]) for p in d["results"]}

    def cell(name):
        m = data[name]
        tot = sum(ok for ok, _ in m.values())
        hold = [i for i in m if i in HOLDOUT_IDS]
        tr = [i for i in m if i not in HOLDOUT_IDS]
        t = defaultdict(lambda: [0, 0])
        for i, (ok, tier) in m.items():
            t[tier][1] += 1
            t[tier][0] += ok
        tiers = " ".join(f"T{k}:{v[0]}/{v[1]}" for k, v in sorted(t.items()))
        h = f"{sum(m[i][0] for i in hold)}/{len(hold)}"
        p = f"{sum(m[i][0] for i in tr)}/{len(tr)}"
        return f"{tot}/{len(m)}", tiers, h, p

    w = max(len(n) for n in data) + 1
    print(f"{'model':<{w}} {'overall':<9} {'held-out':<9} {'train-pool':<11} tiers")
    scores = {}
    for name in data:
        tot, tiers, h, p = cell(name)
        scores[name] = data[name]
        print(f"{name:<{w}} {tot:<9} {h:<9} {p:<11} {tiers}")

    if "Ling-mixed-r1 (LoRA)" in scores and "Ling-mixed" in scores:
        a, b = scores["Ling-mixed"], scores["Ling-mixed-r1 (LoRA)"]
        common = set(a) & set(b)
        gained = [i for i in sorted(common) if b[i][0] and not a[i][0]]
        lost = [i for i in sorted(common) if a[i][0] and not b[i][0]]
        print(f"\nvs pre-LoRA mixed:  GAINED {gained or '-'}")
        print(f"                    LOST   {lost or '-'}")
        hg = [i for i in gained if i in HOLDOUT_IDS]
        hl = [i for i in lost if i in HOLDOUT_IDS]
        print(f"held-out gained: {hg or '-'}   held-out lost: {hl or '-'}")

if __name__ == "__main__":
    main()
