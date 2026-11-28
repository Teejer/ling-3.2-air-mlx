"""Score a donor run against what the student side already does.

The donor question is not "what did GLM score" but "which problems does it solve
that nothing local can" — those are where free local traces replace a paid batch,
and the complement is the list a Kimi batch is actually worth money for. Every
problem id on the union of the runs is bucketed:

  DONOR_ONLY  donor PASS, every local run FAIL   -> harvest here, no API spend
  BOTH        donor PASS, some local run PASS    -> student can self-supply
  LOCAL_ONLY  donor FAIL, some local run PASS    -> student edge, ignore donor
  CEILING     donor FAIL, every local run FAIL   -> strongest-teacher territory
  NEW_*       no local run covered this id (suite grew past the last student run)

Donor failures are also tallied by status: a wall of NO_CODE means the token
budget is clipping the chain of thought, not that the model cannot solve them.

Usage:
  python donor_value.py --donor results/donor-glm-merged.json   # best-of donor passes
                        [--ref GLOB ...]        # repeatable, newest match each
                        [--donor-match GLM]
"""

import argparse
import glob
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_REFS = ("*mixed-4-8bit-r3*", "*mixed-4-8bit-pyserver*", "*Qwen3.8*")


def newest(pattern):
    hits = sorted(glob.glob(str(HERE / "results" / pattern)), key=lambda p: Path(p).stat().st_mtime)
    return hits[-1] if hits else None


def load(path):
    d = json.loads(Path(path).read_text())
    per = {}
    for r in d["results"]:
        prev = per.get(r["id"])
        ok = bool(r.get("ok"))
        # keep the best attempt per id, but remember a failing status for forensics
        if prev is None or ok:
            per[r["id"]] = dict(ok=ok, status=r.get("status", ""),
                                tok=r.get("completion_tokens"), secs=r.get("seconds"))
    return d, per


def label(path, model):
    n = Path(path).name
    if "r3" in n:
        return "ling-r3"
    if "r2" in n:
        return "ling-r2"
    if "r1" in n:
        return "ling-r1"
    if "pyserver" in n:
        return "ling-ctl"
    if "qwen" in n.lower():
        return "qwen"
    return (model.split("/")[-1][:10] or "ref")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", default=None)
    ap.add_argument("--donor-match", default="GLM")
    ap.add_argument("--ref", action="append", default=None,
                    help="glob under results/, newest match used; repeatable. "
                         f"Default: {' '.join(DEFAULT_REFS)}")
    ap.add_argument("--holdout", action="store_true",
                    help="include held-out ids in the ask-lists (default: excluded, "
                         "they are never trainable)")
    args = ap.parse_args()

    from problems import HOLDOUT_IDS

    donor_path = args.donor or newest(f"*{args.donor_match}*")
    if not donor_path:
        raise SystemExit(f"no donor results matching *{args.donor_match}* in results/")
    dd, donor = load(donor_path)

    ref_patterns = args.ref or DEFAULT_REFS
    refs = {}
    for pat in ref_patterns:
        p = newest(pat)
        if not p:
            print(f"note: no results match {pat}")
            continue
        md, per = load(p)
        refs[label(p, md.get("model", ""))] = per
    print(f"donor: {donor_path}\n       {dd.get('model')}  "
          f"{dd.get('passed')}/{dd.get('total')}  temp={dd.get('temperature')} "
          f"max_tokens={dd.get('max_tokens')} extra={dd.get('extra')}")
    for name, per in refs.items():
        print(f"ref  : {name:9s} {sum(r['ok'] for r in per.values())}/{len(per)} covered")

    ids = list(dict.fromkeys(list(donor) + [i for per in refs.values() for i in per]))
    tiers = {}
    try:
        from problems import PROBLEMS
        tiers = {p["id"]: p.get("tier", 1) for p in PROBLEMS}
    except Exception:
        pass

    rows = []
    for pid in ids:
        d = donor.get(pid)
        lok = {n: per[pid]["ok"] for n, per in refs.items() if pid in per}
        dpass = bool(d and d["ok"])
        any_local = any(lok.values()) if lok else None
        if any_local is None:
            bucket = "NEW_DONOR_PASS" if dpass else "NEW_DONOR_FAIL"
        elif dpass and not any_local:
            bucket = "DONOR_ONLY"
        elif dpass and any_local:
            bucket = "BOTH"
        elif not dpass and any_local:
            bucket = "LOCAL_ONLY"
        else:
            bucket = "CEILING"
        rows.append(dict(id=pid, tier=tiers.get(pid, 0), holdout=pid in HOLDOUT_IDS,
                         donor_pass=dpass, donor_status=(d or {}).get("status", "-"),
                         donor_tok=(d or {}).get("tok"), local_pass=any_local,
                         locals={n: v for n, v in lok.items() if v}, bucket=bucket))

    counts = Counter(r["bucket"] for r in rows)
    print("\n" + "=" * 84)
    for b in ("DONOR_ONLY", "BOTH", "LOCAL_ONLY", "CEILING",
              "NEW_DONOR_PASS", "NEW_DONOR_FAIL"):
        if counts.get(b):
            print(f"  {b:15s} {counts[b]:4d}")
    print("=" * 84)

    fails = Counter(r["donor_status"] for r in rows if not r["donor_pass"])
    print("\ndonor failure modes: " + (", ".join(f"{k}={v}" for k, v in fails.most_common())
                                       or "none"))
    if fails.get("NO_CODE"):
        print("  NO_CODE is a budget symptom, not an ability result — re-probe with "
              "donor_probe.py and raise --max-tokens or cap --extra reasoning_budget.")

    refnames = list(refs)
    covered = {n: set(per) for n, per in refs.items()}
    print(f"\n{'id':24s} {'t':>2s} {'GLM':>4s} " +
          " ".join(f"{n[:8]:>8s}" for n in refnames) + "  bucket            hold")
    for r in sorted(rows, key=lambda r: (r["bucket"], r["id"])):
        cols = " ".join(
            f"{('PASS' if r['locals'].get(n) else 'fail' if r['id'] in covered[n] else '-'):>8s}"
            for n in refnames)
        print(f"{r['id']:24s} {r['tier']:>2d} {('PASS' if r['donor_pass'] else 'fail'):>4s} "
              + cols + f"  {r['bucket']:15s} {'H' if r['holdout'] else ''}")

    trainable = [r for r in rows if not r["holdout"]]
    ask = [r["id"] for r in trainable if r["bucket"] in ("CEILING", "NEW_DONOR_FAIL")]
    targets = [r["id"] for r in trainable if r["bucket"] in ("DONOR_ONLY", "NEW_DONOR_PASS")]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = HERE / "results" / f"donor-value-{stamp}.json"
    out.write_text(json.dumps(dict(
        donor=dd.get("model"), donor_file=str(donor_path),
        refs={n: sorted(per) for n, per in refs.items()},
        counts=dict(counts), donor_failure_modes=dict(fails),
        donor_only_ids=[r["id"] for r in trainable if r["bucket"] == "DONOR_ONLY"],
        free_targets=targets, kimi_ask=ask, rows=rows,
    ), indent=2))

    print(f"\nfree local trace supply now exists for {len(targets)} trainable problems")
    print(f"still needing a stronger teacher (Kimi ask-list): {len(ask)}")
    print("  " + ", ".join(ask) if ask else "  (none)")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
