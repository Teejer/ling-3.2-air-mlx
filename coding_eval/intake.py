"""Intake for externally-authored problems (curriculum batches from ChatGPT/Claude).

Input: JSON file — a list of candidates, each
    {"id", "tier", "tag", "title", "prompt", "tests", "ref"}
where `ref` is a complete reference solution. A candidate is VALIDATED only if
its ref passes its tests under the same sandbox run_eval uses (this catches the
#1 external-author bug: asserts with wrong expected values).

Usage:
  python intake.py candidates.json            # validate + report
  python intake.py candidates.json --promote  # also append survivors to
                                              # problems.py + references.py
The BROKEN report is copy-pasteable into the teacher chat as a correction ask.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fix_indent import looks_flattened, repair_ref  # noqa: E402
from problems import PROBLEMS  # noqa: E402
from run_eval import run_tests  # noqa: E402

HERE = Path(__file__).parent
REQUIRED = ("id", "tag", "title", "prompt", "tests", "ref")


def check(cand, known):
    missing = [k for k in REQUIRED if k not in cand]
    if missing:
        return f"SKIP missing fields: {missing}"
    if cand["id"] in known:
        return "SKIP duplicate id"
    if '"""' in cand["tests"] or '"""' in cand["prompt"]:
        return 'SKIP contains triple quotes (cannot embed)'
    if "assert" not in cand["tests"]:
        return "BROKEN no asserts in tests"
    with tempfile.TemporaryDirectory(dir="/private/tmp/opencode") as wd:
        status, detail = run_tests(wd, cand["ref"], cand["tests"])
    if status == "PASS":
        return "OK"
    if "SYNTAX" in status:
        got = repair_ref(cand["ref"], cand["tests"])
        if got:
            cand["ref"], cand["tests"] = got
            return "OK (reindented)"
    return f"BROKEN {status} {detail[:250]}"


def fmt_problem(c):
    tier = f"tier={c.get('tier', 2)}, intaked=True, "
    return (
        f"    dict(\n        id={c['id']!r}, {tier}tag={c['tag']!r},\n"
        f"        title={c['title']!r},\n"
        f'        prompt="""{c["prompt"].strip()}""",\n'
        f'        tests="""\n{c["tests"].strip()}\n""",\n    ),\n'
    )


def promote(items):
    ptxt = (HERE / "problems.py").read_text()
    assert ptxt.rstrip().endswith("]"), "problems.py must end with the PROBLEMS list"
    ptxt = ptxt.rstrip()[:-1].rstrip() + "\n\n" + "".join(fmt_problem(c) for c in items) + "]\n"
    (HERE / "problems.py").write_text(ptxt)
    refs = (HERE / "references.py").read_text().rstrip()
    refs += "\n\n\n" + "\n\n".join(
        f"# --- {c['id']} (intake) ---\n{c['ref'].rstrip()}" for c in items) + "\n"
    (HERE / "references.py").write_text(refs)
    print(f"promoted {len(items)} problems into problems.py + references.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()
    cands = json.loads(Path(args.candidates).read_text())
    known = {p["id"] for p in PROBLEMS}

    ok, broken = [], []
    for c in cands:
        verdict = check(c, known | {x["id"] for x in ok})
        print(f"{c.get('id', '?'):<20} {verdict}")
        if verdict.startswith("OK"):
            ok.append(c)
        elif verdict.startswith("BROKEN"):
            broken.append((c, verdict))

    print(f"\n{len(ok)}/{len(cands)} validated")
    if broken:
        print("\n--- BROKEN block (paste back to the teacher for correction) ---")
        for c, v in broken:
            print(f"\n[{c.get('id')}] {v}\nref: {c.get('ref', '')[:400]}\ntests: {c.get('tests', '')[:400]}")
    if args.promote and ok:
        promote(ok)


if __name__ == "__main__":
    main()
