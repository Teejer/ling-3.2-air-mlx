"""Self-test the problem set: run each tier-2 problem's asserts against the
reference solution in references.py. A problem whose asserts reject a correct
solution is a harness bug (this bit us in the first eval runs), so this must
pass before any eval run is trusted.

Usage: python validate_problems.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from problems import PROBLEMS  # noqa: E402
from run_eval import run_tests  # noqa: E402

REF_SRC = (Path(__file__).parent / "references.py").read_text()

fail = 0
for prob in PROBLEMS:
    if prob.get("tier", 1) < 2 and not prob.get("intaked"):
        continue
    with tempfile.TemporaryDirectory(dir="/private/tmp/opencode") as wd:
        status, detail = run_tests(wd, REF_SRC, prob["tests"])
    ok = status == "PASS"
    fail += not ok
    print(f"{prob['id']:<18} {'OK' if ok else 'BROKEN: ' + status} {'' if ok else detail[:300]}")

targets = [p for p in PROBLEMS if p.get("tier", 1) >= 2 or p.get("intaked")]
print(f"\n{len(targets) - fail}/{len(targets)} tier-2+/intaked problems verified")
sys.exit(1 if fail else 0)
