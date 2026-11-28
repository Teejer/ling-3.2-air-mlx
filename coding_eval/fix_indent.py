"""Repair flattened Python indentation in teacher-chat batches.

Both ChatGPT and Claude commonly return `ref`/`tests` JSON strings whose
leading indentation collapsed to one space. Where the collapse is recoverable
by keywords alone we reindent directly; where it is not (mid-block dedents),
we DFS over all legal indent assignments and let the harness decide: the
first reconstruction that COMPILES and PASSES the tests is a correct
reference, full stop — it need not match the original formatting.

Used automatically by intake.py; can also run standalone:
  python fix_indent.py batch1.json [batch1_fixed.json]
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_eval import run_tests  # noqa: E402

DEDENT = ("else", "elif", "except", "finally")
MAX_CANDIDATES = 4000
MAX_TESTED = 120


def looks_flattened(src):
    lines = [l for l in src.splitlines() if l.strip()]
    return (any(l.startswith(" ") and not l.startswith("  ") for l in lines)
            and not any(l.startswith("  ") for l in lines))


def _candidates(src):
    """All indent assignments consistent with block openers/dedent keywords."""
    lines = [l.strip() for l in src.splitlines() if l.strip()]
    out, results = [], []

    def dfs(i, prev_level, forced):
        if len(results) >= MAX_CANDIDATES:
            return
        if i == len(lines):
            results.append("\n".join(out))
            return
        line = lines[i]
        if forced is not None:
            lo, hi = forced, forced
        else:
            lo, hi = 0, prev_level
        if line.startswith(DEDENT) and i > 0:
            hi = min(hi, prev_level)  # for/while-else aligns; if-else dedents
            order = list(range(hi, -1, -1))
        else:
            order = list(range(hi, lo - 1, -1))  # keep deep indents first
        for lvl in order:
            indent = "    " * lvl + line
            out.append(indent)
            partial_ok = True
            if not line.endswith(":") and not line.endswith(("\\", "(", "[", "{", ",")):
                try:
                    compile("\n".join(out), "<r>", "exec")
                except SyntaxError:
                    partial_ok = False
            if partial_ok:
                dfs(i + 1, lvl, lvl + 1 if line.endswith(":") else None)
            out.pop()

    if lines:
        dfs(0, 0, 0)
    return results


def llm_reindent(src, base_url="http://127.0.0.1:11234"):
    """Ask the locally served model to restore indentation. Safe because every
    result is still gated by run_tests/compile; returns None if no server."""
    try:
        from run_eval import extract_code, loaded_model, post
        model = loaded_model(base_url)
        resp = post(base_url, {
            "model": model,
            "messages": [
                {"role": "system", "content": "You repair lost Python indentation. "
                 "Output only the corrected code in one ```python block, changing no token."},
                {"role": "user", "content": "Every line of this lost its indentation "
                 "(lines sit at column 0 or one space). Restore correct 4-space Python "
                 "indentation without changing any token:\n\n" + src},
            ],
            "temperature": 0.0, "max_tokens": 2000, "stream": False,
        }, timeout=180)
        return extract_code(resp["choices"][0]["message"].get("content") or "")
    except Exception:
        return None


def fix_tests(tests):
    """Return a compilable version of (possibly flattened) tests, or None.
    LLM first (compiling-but-misnestled DFS reconstructions poison every ref
    test afterwards); DFS as offline fallback."""
    try:
        compile(tests, "<t>", "exec")
        return tests
    except SyntaxError:
        pass
    llm = llm_reindent(tests)
    if llm:
        try:
            compile(llm, "<t>", "exec")
            return llm
        except SyntaxError:
            pass
    for cand in _candidates(tests):
        try:
            compile(cand, "<t>", "exec")
            return cand
        except SyntaxError:
            continue
    return None


def repair_ref(ref, tests):
    """Find an indentation of `ref` that passes `tests`; returns (ref, tests) or None."""
    tests_fixed = fix_tests(tests)
    if tests_fixed is None:
        return None
    try:
        compile(ref, "<r>", "exec")
        sources = [ref]
    except SyntaxError:
        sources = _candidates(ref)
    tested = 0
    for cand in sources:
        if tested >= MAX_TESTED:
            break
        try:
            compile(cand, "<r>", "exec")
        except SyntaxError:
            continue
        tested += 1
        with tempfile.TemporaryDirectory(dir="/private/tmp/opencode") as wd:
            status, _ = run_tests(wd, cand, tests_fixed)
        if status == "PASS":
            return cand, tests_fixed
    llm = llm_reindent(ref)
    if llm:
        try:
            compile(llm, "<r>", "exec")
        except SyntaxError:
            return None
        with tempfile.TemporaryDirectory(dir="/private/tmp/opencode") as wd:
            status, _ = run_tests(wd, llm, tests_fixed)
        if status == "PASS":
            return llm, tests_fixed
    return None


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_fixed.json")
    cands = json.loads(src.read_text())
    fixed = 0
    for c in cands:
        if looks_flattened(c.get("ref", "")) or looks_flattened(c.get("tests", "")):
            got = repair_ref(c.get("ref", ""), c.get("tests", ""))
            if got:
                c["ref"], c["tests"] = got
                fixed += 1
    dst.write_text(json.dumps(cands, indent=1))
    print(f"repaired {fixed}/{len(cands)} -> {dst} (run intake.py on it)")


if __name__ == "__main__":
    main()
