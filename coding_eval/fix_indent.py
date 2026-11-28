"""Repair flattened Python indentation in a candidates file (one common
teacher-chat artifact: refs arrive with every line indented by one space).

Keyword heuristic: indent after lines ending in ':', dedent before
else/elif/except/finally. Where the original nesting had a mid-block dedent
(no keyword marker) it cannot be recovered — those candidates will still fail
intake validation and belong in the correction message. Safe: nothing is
trusted, everything still goes through intake.py's execution gate.

Usage: python fix_indent.py batch1.json [batch1_fixed.json]
"""
import json
import sys
from pathlib import Path

DEDENT = ("else", "elif", "except", "finally")


def reindent(src):
    out, level = [], 0
    for raw in src.splitlines():
        line = raw.strip()
        if not line:
            out.append("")
            continue
        lvl = max(level - 1, 0) if line.startswith(DEDENT) else level
        out.append("    " * lvl + line)
        if line.endswith(":"):
            level += 1
    return "\n".join(out)


def looks_flattened(src):
    lines = [l for l in src.splitlines() if l.strip()]
    return (any(l.startswith(" ") and not l.startswith("  ") for l in lines)
            and not any(l.startswith("  ") for l in lines))


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_fixed.json")
    cands = json.loads(src.read_text())
    changed = 0
    for c in cands:
        if looks_flattened(c.get("ref", "")):
            c["ref"] = reindent(c["ref"])
            changed += 1
        if looks_flattened(c.get("tests", "")):
            c["tests"] = reindent(c["tests"])
    dst.write_text(json.dumps(cands, indent=1))
    print(f"reindented {changed}/{len(cands)} refs -> {dst}")
    print("now run: python intake.py", dst)


if __name__ == "__main__":
    main()
