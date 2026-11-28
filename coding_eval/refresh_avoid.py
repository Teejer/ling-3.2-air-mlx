"""Rewrite the AVOID id list in every teacher prompt from problems.py.

The list went stale twice (95 ids pasted while the suite held 104), which is
exactly how a teacher gets asked to re-author a problem we already grade. The
line it replaces is the one starting `AVOID — these ids already exist` and
running to the next blank line.

Usage: python refresh_avoid.py [--check]
"""

import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
PROMPTS = ("curriculum_seed_prompt.md", "kimi_batch1_prompt.md")
HEAD = "AVOID — these ids already exist, no duplicates or near-duplicates:"
WRAP = 92


def render(ids):
    """One wrapped paragraph, ids sorted, comma+space separated, period last."""
    lines, cur = [], ""
    for i, pid in enumerate(ids):
        piece = pid + ("," if i + 1 < len(ids) else ".")
        if cur and len(cur) + 1 + len(piece) > WRAP:
            lines.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def block_end(lines, start):
    """Index of the first line after the AVOID paragraph."""
    j = start + 1
    while j < len(lines) and lines[j].strip():
        j += 1
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report staleness, write nothing")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    from problems import PROBLEMS

    ids = sorted({p["id"] for p in PROBLEMS})
    body = render(ids)
    stale = 0
    for name in PROMPTS:
        path = HERE / name
        lines = path.read_text().splitlines()
        try:
            start = next(n for n, l in enumerate(lines) if l.startswith(HEAD))
        except StopIteration:
            print(f"{name}: no AVOID block — skipped")
            stale += 1
            continue
        end = block_end(lines, start)
        if lines[start + 1 : end] == body.splitlines():
            print(f"{name}: current ({len(ids)} ids)")
            continue
        stale += 1
        if args.check:
            have = sum(1 for l in lines[start + 1 : end] for t in re.findall(r"[a-z0-9_]+", l))
            print(f"{name}: STALE — lists ~{have} ids, suite has {len(ids)}")
        else:
            path.write_text("\n".join(lines[: start + 1] + body.splitlines() + lines[end:]) + "\n")
            print(f"{name}: rewrote AVOID with {len(ids)} ids")
    return 1 if (stale and args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
