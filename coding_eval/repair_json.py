"""Repair teacher-chat replies that are line-per-key quasi-JSON with unescaped
double quotes inside string values (ChatHub/Kimi class of transport damage).

The chat UI writes each object as `{"key": "value", ...}` one key per line,
but forgets to escape `"` inside prompts/tests (e.g. == "110"). Because every
field sits on its own line with a known key set, we can split on line
structure, unescape what IS escaped, keep stray quotes as literal characters,
and re-emit with json.dumps. Content is untouched — this is transport repair,
not authoring.

Usage: python repair_json.py batch1kimi.json [batch1kimi_repaired.json]
"""
import json
import re
import sys
from pathlib import Path

KEYS = ("id", "tier", "tag", "title", "prompt", "tests", "ref")
KEY_RE = re.compile(r'^"(id|tier|tag|title|prompt|tests|ref)": (.*)$')


def unescape(s):
    """Decode valid JSON escapes; leave stray backslashes/quotes literal."""
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            if n == "n":
                out.append("\n"); i += 2; continue
            if n == "t":
                out.append("\t"); i += 2; continue
            if n == "r":
                out.append("\r"); i += 2; continue
            if n == '"':
                out.append('"'); i += 2; continue
            if n == "\\":
                out.append("\\"); i += 2; continue
            if n == "/":
                out.append("/"); i += 2; continue
            if n == "u" and i + 5 < len(s):
                try:
                    out.append(chr(int(s[i + 2:i + 6], 16))); i += 6; continue
                except ValueError:
                    pass
            out.append(c); i += 1; continue
        out.append(c); i += 1
    return "".join(out)


def repair(text):
    objs, cur, buf = [], {}, None
    for line in text.splitlines():
        if line.strip() in ("[", "]", "") and not cur:
            continue
        if line.strip() == "{":
            cur, buf = {}, None
            continue
        if line.strip().rstrip(",") == "}":
            if buf is not None and cur:
                cur[buf[0]] = finish(buf[0], buf[1])
            if cur.get("id"):
                objs.append(cur)
            cur, buf = {}, None
            continue
        m = KEY_RE.match(line)
        if m and cur is not None:
            if buf is not None:  # flush previous field
                cur[buf[0]] = finish(buf[0], buf[1])
            buf = (m.group(1), m.group(2))
        elif buf is not None:  # continuation of a value (raw newline in string)
            buf = (buf[0], buf[1] + "\n" + line)
    return objs


def finish(key, raw):
    if key == "tier":
        return int(re.sub(r"[,\s]", "", raw))
    v = raw.rstrip()
    if v.endswith(","):
        v = v[:-1]
    if v.startswith('"'):
        v = v[1:]
    if v.endswith('"'):
        v = v[:-1]
    return unescape(v)


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(
        src.stem.replace("_repaired", "") + "_repaired.json")
    objs = repair(src.read_text())
    seen, uniq = set(), []
    for o in objs:
        if o["id"] in seen:
            print(f"duplicate id dropped: {o['id']}")
            continue
        seen.add(o["id"])
        uniq.append({k: o.get(k, "") for k in KEYS})
    for o in uniq:
        missing = [k for k in KEYS if not str(o[k])]
        if missing:
            print(f"{o['id']}: MISSING {missing}")
    out.write_text(json.dumps(uniq, indent=1, ensure_ascii=False))
    json.loads(out.read_text())  # prove it parses
    print(f"{len(uniq)} objects -> {out}")
    print("ids:", ", ".join(o["id"] for o in uniq))


if __name__ == "__main__":
    main()
