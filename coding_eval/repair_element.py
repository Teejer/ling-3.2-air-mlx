"""Recover individual problem elements from a teacher batch that won't json-parse.

Teacher replies often contain unescaped `"` inside string values (Claude does it in
`tests`, ChatGPT in prompts), which kills the whole-array parse. repair_json.py
handles the array; this works per element and additionally reindents a `ref` whose
nested indentation was flattened (1-space bodies), using the bracket/`:` depth of
each logical line to rebuild the block structure.

    python repair_element.py batches/batch2claud.json decode_nested_repeat > fixed.json
    python repair_element.py batches/batch2claud.json --list
"""
import json
import re
import sys
from pathlib import Path


def split_elements(raw):
    """Raw text of each top-level object, balanced on braces outside strings."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                out.append(raw[start:i + 1])
                start = None
    return out


def fix_string_value(frag, key):
    """Escape the interior quotes of "key": "...", handling \\n sequences and \\\"."""
    m = re.search(r'"%s"\s*:\s*"' % key, frag)
    if not m:
        return frag
    i = m.end()
    buf = []
    while i < len(frag):
        ch = frag[i]
        if ch == "\\":
            buf.append(frag[i:i + 2]); i += 2; continue
        if ch == '"':
            # A real closing quote is followed by a real next key (or end of
            # object). Checking the key NAME matters: Python test code like
            # `["3[a", "a]"]` has `", "` inside the value too.
            if re.match(r'\s*(,\s*"(?:id|tier|tag|title|prompt|tests|ref)"\s*:|\}\s*$|\}\s*,)',
                        frag[i + 1:]):
                break  # closing quote itself is re-added by the rebuild
            buf.append('\\"'); i += 1; continue
        buf.append(ch); i += 1
    return frag[:m.start()] + '"%s": "%s"' % (key, "".join(buf)) + frag[i + 1:]


def reindent_ref(code):
    """Rebuild block indentation for a ref written with flat 1-space bodies.

    Reliable only when every block opener is a single-line compound (`if x: y`)
    or its body ends on a line that closes the brackets it opened — a multi-line
    body (`if x:` / `y` / `z`) carries no dedent marker, so anything after it
    stays indented unless bracket depth says otherwise.
    """
    out, stack = [], []
    for ln in code.split("\n"):
        text = ln.strip()
        if not text:
            out.append("")
            continue
        depth = len(stack)
        if re.match(r"^(elif|else|except|finally)\b", text):
            # pop only if a real block is open to close (a single-line
            # `elif x: y` never pushed, and popping for it breaks the chain)
            kw = text.split(None, 1)[0].rstrip(":")
            opener = {"elif": ("if", "elif"), "else": ("if", "for", "while", "try"),
                      "except": ("try",),
                      # finally's try-opener was already popped by except/else;
                      # any sibling arm means we're at the right level
                      "finally": ("try", "except", "else")}[kw]
            while stack and stack[-1][1] not in opener and stack[-1][1] not in ("def", "class"):
                stack.pop()  # close any blocks nested inside the previous arm
            if stack and stack[-1][1] in opener:
                stack.pop()
            depth = len(stack)
        out.append("    " * depth + text)
        bal = _bracket_delta(text)
        if text.endswith(":") and bal == 0:
            kw0 = text.split()[0].rstrip(":")
            stack.append((depth + 1, kw0 if kw0 in ("if", "elif", "else", "for",
                                                    "while", "try", "except",
                                                    "finally", "with", "def",
                                                    "class") else "_"))
        elif bal < 0:
            for _ in range(-bal):
                if stack:
                    stack.pop()
    return "\n".join(out)


def _bracket_delta(text):
    """Net unclosed brackets, skipping string literals (incl. r/b prefixes)."""
    bal = 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "#":
            break
        if text[i:i + 2].lower() in ("r'", 'r"') or text[i:i + 2].lower() in ("b'", 'b"'):
            i += 1
            ch = text[i]
        if ch in "\"'":
            q, i = ch, i + 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == q:
                    i += 1
                    break
                i += 1
            continue
        if ch in "([{":
            bal += 1
        elif ch in ")]}":
            bal -= 1
        i += 1
    return bal


def normalize_escapes(frag, key):
    """Double backslashes that form invalid JSON escapes, inside ONE value.

    Idempotent: `\\[` -> `\\\\[` on the first pass, then it is a valid escape and
    is left alone. Scoped per key on purpose — a global pass would re-double
    values that were already correct on every retry.
    """
    m = re.search(r'"%s"\s*:\s*"' % key, frag)
    if not m:
        return frag
    i = m.end()
    buf = []
    while i < len(frag):
        ch = frag[i]
        if ch == "\\":
            nxt = frag[i + 1:i + 2]
            if nxt and nxt not in '"\\/bfnrtu':
                buf.append("\\\\"); i += 1; continue
            buf.append(frag[i:i + 2]); i += 2; continue
        if ch == '"':
            break  # closing quote re-added by the rebuild
        buf.append(ch); i += 1
    return frag[:m.start()] + '"%s": "%s"' % (key, "".join(buf)) + frag[i + 1:]


def load(frag):
    # iterate: fixing one value shifts offsets, and a mangled value can hide the
    # next key from the regex scan. Each pass changes something or we give up.
    for _ in range(24):
        try:
            return json.loads(frag)
        except json.JSONDecodeError:
            for fix in (fix_string_value, normalize_escapes):
                for key in ("id", "tier", "tag", "title", "prompt", "tests", "ref"):
                    nxt = fix(frag, key)
                    if nxt != frag:
                        frag = nxt
                        break
                else:
                    continue
                break
            else:
                raise
    raise ValueError("repair did not converge")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    if not path.is_absolute():
        path = Path(__file__).parent / path
    raw = path.read_text(encoding="utf-8")
    frags = split_elements(raw)
    wanted = sys.argv[2:]
    if wanted == ["--list"]:
        for f in frags:
            m = re.search(r'"id"\s*:\s*"([^"]+)"', f)
            print(m.group(1) if m else "?")
        return
    out = []
    for f in frags:
        m = re.search(r'"id"\s*:\s*"([^"]+)"', f)
        if not m or m.group(1) not in wanted:
            continue
        el = load(f)
        # only reindent genuinely flattened refs (flat teachers use 1-space
        # bodies); re-indenting properly indented code destroys it
        leads = [len(l) - len(l.lstrip()) for l in el["ref"].split("\n") if l.strip()]
        if leads and max(leads) <= 1:
            el["ref"] = reindent_ref(el["ref"])
        out.append(el)
    if not out:
        sys.exit("no matching ids")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
