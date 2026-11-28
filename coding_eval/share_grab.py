"""Pull teacher replies out of public share links — no browser, no paste.

  ChatGPT: https://chatgpt.com/share/<id>  (must be created while logged in,
           but is then world-readable)  -> FULLY AUTOMATED here via the
           public backend-api/share/<id> JSON endpoint.
  Claude:  https://claude.ai/share/<id>   -> API is Cloudflare-gated; falls
           back to rendering the SSR page, which usually lacks the messages.
           Keep copy-pasting Claude replies manually for now.

Any assistant reply that parses as a JSON array of problem objects is saved
to coding_eval/batches/<out>.json (latest reply wins; --all numbers them).

Usage:
  python share_grab.py <share-url> batch3.json [--all]
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")
BATCH_DIR = Path(__file__).parent / "batches"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")


def problem_arrays(text):
    """Every JSON array of problem objects embedded in a reply."""
    out = []
    for m in re.finditer(r"\[", text):
        frag = text[m.start():]
        j = frag.rfind("]")
        if j < 0:
            continue
        try:
            arr = json.loads(frag[: j + 1])
        except Exception:
            continue
        if (isinstance(arr, list) and arr and isinstance(arr[0], dict)
                and {"id", "tests", "ref"} <= set(arr[0])):
            out.append(arr)
            break  # first balanced array in the reply is the whole batch
    return out


def from_chatgpt(share_id):
    d = json.loads(get(f"https://chatgpt.com/backend-api/share/{share_id}"))
    replies = []
    def walk(n):
        if isinstance(n, dict):
            m = n.get("message")
            if isinstance(m, dict) and m.get("author", {}).get("role") == "assistant":
                c = m.get("content")
                if isinstance(c, dict) and c.get("content_type") == "text":
                    txt = "".join(c.get("parts") or []).strip()
                    if txt:
                        replies.append(txt)
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(d.get("mapping") or d)
    return replies


def from_claude(share_id):
    print("note: claude.ai share content is Cloudflare-gated; trying SSR page")
    html = get(f"https://claude.ai/share/{share_id}")
    return [html]  # problem_arrays may still find an embedded reply


def main():
    args = [a for a in sys.argv[1:] if a != "--all"]
    grab_all = "--all" in sys.argv
    if len(args) < 2:
        sys.exit(__doc__)
    url, out_name = args[0], args[1]
    if "chatgpt.com/share/" in url:
        replies = from_chatgpt(url.rstrip("/").rsplit("/", 1)[1])
    elif "claude.ai/share/" in url:
        replies = from_claude(url.rstrip("/").rsplit("/", 1)[1])
    else:
        sys.exit("unsupported host — known: chatgpt.com/share, claude.ai/share")
    found = []
    for r in replies:
        found.extend(problem_arrays(r))
    if not found:
        sys.exit(f"no problem-array reply found in {len(replies)} assistant replies")
    targets = found if grab_all else [found[-1]]
    for i, arr in enumerate(targets):
        name = out_name if len(targets) == 1 else out_name.replace(".json", f"_{i}.json")
        path = BATCH_DIR / name
        path.write_text(json.dumps(arr, indent=1, ensure_ascii=False))
        print(f"{path.name}: {len(arr)} problems: "
              + ", ".join(p.get("id", "?") for p in arr))


if __name__ == "__main__":
    main()
