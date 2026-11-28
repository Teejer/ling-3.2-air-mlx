"""Coding eval runner against an OpenAI-compatible server (mlx-serve).

Serves each problem, extracts the python block, executes the asserts in a
resource-limited subprocess, and writes a results JSON tagged with the loaded
model id. Run one model per invocation (switch the server between runs).

Usage:
  python run_eval.py [--base-url http://127.0.0.1:11234] [--expect qwen]
                     [--max-tokens 800] [--temperature 0.0] [--only id,id]
                     [--limit N] [--reps 1]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from problems import HOLDOUT_IDS, PROBLEMS  # noqa: E402

SYSTEM = (
    "You are an expert Python programmer. Implement exactly what is asked, "
    "std-lib only, no tests, no example calls. Finish with a single "
    "```python block containing the complete implementation and nothing after it."
)

CODE_RE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S)


def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def loaded_model(base, expect=None):
    """Pick the model to drive. mlx-serve can hold several models resident, so
    prefer an id matching --expect when given, else the loaded one."""
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
        data = json.loads(r.read().decode())["data"]
    ready = [m for m in data if m.get("loaded")] or data
    if expect:
        hit = [m for m in ready if expect.lower() in m["id"].lower()]
        if hit:
            return hit[0]["id"]
    return ready[0]["id"]


def extract_code(text):
    blocks = CODE_RE.findall(text)
    if not blocks:
        return text if "def " in text or "class " in text else ""
    if len(blocks) > 1:
        # models that split imports/funcs across blocks: concatenate in order
        return "\n".join(blocks)
    code = blocks[-1]
    lines = code.splitlines()
    while lines and lines[0].strip().lower() in ("python", "py"):
        lines = lines[1:]
    return "\n".join(lines)


def run_tests(workdir, code, tests):
    sol = Path(workdir, "sol.py")
    sol.write_text(code)
    harness = f"""
import resource, sys, traceback
resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
try:
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
except ValueError:
    pass
try:
    import sol
except SyntaxError as e:
    print("SYNTAX_ERROR", e); sys.exit(2)
except Exception as e:
    print("IMPORT_ERROR", type(e).__name__, e); sys.exit(2)
ns = dict(vars(sol))
try:
    exec({tests!r}, ns)
    print("ALL_OK")
except AssertionError:
    tb = traceback.format_exc().strip().splitlines()[-3:]
    print("ASSERT_FAIL"); print("\\n".join(tb)); sys.exit(1)
except Exception as e:
    print("RUNTIME_ERROR", type(e).__name__, e); sys.exit(1)
"""
    Path(workdir, "harness.py").write_text(harness)
    try:
        p = subprocess.run(
            [sys.executable, "harness.py"], cwd=workdir,
            capture_output=True, text=True, timeout=25,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "timeout"
    out = (p.stdout + p.stderr).strip()
    first = out.splitlines()[0] if out else "NO_OUTPUT"
    return ("PASS" if p.returncode == 0 else first), out[:400]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:11234")
    ap.add_argument("--expect", default=None,
                    help="substring the loaded model id must contain (safety against mislabeling)")
    ap.add_argument("--model-name", default=None,
                    help="override the model field sent in requests (mlx_lm.server "
                         "registers the --model path, while /v1/models reports the "
                         "upstream id — echoing the id back yields 404s)")
    ap.add_argument("--extra", default=None,
                    help="JSON object merged into every request payload — mlx-serve "
                         "knobs such as {\"reasoning_budget\": 4096} or "
                         "{\"thinking\": {\"type\": \"disabled\"}}")
    ap.add_argument("--max-tokens", type=int, default=1400)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--only", default=None)
    ap.add_argument("--tier", type=int, default=0, choices=[0, 1, 2],
                    help="1 or 2 to run one tier; 0 (default) runs all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--harvest", action="store_true",
                    help="LoRA Phase-0 mode: sample each train-pool problem "
                         "--samples times, keep execution-PASSING solutions as "
                         "chat-template JSONL (held-out split excluded)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-per-problem", type=int, default=6,
                    help="cap distinct passing traces per problem (overfit guard)")
    ap.add_argument("--keep-holdout", action="store_true",
                    help="harvest held-out problems too (NOT recommended)")
    args = ap.parse_args()
    extra = json.loads(args.extra) if args.extra else {}
    if args.harvest and args.temperature == 0.0:
        args.temperature = 0.8  # sampling needs heat; greedy harvest = 1 unique trace
        print("harvest: defaulting temperature to 0.8")

    model = loaded_model(args.base_url, args.expect)
    if args.expect and args.expect.lower() not in model.lower():
        sys.exit(f"loaded model {model!r} does not contain {args.expect!r} — refusing")
    slug = re.sub(r"[^a-zA-Z0-9.+-]", "_", model)
    print(f"model: {model}" + (f"  extra: {extra}" if extra else ""))

    probs = PROBLEMS
    if args.tier:
        probs = [p for p in probs if p.get("tier", 1) == args.tier]
    if args.only:
        want = set(args.only.split(","))
        probs = [p for p in probs if p["id"] in want]
    if args.limit:
        probs = probs[: args.limit]

    if args.harvest:
        if not args.keep_holdout:
            probs = [p for p in probs if p["id"] not in HOLDOUT_IDS]
        hdir = Path(__file__).parent / "harvest"
        hdir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        hpath = hdir / f"{slug}-{stamp}.jsonl"
        tally, total = {}, 0
        consec_err = 0
        with hpath.open("a") as fh:
            for prob in probs:
                seen_codes, kept = set(), 0
                for s in range(args.samples):
                    if kept >= args.max_per_problem:
                        break
                    try:
                        resp = post(args.base_url, {
                            "model": args.model_name or model,
                            "messages": [
                                {"role": "system", "content": SYSTEM},
                                {"role": "user", "content": prob["prompt"]},
                            ],
                            "temperature": args.temperature,
                            "max_tokens": args.max_tokens,
                            "stream": False,
                            **extra,
                        })
                    except Exception as e:
                        consec_err += 1
                        print(f"{prob['id']:<18} sample{s} REQUEST_ERROR {e}")
                        if consec_err >= 3:
                            print("SERVER_LOST — aborting harvest")
                            sys.exit(2)
                        continue
                    consec_err = 0
                    ch = resp["choices"][0]
                    content = ch["message"].get("content") or ""
                    if not content.strip():
                        content = (ch["message"].get("reasoning_content")
                                   or ch["message"].get("reasoning") or "")
                    code = extract_code(content)
                    if not code:
                        continue
                    with tempfile.TemporaryDirectory(dir="/private/tmp/opencode") as wd:
                        status, _ = run_tests(wd, code, prob["tests"])
                    norm = re.sub(r"\s+", " ", code).strip()
                    if status != "PASS" or norm in seen_codes:
                        continue
                    seen_codes.add(norm)
                    kept += 1
                    total += 1
                    tally[prob["id"]] = kept
                    fh.write(json.dumps({
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prob["prompt"]},
                            {"role": "assistant",
                             "content": "```python\n" + code + "\n```"},
                        ],
                        "meta": {"id": prob["id"], "tier": prob.get("tier", 1),
                                 "model": model, "sample": s,
                                 "temperature": args.temperature},
                    }, ensure_ascii=False) + "\n")
                    fh.flush()
                print(f"{prob['id']:<18} kept {kept}/{args.samples}")
        print(f"\nharvested {total} verified traces over {len(tally)} problems -> {hpath}")
        return

    results, passed = [], 0
    consec_err = 0
    for prob in probs:
        for rep in range(args.reps):
            t0 = time.time()
            try:
                resp = post(args.base_url, {
                    "model": args.model_name or model,
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prob["prompt"]},
                    ],
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "stream": False,
                    **extra,
                })
            except Exception as e:
                consec_err += 1
                results.append(dict(id=prob["id"], rep=rep, status="REQUEST_ERROR",
                                    err=str(e)[:200]))
                print(f"{prob['id']:<18} REQUEST_ERROR {e}")
                if consec_err >= 3:
                    print("SERVER_LOST: 3 consecutive request failures — aborting run")
                    sys.exit(2)
                continue
            consec_err = 0
            ch = resp["choices"][0]
            content = ch["message"].get("content") or ""
            if not content.strip():  # some engines put the answer in reasoning
                content = (ch["message"].get("reasoning_content")
                           or ch["message"].get("reasoning") or "")
            code = extract_code(content)
            usage = resp.get("usage") or {}
            secs = time.time() - t0
            with tempfile.TemporaryDirectory(dir="/private/tmp/opencode") as wd:
                status, detail = run_tests(wd, code, prob["tests"]) if code else ("NO_CODE", "")
            ok = status == "PASS"
            passed += ok
            results.append(dict(
                id=prob["id"], rep=rep, tier=prob.get("tier", 1), status=status, ok=ok,
                completion_tokens=usage.get("completion_tokens"),
                seconds=round(secs, 1),
                detail="" if ok else (detail + "\n--code-head--\n" + code[:400])[:900],
            ))
            mark = "PASS" if ok else f"FAIL({status})"
            tok = usage.get("completion_tokens") or 0
            print(f"{prob['id']:<18} {mark:<18} {tok:>4} tok  {secs:5.1f}s")

    outdir = Path(__file__).parent / "results"
    outdir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = outdir / f"{slug}-{stamp}.json"
    path.write_text(json.dumps(dict(
        model=model, base_url=args.base_url, temperature=args.temperature,
        max_tokens=args.max_tokens, extra=extra, passed=passed, total=len(results),
        results=results,
    ), indent=2))
    print(f"\n{passed}/{len(results)} passed -> {path}")


if __name__ == "__main__":
    main()
