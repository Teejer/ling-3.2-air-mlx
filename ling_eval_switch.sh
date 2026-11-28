#!/bin/bash
# Eval switch sequence (user-approved): serve each Ling pack on 11234 with the
# dev mlx-serve binary, run the coding eval, then restore MLX-Serve.app.
set -uo pipefail
DEV_BIN=/Users/teejer/Documents/ai/inference/mlx-serve/zig-out/bin/mlx-serve
MDIR=/Volumes/NVME/models
cd /Volumes/NVME/ling-3.2-air

stop_server() {
  osascript -e 'quit app "MLX-Serve"' 2>/dev/null
  pkill -f "$DEV_BIN" 2>/dev/null
  for _ in $(seq 1 90); do
    pgrep -f "port 11234" >/dev/null || return 0
    sleep 1
  done
  echo "STOP_TIMEOUT: something still holds 11234"; exit 1
}

start_server() {  # $1=pack name  $2=log path
  nohup "$DEV_BIN" --model "$MDIR/$1" --serve --port 11234 --host 0.0.0.0 \
    --log-level info --model-dir "$MDIR" --timeout 1200 --metrics \
    --no-w8a8 --max-concurrent 16 --kv-quant 8 \
    --prefix-cache-entries 32 --prefix-cache-mem 96GB --os-reserve-gib 0 \
    >"$2" 2>&1 &
  for _ in $(seq 1 450); do
    curl -sf http://127.0.0.1:11234/v1/models 2>/dev/null \
      | "$PWD/.venv/bin/python" -c "import json,sys; d=json.load(sys.stdin)['data']; sys.exit(0 if any(m.get('loaded') for m in d) else 1)" \
      && return 0
    sleep 2
  done
  echo "LOAD_TIMEOUT waiting for $1"; tail -25 "$2"; exit 1
}

run_one() {  # $1=pack name  $2=expect substring  $3=tag
  LOG="/tmp/mlx-serve-11234-eval-$3.log"
  stop_server
  start_server "$1" "$LOG"
  echo "== $1 up; speculative:"
  grep -m2 -iE "mtp head ready|nextn|speculat" "$LOG" || echo "  (no MTP log line found)"
  .venv/bin/python coding_eval/run_eval.py --expect "$2" > "coding_eval/ling-$3.log" 2>&1 \
    && echo "EVAL_OK $3: $(tail -1 coding_eval/ling-$3.log)" \
    || { echo "EVAL_FAIL $3"; tail -10 "coding_eval/ling-$3.log"; }
}

run_one Ling-3.2-air-MLX-4bit air-mlx-4bit 4bit
run_one Ling-3.2-air-MLX-mixed-4-8bit mixed-4-8bit mixed

stop_server
open -a MLX-Serve && echo "app relaunched (it restores its last model)"
echo "SWITCH_EVAL_DONE"
