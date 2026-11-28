#!/bin/bash
# Eval switch, v2: dev mlx-serve on port 11250 (11234 belongs to
# MLX-Serve.app, which supervises itself and SIGKILLed the v1 attempt).
# The app is NOT touched; speed columns are contention-skewed, quality is not.
set -uo pipefail
DEV_BIN=/Users/teejer/Documents/ai/inference/mlx-serve/zig-out/bin/mlx-serve
PORT=11250
MDIR=/Volumes/NVME/models
cd /Volumes/NVME/ling-3.2-air

start_server() {  # $1=pack name  $2=log
  nohup "$DEV_BIN" --model "$MDIR/$1" --serve --port $PORT --host 0.0.0.0 \
    --log-level info --model-dir "$MDIR" --timeout 1200 --metrics \
    --no-w8a8 --max-concurrent 4 --kv-quant 8 \
    --prefix-cache-entries 16 --prefix-cache-mem 32GB --os-reserve-gib 0 \
    >"$2" 2>&1 &
  for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"loaded":true' && return 0
    sleep 2
  done
  echo "LOAD_TIMEOUT waiting for $1"; tail -25 "$2"; exit 1
}

stop_server() {
  pkill -f "serve --port $PORT" 2>/dev/null
  for _ in $(seq 1 60); do pgrep -f "serve --port $PORT" >/dev/null || return 0; sleep 1; done
  echo "STOP_TIMEOUT port $PORT still held"; exit 1
}

run_one() {  # $1=pack  $2=expect  $3=tag
  LOG="/tmp/mlx-serve-11250-$3.log"
  start_server "$1" "$LOG"
  echo "== $1 up on $PORT; speculative:"
  grep -m2 -iE "mtp head ready|nextn|speculat" "$LOG" | sed 's/^/   /' || echo "   (no MTP log line)"
  .venv/bin/python coding_eval/run_eval.py --base-url "http://127.0.0.1:$PORT" \
    --expect "$2" > "coding_eval/ling-$3.log" 2>&1 \
    && echo "EVAL_OK $3: $(tail -1 "coding_eval/ling-$3.log")" \
    || { echo "EVAL_FAIL $3"; tail -8 "coding_eval/ling-$3.log"; }
  stop_server
}

run_one Ling-3.2-air-MLX-4bit air-mlx-4bit 4bit
run_one Ling-3.2-air-MLX-mixed-4-8bit mixed-4-8bit mixed
echo "SWITCH2_DONE"
