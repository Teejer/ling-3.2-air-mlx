#!/bin/bash
# Phase-0 harvest round 1: mixed pack on the dev binary (:11250), sample the
# 43-problem train pool at temp 0.8, executor-gate, emit verified traces.
# 11234 (MLX-Serve.app / Qwen) is untouched.
set -uo pipefail
DEV_BIN=/Users/teejer/Documents/ai/inference/mlx-serve/zig-out/bin/mlx-serve
PORT=11250
MDIR=/Volumes/NVME/models
PACK=Ling-3.2-air-MLX-mixed-4-8bit
cd /Volumes/NVME/ling-3.2-air

LOG=coding_eval/harvest-server.log
nohup "$DEV_BIN" --model "$MDIR/$PACK" --serve --port $PORT --host 0.0.0.0 \
  --log-level info --model-dir "$MDIR" --timeout 1200 --metrics \
  --no-w8a8 --max-concurrent 4 --kv-quant 8 \
  --prefix-cache-entries 16 --prefix-cache-mem 32GB --os-reserve-gib 0 \
  >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for _ in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"loaded":true' && break
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q '"loaded":true' \
  || { echo "LOAD_TIMEOUT"; tail -25 "$LOG"; exit 1; }
echo "mixed up on $PORT"

.venv/bin/python coding_eval/run_eval.py --base-url "http://127.0.0.1:$PORT" \
  --expect mixed --harvest --samples 32 --max-per-problem 6 \
  > coding_eval/harvest-round1.log 2>&1 \
  && echo "HARVEST_OK: $(tail -1 coding_eval/harvest-round1.log)" \
  || { echo "HARVEST_FAIL"; tail -10 coding_eval/harvest-round1.log; exit 1; }
