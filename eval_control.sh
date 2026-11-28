#!/bin/bash
# Control run for the round-1 A/B: base pack, NO adapter, same mlx_lm.server,
# same 66-problem suite. r1-vs-control on identical serving = pure adapter
# effect. Chained before harvest round 2 (GPU-sequential).
set -uo pipefail
cd /Volumes/NVME/ling-3.2-air
PORT=11250
pkill -f "serve --port $PORT" 2>/dev/null
pkill -f "mlx_lm.server.*$PORT" 2>/dev/null
sleep 3

.venv/bin/mlx_lm.server \
  --model /Volumes/NVME/ling-3.2-air/Ling-3.2-air-MLX-mixed-4-8bit \
  --port $PORT > coding_eval/eval-control-server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
  || { echo "LOAD_FAIL"; tail -20 coding_eval/eval-control-server.log; exit 1; }

.venv/bin/python coding_eval/run_eval.py --base-url "http://127.0.0.1:$PORT" \
  --expect Ling --model-name default_model > coding_eval/eval-control.log 2>&1 \
  || { echo "CONTROL_FAIL"; tail -10 coding_eval/eval-control.log; exit 1; }

LAST=$(ls -t coding_eval/results/*.json | head -1)
STAMP=$(basename "$LAST" | grep -o '[0-9]\{8\}-[0-9]\{6\}')
mv "$LAST" "coding_eval/results/Ling-3.2-air-MLX-mixed-4-8bit-pyserver-$STAMP.json"
echo "CONTROL_OK: $(tail -1 coding_eval/eval-control.log) -> retagged pyserver-$STAMP"
