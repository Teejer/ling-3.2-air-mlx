#!/bin/bash
# Round-4 eval: py-server + round-4 adapter, full 104-problem suite, retagged -r4-.
# Same harness as eval_round3_server.sh on purpose — r3's 61/66 and 10/12 were measured
# through mlx_lm.server with an adapter, so r4 must be too or the comparison is hollow.
set -uo pipefail
cd /Volumes/NVME/ling-3.2-air
PORT=11250
pkill -f "serve --port $PORT" 2>/dev/null
pkill -f "mlx_lm.server.*$PORT" 2>/dev/null
sleep 3

# A cold mmap pack makes MLX die with a METAL GPU timeout; 10s of cat is the whole fix.
cat Ling-3.2-air-MLX-mixed-4-8bit/*.safetensors > /dev/null

.venv/bin/mlx_lm.server \
  --model /Volumes/NVME/ling-3.2-air/Ling-3.2-air-MLX-mixed-4-8bit \
  --adapter-path adapters/round4 \
  --port $PORT > coding_eval/eval-round4-server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
  || { echo "LOAD_FAIL"; tail -20 coding_eval/eval-round4-server.log; exit 1; }

.venv/bin/python coding_eval/run_eval.py --base-url "http://127.0.0.1:$PORT" \
  --expect Ling --model-name default_model > coding_eval/eval-round4.log 2>&1 \
  || { echo "EVALR4_FAIL"; tail -10 coding_eval/eval-round4.log; exit 1; }

LAST=$(ls -t coding_eval/results/*.json | head -1)
STAMP=$(basename "$LAST" | grep -o '[0-9]\{8\}-[0-9]\{6\}')
mv "$LAST" "coding_eval/results/Ling-3.2-air-MLX-mixed-4-8bit-r4-$STAMP.json"
echo "EVALR4_OK: $(tail -1 coding_eval/eval-round4.log) -> retagged r4-$STAMP"
