#!/bin/bash
# Round-1 eval, server pivot: the fused pack lacks engine-named kv_b_proj
# (mlx-lm saves embed_q/unembed_out; Zig engine wants kv_b_proj — pack
# rename is a morning task). For the QUALITY signal tonight, serve the base
# pack + adapter through mlx_lm.server (same weights, same chat template,
# greedy eval unaffected by the slower Python server).
set -uo pipefail
cd /Volumes/NVME/ling-3.2-air
PORT=11250
pkill -f "serve --port $PORT" 2>/dev/null
pkill -f "mlx_lm.server.*$PORT" 2>/dev/null
sleep 3

.venv/bin/mlx_lm.server \
  --model /Volumes/NVME/ling-3.2-air/Ling-3.2-air-MLX-mixed-4-8bit \
  --adapter-path adapters/round1 \
  --port $PORT > coding_eval/eval-round1-server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for _ in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
  || { echo "LOAD_FAIL"; tail -20 coding_eval/eval-round1-server.log; exit 1; }
echo "mlx_lm.server + adapter up on $PORT"

.venv/bin/python coding_eval/run_eval.py --base-url "http://127.0.0.1:$PORT" \
  --expect Ling --model-name default_model > coding_eval/eval-round1.log 2>&1 \
  || { echo "EVALR1_FAIL"; tail -10 coding_eval/eval-round1.log; exit 1; }

# retag the result file so compare_round1 sees it as the r1 model, not the
# pre-LoRA baseline (mlx_lm.server reports the base model id)
LAST=$(ls -t coding_eval/results/*.json | head -1)
STAMP=$(basename "$LAST" | grep -o '[0-9]\{8\}-[0-9]\{6\}')
mv "$LAST" "coding_eval/results/Ling-3.2-air-MLX-mixed-4-8bit-r1-$STAMP.json"
echo "EVALR1_OK: $(tail -1 coding_eval/eval-round1.log) -> retagged r1-$STAMP"
.venv/bin/python coding_eval/compare_round1.py
