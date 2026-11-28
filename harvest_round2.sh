#!/bin/bash
# Harvest round 2: the problems added AFTER round 1 (Kimi batch + ChatGPT
# survivors) that the first harvest never saw. Same gate, same holdout rule.
set -uo pipefail
DEV_BIN=/Users/teejer/Documents/ai/inference/mlx-serve/zig-out/bin/mlx-serve
PORT=11250
MDIR=/Volumes/NVME/models
cd /Volumes/NVME/ling-3.2-air
IDS="weighted_interval_schedule,longest_unique_window,count_coprime_inclusion_exclusion,min_station_refuels,min_eating_speed,tsp_exact_bitmask,predict_the_winner,base_neg2_conversion,min_path_effort_grid,render_escaped_template,matrix_chain_multiplication,subarrays_sum_equals_k"

LOG=coding_eval/harvest-server2.log
nohup "$DEV_BIN" --model "$MDIR/Ling-3.2-air-MLX-mixed-4-8bit" --serve --port $PORT --host 0.0.0.0 \
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

.venv/bin/python coding_eval/run_eval.py --base-url "http://127.0.0.1:$PORT" \
  --expect mixed --harvest --only "$IDS" --samples 24 --max-per-problem 6 \
  > coding_eval/harvest-round2.log 2>&1 \
  && echo "HARVEST2_OK: $(tail -1 coding_eval/harvest-round2.log)" \
  || { echo "HARVEST2_FAIL"; tail -10 coding_eval/harvest-round2.log; exit 1; }
