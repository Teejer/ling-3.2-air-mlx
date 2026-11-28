#!/bin/bash
# Round-5 harvest — targeted, not blanket.
#
# Round 4 proved that donating traces for problems the student ALREADY passes is pure
# interference (r3 already solved 31 of the 38 problems we donated for; score 92 -> 88).
# So round 5 harvests only the two buckets that have a demonstrated gap, and takes the
# teacher only where the teacher is the only source:
#
#   A  ADDITIVE  — r3 fails, the untrained pack fails, GLM-5.3-Flash passes:
#      csv_line, decode_nested_repeat, expand_counted_pattern, shortest_threshold_slice
#      -> donor traces (GLM, this box, no licence question: open weights, run locally)
#   B  ANTI-DRIFT — r3 fails but the UNTRAINED pack passes, i.e. rounds 1-3 broke them:
#      directed_euler_walk, grid_keys_shortest_path, min_assignment_cost, roman,
#      solve_linear_congruence
#      -> the STUDENT'S OWN passing samples from the untrained pack. No teacher, no style
#      tax; the objective is "stop drifting", not "imitate GLM".
#
# Nothing here touches HOLDOUT_IDS (run_eval refuses them under --harvest).
set -uo pipefail
DEV_BIN=/Users/teejer/Documents/ai/inference/mlx-serve/zig-out/bin/mlx-serve
PORT=11250
MDIR=/Volumes/NVME/models
APP=http://127.0.0.1:11234
cd /Volumes/NVME/ling-3.2-air

ADDITIVE="csv_line,decode_nested_repeat,expand_counted_pattern,shortest_threshold_slice"
REGRESSION="directed_euler_walk,grid_keys_shortest_path,min_assignment_cost,roman,solve_linear_congruence"

pkill -f "serve --port $PORT" 2>/dev/null
pkill -f "mlx_lm.server.*$PORT" 2>/dev/null
sleep 3

# ---------------------------------------------------------------- A: donor, additive only
if ! curl -sf -m 10 "$APP/v1/models" | grep -q "GLM-5.3-Flash"; then
  echo "SKIP-A: GLM-5.3-Flash is not served on $APP — boot it with /private/tmp/opencode/glm_boot.sh"
else
  .venv/bin/python coding_eval/run_eval.py --base-url "$APP" --expect GLM --harvest \
    --only "$ADDITIVE" --samples 8 --max-per-problem 4 --temperature 0.8 \
    --extra '{"enable_thinking": false}' \
    > coding_eval/harvest-round5-donor.log 2>&1 \
    && echo "HARVEST5-A OK: $(tail -1 coding_eval/harvest-round5-donor.log)" \
    || { echo "HARVEST5-A FAIL"; tail -10 coding_eval/harvest-round5-donor.log; }
fi

# ------------------------------------------------- B: the student's own pre-training output
LOG=coding_eval/harvest-round5-server.log
cat Ling-3.2-air-MLX-mixed-4-8bit/*.safetensors > /dev/null   # cold mmap = METAL GPU timeout
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
  --expect mixed --harvest --only "$REGRESSION" --samples 24 --max-per-problem 6 \
  --temperature 0.8 > coding_eval/harvest-round5-replay.log 2>&1 \
  || { echo "HARVEST5-B FAIL"; tail -10 coding_eval/harvest-round5-replay.log; exit 1; }

# The engine serves the same pack name as the r1-r3 self-harvest, so the file would be
# indistinguishable from the champion's own bank. Rename it: provenance by filename.
NEW=$(ls -t coding_eval/harvest/Ling-3.2-air-MLX-mixed-4-8bit-*.jsonl | head -1)
mv "$NEW" "${NEW/Ling-3.2-air-MLX-mixed-4-8bit-/REPLAY-untrained-}"
echo "HARVEST5-B OK: $(tail -1 coding_eval/harvest-round5-replay.log) -> $(basename "${NEW/Ling-3.2-air-MLX-mixed-4-8bit-/REPLAY-untrained-}")"
