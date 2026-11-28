#!/bin/bash
# Round-1 eval: fuse adapter into a pack copy, serve on :11250, score the
# full 55-problem suite (held-out subset is the honest number).
set -uo pipefail
DEV_BIN=/Users/teejer/Documents/ai/inference/mlx-serve/zig-out/bin/mlx-serve
PORT=11250
MDIR=/Volumes/NVME/models
cd /Volumes/NVME/ling-3.2-air
BASE=Ling-3.2-air-MLX-mixed-4-8bit
R1=Ling-3.2-air-MLX-mixed-4-8bit-r1

pkill -f "serve --port $PORT" 2>/dev/null; sleep 5

[ -d "adapters/round1" ] && [ -f adapters/round1/adapters.safetensors ] \
  || { echo "NO_ADAPTER"; exit 1; }

if [ ! -d "$R1" ]; then
  .venv/bin/mlx_lm.fuse --model "$BASE" --adapter-path adapters/round1 \
    --save-path "$R1" > coding_eval/fuse-round1.log 2>&1 \
    || { echo "FUSE_FAIL"; tail -15 coding_eval/fuse-round1.log; exit 1; }
  # keep config metadata (chat template, tokenizer) from the base pack
  for f in chat_template.jinja tokenizer.json tokenizer_config.json vocab.json \
           merges.txt generation_config.json; do
    [ -e "$BASE/$f" ] && [ ! -e "$R1/$f" ] && cp "$BASE/$f" "$R1/$f"
  done
fi
ln -sfn "/Volumes/NVME/ling-3.2-air/$R1" "$MDIR/$R1"

LOG=coding_eval/eval-round1-server.log
nohup "$DEV_BIN" --model "$MDIR/$R1" --serve --port $PORT --host 0.0.0.0 \
  --log-level info --model-dir "$MDIR" --timeout 1200 --metrics \
  --no-w8a8 --max-concurrent 4 --kv-quant 8 \
  --prefix-cache-entries 16 --prefix-cache-mem 32GB --os-reserve-gib 0 \
  >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 1 900); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"loaded":true' && break
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q '"loaded":true' \
  || { echo "LOAD_FAIL (engine may reject fused tensors — see $LOG)"; tail -25 "$LOG"; exit 1; }

.venv/bin/python coding_eval/run_eval.py --base-url "http://127.0.0.1:$PORT" \
  --expect r1 > coding_eval/eval-round1.log 2>&1 \
  && echo "EVALR1_OK: $(tail -1 coding_eval/eval-round1.log)" \
  || { echo "EVALR1_FAIL"; tail -10 coding_eval/eval-round1.log; exit 1; }
