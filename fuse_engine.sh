#!/bin/bash
# Engine-path eval for an adapter: fuse -> rename surgery -> serve on the
# fast Zig dev binary -> full suite. Usage: ./fuse_engine.sh adapters/round2 \
#   Ling-3.2-air-MLX-mixed-4-8bit-r2e [expect-substring]
set -uo pipefail
DEV_BIN=/Users/teejer/Documents/ai/inference/mlx-serve/zig-out/bin/mlx-serve
PORT=11250
MDIR=/Volumes/NVME/models
cd /Volumes/NVME/ling-3.2-air
ADAPTER=${1:?usage: fuse_engine.sh ADAPTER_DIR OUT_PACK [expect]}
OUT=${2:?usage: fuse_engine.sh ADAPTER_DIR OUT_PACK [expect]}
EXPECT=${3:-$(basename "$OUT" | sed 's/.*-r/r/')}
BASE=Ling-3.2-air-MLX-mixed-4-8bit

pkill -f "serve --port $PORT" 2>/dev/null
pkill -f "mlx_lm.server.*$PORT" 2>/dev/null
sleep 3

if [ ! -d "$OUT" ]; then
  .venv/bin/mlx_lm.fuse --model "$BASE" --adapter-path "$ADAPTER" \
    --save-path "$OUT" > coding_eval/fuse-$EXPECT.log 2>&1 \
    || { echo "FUSE_FAIL"; tail -15 coding_eval/fuse-$EXPECT.log; exit 1; }
  for f in chat_template.jinja tokenizer.json tokenizer_config.json vocab.json \
           merges.txt generation_config.json; do
    [ -e "$BASE/$f" ] && [ ! -e "$OUT/$f" ] && cp "$BASE/$f" "$OUT/$f"
  done
  .venv/bin/python coding_eval/fix_fused_pack.py "$OUT" \
    || { echo "RENAME_FAIL"; exit 1; }
fi
ln -sfn "/Volumes/NVME/ling-3.2-air/$OUT" "$MDIR/$OUT"

LOG=coding_eval/eval-$EXPECT-server.log
nohup "$DEV_BIN" --model "$MDIR/$OUT" --serve --port $PORT --host 0.0.0.0 \
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
  || { echo "LOAD_FAIL — engine log tail:"; tail -25 "$LOG"; exit 1; }

.venv/bin/python coding_eval/run_eval.py --base-url "http://127.0.0.1:$PORT" \
  --expect "$EXPECT" > coding_eval/eval-$EXPECT.log 2>&1 \
  || { echo "EVAL_FAIL"; tail -10 coding_eval/eval-$EXPECT.log; exit 1; }
echo "FUSE_ENGINE_OK: $(tail -1 coding_eval/eval-$EXPECT.log)"
