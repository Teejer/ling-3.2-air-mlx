#!/bin/bash
# Re-eval with fixed tests + MTP-enabled dev binary.
set -uo pipefail
cd /Volumes/NVME/ling-3.2-air

echo "== Qwen (app, :11234)"
.venv/bin/python coding_eval/run_eval.py --expect qwen \
  > coding_eval/qwen-fixed.log 2>&1 \
  && echo "EVAL_OK qwen: $(tail -1 coding_eval/qwen-fixed.log)" \
  || { echo "EVAL_FAIL qwen"; tail -5 coding_eval/qwen-fixed.log; }

echo "== Ling packs (rebuilt dev binary, :11250)"
./ling_eval_switch2.sh

echo "== MTP evidence from last Ling run:"
grep -m1 -i "mtp head ready" /tmp/mlx-serve-11250-mixed.log || echo "  (no MTP ready line)"
grep -iE "accept" /tmp/mlx-serve-11250-mixed.log | tail -2 || true
echo "RUN3_DONE"
