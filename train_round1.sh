#!/bin/bash
# Round-1 LoRA SFT on verified harvest traces. Run AFTER harvest finishes;
# frees the GPU from the :11250 server first.
set -uo pipefail
cd /Volumes/NVME/ling-3.2-air

pkill -f "serve --port 11250" 2>/dev/null
sleep 5

.venv/bin/python coding_eval/prepare_round1_data.py /private/tmp/opencode/lora_round1 \
  || { echo "DATA_FAIL"; exit 1; }
N=$(wc -l < /private/tmp/opencode/lora_round1/train.jsonl)
echo "train traces: $N"
if [ "$N" -lt 30 ]; then echo "TOO_FEW_TRACES ($N) — skipping training, inspect harvest"; exit 1; fi

mkdir -p adapters/round1
.venv/bin/mlx_lm.lora \
  --model /Volumes/NVME/ling-3.2-air/Ling-3.2-air-MLX-mixed-4-8bit \
  --config coding_eval/lora_round1.yaml \
  --train \
  --data /private/tmp/opencode/lora_round1 \
  --adapter-path adapters/round1 \
  --iters 250 --batch-size 1 --num-layers 16 --max-seq-length 2048 \
  --learning-rate 5e-5 --steps-per-report 10 --steps-per-eval 50 \
  > coding_eval/train-round1.log 2>&1 \
  && echo "TRAIN_OK: $(tail -3 coding_eval/train-round1.log | tr '\n' ' ')" \
  || { echo "TRAIN_FAIL"; tail -15 coding_eval/train-round1.log; exit 1; }
