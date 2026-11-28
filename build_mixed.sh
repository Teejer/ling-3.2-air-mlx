#!/bin/bash
# Chain: wait for the blob mover to finish (so the converter never reads a
# blob mid-symlink-swap), then build the mixed 4/8-bit pack, then validate it
# through mlx-lm.
set -uo pipefail
cd /Volumes/NVME/ling-3.2-air

while pgrep -f move_hf_blobs_to_nvme >/dev/null; do sleep 15; done
if ! grep -q "^DONE" move-blobs.log; then
  echo "MOVER_DID_NOT_FINISH_OK"; tail -5 move-blobs.log; exit 1
fi
echo "MOVER_OK: $(grep '^DONE' move-blobs.log)"

.venv/bin/python convert_ling_to_mlx.py \
  --hf-dir inclusionAI/Ling-3.2-air \
  --out Ling-3.2-air-MLX-mixed-4-8bit \
  --expert8-profile expert_profile_coding.json --expert8-top 12 \
  > build-mixed.log 2>&1
if [ $? -ne 0 ]; then echo "BUILD_FAIL"; tail -20 build-mixed.log; exit 1; fi
echo "BUILD_OK: $(tail -2 build-mixed.log | head -1)"

.venv/bin/python check_pack_python.py Ling-3.2-air-MLX-mixed-4-8bit > check-mixed.log 2>&1
if [ $? -ne 0 ]; then echo "CHECK_FAIL"; tail -20 check-mixed.log; exit 1; fi
echo "CHECK_OK: $(grep GREEDY check-mixed.log)"
