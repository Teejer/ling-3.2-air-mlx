#!/bin/bash
# Upload both mixed packs (model cards already written). Resumable: re-run on
# interruption.
set -uo pipefail
cd /Volumes/NVME/ling-3.2-air

echo "UPLOAD_START text $(date +%H:%M:%S)"
( cd Ling-3.2-air-MLX-mixed-4-8bit && \
  hf upload teejer/Ling-3.2-air-mlx-mixed-4-8bit . . \
    --commit-message "Ling-3.2-air mixed 4/8-bit: 8-bit experts on 12 coding-specialist layers" ) \
  && echo "UPLOAD_OK text" || echo "UPLOAD_FAIL text"

echo "UPLOAD_START vl $(date +%H:%M:%S)"
( cd Ling-3.2-air-MLX-mixed-4-8bit-VL && \
  hf upload teejer/Ling-3.2-air-mlx-mixed-4-8bit-vl . . \
    --commit-message "Ling-3.2-air mixed 4/8-bit VL: coding-expert trunk + bf16 vision tower" ) \
  && echo "UPLOAD_OK vl" || echo "UPLOAD_FAIL vl"

echo "UPLOADS_DONE"
