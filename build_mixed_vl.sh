#!/bin/bash
# Chain 2: after the text mixed build+check finishes, build the VL variant of
# the mixed pack (same 8-bit coding-expert trunk + bf16 vision tower) and
# validate it through mlx-lm.
set -uo pipefail
cd /Volumes/NVME/ling-3.2-air

while pgrep -f build_mixed.sh >/dev/null; do sleep 20; done
if ! grep -q "^CHECK_OK" build-chain.log 2>/dev/null \
   && ! grep -q "^CHECK_OK" /Users/teejer/.local/share/opencode/shell/*/sh_21f98376f0010IZ3K1UIWw3kxL.out 2>/dev/null; then
  echo "TEXT_CHAIN_NOT_OK - skipping VL build; inspect build-mixed.log / check-mixed.log"; exit 1
fi
echo "TEXT_CHAIN_OK, starting VL mixed build"

.venv/bin/python convert_ling_to_mlx.py \
  --hf-dir inclusionAI/Ling-3.2-air \
  --out Ling-3.2-air-MLX-mixed-4-8bit-VL \
  --vl --expert8-profile expert_profile_coding.json --expert8-top 12 \
  > build-mixed-vl.log 2>&1
if [ $? -ne 0 ]; then echo "VL_BUILD_FAIL"; tail -20 build-mixed-vl.log; exit 1; fi
echo "VL_BUILD_OK: $(tail -2 build-mixed-vl.log | head -1)"

.venv/bin/python check_pack_python.py Ling-3.2-air-MLX-mixed-4-8bit-VL > check-mixed-vl.log 2>&1
if [ $? -ne 0 ]; then echo "VL_CHECK_FAIL (VL needs the v3_vl loader — expected for this text-only validator)"; tail -8 check-mixed-vl.log; exit 1; fi
echo "VL_CHECK_OK: $(grep GREEDY check-mixed-vl.log)"
