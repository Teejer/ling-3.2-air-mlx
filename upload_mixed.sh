#!/bin/bash
# Chain 3: once both mixed builds + checks are done, write model cards and
# upload the packs to the hub as teejer/Ling-3.2-air-mlx-mixed-4-8bit[-vl].
set -uo pipefail
cd /Volumes/NVME/ling-3.2-air

# wait for both build chains to exit
while pgrep -f "build_mixed(_vl)?\.sh" >/dev/null; do sleep 30; done

ok_text=0; ok_vl=0
[ -f Ling-3.2-air-MLX-mixed-4-8bit/model.safetensors.index.json ] \
  && grep -q GREEDY check-mixed.log && ok_text=1
[ -f Ling-3.2-air-MLX-mixed-4-8bit-VL/model.safetensors.index.json ] \
  && grep -q GREEDY check-mixed-vl.log 2>/dev/null && ok_vl=1
echo "gates: text=$ok_text vl=$ok_vl"

gen_readme() {  # $1=pack dir  $2=repo id  $3=vl(1/0)
  .venv/bin/python - "$1" "$2" "$3" <<'PYEOF'
import json, sys, pathlib
pack, repo, is_vl = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
mq = json.loads((pathlib.Path(pack) / "mixed-quant.json").read_text())
layers = mq["expert8_layers"]
greedy = ""
try:
    for line in open(f"check-{'mixed-vl' if is_vl else 'mixed'}.log"):
        if "GREEDY" in line:
            greedy = line.strip()
except FileNotFoundError:
    pass
import os
size_gb = sum(f.stat().st_size for f in pathlib.Path(pack).glob("*.safetensors")) / 1e9
card = f"""---
base_model: inclusionAI/Ling-3.2-air
tags:
- mlx
- quantization
- moe
- mtp
language: en
---

# {"Ling-3.2-air MLX mixed 4/8-bit (VL)" if is_vl else "Ling-3.2-air MLX mixed 4/8-bit"} — coding-expert quant

MLX conversion of [inclusionAI/Ling-3.2-air](https://huggingface.co/inclusionAI/Ling-3.2-air)
(124B-A5.5B hybrid-MoE, FP8 block-128) with **mixed-precision experts**: the
MoE layers that routing profiles show carrying code are stored at
{mq['expert_bits']}-bit (group {mq['base']['group_size']}, affine), everything else at
{mq['base']['bits']}-bit g{mq['base']['group_size']}.{" Includes the bf16 vision tower (`model.visual.*`), `vision_config`, M-RoPE (`mrope_section [8,12,12]`) and the vision token ids — served by mlx-serve's `ling-vl` engine branch." if is_vl else " Text-only (vision tower dropped); the VL sibling is `teejer/Ling-3.2-air-mlx-mixed-4-8bit-vl`."}

**8-bit expert layers:** `{layers}`

Selected by profiling router top-k activations over a code corpus vs a prose
baseline (mlx-lm `bailing_moe_v3`, per-layer score = top-10 code-routing
concentration x code-specific mass) — see `mixed-quant.json` for provenance.
The NextN MTP head is kept at `model.layers.42.*` (works with the
mlx-serve MTP-for-MLX-packs patch, ~+31% decode).

Per-tensor bits are solvable from the packed geometry (weights pack into
uint32 words), so mlx-serve loads this pack unmodified; `config.json`
carries the base 4-bit defaults.

Built with the kit at [Teejer/ling-3.2-air-mlx](https://github.com/Teejer/ling-3.2-air-mlx)
(converter + profiler + validator): FP8 e4m3fn dequant verified exact vs torch; pack validated
greedy through mlx-lm's independent implementation.
{f"Greedy sample from validation: `{greedy}`" if greedy else ""}

Size: {size_gb:.1f} GB. Served with [mlx-serve](https://github.com/ddalcu/mlx-serve)
(`bailing_hybrid`{"/`ling-vl`" if is_vl else ""}); pass the pack dir as an absolute path.
"""
pathlib.Path(pack, "README.md").write_text(card)
print(f"README written for {pack} ({size_gb:.1f} GB)")
PYEOF
}

upload() {  # $1=repo  $2=dir
  echo "UPLOAD_START $1 ($(du -sh "$2" | cut -f1))"
  .venv/bin/python - "$1" "$2" <<'PYEOF'
import sys
from huggingface_hub import create_repo, upload_large_folder
repo, folder = sys.argv[1], sys.argv[2]
create_repo(repo, repo_type="model", exist_ok=True)
upload_large_folder(
    repo_id=repo, folder_path=folder, repo_type="model",
    commit_description=f"mixed 4/8-bit coding-expert pack",
)
print(f"UPLOAD_DONE {repo}")
PYEOF
}

if [ "$ok_text" = 1 ]; then
  gen_readme Ling-3.2-air-MLX-mixed-4-8bit teejer/Ling-3.2-air-mlx-mixed-4-8bit 0
  upload teejer/Ling-3.2-air-mlx-mixed-4-8bit Ling-3.2-air-MLX-mixed-4-8bit || echo "TEXT_UPLOAD_FAIL"
else
  echo "TEXT_PACK_NOT_READY - not uploading (inspect build-mixed.log / check-mixed.log)"
fi

if [ "$ok_vl" = 1 ]; then
  gen_readme Ling-3.2-air-MLX-mixed-4-8bit-VL teejer/Ling-3.2-air-mlx-mixed-4-8bit-vl 1
  upload teejer/Ling-3.2-air-mlx-mixed-4-8bit-vl Ling-3.2-air-MLX-mixed-4-8bit-VL || echo "VL_UPLOAD_FAIL"
else
  echo "VL_PACK_NOT_READY - not uploading (inspect build-mixed-vl.log / check-mixed-vl.log)"
fi

# serve-ready links next to the existing packs
ln -sfn "/Volumes/NVME/ling-3.2-air/Ling-3.2-air-MLX-mixed-4-8bit" /Volumes/NVME/models/Ling-3.2-air-MLX-mixed-4-8bit
ln -sfn "/Volumes/NVME/ling-3.2-air/Ling-3.2-air-MLX-mixed-4-8bit-VL" /Volumes/NVME/models/Ling-3.2-air-MLX-mixed-4-8bit-VL
echo "CHAIN3_DONE"
