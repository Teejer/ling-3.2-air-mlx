"""Fix a fused MLX pack for the Zig engine: re-materialize kv_b_proj.

mlx_lm.fuse saves the training-side view of bailing_moe_v3, where sanitize
split kv_b_proj into embed_q/unembed_out (bf16, unquantized). The engine's
bailing_hybrid plugin wants model.layers.N.attention.kv_b_proj.weight.
This inverts the split in-place, shard by shard, preserving quantization
metadata for everything else:

  value[:, :nope, :] = embed_q.swapaxes(-1, -2)      # (heads, nope, hidden)
  value[:, nope:, :] = unembed_out                    # (heads, v, hidden)
  kv_b_proj.weight = value.reshape(heads*(nope+v), hidden)

Usage: python fix_fused_pack.py PACK_DIR [--dry]
"""
import json
import sys
from pathlib import Path

import mlx.core as mx


def main():
    pack = Path(sys.argv[1])
    dry = "--dry" in sys.argv
    index = json.loads((pack / "model.safetensors.index.json").read_text())
    shards = sorted({v for k, v in index["weight_map"].items() if "embed_q" in k})
    print(f"{len(shards)} shard(s) contain embed_q/unembed_out")
    total = 0
    for shard in shards:
        path = pack / shard
        tensors, meta = mx.load(str(path), return_metadata=True)
        eq_keys = [k for k in tensors if k.endswith("embed_q.weight")]
        changed = {}
        for eqk in eq_keys:
            base = eqk[: -len("embed_q.weight")]
            eok = base + "unembed_out.weight"
            if eok not in tensors:
                print(f"  SKIP {base}: unembed_out missing")
                continue
            eq, eo = tensors[eqk], tensors[eok]
            heads, hidden, nope = eq.shape          # (heads, hidden, nope)
            v = eo.shape[1]
            assert eo.shape == (heads, v, hidden), eo.shape
            value = mx.concatenate(
                [eq.swapaxes(-1, -2), eo], axis=1
            )  # (heads, nope+v, hidden)
            kv = value.reshape(heads * (nope + v), hidden).astype(mx.bfloat16)
            changed[base + "kv_b_proj.weight"] = kv
            for drop in (eqk, eok):
                tensors.pop(drop)
            total += 1
        if changed and not dry:
            tensors.update(changed)
            mx.save_safetensors(str(path), tensors, metadata=meta or {})
        print(f"  {shard}: {len(changed)} kv_b_proj rebuilt"
              + (" [dry]" if dry else ""))
    # update the index weight_map
    if total and not dry:
        wm = index["weight_map"]
        for k in list(wm):
            if "embed_q" in k or "unembed_out" in k:
                shard = wm.pop(k)
                base = k[: -len("embed_q.weight")] if k.endswith("embed_q.weight") else k[: -len("unembed_out.weight")]
                newk = base + "kv_b_proj.weight"
                if newk not in wm:
                    wm[newk] = shard
        (pack / "model.safetensors.index.json").write_text(json.dumps(index))
    print(f"done: {total} kv_b_proj tensors" + (" (dry run, nothing written)" if dry else ""))


if __name__ == "__main__":
    main()
