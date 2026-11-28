"""Load our Ling MLX pack into mlx-lm's INDEPENDENT python implementation
(bailing_moe_v3) and run a greedy generation. Coherent text => the pack's
numerics are fine and any garbage under mlx-serve is the engine's business;
garbage here => the conversion is wrong."""
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.bailing_moe_v3 import Model, ModelArgs

PACK = Path(sys.argv[1] if len(sys.argv) > 1 else "Ling-3.2-air-MLX-4bit")
cfg = json.load(open(PACK / "config.json"))
q = cfg["quantization"]
args = ModelArgs.from_dict(cfg)
model = Model(args)

weights = {}
for f in sorted(PACK.glob("model-*.safetensors")):
    weights.update(mx.load(str(f)))

def infer_bits(prefix):
    """Bits of a packed tensor from its geometry (mixed 4/8-bit packs vary
    bits per tensor; this build packs into uint32 words)."""
    w = weights.get(f"{prefix}.weight")
    s = weights.get(f"{prefix}.scales")
    if w is None or s is None:
        return q["bits"]
    in_dim = s.shape[-1] * q["group_size"]
    per_elem = {mx.uint8: 8, mx.uint16: 16, mx.uint32: 32}.get(w.dtype, 8)
    return w.shape[-1] * per_elem // in_dim

# Build quantized modules where the pack stores packed weights (as mlx-lm's
# quantize_model does), so load_weights fills them natively.
import mlx.nn as nn

vp = model.quant_predicate

def class_predicate(path, mod):
    if not hasattr(mod, "to_quantized"):
        return False
    p = vp(path, mod)
    return p if p is not True else {"group_size": q["group_size"], "bits": infer_bits(path)}

nn.quantize(model, group_size=q["group_size"], bits=q["bits"], mode="affine",
            class_predicate=class_predicate)

# v3 modules want absorbed embed_q/unembed_out; drop the NextN head (no module).
n = args.num_hidden_layers
w2 = {}
for k, v in weights.items():
    li = k.split(".")[2] if k.startswith("model.layers.") else None
    if li is not None and li.isdigit() and int(li) >= n:
        continue
    w2[k] = v
for li in range(n):
    pre = f"model.layers.{li}.attention"
    if f"{pre}.kv_b_proj.weight" not in w2 or f"{pre}.embed_q.weight" in w2:
        continue
    kw = w2.pop(f"{pre}.kv_b_proj.weight")
    ks, kb = w2.pop(f"{pre}.kv_b_proj.scales", None), w2.pop(f"{pre}.kv_b_proj.biases", None)
    kvb = kw if ks is None else mx.dequantize(
        kw, ks, kb, group_size=q["group_size"],
        bits=infer_bits(f"{pre}.kv_b_proj"), mode="affine")
    heads, nope, vd = args.num_attention_heads, args.qk_nope_head_dim, args.v_head_dim
    v = kvb.reshape(heads, nope + vd, -1)
    w2[f"{pre}.embed_q.weight"] = mx.contiguous(v[:, :nope, :].swapaxes(-1, -2))
    w2[f"{pre}.unembed_out.weight"] = mx.contiguous(v[:, nope:, :])

missing = model.load_weights(list(w2.items()), strict=False)
missing = list(missing) if missing else []
print("missing/unexpected:", missing[:8], "total", len(missing))
# eval per layer so a big pack doesn't trip the Metal command-buffer watchdog
for layer in model.model.layers:
    mx.eval(layer.parameters())
mx.eval(model.model.word_embeddings.parameters(), model.lm_head.parameters())

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(str(PACK))
prompt = (
    "<role>SYSTEM</role>detailed thinking off<|role_end|>"
    "<role>HUMAN</role>What is 21 + 21? Answer with just the number.<|role_end|>"
    "<role>ASSISTANT</role>\n"
    "<|" + "think" + "|>"
)
ids = mx.array([tok.encode(prompt)])
cache = model.make_cache()
tokens = []
for _ in range(24):
    y = model(ids[:, -1:], cache)
    tok_id = int(y[:, -1, :].argmax(axis=-1))
    mx.eval(y)
    tokens.append(tok_id)
    if tok_id == 156895:
        break
    ids = mx.concatenate([ids, mx.array([[tok_id]])], axis=1)
print("GREEDY:", repr(tok.decode(tokens)))
