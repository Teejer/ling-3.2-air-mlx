"""Round-2 LoRA training: target SPECIFIC layers, not just the last N.

The coding story (expert_profile_coding.json) puts the action at layers
{11,12,17,20,23,24,27,29,30,32,35,36}, but mlx-lm's CLI only wraps the last
--num-layers blocks. Strategy: keep references to the ORIGINAL Linear /
QuantizedLinear modules outside the target set, let mlx-lm wrap ALL layers,
then put the untouched originals back (their LoRA wrappers are discarded
before the optimizer is built, so no params train there), and train via
mlx_lm.lora's entrypoint.

Usage (same flags as mlx_lm.lora, plus --layers):
  python train_round2.py --model DIR --train --data DIR --adapter-path DIR \
      --iters 250 [--layers 11,12,...]
"""
import sys

import mlx.nn as nn
import mlx_lm.lora as lora_mod
from mlx_lm.tuner.utils import linear_to_lora_layers as _orig_apply

DEFAULT_LAYERS = [11, 12, 17, 20, 23, 24, 27, 29, 30, 32, 35, 36]
ALLOWED = set(DEFAULT_LAYERS)
RANK = 16   # mlx-lm 0.32.2 silently ignored the yaml rank (round 1 got 8)
SCALE = 32


def _layer_index(module_path):
    parts = module_path.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def _unwrap_lora(module):
    """Return the plain Linear/QuantizedLinear underlying a LoRA wrapper."""
    base = getattr(module, "linear", None)
    if base is not None:
        return base  # mlx-lm 0.32.2: LoRA wraps by composition (.linear)
    params = dict(module.parameters())
    params = {k: v for k, v in params.items() if k not in ("lora_a", "lora_b")}
    if "scales" in params:  # quantized
        w, s = params["weight"], params["scales"]
        out, packed = w.shape
        group_size = getattr(module, "group_size", 64)
        in_features = s.shape[-1] * group_size
        bits = (32 * packed) // in_features
        new = nn.QuantizedLinear(in_features, out, bias=False,
                                 group_size=group_size, bits=bits)
    else:
        out, in_features = params["weight"].shape
        new = nn.Linear(in_features, out, bias="bias" in params)
    new.update(params)
    return new


def selective_apply(model, num_layers, config, use_dora=False):
    """Wrap all, then restore non-target layers to their original modules."""
    config = dict(config)
    config["rank"] = RANK
    config["scale"] = SCALE
    print(f"[round2] forcing rank={RANK} scale={SCALE} (yaml rank ignored upstream)")

    originals = {}
    for path, mod in list(model.named_modules()):
        li = _layer_index(path)
        if li is None or li in ALLOWED:
            continue
        if isinstance(mod, (nn.Linear, nn.QuantizedLinear)):
            originals[path] = mod

    _orig_apply(model, len(model.layers), config, use_dora)

    restored = 0
    for path, mod in originals.items():
        parent_path, attr = path.rsplit(".", 1)
        parent = model
        for part in parent_path.split("."):
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        setattr(parent, attr, mod)  # unwrapped original; wrapper dropped
        restored += 1

    # Deterministic second sweep: anything LoRA'd outside ALLOWED (e.g.
    # shared_experts whose restore-by-path missed) gets unwrapped from params.
    swept = 0
    for path, mod in list(model.named_modules()):
        if not hasattr(mod, "lora_a"):
            continue
        li = _layer_index(path)
        if li is not None and li in ALLOWED:
            continue
        parent_path, attr = path.rsplit(".", 1)
        parent = model
        for part in parent_path.split("."):
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        setattr(parent, attr, _unwrap_lora(mod))
        swept += 1

    kept = sum(1 for _, m in model.named_modules() if hasattr(m, "lora_a"))
    print(f"[round2] LoRA kept on {kept} modules; restored {restored}, swept {swept}")
    return model


def main():
    argv = sys.argv[1:]
    layers = DEFAULT_LAYERS
    if "--layers" in argv:
        i = argv.index("--layers")
        layers = [int(x) for x in argv[i + 1].split(",")]
        del argv[i:i + 2]
    ALLOWED.clear()
    ALLOWED.update(layers)
    print(f"[round2] target layers: {sorted(ALLOWED)}")

    lora_mod.linear_to_lora_layers = selective_apply
    sys.argv = ["train_round2.py"] + argv
    lora_mod.main()


if __name__ == "__main__":
    main()
