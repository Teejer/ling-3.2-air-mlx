"""Round-2 LoRA training: target SPECIFIC layers, not just the last N.

The coding story (expert_profile_coding.json) puts the action at layers
{11,12,17,20,23,24,27,29,30,32,35,36}, but mlx-lm's CLI only wraps the last
--num-layers blocks. Strategy: snapshot every Linear/QuantizedLinear's
parameters (lazy arrays — cheap), let mlx-lm wrap ALL layers, then rebuild
the non-target layers from their snapshots (dropping the LoRA params), and
train via mlx_lm.lora.run with num_layers = all.

Usage (same flags as mlx_lm.lora, plus --layers):
  python train_round2.py --model DIR --train --data DIR --adapter-path DIR \
      --iters 250 --layers 11,12,17,20,23,24,27,29,30,32,35,36
"""
import argparse
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.lora as lora_mod
from mlx_lm.tuner.utils import linear_to_lora_layers as _orig_apply

DEFAULT_LAYERS = [11, 12, 17, 20, 23, 24, 27, 29, 30, 32, 35, 36]
ALLOWED = set(DEFAULT_LAYERS)


def _layer_index(module_path):
    parts = module_path.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def selective_apply(model, num_layers, config, use_dora=False):
    """Wrap all, then restore non-target layers from snapshots."""
    # snapshot originals of every leaf linear under model.layers
    snapshots = {}
    for path, mod in list(model.named_modules()):
        if not path.startswith("layers."):
            continue
        li = _layer_index(path)
        if li is None or li in ALLOWED:
            continue
        if isinstance(mod, nn.QuantizedLinear):
            snapshots[path] = ("q", mod.asdict())
        elif isinstance(mod, nn.Linear):
            snapshots[path] = ("l", mod.asdict())

    _orig_apply(model, len(model.layers), config, use_dora)

    # rebuild non-target layers from snapshots (drops LoRA params)
    restored = 0
    for path, (kind, d) in snapshots.items():
        parent_path, attr = path.rsplit(".", 1)
        parent = model
        for part in parent_path.split("."):
            parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
        cls = nn.QuantizedLinear if kind == "q" else nn.Linear
        d.pop("lora_a", None)
        d.pop("lora_b", None)
        d.pop("scale", None)
        d.pop("dropout", None)
        try:
            new = cls(**{k: v for k, v in d.items()
                         if k in ("bias", "weight", "scales", "biases",
                                  "group_size", "bits", "mode")})
        except TypeError:
            # older/newer ctor: build bare and set params
            new = cls(d.get("weight", d.get("scales")).shape[-1], 0) if kind == "l" else cls()
        setattr(parent, attr, new)
        restored += 1
    kept = sum(1 for _, m in model.named_modules() if hasattr(m, "lora_a"))
    print(f"[round2] LoRA kept on {kept} modules; restored {restored} non-target modules")
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
