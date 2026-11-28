#!/usr/bin/env python3
"""Convert an inclusionAI Ling release checkpoint to a text-only MLX pack.

Targets the layout mlx-lm's `bailing_moe_v3` module ships and mlx-serve's
`bailing_hybrid` engine reads:

  * drop the vision tower (`model.visual.*`, `linear_proj`) and keep the NextN
    layer (index == num_hidden_layers) as the MTP head (the engine ignores it
    on MLX packs today; carrying it is cheap insurance for when it is wired);
  * dequantize the release's block-FP8 (`weight_scale_inv`, 128x128 blocks)
    tensors to bfloat16;
  * stack `mlp.experts.{e}.{gate,up,down}_proj` into `mlp.switch_mlp.*`;
  * absorb MLA's `kv_b_proj` into `embed_q` / `unembed_out`;
  * lay conv1d weights out flat (`<x>_conv1d.weight`, [dim, K, 1]);
  * affine-quantize the release's fp8 matrices (4-bit group 64 by default),
    keeping embeddings, lm_head, MLA projections, router and per-channel gates
    wide, as mlx-lm's quant_predicate does;
  * write a flattened top-level `text_config` as `model_type: bailing_hybrid`
    (with `rope_interleave` explicit: the release omits it, the engine
    defaults it to false, the reference defaults it to true).

The layer-type rule follows the released checkpoint: MLA iff
(layer + 1) % layer_group_size == 0 (42 layers / group 6 -> MLA at 5, 11, .., 41;
mlx-lm's `_is_kda_layer` tail clause must not fire — it would for n % group != 0).

Usage:
  python convert_ling_to_mlx.py --hf-dir <local dir or HF repo id> --out <dir> \
      [--bits 4] [--group-size 64] [--no-head]
"""

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

SHARD_BYTES = 5 * 1024**3  # mlx-lm's make_shards size


# ----------------------------------------------------------------- utilities


def find_hf_dir(spec: str) -> Path:
    p = Path(spec)
    if p.exists():
        return p
    name = "models--" + spec.replace("/", "--")
    snap = Path.home() / ".cache" / "huggingface" / "hub" / name / "snapshots"
    if snap.exists():
        return sorted(snap.iterdir())[-1]
    raise FileNotFoundError(f"neither a directory nor a cached repo: {spec}")


def layer_index(key: str):
    m = re.match(r"model\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def _e4m3_table() -> mx.array:
    """256-entry e4m3fn decode table (this MLX build loads FP8 as raw uint8
    and exposes no float8 dtype)."""
    i = np.arange(256)
    e, m = (i >> 3) & 0xF, i & 7
    s = np.where(i & 0x80, -1.0, 1.0)
    vals = np.where(e > 0, (1 + m / 8) * 2.0 ** (e - 7), m / 8 * 2.0**-6) * s
    vals[(i & 0x7F) == 0x7F] = np.nan  # e4m3fn has no inf; 0x7F/0xFF are NaN
    return mx.array(vals)


E4M3 = None


def dequant_fp8(w: mx.array, s: mx.array) -> mx.array:
    """e4m3 weight + 128x128 block scale grid (weight_scale_inv) -> bfloat16.
    Matches torch's float8_e4m3fn decode exactly (verified on the release)."""
    global E4M3
    if E4M3 is None:
        E4M3 = _e4m3_table()
    out, inn = w.shape
    if w.dtype == mx.uint8:
        w = mx.take(E4M3, w.reshape(-1)).reshape(out, inn)
    sc = mx.repeat(mx.repeat(s.astype(mx.float32), 128, axis=0), 128, axis=1)[:out, :inn]
    return (w.astype(mx.float32) * sc).astype(mx.bfloat16)


def quantize2d(w: mx.array, bits: int, group: int):
    """(packed, scales, biases) exactly as mlx-lm's nn.quantize stores them
    (bf16 in, bf16 scales/biases out, like every published MLX pack)."""
    if w.dtype != mx.bfloat16:
        w = w.astype(mx.bfloat16)
    return mx.quantize(w, group_size=group, bits=bits)


# ---------------------------------------------------------------- quant rules


class Plan:
    """Which tensors quantize, mirroring mlx-lm bailing_moe_v3.quant_predicate
    with the released checkpoint's layer-type rule. Layers listed in
    `expert8` store their stacked routed experts at `expert_bits` (mixed
    4/8-bit pack); the engine solves bits per tensor from the packed
    geometry, so the config keeps the base bits as the default."""

    def __init__(self, cfg: dict, bits: int, group: int,
                 expert8: frozenset[int] = frozenset(), expert_bits: int = 8):
        self.bits = bits
        self.group = group
        self.expert8 = expert8
        self.expert_bits = expert_bits
        self.n_layers = cfg["num_hidden_layers"]
        self.group_size_layers = cfg["layer_group_size"]
        self.num_experts = cfg["num_experts"]

    force_mla: bool | None = None  # set while processing the NextN head

    def is_mla(self, layer: int) -> bool:
        if self.force_mla is not None:
            return self.force_mla
        return (layer + 1) % self.group_size_layers == 0

    def expert_bits_for(self, layer: int) -> int:
        return self.expert_bits if layer in self.expert8 else self.bits

    def quantize(self, key: str) -> bool:
        if not key.endswith(".weight"):
            return False  # norms, A_log, dt_bias, expert_bias
        if key.startswith(("model.word_embeddings", "lm_head")):
            return False
        if ".mlp.gate." in key:
            return False  # router stays wide
        if ".attention." in key:
            name = key[: -len(".weight")].split(".")[-1]
            if name in ("b_proj", "o_norm") or name.endswith("_conv1d"):
                return False
            if self.is_mla(layer_index(key)):
                return name in ("dense", "kv_b_proj")
            return name in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj", "o_proj")
        if ".mlp." in key:  # dense MLP, shared experts, (experts handled apart)
            return True
        return False


# ------------------------------------------------------------------- writer


class ShardWriter:
    def __init__(self, out_dir: Path):
        self.out = out_dir
        self.buf, self.buf_bytes, self.parts, self.index = {}, 0, 0, {}

    def add(self, name: str, arr: mx.array):
        self.buf[name] = arr
        self.buf_bytes += arr.nbytes
        if self.buf_bytes >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        self.parts += 1
        part = self.out / f"_part{self.parts:05d}.safetensors"
        mx.save_safetensors(str(part), self.buf, metadata={"format": "mlx"})
        for name in self.buf:
            self.index[name] = part.name
        self.buf, self.buf_bytes = {}, 0

    def finish(self, total_params: int) -> int:
        self.flush()
        shards = sorted(self.out.glob("_part*.safetensors"))
        count = len(shards)
        fmt = (
            "model-{:05d}-of-{:05d}.safetensors"
            if count > 1
            else "model.safetensors"
        )
        total_size = 0
        remap = {}
        for i, part in enumerate(shards):
            name = fmt.format(i + 1, count)
            part.rename(self.out / name)
            remap[part.name] = name
        for name, shard in self.index.items():
            self.index[name] = remap[shard]
        total_size = sum((self.out / s).stat().st_size for s in set(remap.values()))
        index_data = {
            "metadata": {"total_size": total_size, "total_parameters": total_params},
            "weight_map": dict(sorted(self.index.items())),
        }
        (self.out / "model.safetensors.index.json").write_text(
            json.dumps(index_data, indent=2)
        )
        return total_size


# ----------------------------------------------------------------- transform


def emit(name: str, arr: mx.array, plan: Plan, out: ShardWriter):
    """One final (post-transform) tensor: quantize per plan, buffer for save.
    Kept-wide tensors land in bf16, as mlx-lm's convert casts the model."""
    if plan.quantize(name):
        w, s, b = quantize2d(arr, plan.bits, plan.group)
        mx.eval(w, s, b)
        stem = name[: -len(".weight")]
        out.add(f"{stem}.weight", w)
        out.add(f"{stem}.scales", s)
        out.add(f"{stem}.biases", b)
    else:
        if arr.dtype not in (mx.bfloat16, mx.float16):
            arr = arr.astype(mx.bfloat16)
        mx.eval(arr)
        out.add(name, arr)


def process_layer(b: dict, L: int, cfg: dict, plan: Plan, out: ShardWriter):
    p = f"model.layers.{L}."
    is_mla = plan.is_mla(L)

    # 1. routed experts: dequant -> stack -> quantize per expert
    #    (mixed 4/8-bit: a promoted layer's whole stack lands at expert_bits)
    ebits = plan.expert_bits_for(L)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        first = f"{p}mlp.experts.0.{proj}.weight"
        if first not in b:
            continue
        ws, ss, bs = [], [], []
        for e in range(plan.num_experts):
            base = f"{p}mlp.experts.{e}.{proj}.weight"
            w = b.pop(base)
            s = b.pop(base + "_scale_inv", None)
            if s is not None:
                w = dequant_fp8(w, s)
            if plan.quantize(base):
                qw, qs, qb = quantize2d(w, ebits, plan.group)
                ws.append(qw), ss.append(qs), bs.append(qb)
            else:
                ws.append(w.astype(mx.bfloat16))
            del w
        key = f"{p}mlp.switch_mlp.{proj}"
        if ss:
            for suffix, parts in (("weight", ws), ("scales", ss), ("biases", bs)):
                stacked = mx.stack(parts)
                mx.eval(stacked)
                out.add(f"{key}.{suffix}", stacked)
        else:
            stacked = mx.stack(ws)
            mx.eval(stacked)
            out.add(f"{key}.weight", stacked)
        del ws, ss, bs

    # 2. (no MLA absorb: mlx-serve's bailing loader binds the raw kv_b_proj;
    #    it flows through step 3 with its fp8 pair applied and quantizes like
    #    the published packs)

    # 3. everything else: dequant fp8 pairs, fix conv layout, quantize per plan
    for name in sorted(b):
        if name.endswith("_scale_inv"):
            continue
        arr = b[name]
        if name.endswith(".weight") and (name + "_scale_inv") in b:
            arr = dequant_fp8(arr, b[name + "_scale_inv"])
        if "_conv1d.weight" in name and arr.ndim == 3 and arr.shape[-1] != 1:
            arr = arr.moveaxis(2, 1)
        emit(name, arr, plan, out)


# --------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, help="local dir or HF repo id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--no-head", action="store_true", help="drop the NextN layer")
    ap.add_argument("--vl", action="store_true",
                    help="VL pack: keep the vision tower (bf16) + linear_proj, keep vision_config/"
                         "mrope/token ids, declare model_type bailing_moe_v3_vl")
    ap.add_argument("--expert-bits", type=int, default=8,
                    help="bits for the stacked experts of --expert8-layers layers")
    ap.add_argument("--expert8-layers", default=None,
                    help="comma-separated layer indices whose routed experts take --expert-bits")
    ap.add_argument("--expert8-profile", default=None,
                    help="expert_profile_coding.json from profile_expert_coding.py; "
                         "takes --expert8-top layers by score (or its recommended_8bit_layers)")
    ap.add_argument("--expert8-top", type=int, default=12)
    args = ap.parse_args()

    expert8 = frozenset()
    if args.expert8_layers:
        expert8 = frozenset(int(x) for x in args.expert8_layers.split(","))
    elif args.expert8_profile:
        prof = json.loads(Path(args.expert8_profile).read_text())
        ranked = sorted(prof["layers"], key=lambda d: -d["score"])
        expert8 = frozenset(d["layer"] for d in ranked[: args.expert8_top])

    hf, out = find_hf_dir(args.hf_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = json.loads((hf / "config.json").read_text())
    src = raw.get("text_config", raw)
    n = src["num_hidden_layers"]

    # config: flatten text_config to the top level
    cfg = dict(src)
    cfg["model_type"] = "bailing_hybrid"
    cfg.setdefault("rope_interleave", True)  # release default; engine needs it explicit
    cfg.setdefault("num_shared_experts", 1)
    cfg.setdefault("tie_word_embeddings", False)
    cfg["torch_dtype"] = "bfloat16"
    cfg["architectures"] = ["BailingMoeV3VLForConditionalGeneration" if args.vl else "BailingMoeV3ForCausalLM"]
    if args.vl:
        cfg["model_type"] = "bailing_moe_v3_vl"
        vc = dict(raw["vision_config"])
        vc.setdefault("min_pixels", 4096)      # the release processor's resize bounds,
        vc.setdefault("max_pixels", 4194304)   # so the engine config carries them without the preprocessor
        cfg["vision_config"] = vc
        for k in ("image_token_id", "video_token_id", "vision_start_token_id",
                  "vision_end_token_id", "video_start_token_id", "video_end_token_id"):
            if k in raw:
                cfg[k] = raw[k]
    else:
        cfg.pop("mrope_section", None)
        cfg.pop("vision_config", None)
    cfg.pop("quantization_config", None)  # the release's fp8 one; replaced below
    cfg["quantization"] = {
        "group_size": args.group_size,
        "bits": args.bits,
        "mode": "affine",
    }
    cfg["quantization_config"] = dict(cfg["quantization"])
    if args.no_head:
        cfg["num_nextn_predict_layers"] = 0
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    plan = Plan(src, args.bits, args.group_size, expert8, args.expert_bits)
    if expert8:
        print(f"mixed pack: experts at {args.expert_bits}-bit on layers "
              f"{sorted(expert8)}, {args.bits}-bit elsewhere", flush=True)
    writer = ShardWriter(out)

    shards = sorted(hf.glob("model-*.safetensors")) or [hf / "model.safetensors"]
    buckets: dict[int, dict] = {}
    loose: dict[str, mx.array] = {}
    total_params = 0
    t0 = time.time()

    for shard in shards:
        print(f"read  {shard.name}", flush=True)
        tensors = mx.load(str(shard))
        for name, arr in tensors.items():
            if name.startswith(("model.visual.", "linear_proj")):
                if not args.vl:
                    continue
                # VL pack: the tower ships bf16 — route it through `loose` (kept wide, never quantized)
                loose[name] = arr
                continue
            if name.endswith("_scale_inv"):
                (buckets.setdefault(layer_index(name), {}))[name] = arr
                continue
            li = layer_index(name)
            if li is not None and li >= n + (0 if args.no_head else 1):
                continue  # anything past the optional NextN layer
            if li is not None and li < n:
                (buckets.setdefault(li, {}))[name] = arr
            elif li == n:
                (buckets.setdefault(n, {}))[name] = arr  # the head
            else:
                loose[name] = arr
        tensors.clear()

    # embed / lm_head / final norm (kept wide)
    for name in sorted(loose):
        arr = loose[name]
        mx.eval(arr)
        writer.add(name, arr if arr.dtype == mx.bfloat16 else arr.astype(mx.bfloat16))
        total_params += arr.size
    loose.clear()

    # layers, in order (0..n-1 trunk, n = the NextN head, kept as layer n)
    head_layer = n
    for L in sorted(buckets):
        b = buckets.pop(L)
        is_head = L == head_layer
        before = len(writer.index)
        if is_head:
            # the head's own matrices: keep wide like an MLA layer's projections
            arr = b.pop(f"model.layers.{L}.eh_proj.weight", None)
            if arr is not None:
                mx.eval(arr)
                writer.add(f"model.layers.{L}.eh_proj.weight", arr.astype(mx.bfloat16))
            plan.force_mla = True  # the head is an MLA block (mlx-serve head.zig)
            process_layer(b, L, src, plan, writer)
            plan.force_mla = None
        else:
            process_layer(b, L, src, plan, writer)
        print(
            f"layer {L:>3} done  ({len(writer.index) - before} tensors, "
            f"{(time.time() - t0) / 60:.1f} min)",
            flush=True,
        )

    total = writer.finish(total_params)
    print(f"\nwrote {len(writer.index)} tensors, {total / 1e9:.1f} GB -> {out}")

    if expert8:
        # provenance sidecar (config.json keeps the base quantization dict;
        # the engine solves per-tensor bits from the packed geometry)
        (out / "mixed-quant.json").write_text(json.dumps({
            "base": {"bits": args.bits, "group_size": args.group_size, "mode": "affine"},
            "expert_bits": args.expert_bits,
            "expert8_layers": sorted(expert8),
            "profile": args.expert8_profile,
        }, indent=2) + "\n")

    # tokenizer + generation files
    for f in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        src_f = hf / f
        if src_f.exists():
            shutil.copy(src_f, out / f)
    print("copied tokenizer/generation files")


if __name__ == "__main__":
    main()
