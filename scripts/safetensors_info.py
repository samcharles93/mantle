#!/usr/bin/env python3
"""
Quick-and-dirty safetensors model inspector.

Accepts a safetensors directory (or a .safetensors file) and prints the pieces
that matter for Mantle packing/running: config/tokenizer presence, architecture,
shards, tensor counts, dtypes, and approximate payload size.

Pure stdlib, no MCF interaction.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

INDEX_FILE = "model.safetensors.index.json"


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return max(0, self.end - self.start)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_safetensors_header(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"file too small: {path}")
        (header_len,) = struct.unpack("<Q", raw)
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"truncated header: {path}")
    try:
        return json.loads(header_bytes.decode("utf-8"))
    except Exception as e:  # pragma: no cover - quick utility
        raise ValueError(f"invalid safetensors header JSON: {path}: {e}") from e


def tensors_from_header(header: Dict[str, Any]) -> List[TensorInfo]:
    out: List[TensorInfo] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        try:
            dtype = str(entry.get("dtype", ""))
            shape = tuple(int(x) for x in entry.get("shape", []))
            offsets = entry.get("data_offsets", [0, 0])
            start = int(offsets[0])
            end = int(offsets[1])
        except Exception:
            continue
        out.append(TensorInfo(name=name, dtype=dtype, shape=shape, start=start, end=end))
    out.sort(key=lambda t: t.name)
    return out


def discover_model(path: str) -> Tuple[str, Dict[str, str]]:
    """
    Returns (base_dir, tensor_to_shard).

    tensor_to_shard maps tensor name -> shard filename (basename). For single-file
    models, every tensor maps to that file's basename.
    """
    path = os.path.abspath(path)
    if os.path.isfile(path):
        base_dir = os.path.dirname(path)
        header = read_safetensors_header(path)
        tensors = tensors_from_header(header)
        shard = os.path.basename(path)
        return base_dir, {t.name: shard for t in tensors}

    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    idx_path = os.path.join(path, INDEX_FILE)
    if os.path.isfile(idx_path):
        idx = read_json(idx_path)
        weight_map = idx.get("weight_map") or {}
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid weight_map in {idx_path}")
        tensor_to_shard = {str(k): str(v) for k, v in weight_map.items()}
        return path, tensor_to_shard

    safetensors_files = sorted(
        p for p in os.listdir(path) if p.lower().endswith(".safetensors")
    )
    if len(safetensors_files) != 1:
        raise ValueError(
            f"expected exactly one .safetensors file or {INDEX_FILE} in {path}; "
            f"found {len(safetensors_files)}"
        )
    sf = os.path.join(path, safetensors_files[0])
    header = read_safetensors_header(sf)
    tensors = tensors_from_header(header)
    shard = os.path.basename(sf)
    return path, {t.name: shard for t in tensors}


def shard_headers(base_dir: str, shards: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    headers: Dict[str, Dict[str, Any]] = {}
    for shard in sorted(set(shards)):
        shard_path = os.path.join(base_dir, shard)
        headers[shard] = read_safetensors_header(shard_path)
    return headers


def summarize_tensors(base_dir: str, tensor_to_shard: Dict[str, str]) -> Dict[str, Any]:
    headers = shard_headers(base_dir, tensor_to_shard.values())

    tensors: List[TensorInfo] = []
    missing: List[str] = []
    for name, shard in sorted(tensor_to_shard.items()):
        header = headers.get(shard, {})
        entry = header.get(name)
        if not isinstance(entry, dict):
            missing.append(name)
            continue
        tensors.extend(tensors_from_header({name: entry}))

    dtype_counts = Counter(t.dtype for t in tensors)
    total_bytes = sum(t.nbytes for t in tensors)

    return {
        "tensor_count": len(tensors),
        "missing_from_headers": missing,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "total_payload_bytes": total_bytes,
        "total_payload_gib": total_bytes / (1 << 30),
        "shard_count": len(headers),
        "shards": sorted(headers.keys()),
    }


def read_config(base_dir: str) -> Dict[str, Any] | None:
    cfg_path = os.path.join(base_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        return read_json(cfg_path)
    except Exception:
        return {"_error": f"failed to parse {cfg_path}"}


def read_tokenizer_config(base_dir: str) -> Dict[str, Any] | None:
    tok_cfg_path = os.path.join(base_dir, "tokenizer_config.json")
    if not os.path.isfile(tok_cfg_path):
        return None
    try:
        return read_json(tok_cfg_path)
    except Exception:
        return {"_error": f"failed to parse {tok_cfg_path}"}


def path_exists(base_dir: str, name: str) -> bool:
    return os.path.isfile(os.path.join(base_dir, name))


def infer_arch(config: Dict[str, Any] | None) -> Dict[str, Any]:
    if not config:
        return {"model_type": None, "architectures": []}
    model_type = config.get("model_type")
    archs = config.get("architectures") or []
    if not isinstance(archs, list):
        archs = [archs]
    return {"model_type": model_type, "architectures": archs}


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Inspect a safetensors model directory")
    ap.add_argument("path", help="Path to a safetensors directory or a .safetensors file")
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human output",
    )
    args = ap.parse_args(argv)

    base_dir, tensor_to_shard = discover_model(args.path)

    config = read_config(base_dir)
    tok_config = read_tokenizer_config(base_dir)
    arch = infer_arch(config)
    tensor_summary = summarize_tensors(base_dir, tensor_to_shard)

    required = {
        "config.json": path_exists(base_dir, "config.json"),
        "tokenizer.json": path_exists(base_dir, "tokenizer.json"),
    }
    optional = {
        "tokenizer_config.json": path_exists(base_dir, "tokenizer_config.json"),
        "generation_config.json": path_exists(base_dir, "generation_config.json"),
        "chat_template.jinja": path_exists(base_dir, "chat_template.jinja"),
        INDEX_FILE: path_exists(base_dir, INDEX_FILE),
    }

    out = {
        "base_dir": base_dir,
        "arch": arch,
        "tokenizer": {
            "tokenizer_config_chat_template": (
                tok_config.get("chat_template") if isinstance(tok_config, dict) else None
            ),
        },
        "required_files": required,
        "optional_files": optional,
        "tensor_summary": tensor_summary,
    }

    if args.json:
        json.dump(out, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    print(f"base_dir: {out['base_dir']}")
    print("arch:")
    print(f"  model_type: {arch['model_type']}")
    if arch["architectures"]:
        print(f"  architectures: {', '.join(str(a) for a in arch['architectures'])}")
    else:
        print("  architectures: (none)")

    chat_tmpl = out["tokenizer"]["tokenizer_config_chat_template"]
    print("tokenizer:")
    if chat_tmpl:
        print("  tokenizer_config.chat_template: present")
    else:
        print("  tokenizer_config.chat_template: missing/empty")

    print("required_files:")
    for k, v in required.items():
        print(f"  {k}: {'yes' if v else 'NO'}")

    print("optional_files:")
    for k, v in optional.items():
        print(f"  {k}: {'yes' if v else 'no'}")

    ts = tensor_summary
    print("tensor_summary:")
    print(f"  shard_count: {ts['shard_count']}")
    print(f"  tensor_count: {ts['tensor_count']}")
    print(f"  total_payload_gib: {ts['total_payload_gib']:.2f}")
    if ts["dtype_counts"]:
        dtypes = ", ".join(f"{k}:{v}" for k, v in ts["dtype_counts"].items())
        print(f"  dtype_counts: {dtypes}")
    if ts["missing_from_headers"]:
        print(f"  missing_from_headers: {len(ts['missing_from_headers'])}")
    
    # Print list of all tensors for inspection
    print("\ntensors:")
    all_tensors = ts.get("shards", [])
    headers = shard_headers(base_dir, all_tensors)
    for shard, header in headers.items():
        for name, info in header.items():
            if name == "__metadata__": continue
            shape = info.get("shape", [])
            print(f"  {name}: {shape}")

    if not all(required.values()):
        print("\nwarning: required files missing for Mantle pack/run", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
