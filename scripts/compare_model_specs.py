#!/usr/bin/env python3
import json
import re
import struct
from pathlib import Path


RE_FUNC = re.compile(r"func\s+(\w+Spec)\s*\(\)\s*\*ArchSpec\s*\{", re.M)


def read_safetensors_header(path: Path):
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header

def collect_tensor_names(root: Path):
    names = set()
    for st_path in sorted(root.glob("*.safetensors")):
        header = read_safetensors_header(st_path)
        for k in header.keys():
            if k != "__metadata__":
                names.add(k)
    return names

def pick_attn_layer_index(names, layer_types):
    if not layer_types:
        return 0
    max_check = min(len(layer_types), 128)
    prefixes = ["model.layers.", "model.language_model.layers.", "language_model.model.layers."]
    for i in range(max_check):
        for p in prefixes:
            prefix = f"{p}{i}."
            if any(n.startswith(prefix + "self_attn.") or n.startswith(prefix + "attention.") for n in names):
                return i
    return 0

def pick_recurrent_layer_index(names, layer_types):
    if not layer_types:
        return 0
    max_check = min(len(layer_types), 128)
    prefixes = ["model.layers.", "model.language_model.layers.", "language_model.model.layers."]
    for i in range(max_check):
        for p in prefixes:
            prefix = f"{p}{i}."
            if any(n.startswith(prefix + "linear_attn.") or n.startswith(prefix + "conv.") for n in names):
                return i
    return 0

def pick_conv_layer_index(names, layer_types):
    return pick_recurrent_layer_index(names, layer_types)


def load_spec_files(paths):
    parts = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        parts.append(path.read_text())
    return "\n".join(parts)


def extract_spec_body(text, func_name):
    # naive brace matching from func declaration
    m = re.search(r"func\s+%s\s*\(\)\s*\*ArchSpec\s*\{" % re.escape(func_name), text)
    if not m:
        return None
    i = m.end() - 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                return text[i+1:j]
    return None


def parse_arch_names(body):
    # Extract ArchNames block
    m = re.search(r"Names:\s*ArchNames\s*\{", body)
    if not m:
        return {}
    start = m.end()
    # naive brace matching for ArchNames
    depth = 1
    for j in range(start, len(body)):
        if body[j] == '{':
            depth += 1
        elif body[j] == '}':
            depth -= 1
            if depth == 0:
                block = body[start:j]
                break
    else:
        return {}

    out = {}
    # Simple string fields
    for k, v in re.findall(r"(\w+):\s*\"([^\"]+)\"", block):
        out[k] = v
    # Func format strings
    for k, v in re.findall(r"(\w+):\s*func\([^)]*\)\s*string\s*\{\s*return\s*fmt\.Sprintf\(\"([^\"]+)\"", block):
        out[k] = v
    # Candidates: return []string { "a", fmt.Sprintf("..."), ... }
    for k, arr in re.findall(r"(\w+):\s*func\([^)]*\)\s*\[\]string\s*\{\s*return\s*\[\]string\s*\{([^}]*)\}\s*\}", block, re.S):
        vals = []
        vals.extend(re.findall(r"\"([^\"]+)\"", arr))
        vals.extend(re.findall(r"fmt\.Sprintf\(\"([^\"]+)\"", arr))
        out[k] = vals
    return out


def detect_spec(cfg):
    model_type = (cfg.get("model_type") or "").lower()
    archs = [a.lower() for a in cfg.get("architectures", [])]

    def has(substr):
        if substr in model_type:
            return True
        return any(substr in a for a in archs)

    if has("lfm"):
        return "lfm2Spec"
    if has("qwen3_5"):
        return "qwen35Spec"
    if has("qwen3"):
        return "qwen3Spec"
    if has("afmoe"):
        return "afmoeSpec"
    if has("gemma3"):
        return "gemma3Spec"
    if has("granite"):
        return "graniteSpec"
    if has("mistral3"):
        return "mistral3Spec"
    if has("mistral"):
        return "mistralSpec"
    if has("llama"):
        return "llamaSpec"
    return None


def check_presence(header_names, spec, attn_layer_index, recurrent_layer_index):
    missing = []
    def has(name):
        return name in header_names

    def has_any(names, layer_idx):
        expanded = []
        for n in names:
            if "%d" in n:
                expanded.append(n % layer_idx)
            else:
                expanded.append(n)
        return any(n in header_names for n in expanded)

    # string fields
    for key in ["Embedding", "OutputNorm"]:
        if key in spec and not has(spec[key]):
            missing.append((key, spec[key]))
    # candidates
    for key in ["OutputCandidates"]:
        if key in spec:
            vals = spec[key]
            if isinstance(vals, list):
                if not has_any(vals, 0):
                    missing.append((key, vals))
    
    for key in ["AttnNormCandidates", "FfnNormCandidates", "PostAttnNormCandidates", "PostFfnNormCandidates"]:
        if key in spec:
            vals = spec[key]
            if isinstance(vals, list):
                if not (has_any(vals, attn_layer_index) or has_any(vals, recurrent_layer_index)):
                    missing.append((key, vals))

    for key in ["QNormCandidates", "KNormCandidates"]:
        if key in spec:
            vals = spec[key]
            if isinstance(vals, list):
                if not has_any(vals, attn_layer_index):
                    missing.append((key, vals))

    # layer format strings
    for key in ["Wq", "Wk", "Wv", "Wo", "AttnGate"]:
        if key in spec:
            fmt = spec[key]
            name = fmt % attn_layer_index if "%d" in fmt else fmt
            if not has(name):
                missing.append((key, name))

    for key in ["FfnUp", "FfnGate", "FfnDown"]:
        if key in spec:
            fmt = spec[key]
            name = fmt % attn_layer_index if "%d" in fmt else fmt
            if not has(name):
                # try recurrent index too
                name2 = fmt % recurrent_layer_index if "%d" in fmt else fmt
                if not has(name2):
                    missing.append((key, name))

    for key in ["ShortConvKernel", "ShortConvInProj", "ShortConvOutProj"]:
        if key in spec:
            fmt = spec[key]
            name = fmt % recurrent_layer_index if "%d" in fmt else fmt
            if not has(name):
                missing.append((key, name))

    # Bias tensors
    for key in ["WqBias", "WkBias", "WvBias"]:
        if key in spec:
            fmt = spec[key]
            name = fmt % attn_layer_index if "%d" in fmt else fmt
            if has(name):
                continue
    return missing


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="/work/models/raw")
    args = ap.parse_args()

    spec_text = load_spec_files(["internal/model/modelspec.go", "internal/model/afmoe.go"])
    if not spec_text:
        raise SystemExit("No spec files found to parse")

    root = Path(args.root)
    if (root / "config.json").exists():
        dirs = [root]
    else:
        dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for d in dirs:
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        text_cfg = cfg.get("text_config", {})
        layer_types = cfg.get("layer_types") or text_cfg.get("layer_types")
        spec_func = detect_spec(cfg)
        if spec_func is None:
            print(f"{d.name}: no spec mapping for model_type={cfg.get('model_type')}")
            continue
        body = extract_spec_body(spec_text, spec_func)
        if body is None:
            print(f"{d.name}: spec func {spec_func} not found")
            continue
        arch = parse_arch_names(body)
        names = collect_tensor_names(d)
        attn_layer_index = pick_attn_layer_index(names, layer_types)
        recurrent_layer_index = pick_recurrent_layer_index(names, layer_types)
        missing = check_presence(names, arch, attn_layer_index, recurrent_layer_index)
        if missing:
            print(f"{d.name}: {spec_func} MISSING {len(missing)} entries")
            for k, v in missing[:12]:
                print("  -", k, ":", v)
            if len(missing) > 12:
                print("  ...")
        else:
            print(f"{d.name}: {spec_func} OK")


if __name__ == "__main__":
    main()
