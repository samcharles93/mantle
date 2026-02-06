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


def load_spec_files(paths):
    text = "\n".join(Path(p).read_text() for p in paths)
    return text


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
    # Candidates: return []string { "a", "b" }
    for k, arr in re.findall(r"(\w+):\s*func\([^)]*\)\s*\[\]string\s*\{\s*return\s*\[\]string\s*\{([^}]*)\}\s*\}", block, re.S):
        vals = re.findall(r"\"([^\"]+)\"", arr)
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
    if has("qwen3"):
        return "qwen3Spec"
    if has("afmoe"):
        return "afmoeSpec"
    if has("gemma3"):
        return "gemmaSpec"
    if has("granite"):
        return "graniteSpec"
    if has("mistral3"):
        return "mistral3Spec"
    if has("mistral"):
        return "mistralSpec"
    if has("llama"):
        return "llamaSpec"
    return None


def check_presence(header_names, spec):
    missing = []
    def has(name):
        return name in header_names

    def has_any(names):
        return any(n in header_names for n in names)

    # string fields
    for key in ["Embedding", "OutputNorm"]:
        if key in spec and not has(spec[key]):
            missing.append((key, spec[key]))
    # candidates
    for key in ["OutputCandidates", "AttnNormCandidates", "FfnNormCandidates", "PostAttnNormCandidates", "PostFfnNormCandidates", "QNormCandidates", "KNormCandidates"]:
        if key in spec:
            vals = spec[key]
            if isinstance(vals, list):
                if not has_any(vals):
                    missing.append((key, vals))
    # layer0 format strings
    for key in ["Wq", "Wk", "Wv", "Wo", "FfnUp", "FfnGate", "FfnDown", "AttnGate", "WqBias", "WkBias", "WvBias", "ShortConvKernel", "ShortConvInProj", "ShortConvOutProj"]:
        if key in spec:
            fmt = spec[key]
            name = fmt % 0 if "%d" in fmt else fmt
            if not has(name):
                missing.append((key, name))
    return missing


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="/work/models/raw")
    args = ap.parse_args()

    spec_text = load_spec_files(["internal/model/modelspec.go", "internal/model/afmoe.go"])

    root = Path(args.root)
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for d in dirs:
        cfg_path = d / "config.json"
        st_path = d / "model.safetensors"
        if not cfg_path.exists() or not st_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        spec_func = detect_spec(cfg)
        if spec_func is None:
            print(f"{d.name}: no spec mapping for model_type={cfg.get('model_type')}")
            continue
        body = extract_spec_body(spec_text, spec_func)
        if body is None:
            print(f"{d.name}: spec func {spec_func} not found")
            continue
        arch = parse_arch_names(body)
        header = read_safetensors_header(st_path)
        names = set(k for k in header.keys() if k != "__metadata__")
        missing = check_presence(names, arch)
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
