#!/usr/bin/env python3
import json
import struct
import sys
import re
from pathlib import Path
from collections import defaultdict

# -----------------------------------------------------------------------------
# Mapping definitions
# -----------------------------------------------------------------------------

# Heuristics to map suffixes to ArchNames fields
# We look for these substrings (or exact matches) in the layer-level suffixes.
LAYER_MAPPINGS = {
    "Wq": ["q_proj", "self_attn.q_proj", "query.weight"],
    "Wk": ["k_proj", "self_attn.k_proj", "key.weight"],
    "Wv": ["v_proj", "self_attn.v_proj", "value.weight"],
    "Wo": ["o_proj", "self_attn.o_proj", "self_attn.out_proj", "dense.weight"],
    "WqBias": ["q_proj.bias", "self_attn.q_proj.bias"],
    "WkBias": ["k_proj.bias", "self_attn.k_proj.bias"],
    "WvBias": ["v_proj.bias", "self_attn.v_proj.bias"],
    "AttnGate": ["gate_proj", "self_attn.gate_proj"], # Some models have this?
    
    "FfnUp": ["up_proj", "w3", "mlp.up_proj", "feed_forward.w3"],
    "FfnGate": ["gate_proj", "w1", "mlp.gate_proj", "feed_forward.w1"],
    "FfnDown": ["down_proj", "w2", "mlp.down_proj", "feed_forward.w2"],
    
    "AttnNorm": ["input_layernorm", "attn_norm", "ln_1", "operator_norm"],
    "FfnNorm": ["post_attention_layernorm", "ffn_norm", "ln_2", "pre_mlp_layernorm", "pre_feedforward_layernorm"],
    "PostAttnNorm": ["post_attention_layernorm", "post_attn_norm", "post_attention_norm"],
    "PostFfnNorm": ["post_feedforward_layernorm", "post_mlp_layernorm", "post_ffn_norm"],
    
    # MoE
    "MoERouter": ["router.gate", "gate.weight"],
    "MoEExpertBias": ["expert_bias"],
    
    # QK Norms
    "QNorm": ["q_norm", "q_layernorm"],
    "KNorm": ["k_norm", "k_layernorm"],
    
    # ShortConv (LFM/Mamba)
    "ShortConvKernel": ["conv.conv.weight"],
    "ShortConvInProj": ["conv.in_proj.weight"],
    "ShortConvOutProj": ["conv.out_proj.weight"],
    
    # Mamba
    "MambaInProj": ["mamba.in_proj.weight"],
    "MambaOutProj": ["mamba.out_proj.weight"],
    "MambaConv": ["mamba.conv1d.weight"],
    "MambaConvBias": ["mamba.conv1d.bias"],
    "MambaALog": ["mamba.A_log"],
    "MambaD": ["mamba.D"],
    "MambaDTBias": ["mamba.dt_bias"],
}
    
# Heuristics for global tensors
GLOBAL_MAPPINGS = {
    "Embedding": ["embed_tokens", "wte", "word_embeddings"],
    "OutputNorm": ["norm.weight", "ln_f.weight", "final_layernorm", "embedding_norm"],
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def read_safetensors_header(path: Path):
    with path.open("rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            return {}
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_json = f.read(header_len)
        return json.loads(header_json)

def collect_tensor_names(root: Path):
    names = set()
    
    # Check for index first (sharded)
    index_path = root / "model.safetensors.index.json"
    safetensors_files = sorted(list(root.glob("*.safetensors")))
    
    if not safetensors_files:
        raise SystemExit(f"No .safetensors files found in {root}")

    print(f"// Found {len(safetensors_files)} safetensors files...", file=sys.stderr)

    for st_path in safetensors_files:
        try:
            header = read_safetensors_header(st_path)
            for k in header.keys():
                if k != "__metadata__":
                    names.add(k)
        except Exception as e:
            print(f"// Error reading {st_path}: {e}", file=sys.stderr)
            
    return sorted(list(names))

def find_layer_pattern(names):
    # We look for patterns like "model.layers.0.foo"
    # Regex: Start, capture prefix, capture number, capture suffix
    # We want the prefix that covers the most keys and implies a sequence of layers.
    
    # (prefix, number, suffix)
    candidates = []
    regex = re.compile(r"^(.*?)(\d+)\.(.*)$")
    
    for n in names:
        m = regex.match(n)
        if m:
            candidates.append(m.groups())
            
    if not candidates:
        return None, None
        
    # Count occurrences of prefixes
    counts = defaultdict(set) # prefix -> set(layer_nums)
    samples = {} # prefix -> sample suffix
    
    for prefix, num, suffix in candidates:
        counts[prefix].add(int(num))
        samples[prefix] = suffix
        
    # Pick the prefix with the most layers (max(layer_num)) or most distinct layers
    best_prefix = None
    max_layers = -1
    
    for prefix, layers in counts.items():
        if len(layers) > max_layers:
            max_layers = len(layers)
            best_prefix = prefix
            
    if best_prefix is None:
        return None, None
        
    print(f"// Detected layer prefix: '{best_prefix}' with {max_layers} layers.", file=sys.stderr)
    return best_prefix, max_layers

def to_go_func_name(name):
    # simple camelCase
    parts = re.split(r'[_\-.]', name)
    return parts[0].lower() + "".join(x.title() for x in parts[1:]) + "Spec"

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate a Mantle model spec snippet from a safetensors directory")
    ap.add_argument("dir", help="Path to model directory containing config.json + *.safetensors")
    args = ap.parse_args()

    root = Path(args.dir)
    cfg_path = root / "config.json"
    
    if not cfg_path.exists():
        raise SystemExit("config.json missing")

    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception as e:
        raise SystemExit(f"Failed to parse config.json: {e}")

    model_type = cfg.get("model_type", "unknown")
    archs = cfg.get("architectures", [])
    layer_types = cfg.get("layer_types")
    
    names = collect_tensor_names(root)
    
    print("// === Generated model spec ===")
    print(f"// Source: {root}")
    print(f"// Model Type: {model_type}")
    print(f"// Architectures: {archs}")
    
    # 1. Identify Layer Pattern
    prefix, num_layers = find_layer_pattern(names)
    
    found_layer_map = {} # GoField -> Suffix
    found_global_map = {} # GoField -> FullName
    
    # 2. Analyze Layer Tensors
    if prefix:
        # Get all suffixes that exist for layer 0 (or layer 1 if 0 missing, but usually 0)
        layer_0_suffixes = []
        regex = re.compile(fr"^{re.escape(prefix)}0\.(.*)$")
        for n in names:
            m = regex.match(n)
            if m:
                layer_0_suffixes.append(m.group(1))
        
        # Match suffixes to fields
        for field, keywords in LAYER_MAPPINGS.items():
            for s in layer_0_suffixes:
                # Check for exact match or substring match
                # Prefer exact match if possible, or matches ending with keyword
                matched = False
                for k in keywords:
                    if k == s or s.endswith("." + k) or s == k + ".weight":
                        # Be reasonably strict to avoid false positives
                        found_layer_map[field] = s
                        matched = True
                        break
                    # fuzzy match for things like "q_proj" inside "self_attn.q_proj.weight"
                    if k in s:
                        found_layer_map[field] = s
                        matched = True
                        break
                if matched:
                    pass

    # Refine matching: Inverse search
    # For each desired field, search through available suffixes
    final_layer_map = {}
    if prefix:
        # Gather all layer 0 names again
        l0_names = [n for n in names if n.startswith(f"{prefix}0.")]
        
        for field, candidates in LAYER_MAPPINGS.items():
            best_match = None
            for n in l0_names:
                suffix = n[len(prefix)+2:] # remove "prefix" + "0."
                
                for c in candidates:
                    # Exact match of keyword to suffix part (ignoring .weight)
                    stem = suffix.replace(".weight", "")
                    if stem == c or suffix == c:
                        best_match = suffix
                        break
                    if c in suffix:
                        best_match = suffix
                        # Keep looking for better matches?
                        
                if best_match: break
            if best_match:
                final_layer_map[field] = best_match

    # 3. Analyze Global Tensors
    # Remove layer tensors from consideration roughly
    global_candidates = [n for n in names if not (prefix and n.startswith(prefix))]
    
    for field, keywords in GLOBAL_MAPPINGS.items():
        for n in global_candidates:
            for k in keywords:
                if k in n:
                    found_global_map[field] = n
                    break
            if field in found_global_map: break

    # 4. Generate Code
    func_name = to_go_func_name(model_type)
    
    RED = "\033[91m"
    RESET = "\033[0m"
    
    print("")
    print(f"func {func_name}() *ArchSpec {{")
    print(f"\treturn &ArchSpec{{")
    print(f"\t\tName:           \"{model_type}\",")
    
    # Heuristics for flags
    has_qk_norm = "QNorm" in final_layer_map or "KNorm" in final_layer_map
    print(f"\t\tHasQKNorm:     {'true' if has_qk_norm else 'false'},")
    use_layer_types = bool(layer_types)
    print(f"\t\tUseLayerTypes: {'true' if use_layer_types else 'false'}, // Verify manually")
    print(f"\t\tRopeLocalOnly: false, // Verify manually")
    
    print(f"\t\tNames: ArchNames{{")
    
    # Global
    if "Embedding" in found_global_map:
        print(f"\t\t\tEmbedding:  \"{found_global_map['Embedding']}\",")
    else:
        print(f"\t\t\t// {RED}MISSING: Embedding not found (candidates: {GLOBAL_MAPPINGS['Embedding']}){RESET}")

    if "OutputNorm" in found_global_map:
        print(f"\t\t\tOutputNorm: \"{found_global_map['OutputNorm']}\",")
    else:
        print(f"\t\t\t// {RED}MISSING: OutputNorm not found{RESET}")

    # Output Candidates (Usually just lm_head or same as embedding if tied)
    output_candidates = []
    for n in names:
        if "lm_head" in n or n.endswith("output.weight") or n.endswith("output_proj.weight"):
            output_candidates.append(n)
    # Keep order stable but unique
    output_candidates = list(dict.fromkeys(output_candidates))
    print(f"\t\t\tOutputCandidates: func() []string {{")
    print(f"\t\t\t\treturn []string{{")
    for n in output_candidates:
        print(f"\t\t\t\t\t\"{n}\",")
    if "Embedding" in found_global_map:
        print(f"\t\t\t\t\t\"{found_global_map['Embedding']}\", // often tied")
    print(f"\t\t\t\t}}")
    print(f"\t\t\t}},")

    # Layer Tensors
    def print_layer_func(field, go_field_name=None):
        fname = go_field_name or field
        if field in final_layer_map:
            fmt_str = f"{prefix}%d.{final_layer_map[field]}"
            print(f"\t\t\t{fname}: func(layer int) string {{")
            print(f"\t\t\t\treturn fmt.Sprintf(\"{fmt_str}\", layer)")
            print(f"\t\t\t}},")
        else:
            # Only print RED for critical fields
            if field in ["Wq", "Wk", "Wv", "Wo", "FfnUp", "FfnGate", "FfnDown", "AttnNorm", "FfnNorm"]:
                print(f"\t\t\t// {RED}MISSING: {fname} not found{RESET}")
            else:
                    # Optional fields, just ignore
                    pass

    # Norm Candidates logic
    # We often have AttnNormCandidates which is a list.
    if "AttnNorm" in final_layer_map:
        fmt_str = f"{prefix}%d.{final_layer_map['AttnNorm']}"
        print(f"\t\t\tAttnNormCandidates: func(layer int) []string {{")
        print(f"\t\t\t\treturn []string{{")
        print(f"\t\t\t\t\tfmt.Sprintf(\"{fmt_str}\", layer),")
        print(f"\t\t\t\t}}")
        print(f"\t\t\t}},")
        # Also map the singleton for convenience if needed, though ArchSpec has Func(layer) string for single
        print_layer_func("AttnNorm")

    if "FfnNorm" in final_layer_map:
        fmt_str = f"{prefix}%d.{final_layer_map['FfnNorm']}"
        print(f"\t\t\tFfnNormCandidates: func(layer int) []string {{")
        print(f"\t\t\t\treturn []string{{")
        print(f"\t\t\t\t\tfmt.Sprintf(\"{fmt_str}\", layer),")
        print(f"\t\t\t\t}}")
        print(f"\t\t\t}},")
        print_layer_func("FfnNorm")

    if "PostAttnNorm" in final_layer_map:
        fmt_str = f"{prefix}%d.{final_layer_map['PostAttnNorm']}"
        print(f"\t\t\tPostAttnNormCandidates: func(layer int) []string {{")
        print(f"\t\t\t\treturn []string{{")
        print(f"\t\t\t\t\tfmt.Sprintf(\"{fmt_str}\", layer),")
        print(f"\t\t\t\t}}")
        print(f"\t\t\t}},")

    if "PostFfnNorm" in final_layer_map:
        fmt_str = f"{prefix}%d.{final_layer_map['PostFfnNorm']}"
        print(f"\t\t\tPostFfnNormCandidates: func(layer int) []string {{")
        print(f"\t\t\t\treturn []string{{")
        print(f"\t\t\t\t\tfmt.Sprintf(\"{fmt_str}\", layer),")
        print(f"\t\t\t\t}}")
        print(f"\t\t\t}},")

    # QK Norms
    if "QNorm" in final_layer_map:
        fmt_str = f"{prefix}%d.{final_layer_map['QNorm']}"
        print(f"\t\t\tQNormCandidates: func(layer int) []string {{")
        print(f"\t\t\t\treturn []string{{fmt.Sprintf(\"{fmt_str}\", layer)}}")
        print(f"\t\t\t}},")
        
    if "KNorm" in final_layer_map:
        fmt_str = f"{prefix}%d.{final_layer_map['KNorm']}"
        print(f"\t\t\tKNormCandidates: func(layer int) []string {{")
        print(f"\t\t\t\treturn []string{{fmt.Sprintf(\"{fmt_str}\", layer)}}")
        print(f"\t\t\t}},")

    # Projections
    print_layer_func("Wq")
    print_layer_func("Wk")
    print_layer_func("Wv")
    print_layer_func("Wo")
    print_layer_func("WqBias")
    print_layer_func("WkBias")
    print_layer_func("WvBias")
    
    print_layer_func("FfnUp")
    print_layer_func("FfnGate")
    print_layer_func("FfnDown")
    
    print_layer_func("MoERouter")
    print_layer_func("MoEExpertBias")
    
    # Mamba / ShortConv
    print_layer_func("ShortConvKernel")
    print_layer_func("ShortConvInProj")
    print_layer_func("ShortConvOutProj")
    
    print_layer_func("MambaInProj")
    print_layer_func("MambaOutProj")
    print_layer_func("MambaConv")
    print_layer_func("MambaConvBias")
    print_layer_func("MambaALog")
    print_layer_func("MambaD")
    print_layer_func("MambaDTBias")
    
    print(f"\t\t}},")
    print(f"\t}}")
    print(f"}}")

    # 5. Scan for Unmapped Tensors
    # This helps identifying missing features/biases/rotary embeddings etc.
    
    mapped_suffixes = set(final_layer_map.values())
    mapped_globals = set(found_global_map.values())
    mapped_globals.update(output_candidates)
    
    unmapped_global = []
    unmapped_layer_0 = []
    
    for n in names:
        if prefix and n.startswith(f"{prefix}0."):
            suffix = n[len(prefix)+2:] # remove "model.layers.0."
            if suffix not in mapped_suffixes:
                # double check if we mapped it but logic above missed adding to set? no
                unmapped_layer_0.append(suffix)
        elif prefix and n.startswith(prefix):
                # Other layers, assume same pattern
                pass
        else:
                # Global
                if n not in mapped_globals:
                    unmapped_global.append(n)
                    
    if unmapped_global or unmapped_layer_0:
        print(f"\n// {RED}--- UNMAPPED TENSORS WARNING ---{RESET}")
        print("// These tensors exist in the model but were not mapped to any ArchSpec field.")
        print("// Please check if they are important (e.g. biases, scales, rotary embeddings).")
        
        if unmapped_global:
            print("\n// [Global Unmapped]")
            for n in unmapped_global[:20]:
                print(f"// {n}")
            if len(unmapped_global) > 20: print(f"// ... and {len(unmapped_global)-20} more")
            
        if unmapped_layer_0:
            print("\n// [Layer 0 Unmapped]")
            for n in unmapped_layer_0[:20]:
                print(f"// {n}")
            if len(unmapped_layer_0) > 20: print(f"// ... and {len(unmapped_layer_0)-20} more")

if __name__ == "__main__":
    main()
