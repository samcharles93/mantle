# Mantle CLI Quickstart

## mantle run - Run inference for LLM models

Basic usage:
```bash
bin/mantle run -m <model.mcf> -p "prompt"
```

Options:

| Flag | Alias | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | required | Path to .mcf file |
| `--prompt` | `-p` | "" | Prompt text to generate |
| `--system` | `-sys` | "" | System prompt |
| `--steps` | `-n`, `--num-tokens` | -1 | Number of tokens to generate (-1 = infinite) |
| `--temp` | `-t` | 0.8 | Sampling temperature (higher = more creative) |
| `--top-k` | | 40 | Top-K sampling cutoff |
| `--top-p` | | 0.95 | Top-P (nucleus) sampling threshold |
| `--min-p` | | 0.05 | Min-P sampling (minimum probability threshold) |
| `--repeat-penalty` | | 1.1 | Penalty for repeating tokens |
| `--repeat-last-n` | | 64 | Lookback window for repeat penalty |
| `--seed` | | random | RNG seed for reproducibility |
| `--max-context` | `-c`, `--ctx` | 4096 | Maximum context length |
| `--no-template` | | false | Disable chat template rendering |
| `--echo-prompt` | | false | Print prompt before generation |
| `--show-config` | | true | Print model + sampler config |
| `--show-tokens` | | true | Print input token IDs |
| `--cpuprofile` | | "" | Write CPU profile to file |
| `--memprofile` | | "" | Write memory profile to file |

RoPE scaling options (for extended context):
| Flag | Description |
|------|-------------|
| `--rope-scaling` | Type: `linear`, `yarn`, `none` |
| `--rope-scale` | Scaling factor |
| `--rope-freq-base` | Base frequency override |
| `--yarn-orig-ctx` | YaRN original context size |

---

## mantle pack - Pack Safetensors model into MCF

```bash
bin/mantle pack --input <model_dir> --output <model.mcf>
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `--in` | required | Model directory containing config/tokenizer |
| `--output`, `--out` | auto | Output .mcf path |
| `--dedup` | true | Deduplicate identical tensors |
| `--model-safetensors` | auto | Override safetensors filename |
| `--tensor-align` | 64 | Alignment between tensors (0 to disable) |
| `--cast` | `keep` | Float casting: `keep`, `f16`, `bf16` |
| `--no-resources` | false | Don't embed config/tokenizer/vocab sections |
| `--progress-every` | 50 | Log progress every N tensors |

Resource overrides:
| Flag | Description |
|------|-------------|
| `--config-json` | Override config.json path |
| `--generation-config-json` | Override generation_config.json path |
| `--tokenizer-json` | Override tokenizer.json path |
| `--tokenizer-config-json` | Override tokenizer_config.json path |

---

## mantle quantize - Quantize existing MCF

```bash
bin/mantle quantize --input <model.mcf> --output <model.q4.mcf> --method k4
```

Options:

| Flag | Alias | Default | Description |
|------|-------|---------|-------------|
| `--input` | `-in` | required | Input .mcf path |
| `--output` | `-out` | auto | Output .mcf path |
| `--method` | | `k4` | Quantization method |

**Quantization methods:**
- `k6` - 6-bit K-quants (block size 32, superblock 256)
- `k4` - 4-bit K-quants (default)
- `k3` - 3-bit K-quants
- `k2` - 2-bit K-quants
- `q8` - 8-bit Q-quants (simple per-block)
- `q4` - 4-bit Q-quants

Optional overrides:
| Flag | Description |
|------|-------------|
| `--min-clip` | Min clipping value (must set with `--max-clip`) |
| `--max-clip` | Max clipping value (must set with `--min-clip`) |