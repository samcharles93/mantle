package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/samcharles93/mantle/internal/gguf"
	"github.com/samcharles93/mantle/internal/model"
)

func main() {
	var (
		showKV      = flag.Bool("kv", false, "show all metadata key/values")
		showTensors = flag.Int("tensors", 20, "number of tensors to list (0 to skip, -1 for all)")
		showLayers  = flag.Bool("layers", true, "show per-layer tensor summary (LFM2 only)")
	)
	flag.Parse()

	if flag.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: gguf_inspect [--kv] [--tensors N] <path.gguf>")
		os.Exit(2)
	}

	path := flag.Arg(0)
	f, err := gguf.Open(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}

	fmt.Printf("File: %s\n", path)
	fmt.Printf("GGUF v%d | tensors=%d | kv=%d | alignment=%d | data_offset=%d\n",
		f.Header.Version, f.Header.TensorCount, f.Header.KVCount, f.Alignment, f.DataOffset)

	printKey(f, "general.name")
	printKey(f, "general.architecture")
	printKey(f, "general.quantization")
	printKey(f, "general.file_type")
	printKey(f, "general.alignment")
	printKey(f, "general.context_length")
	printKey(f, "general.version")
	printKey(f, "tokenizer.ggml.model")
	printKey(f, "tokenizer.ggml.bos_token_id")
	printKey(f, "tokenizer.ggml.eos_token_id")
	printKey(f, "tokenizer.ggml.pad_token_id")
	printKey(f, "tokenizer.ggml.unk_token_id")

	if cfg, err := loadLFM2Config(f); err == nil {
		fmt.Println()
		fmt.Println("LFM2 params:")
		fmt.Printf("  blocks:          %d\n", cfg.Config.BlockCount)
		fmt.Printf("  embd:            %d\n", cfg.Config.EmbeddingLength)
		fmt.Printf("  ffn:             %d\n", cfg.Config.FFNLength)
		fmt.Printf("  heads:           %d\n", cfg.Config.HeadCount)
		if len(cfg.Config.HeadCountKV) > 0 {
			fmt.Printf("  kv_heads:        %v\n", cfg.Config.HeadCountKV)
		}
		fmt.Printf("  rms_eps:         %g\n", cfg.Config.RMSEpsilon)
		fmt.Printf("  rope_freq_base:  %g\n", cfg.Config.RopeFreqBase)
		fmt.Printf("  ctx_len:         %d\n", cfg.Config.ContextLength)
		fmt.Printf("  vocab:           %d\n", cfg.Config.VocabSize)
		fmt.Printf("  shortconv_l:     %d\n", cfg.Config.ShortConvLCache)
		fmt.Printf("  tok.model:       %s\n", cfg.Tokenizer.Model)
		fmt.Printf("  tok.pre:         %s\n", cfg.Tokenizer.Pre)
		fmt.Printf("  tok.bos:         %d\n", cfg.Tokenizer.BOSTokenID)
		fmt.Printf("  tok.eos:         %d\n", cfg.Tokenizer.EOSTokenID)
		fmt.Printf("  tok.add_bos:     %v\n", cfg.Tokenizer.AddBOS)
		fmt.Printf("  tok.add_eos:     %v\n", cfg.Tokenizer.AddEOS)
		fmt.Printf("  tok.vocab_len:   %d\n", len(cfg.Tokenizer.Tokens))
		fmt.Printf("  tok.merges_len:  %d\n", len(cfg.Tokenizer.Merges))

		if *showLayers {
			printLFM2Layers(cfg, f)
		}
	} else {
		fmt.Println()
		fmt.Println("Model params:")
		printKey(f, "llama.block_count")
		printKey(f, "llama.embedding_length")
		printKey(f, "llama.attention.head_count")
		printKey(f, "llama.attention.head_count_kv")
		printKey(f, "llama.attention.layer_norm_rms_epsilon")
		printKey(f, "llama.rope.freq_base")
		printKey(f, "llama.rope.freq_scale")
		printKey(f, "llama.context_length")
		printKey(f, "llama.vocab_size")
	}

	if *showKV {
		fmt.Println()
		fmt.Println("All metadata:")
		keys := make([]string, 0, len(f.KV))
		for k := range f.KV {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Printf("  %s = %s\n", k, formatValue(f.KV[k]))
		}
	}

	n := *showTensors
	if n != 0 {
		fmt.Println()
		fmt.Println("Tensors:")
		count := len(f.Tensors)
		if n < 0 || n > count {
			n = count
		}
		for i := 0; i < n; i++ {
			t := f.Tensors[i]
			fmt.Printf("  %-40s %-6s dims=%s off=%d\n",
				t.Name, t.Type.String(), formatDims(t.Dims), t.Offset)
		}
		if n < count {
			fmt.Printf("  ... (%d more)\n", count-n)
		}
	}
}

func printKey(f *gguf.File, key string) {
	if v, ok := f.KV[key]; ok {
		fmt.Printf("  %-36s %s\n", key+":", formatValue(v))
	}
}

func formatDims(dims []uint64) string {
	if len(dims) == 0 {
		return "[]"
	}
	parts := make([]string, len(dims))
	for i, v := range dims {
		parts[i] = fmt.Sprintf("%d", v)
	}
	return "[" + strings.Join(parts, "x") + "]"
}

func formatValue(v gguf.Value) string {
	switch val := v.Value.(type) {
	case string:
		return val
	case bool:
		if val {
			return "true"
		}
		return "false"
	case gguf.ArrayValue:
		return fmt.Sprintf("array(%s) len=%d", val.ElemType.String(), len(val.Values))
	default:
		return fmt.Sprintf("%v", val)
	}
}

type layerInfo struct {
	hasShortConv bool
	hasAttnQ     bool
	hasAttnK     bool
	hasAttnV     bool
	hasAttnOut   bool
	qDims        []uint64
	kDims        []uint64
	vDims        []uint64
	outDims      []uint64
}

func printLFM2Layers(cfg *model.ModelConfig, f *gguf.File) {
	layers := make([]layerInfo, cfg.Config.BlockCount)
	for _, t := range f.Tensors {
		idx, suffix, ok := parseLayerName(t.Name)
		if !ok || idx < 0 || idx >= len(layers) {
			continue
		}
		info := &layers[idx]
		switch {
		case strings.HasPrefix(suffix, "shortconv."):
			info.hasShortConv = true
		case suffix == "attn_q.weight":
			info.hasAttnQ = true
			info.qDims = t.Dims
		case suffix == "attn_k.weight":
			info.hasAttnK = true
			info.kDims = t.Dims
		case suffix == "attn_v.weight":
			info.hasAttnV = true
			info.vDims = t.Dims
		case suffix == "attn_output.weight":
			info.hasAttnOut = true
			info.outDims = t.Dims
		}
	}

	fmt.Println()
	fmt.Println("LFM2 layer summary:")
	headDim := 0
	if cfg.Config.HeadCount > 0 && cfg.Config.EmbeddingLength%cfg.Config.HeadCount == 0 {
		headDim = cfg.Config.EmbeddingLength / cfg.Config.HeadCount
	}
	for i, info := range layers {
		attn := info.hasAttnQ || info.hasAttnK || info.hasAttnV || info.hasAttnOut
		kvHeads := "-"
		if headDim > 0 && len(info.kDims) == 2 {
			kvHeads = fmt.Sprintf("%d", int(info.kDims[1])/headDim)
		}
		fmt.Printf("  layer %02d: attn=%v shortconv=%v kv_heads=%s\n", i, attn, info.hasShortConv, kvHeads)
	}
}

func parseLayerName(name string) (int, string, bool) {
	if !strings.HasPrefix(name, "blk.") {
		return 0, "", false
	}
	rest := strings.TrimPrefix(name, "blk.")
	parts := strings.SplitN(rest, ".", 2)
	if len(parts) != 2 {
		return 0, "", false
	}
	idx, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, "", false
	}
	return idx, parts[1], true
}
