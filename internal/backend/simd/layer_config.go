package simd

import (
	"fmt"
	"math"
	"strings"

	"github.com/samcharles93/mantle/internal/model"
)

type layerLoadConfig struct {
	AttnType       string
	AttnWindow     int
	HeadKV         int
	HeadDim        int
	AttnScale      float32
	ValueFromKey   bool
	ApplyVNorm     bool
	SharedKVSource int
	StoreFullKV    bool
	RopeInvFreq    []float64
	RopeAttnScale  float32
}

func buildLayerLoadConfigs(cfg *model.HFConfig, spec *model.ArchSpec, blockCount, defaultHeadDim int) ([]layerLoadConfig, []string, error) {
	layerTypes := deriveLayerTypes(cfg, blockCount)
	headCount := cfg.NumAttentionHeads
	if headCount <= 0 {
		return nil, nil, fmt.Errorf("num_attention_heads must be set")
	}
	defaultKVHeads := cfg.NumKeyValueHeads
	if defaultKVHeads <= 0 {
		defaultKVHeads = headCount
	}

	out := make([]layerLoadConfig, blockCount)
	for i := range blockCount {
		lc := layerLoadConfig{
			AttnType:       "full_attention",
			HeadKV:         defaultKVHeads,
			HeadDim:        defaultHeadDim,
			AttnScale:      float32(1.0 / math.Sqrt(float64(defaultHeadDim))),
			SharedKVSource: -1,
		}
		if len(layerTypes) == blockCount && layerTypes[i] != "" {
			lc.AttnType = layerTypes[i]
		}
		if lc.AttnType == "sliding_attention" && cfg.SlidingWindow > 0 {
			lc.AttnWindow = cfg.SlidingWindow
		}
		if spec.UseLayerTypes && isNonAttentionLayerType(lc.AttnType) {
			lc.HeadKV = 0
			lc.AttnScale = 0
			out[i] = lc
			continue
		}
		if err := applyGemmaAttentionConfig(&lc, cfg, spec, i); err != nil {
			return nil, nil, err
		}
		out[i] = lc
	}
	if err := applyGemmaSharedKVConfig(out, layerTypes, cfg, spec); err != nil {
		return nil, nil, err
	}

	return out, layerTypes, nil
}

func lastIndexOfLayerType(layerTypes []string, want string) int {
	for i := len(layerTypes) - 1; i >= 0; i-- {
		if layerTypes[i] == want {
			return i
		}
	}
	return -1
}

func deriveLayerTypes(cfg *model.HFConfig, blockCount int) []string {
	if len(cfg.LayerTypes) == blockCount {
		layerTypes := make([]string, blockCount)
		copy(layerTypes, cfg.LayerTypes)
		return layerTypes
	}
	if len(cfg.SlidingWindowPattern.Pattern) == blockCount && cfg.SlidingWindow > 0 {
		layerTypes := make([]string, blockCount)
		for i := range cfg.SlidingWindowPattern.Pattern {
			if cfg.SlidingWindowPattern.Pattern[i] {
				layerTypes[i] = "sliding_attention"
			} else {
				layerTypes[i] = "full_attention"
			}
		}
		return layerTypes
	}
	if cfg.GlobalAttnEveryN > 0 && cfg.SlidingWindow > 0 {
		layerTypes := make([]string, blockCount)
		for i := range blockCount {
			if (i+1)%cfg.GlobalAttnEveryN == 0 {
				layerTypes[i] = "full_attention"
			} else {
				layerTypes[i] = "sliding_attention"
			}
		}
		return layerTypes
	}
	return nil
}

func isNonAttentionLayerType(layerType string) bool {
	switch strings.ToLower(strings.TrimSpace(layerType)) {
	case "", "attention", "full_attention", "sliding_attention":
		return false
	case "linear_attention", "conv":
		return true
	default:
		normalized := strings.ToLower(strings.TrimSpace(layerType))
		return strings.Contains(normalized, "linear") || strings.Contains(normalized, "conv")
	}
}
