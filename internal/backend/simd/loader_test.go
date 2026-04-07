package simd

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/samcharles93/mantle/internal/model"
	"github.com/samcharles93/mantle/pkg/mcf"
)

type stubTensorSource struct {
	tensors map[string]tensorPayload
}

func (s stubTensorSource) ReadTensor(name string) (tensorPayload, error) {
	p, ok := s.tensors[name]
	if !ok {
		return tensorPayload{}, errors.New("missing tensor")
	}
	return p, nil
}

func (s stubTensorSource) TensorShape(name string) ([]int, bool) {
	p, ok := s.tensors[name]
	if !ok {
		return nil, false
	}
	shape := make([]int, len(p.Shape))
	copy(shape, p.Shape)
	return shape, true
}

func f32Raw(vals ...float32) []byte {
	raw := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	return raw
}

func TestLoadAttentionQAndGateSplitsFusedQProj(t *testing.T) {
	src := stubTensorSource{
		tensors: map[string]tensorPayload{
			"q_proj": {
				DType: mcf.DTypeF32,
				Shape: []int{8, 2},
				Raw: f32Raw(
					1, 2,
					3, 4,
					5, 6,
					7, 8,
					9, 10,
					11, 12,
					13, 14,
					15, 16,
				),
			},
		},
	}

	wq, gate, err := loadAttentionQAndGate(src, &model.HFConfig{AttnOutputGate: true}, "q_proj", 4, 2, 2)
	if err != nil {
		t.Fatalf("loadAttentionQAndGate: %v", err)
	}
	if gate == nil {
		t.Fatalf("expected fused attention gate to be split out")
	} else if gate.R != 4 || gate.C != 2 {
		t.Fatalf("gate shape=%dx%d want 4x2", gate.R, gate.C)
	}
	if wq.R != 4 || wq.C != 2 {
		t.Fatalf("wq shape=%dx%d want 4x2", wq.R, wq.C)
	}
	if got := wq.Data; !equalFloat32s(got, []float32{1, 2, 3, 4, 9, 10, 11, 12}) {
		t.Fatalf("wq data=%v", got)
	}
	if got := gate.Data; !equalFloat32s(got, []float32{5, 6, 7, 8, 13, 14, 15, 16}) {
		t.Fatalf("gate data=%v", got)
	}
}

func TestBuildLayerLoadConfigsGemma4MixedAttention(t *testing.T) {
	t.Parallel()

	cfg, spec := loadArchForTest(t, `{
		"model_type": "gemma4",
		"architectures": ["Gemma4ForConditionalGeneration"],
		"num_hidden_layers": 2,
		"num_attention_heads": 8,
		"num_key_value_heads": 4,
		"head_dim": 256,
		"global_head_dim": 512,
		"sliding_window": 1024,
		"layer_types": ["sliding_attention", "full_attention"],
		"rope_parameters": {
			"sliding_attention": {"rope_type": "default", "rope_theta": 10000},
			"full_attention": {"rope_type": "proportional", "rope_theta": 1000000, "partial_rotary_factor": 0.5}
		}
	}`)

	cfgs, layerTypes, err := buildLayerLoadConfigs(cfg, spec, 2, cfg.HeadDim)
	if err != nil {
		t.Fatalf("buildLayerLoadConfigs: %v", err)
	}
	if got, want := layerTypes, []string{"sliding_attention", "full_attention"}; !equalStrings(got, want) {
		t.Fatalf("layer types=%v want %v", got, want)
	}
	if cfgs[0].AttnType != "sliding_attention" || cfgs[1].AttnType != "full_attention" {
		t.Fatalf("attention types=%q/%q", cfgs[0].AttnType, cfgs[1].AttnType)
	}
	if cfgs[0].HeadDim != 256 {
		t.Fatalf("sliding head_dim=%d want 256", cfgs[0].HeadDim)
	}
	if cfgs[1].HeadDim != 512 {
		t.Fatalf("full head_dim=%d want 512", cfgs[1].HeadDim)
	}
	if cfgs[0].AttnWindow != 1024 {
		t.Fatalf("sliding window=%d want 1024", cfgs[0].AttnWindow)
	}
	if cfgs[0].AttnScale != 1 || cfgs[1].AttnScale != 1 {
		t.Fatalf("gemma4 attention scales=%v/%v want 1", cfgs[0].AttnScale, cfgs[1].AttnScale)
	}
	if !cfgs[0].ApplyVNorm || !cfgs[1].ApplyVNorm {
		t.Fatalf("expected Gemma4 V norm on both attention types")
	}
	if len(cfgs[1].RopeInvFreq) != 256 {
		t.Fatalf("full rope len=%d want 256", len(cfgs[1].RopeInvFreq))
	}
	if cfgs[1].RopeInvFreq[0] == 0 || cfgs[1].RopeInvFreq[127] == 0 {
		t.Fatalf("expected populated proportional rope prefix, got %v", cfgs[1].RopeInvFreq[:128])
	}
	for i := 128; i < len(cfgs[1].RopeInvFreq); i++ {
		if cfgs[1].RopeInvFreq[i] != 0 {
			t.Fatalf("expected zero tail in proportional rope at %d, got %v", i, cfgs[1].RopeInvFreq[i])
		}
	}
}

func TestBuildLayerLoadConfigsGemma4SharedKVAndValueFromKey(t *testing.T) {
	t.Parallel()

	cfg, spec := loadArchForTest(t, `{
		"model_type": "gemma4",
		"architectures": ["Gemma4ForConditionalGeneration"],
		"num_hidden_layers": 4,
		"num_attention_heads": 8,
		"num_key_value_heads": 4,
		"num_global_key_value_heads": 2,
		"head_dim": 256,
		"global_head_dim": 512,
		"attention_k_eq_v": true,
		"num_kv_shared_layers": 2,
		"layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
		"rope_parameters": {
			"sliding_attention": {"rope_type": "default", "rope_theta": 10000},
			"full_attention": {"rope_type": "proportional", "rope_theta": 1000000, "partial_rotary_factor": 0.5}
		}
	}`)

	cfgs, _, err := buildLayerLoadConfigs(cfg, spec, 4, cfg.HeadDim)
	if err != nil {
		t.Fatalf("buildLayerLoadConfigs: %v", err)
	}
	if cfgs[0].SharedKVSource != -1 || cfgs[1].SharedKVSource != -1 {
		t.Fatalf("unexpected shared KV sources for seed layers: %d %d", cfgs[0].SharedKVSource, cfgs[1].SharedKVSource)
	}
	if !cfgs[0].StoreFullKV || !cfgs[1].StoreFullKV {
		t.Fatalf("expected last non-shared layers to retain full KV")
	}
	if cfgs[2].SharedKVSource != 0 {
		t.Fatalf("sliding shared source=%d want 0", cfgs[2].SharedKVSource)
	}
	if cfgs[3].SharedKVSource != 1 {
		t.Fatalf("full shared source=%d want 1", cfgs[3].SharedKVSource)
	}
	if cfgs[1].HeadKV != 2 || cfgs[3].HeadKV != 2 {
		t.Fatalf("full HeadKV=%d/%d want 2", cfgs[1].HeadKV, cfgs[3].HeadKV)
	}
	if !cfgs[1].ValueFromKey || !cfgs[3].ValueFromKey {
		t.Fatalf("expected full Gemma4 layers to derive V from K")
	}
	if cfgs[0].ValueFromKey || cfgs[2].ValueFromKey {
		t.Fatalf("sliding Gemma4 layers must keep explicit V projections")
	}
}

func TestBuildLayerLoadConfigsGemma3nMixedAttentionAndSharedKV(t *testing.T) {
	t.Parallel()

	cfg, spec := loadArchForTest(t, `{
		"model_type": "gemma3n",
		"architectures": ["Gemma3nForConditionalGeneration"],
		"text_config": {
			"model_type": "gemma3n_text",
			"num_hidden_layers": 4,
			"num_attention_heads": 8,
			"num_key_value_heads": 2,
			"head_dim": 256,
			"sliding_window": 512,
			"num_kv_shared_layers": 2,
			"layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
			"rope_theta": 1000000.0,
			"rope_local_base_freq": 10000.0
		}
	}`)

	cfgs, layerTypes, err := buildLayerLoadConfigs(cfg, spec, 4, cfg.HeadDim)
	if err != nil {
		t.Fatalf("buildLayerLoadConfigs: %v", err)
	}
	if got, want := layerTypes, []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}; !equalStrings(got, want) {
		t.Fatalf("layer types=%v want %v", got, want)
	}
	if !cfgs[0].ApplyVNorm || !cfgs[1].ApplyVNorm {
		t.Fatalf("expected Gemma3n V norm on attention layers")
	}
	if cfgs[0].AttnScale != 1 || cfgs[1].AttnScale != 1 {
		t.Fatalf("gemma3n attention scales=%v/%v want 1", cfgs[0].AttnScale, cfgs[1].AttnScale)
	}
	if cfgs[0].AttnWindow != 512 {
		t.Fatalf("sliding window=%d want 512", cfgs[0].AttnWindow)
	}
	if cfgs[0].SharedKVSource != -1 || cfgs[1].SharedKVSource != -1 {
		t.Fatalf("unexpected seed shared KV sources: %d %d", cfgs[0].SharedKVSource, cfgs[1].SharedKVSource)
	}
	if !cfgs[0].StoreFullKV || !cfgs[1].StoreFullKV {
		t.Fatalf("expected last non-shared Gemma3n layers to retain full KV")
	}
	if cfgs[2].SharedKVSource != 0 || cfgs[3].SharedKVSource != 1 {
		t.Fatalf("shared KV sources=%d/%d want 0/1", cfgs[2].SharedKVSource, cfgs[3].SharedKVSource)
	}
	if cfgs[0].ValueFromKey || cfgs[1].ValueFromKey || cfgs[2].ValueFromKey || cfgs[3].ValueFromKey {
		t.Fatalf("Gemma3n must keep explicit V projections")
	}
	if len(cfgs[0].RopeInvFreq) != 128 || len(cfgs[1].RopeInvFreq) != 128 {
		t.Fatalf("rope lens=%d/%d want 128/128", len(cfgs[0].RopeInvFreq), len(cfgs[1].RopeInvFreq))
	}
	wantSliding := 1 / math.Pow(10000, 2.0/256.0)
	wantFull := 1 / math.Pow(1_000_000, 2.0/256.0)
	if math.Abs(cfgs[0].RopeInvFreq[1]-wantSliding) > 1e-12 {
		t.Fatalf("sliding rope invFreq[1]=%g want %g", cfgs[0].RopeInvFreq[1], wantSliding)
	}
	if math.Abs(cfgs[1].RopeInvFreq[1]-wantFull) > 1e-12 {
		t.Fatalf("full rope invFreq[1]=%g want %g", cfgs[1].RopeInvFreq[1], wantFull)
	}
}

func TestBuildLayerLoadConfigsLFM2ConvLayersAreRecurrent(t *testing.T) {
	t.Parallel()

	cfg, spec := loadArchForTest(t, `{
		"model_type": "lfm2",
		"architectures": ["Lfm2ForCausalLM"],
		"hidden_size": 1024,
		"intermediate_size": 6656,
		"num_hidden_layers": 4,
		"num_attention_heads": 16,
		"num_key_value_heads": 8,
		"layer_types": ["conv", "conv", "full_attention", "conv"]
	}`)

	cfgs, layerTypes, err := buildLayerLoadConfigs(cfg, spec, 4, 64)
	if err != nil {
		t.Fatalf("buildLayerLoadConfigs: %v", err)
	}
	if got, want := layerTypes, []string{"conv", "conv", "full_attention", "conv"}; !equalStrings(got, want) {
		t.Fatalf("layer types=%v want %v", got, want)
	}
	for _, idx := range []int{0, 1, 3} {
		if cfgs[idx].HeadKV != 0 {
			t.Fatalf("conv layer %d HeadKV=%d want 0", idx, cfgs[idx].HeadKV)
		}
		if cfgs[idx].AttnScale != 0 {
			t.Fatalf("conv layer %d AttnScale=%v want 0", idx, cfgs[idx].AttnScale)
		}
	}
	if cfgs[2].HeadKV != 8 {
		t.Fatalf("attention layer HeadKV=%d want 8", cfgs[2].HeadKV)
	}
	if cfgs[2].AttnScale == 0 {
		t.Fatalf("attention layer AttnScale must be non-zero")
	}
}

func loadArchForTest(t *testing.T, raw string) (*model.HFConfig, *model.ArchSpec) {
	t.Helper()

	cfg, err := model.LoadHFConfigBytes([]byte(raw))
	if err != nil {
		t.Fatalf("LoadHFConfigBytes: %v", err)
	}
	spec, err := model.DetectArch(cfg)
	if err != nil {
		t.Fatalf("DetectArch: %v", err)
	}
	return cfg, spec
}

func equalFloat32s(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
