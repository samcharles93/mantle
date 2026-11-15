package inference

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/samcharles93/mantle/internal/logger"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// textOnlyTestTensor is one tensor of the tiny synthetic text model.
type textOnlyTestTensor struct {
	Name  string
	Shape []uint64
	Vals  []float32
}

// textOnlySyntheticTensors returns a complete, tiny llama-shaped tensor set so
// the loader can reach the multimodal block with every text tensor present.
func textOnlySyntheticTensors() []textOnlyTestTensor {
	shapes := map[string][]uint64{
		"model.embed_tokens.weight":                      {16, 8},
		"model.norm.weight":                              {8},
		"lm_head.weight":                                 {16, 8},
		"model.layers.0.input_layernorm.weight":          {8},
		"model.layers.0.post_attention_layernorm.weight": {8},
		"model.layers.0.self_attn.q_proj.weight":         {8, 8},
		"model.layers.0.self_attn.k_proj.weight":         {8, 8},
		"model.layers.0.self_attn.v_proj.weight":         {8, 8},
		"model.layers.0.self_attn.o_proj.weight":         {8, 8},
		"model.layers.0.mlp.gate_proj.weight":            {16, 8},
		"model.layers.0.mlp.up_proj.weight":              {16, 8},
		"model.layers.0.mlp.down_proj.weight":            {8, 16},
	}
	names := slices.Sorted(maps.Keys(shapes))
	out := make([]textOnlyTestTensor, 0, len(names))
	for _, name := range names {
		shape := shapes[name]
		n := 1
		for _, d := range shape {
			n *= int(d)
		}
		vals := make([]float32, n)
		for i := range vals {
			vals[i] = 0.01 * float32(i%7+1)
		}
		out = append(out, textOnlyTestTensor{Name: name, Shape: shape, Vals: vals})
	}
	return out
}

// textOnlyHFConfig declares a llama text decoder plus a gemma4_vision tower,
// the shape of the checkpoints this repo cannot build: its vision_config has
// no in_channels, and Gemma-4's tensor layout has no LFM3-VL vision tensors.
func textOnlyHFConfig() []byte {
	cfg := map[string]any{
		"architectures":           []string{"LlamaForCausalLM"},
		"model_type":              "llama",
		"hidden_size":             8,
		"num_hidden_layers":       1,
		"num_attention_heads":     1,
		"num_key_value_heads":     1,
		"intermediate_size":       16,
		"vocab_size":              16,
		"max_position_embeddings": 32,
		"rms_norm_eps":            1e-5,
		"rope_theta":              10000,
		"tie_word_embeddings":     false,
		"vision_config": map[string]any{
			"model_type":          "gemma4_vision",
			"hidden_size":         32,
			"num_attention_heads": 2,
			"num_hidden_layers":   1,
			"patch_size":          16,
		},
	}
	raw, err := json.Marshal(cfg)
	if err != nil {
		panic(err)
	}
	return raw
}

// writeTextOnlyMCF writes a synthetic container carrying the tiny text tensors,
// the unsupported vision_config and enough tokenizer data for Loader.Load.
func writeTextOnlyMCF(t *testing.T, path string) {
	t.Helper()

	out, err := os.Create(path)
	if err != nil {
		t.Fatalf("create mcf: %v", err)
	}
	defer func() { _ = out.Close() }()

	w, err := mcf.NewWriter(out)
	if err != nil {
		t.Fatalf("new writer: %v", err)
	}
	if err := w.WriteSection(mcf.SectionHFConfigJSON, 1, textOnlyHFConfig()); err != nil {
		t.Fatalf("write hf config: %v", err)
	}
	tokJSON := []byte(`{
		"model":{"type":"BPE","vocab":{"<unk>":0,"<s>":1,"</s>":2,"hi":3},"merges":[],"unk_token":"<unk>"},
		"added_tokens":[
			{"id":0,"content":"<unk>","special":true},
			{"id":1,"content":"<s>","special":true},
			{"id":2,"content":"</s>","special":true}
		]
	}`)
	if err := w.WriteSection(mcf.SectionTokenizerJSON, 1, tokJSON); err != nil {
		t.Fatalf("write tokenizer.json: %v", err)
	}
	tokCfg := []byte(`{"bos_token":"<s>","eos_token":"</s>","unk_token":"<unk>"}`)
	if err := w.WriteSection(mcf.SectionTokenizerConfigJSON, 1, tokCfg); err != nil {
		t.Fatalf("write tokenizer_config.json: %v", err)
	}

	sw, err := w.BeginSection(mcf.SectionTensorData, 1)
	if err != nil {
		t.Fatalf("begin tensor data: %v", err)
	}
	dataOff, err := sw.CurrentAbsOffset()
	if err != nil {
		t.Fatalf("tensor data offset: %v", err)
	}
	tensors := textOnlySyntheticTensors()
	records := make([]mcf.TensorIndexRecord, 0, len(tensors))
	for _, tensor := range tensors {
		raw := make([]byte, 4*len(tensor.Vals))
		for i, v := range tensor.Vals {
			binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
		}
		records = append(records, mcf.TensorIndexRecord{
			DType:    mcf.DTypeF32,
			Shape:    tensor.Shape,
			DataOff:  dataOff,
			DataSize: uint64(len(raw)),
			Name:     tensor.Name,
		})
		if _, err := sw.Write(raw); err != nil {
			t.Fatalf("write tensor %s: %v", tensor.Name, err)
		}
		dataOff += uint64(len(raw))
	}
	if err := sw.End(); err != nil {
		t.Fatalf("end tensor data: %v", err)
	}

	indexPayload, err := mcf.EncodeTensorIndexSection(records)
	if err != nil {
		t.Fatalf("encode tensor index: %v", err)
	}
	if err := w.WriteSection(mcf.SectionTensorIndex, mcf.TensorIndexVersion, indexPayload); err != nil {
		t.Fatalf("write tensor index: %v", err)
	}
	if err := w.Finalise(); err != nil {
		t.Fatalf("finalise mcf: %v", err)
	}
}

// captureLogger records the messages a load logs at INFO.
type captureLogger struct {
	lines []string
}

func (l *captureLogger) Debug(string, ...any) {}
func (l *captureLogger) Info(msg string, args ...any) {
	line := msg
	for i := 0; i+1 < len(args); i += 2 {
		line += " " + args[i].(string) + "=" + args[i+1].(string)
	}
	l.lines = append(l.lines, line)
}
func (l *captureLogger) Warn(string, ...any)            {}
func (l *captureLogger) Error(string, ...any)           {}
func (l *captureLogger) With(...any) logger.Logger      { return l }
func (l *captureLogger) WithGroup(string) logger.Logger { return l }

// TestLoadModelTextOnly checks the text-only load path against a synthetic
// container whose hf config declares a vision tower this build cannot
// construct: the plain load fails with advice to use --text-only, the
// text-only load succeeds, and the resulting engine reports a text model whose
// media path is rejected.
func TestLoadModelTextOnly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "text-only.mcf")
	writeTextOnlyMCF(t, path)

	t.Run("plain load refuses the checkpoint", func(t *testing.T) {
		loader := Loader{Backend: "cpu"}
		_, err := loader.Load(context.Background(), path, 32)
		if err == nil {
			t.Fatalf("expected the unbuilt vision tower to stop the load")
		}
		if !strings.Contains(err.Error(), "vision_config: patch_size/in_channels missing") {
			t.Fatalf("error %q lost the underlying cause", err)
		}
		if !strings.Contains(err.Error(), "--text-only") {
			t.Fatalf("error %q does not tell the operator about --text-only", err)
		}
	})

	t.Run("text-only load succeeds", func(t *testing.T) {
		log := &captureLogger{}
		ctx := logger.WithContext(context.Background(), log)
		loader := Loader{Backend: "cpu", TextOnly: true}
		res, err := loader.Load(ctx, path, 32)
		if err != nil {
			t.Fatalf("text-only load: %v", err)
		}
		defer func() { _ = res.Engine.Close() }()

		caps := res.Engine.Capabilities()
		if caps.SupportsVision {
			t.Fatalf("text-only engine reports SupportsVision=true")
		}
		if caps.SupportsAudio {
			t.Fatalf("text-only engine reports SupportsAudio=true")
		}

		cfg := res.Runtime.ModelConfig()
		if cfg == nil {
			t.Fatalf("nil model config")
		}
		if cfg.Vision != nil || cfg.Audio != nil {
			t.Fatalf("text-only load kept a tower config: vision=%v audio=%v", cfg.Vision, cfg.Audio)
		}

		var skipped bool
		for _, line := range log.lines {
			if strings.Contains(line, "skipped vision tower") && strings.Contains(line, "model_type=gemma4_vision") {
				skipped = true
			}
		}
		if !skipped {
			t.Fatalf("load did not log the skipped vision tower; log lines: %v", log.lines)
		}

		// Media supplied to a text-only engine must fail with the existing
		// clear error rather than silently dropping the image.
		_, _, err = prepareMultimodal(res.Engine.(*EngineImpl), &Request{Images: [][]byte{{0x89, 'P', 'N', 'G'}}}, []int{1, 2})
		if err == nil || !strings.Contains(err.Error(), "media was provided") {
			t.Fatalf("image on a text-only engine: err=%v, want a media rejection", err)
		}
	})
}
