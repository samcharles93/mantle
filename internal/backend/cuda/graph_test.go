//go:build cuda

package cuda

import (
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/internal/graph"
)

// testModel wraps *core.Instance to satisfy core.Runtime.
type testModel struct {
	*core.Instance
}

func (m *testModel) ForwardToken(id int) ([]float32, error) { return nil, nil }
func (m *testModel) Reset()                                 {}
func (m *testModel) UpdateRoPE()                            {}

func TestGraphComputeCUDABasic(t *testing.T) {
	count, err := native.DeviceCount()
	if err != nil {
		t.Fatalf("DeviceCount: %v", err)
	}
	if count < 1 {
		t.Skip("no CUDA device available")
	}

	stream, err := native.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	blas, err := native.NewBlasHandle(stream)
	if err != nil {
		t.Fatalf("NewBlasHandle: %v", err)
	}
	defer func() { _ = blas.Destroy() }()

	ops := NewOps(stream, blas)
	defer func() { _ = ops.Close() }()

	const (
		embDim    = 8
		vocabSize = 16
		tokenID   = 5
	)

	embeddings := core.NewMat(vocabSize, embDim)
	for i := range vocabSize * embDim {
		embeddings.Data[i] = float32(i+1) / 10.0
	}

	output := core.NewMat(vocabSize, embDim)
	for i := range vocabSize * embDim {
		output.Data[i] = float32(i+1) / 100.0
	}
	outputNorm := make([]float32, embDim)
	for i := range outputNorm {
		outputNorm[i] = 1.0
	}

	inst := &core.Instance{
		Config: &core.ModelConfig{
			Arch: "test",
			Config: core.Config{
				VocabSize:       vocabSize,
				EmbeddingLength: embDim,
				BlockCount:      0,
			},
		},
		Embeddings:  &embeddings,
		Output:      &output,
		OutputNorm:  outputNorm,
		RMSEpsilon:  1e-6,
		HeadCount:   1,
		MaxKVStride: embDim,
	}
	inst.SetOps(ops)

	rt := &cudaRuntime{
		model:  &testModel{Instance: inst},
		ops:    ops,
		stream: stream,
		blas:   blas,
	}
	gr := NewGraphRuntime(rt, inst)

	graph.ResetTensorID()
	g := &graph.Graph{}

	embedID := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpEmbed,
		Branch: graph.BranchEmbed,
		Name:   "embed",
		Input:  []graph.TensorID{0},
		Output: embedID,
		Params: graph.EmbedParams{
			VocabSize: vocabSize,
			EmbDim:    embDim,
		},
	})

	outputID := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpOutput,
		Branch: graph.BranchOutput,
		Name:   "output",
		Input:  []graph.TensorID{embedID},
		Output: outputID,
		Params: graph.OutputParams{
			Softcap: 0,
		},
	})

	ctx := &graph.ComputeContext{
		Pos:   0,
		KVLen: 1,
		Token: tokenID,
	}

	logits, err := gr.Compute(ctx, g)
	if err != nil {
		t.Fatalf("Compute failed: %v", err)
	}

	if len(logits) != vocabSize {
		t.Fatalf("expected %d logits, got %d", vocabSize, len(logits))
	}

	for i, v := range logits {
		if v == 0 && i != 0 {
			t.Errorf("logit[%d] is zero, expected non-zero from synthetic weights", i)
		}
	}

	t.Logf("logits=%v", logits)
}

func TestParseLayerIndex(t *testing.T) {
	tests := []struct {
		name  string
		input string
		def   int
		want  int
	}{
		{"layer0", "layer0.attn_norm", -1, 0},
		{"layer3", "layer3.ffn", -1, 3},
		{"layer12", "layer12.attention", -1, 12},
		{"no-layer-fallback", "embed", 7, 7},
		{"layer-nonum", "layerX.ffn", 5, 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseLayerIndex(tt.input, tt.def)
			if got != tt.want {
				t.Errorf("parseLayerIndex(%q, %d) = %d, want %d", tt.input, tt.def, got, tt.want)
			}
		})
	}
}

func TestComputeEmbed(t *testing.T) {
	const (
		vocabSize = 10
		embDim    = 4
	)
	mat := core.NewMat(vocabSize, embDim)
	for i := range vocabSize * embDim {
		mat.Data[i] = float32(i)
	}
	inst := &core.Instance{Embeddings: &mat}

	params := graph.EmbedParams{VocabSize: vocabSize, EmbDim: embDim}
	out, err := computeEmbed(inst, 3, params)
	if err != nil {
		t.Fatalf("computeEmbed: %v", err)
	}
	if len(out) != embDim {
		t.Fatalf("expected len %d, got %d", embDim, len(out))
	}
	for j := 0; j < embDim; j++ {
		want := float32(3*embDim + j)
		if out[j] != want {
			t.Errorf("embed[%d] = %f, want %f", j, out[j], want)
		}
	}
}

func TestGraphComputeNoDevice(t *testing.T) {
	const (
		embDim    = 4
		vocabSize = 8
		tokenID   = 2
	)
	mat := core.NewMat(vocabSize, embDim)
	for i := range vocabSize * embDim {
		mat.Data[i] = float32(i+1) / 10.0
	}

	inst := &core.Instance{
		Config: &core.ModelConfig{
			Arch: "test",
			Config: core.Config{
				VocabSize:       vocabSize,
				EmbeddingLength: embDim,
				BlockCount:      0,
			},
		},
		Embeddings: &mat,
	}

	graph.ResetTensorID()
	g := &graph.Graph{}

	embedID := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpEmbed,
		Branch: graph.BranchEmbed,
		Name:   "embed",
		Input:  []graph.TensorID{0},
		Output: embedID,
		Params: graph.EmbedParams{
			VocabSize: vocabSize,
			EmbDim:    embDim,
		},
	})

	ctx := &graph.ComputeContext{Token: tokenID}

	if err := g.Validate(); err != nil {
		t.Fatalf("graph validation failed: %v", err)
	}

	params := g.Nodes[0].Params.(graph.EmbedParams)
	out, err := computeEmbed(inst, ctx.Token, params)
	if err != nil {
		t.Fatalf("computeEmbed: %v", err)
	}
	if len(out) != embDim {
		t.Fatalf("expected len %d, got %d", embDim, len(out))
	}
	for j := 0; j < embDim; j++ {
		want := float32(tokenID*embDim+j+1) / 10.0
		if out[j] != want {
			t.Errorf("embed[%d] = %f, want %f", j, out[j], want)
		}
	}
}

func TestSoftcap(t *testing.T) {
	if s := softcap(1.0, 0.0); s != 1.0 {
		t.Errorf("softcap with 0 cap should be identity: got %f", s)
	}
	s := softcap(100.0, 30.0)
	if s >= 100.0 || s <= 0 {
		t.Errorf("softcap(100, 30) = %f, expected damped value", s)
	}
}

func TestGraphValidateSyntheticGraph(t *testing.T) {
	graph.ResetTensorID()
	g := &graph.Graph{
		Nodes: []graph.Node{
			{
				Op:     graph.OpEmbed,
				Branch: graph.BranchEmbed,
				Name:   "embed",
				Input:  []graph.TensorID{graph.NewTensorID()},
				Output: graph.NewTensorID(),
				Params: graph.EmbedParams{VocabSize: 100, EmbDim: 16},
			},
			{
				Op:     graph.OpAttentionBlock,
				Branch: graph.BranchAttention,
				Name:   "layer0.attention",
				Input:  []graph.TensorID{1},
				Output: graph.NewTensorID(),
				Params: graph.AttentionParams{LayerIndex: 0, HeadDim: 8, NHeadKV: 2},
			},
		},
	}
	if err := g.Validate(); err != nil {
		t.Fatalf("valid synthetic graph rejected: %v", err)
	}
}
