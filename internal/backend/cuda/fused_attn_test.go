//go:build cuda

package cuda

import (
	"testing"

	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/internal/graph"
)

func TestFusedAttentionCUDA(t *testing.T) {
	count, err := native.DeviceCount()
	if err != nil {
		t.Skipf("cannot query CUDA devices: %v", err)
	}
	if count < 1 {
		t.Skip("no CUDA device available")
	}

	gr := &GraphRuntime{}
	ctx := &graph.ComputeContext{Pos: 0, Token: 1}
	params := graph.AttentionParams{LayerIndex: -1}

	_, err = gr.fusedAttentionBlock(ctx, params, nil)
	if err == nil {
		t.Fatal("expected error from nil runtime, got nil")
	}
	t.Logf("fusedAttentionBlock correctly rejected nil runtime: %v", err)
}
