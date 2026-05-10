package simd

import (
	"fmt"
	"testing"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
)

// TestFusedAttentionEquivalence verifies that fusedAttention produces the same
// output as Attention for identical inputs. Since fusedAttention delegates to
// Attention, this is a regression test that the delegation path works correctly.
func TestFusedAttentionEquivalence(t *testing.T) {
	ops := newNoFastPathOps()
	m, layer := newFusedAttentionFixture(ops)
	input := []float32{0.1, 0.2}
	pos := 0
	params := graph.AttentionParams{LayerIndex: 0}

	got := fusedAttention(m, layer, input, pos, params)
	want := Attention(m, layer, input, pos)

	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		diff := got[i] - want[i]
		if diff < -1e-6 || diff > 1e-6 {
			t.Fatalf("mismatch at %d: got %v, want %v (diff=%v)", i, got[i], want[i], diff)
		}
	}
}

// TestFusedAttentionVariedPositions verifies equivalence across multiple
// positions, ensuring the delegation remains correct for different KV cache states.
func TestFusedAttentionVariedPositions(t *testing.T) {
	ops := newNoFastPathOps()
	for pos := 0; pos < 4; pos++ {
		t.Run(fmt.Sprintf("pos=%d", pos), func(t *testing.T) {
			m, layer := newFusedAttentionFixture(ops)
			input := []float32{0.1, 0.2}
			params := graph.AttentionParams{LayerIndex: 0}

			got := fusedAttention(m, layer, input, pos, params)
			want := Attention(m, layer, input, pos)

			if len(got) != len(want) {
				t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
			}
			for i := range got {
				diff := got[i] - want[i]
				if diff < -1e-6 || diff > 1e-6 {
					t.Fatalf("pos=%d idx=%d: got %v, want %v (diff=%v)", pos, i, got[i], want[i], diff)
				}
			}
		})
	}
}

// noFastPathOps disables all fast paths so Attention() takes the scalar
// fallback path, making tests deterministic and backend-agnostic.
type noFastPathOps struct {
	*DefaultOps
}

func (o *noFastPathOps) MatVecQKV(q, k, v []float32, wq, wk, wv *instance.Mat, x []float32) bool {
	return false
}

func (o *noFastPathOps) AttentionInner(attnOut []float32, layer *Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale, softcap float32) bool {
	return false
}

func (o *noFastPathOps) AttentionInnerProjection(projOut []float32, layer, kvLayer *Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale, epsilon, softcap float32) bool {
	return false
}

func (o *noFastPathOps) IncrementalAttention(attnOut []float32, layer *Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale, softcap float32) bool {
	return false
}

func (o *noFastPathOps) FlashAttention(attnOut []float32, layer *Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale, softcap float32) bool {
	return false
}

func (o *noFastPathOps) FlashAttentionMultiHead(attnOut []float32, layer *Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale, softcap float32) bool {
	return false
}

func (o *noFastPathOps) QKVAttentionProjection(projOut []float32, layer, kvLayer *Layer, x []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale, epsilon, softcap float32, invFreq []float64, ropeAttnScale float32, applyRope, mirrorHostKV bool) bool {
	return false
}

func newNoFastPathOps() *noFastPathOps {
	return &noFastPathOps{DefaultOps: &DefaultOps{}}
}

func newFusedAttentionFixture(ops Ops) (*Instance, *Layer) {
	const headDim = 2
	const nHead = 1
	kvStride := nHead * headDim

	wo := instance.NewMatFromData(nHead*headDim, nHead*headDim, []float32{
		1, 0,
		0, 1,
	})
	wq := instance.NewMatFromData(nHead*headDim, nHead*headDim, []float32{
		1, 0,
		0, 1,
	})
	wk := instance.NewMatFromData(nHead*headDim, nHead*headDim, []float32{
		1, 0,
		0, 1,
	})
	wv := instance.NewMatFromData(nHead*headDim, nHead*headDim, []float32{
		1, 0,
		0, 1,
	})

	layer := &Layer{
		HeadKV:  nHead,
		HeadDim: headDim,
		Wq:      &wq,
		Wk:      &wk,
		Wv:      &wv,
		Wo:      &wo,
		AttnCache: AttnCache{
			K:        make([]float32, 0, 256),
			V:        make([]float32, 0, 256),
			KvStride: kvStride,
			CacheLen: 16,
		},
	}

	maxContext := 16
	m := &Instance{
		HeadCount:   nHead,
		HeadDim:     headDim,
		RMSEpsilon:  1e-6,
		RopeInvFreq: []float64{1.0},
		MaxContext:  maxContext,
		Scratch: ScratchBuffers{
			Q:        make([]float32, nHead*headDim),
			K:        make([]float32, kvStride),
			V:        make([]float32, kvStride),
			AttnOut:  make([]float32, nHead*headDim),
			AttnProj: make([]float32, nHead*headDim),
			Scores:   make([]float32, maxContext),
			AttnGate: make([]float32, nHead*headDim),
		},
	}
	m.SetOps(ops)
	return m, layer
}
