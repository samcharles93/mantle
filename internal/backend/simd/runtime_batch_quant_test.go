package simd

import (
	"fmt"
	"math"
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// quantParityEmbd gives two 32-column blocks per hidden row and three per FFN
// row, so the batched quant kernel's paired and single block paths both run.
const quantParityEmbd = 64

// newQuantParityInstance is a batch parity model whose projections and output
// head are stored the way a q8/k4 MCF load leaves them: quantised Raw payloads
// with the loader's int8 cache.
func newQuantParityInstance(tb testing.TB, dtype mcf.TensorDType) *Instance {
	tb.Helper()
	m := newBatchParityInstanceDim(tb, batchParityContext, quantParityEmbd, mcf.DTypeF32)
	weights := []*core.Mat{m.Output}
	for i := range m.Layers {
		l := &m.Layers[i]
		weights = append(weights, l.Wq, l.Wk, l.Wv, l.Wo, l.FfnUp, l.FfnGate, l.FfnDown)
	}
	for wi, w := range weights {
		quantiseTestWeight(tb, w, dtype, wi)
	}
	return m
}

// quantiseTestWeight replaces w with a quantised matrix of the same shape.
// Values come from a fixed pseudo-random pattern rather than w's ramp data,
// whose magnitudes grow with the hidden width until the logits overflow.
func quantiseTestWeight(tb testing.TB, w *core.Mat, dtype mcf.TensorDType, salt int) {
	tb.Helper()
	n := w.R * w.C
	qvals := make([]int8, n)
	var payload []byte
	switch dtype {
	case mcf.DTypeQ8:
		for i := range qvals {
			qvals[i] = int8((i*131+salt*17)%255 - 127)
		}
		payload = buildQPayload(w.R, w.C, 8, 0.0008, qvals)
	case mcf.DTypeK4:
		for i := range qvals {
			qvals[i] = int8((i*7+salt*3)%16 - 8)
		}
		payload = buildK4Payload(w.R, w.C, 0.012, qvals)
	default:
		tb.Fatalf("unsupported dtype %v", dtype)
	}
	*w = core.Mat{R: w.R, C: w.C, Stride: w.C, DType: dtype, Raw: payload}
	qc, err := core.BuildQuantCache(w)
	if err != nil {
		tb.Fatalf("BuildQuantCache: %v", err)
	}
	w.Quant = qc
}

// TestBatchPrefillQuantParity requires the batched prefill over quantised
// weights to reproduce the per-token path exactly, for every logits row and
// the KV cache, at batch sizes that hit each tail of the 2x4 kernel.
func TestBatchPrefillQuantParity(t *testing.T) {
	if !cpu.HasAVX2 {
		t.Skip("the cached int8 quant kernels need AVX2")
	}
	for _, dtype := range []mcf.TensorDType{mcf.DTypeQ8, mcf.DTypeK4} {
		for _, n := range []int{2, 3, 6, 13} {
			t.Run(fmt.Sprintf("%v/n%d", dtype, n), func(t *testing.T) {
				tokens := make([]int, n)
				for i := range tokens {
					tokens[i] = (i*5 + 3) % batchParityVocab
				}
				batched := newQuantParityInstance(t, dtype)
				sequential := newQuantParityInstance(t, dtype)
				if why := batched.batchIneligibleReason(tokens); why != "" {
					t.Fatalf("quantised dense model should be eligible: %s", why)
				}

				want := sequentialLogits(t, sequential, tokens)
				for _, row := range want {
					for _, v := range row {
						if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
							t.Fatalf("oracle logits are not finite: %v", row)
						}
					}
				}
				got, err := batched.ForwardTokens(tokens)
				if err != nil {
					t.Fatalf("ForwardTokens: %v", err)
				}
				compareLogitRows(t, "logits", got, want, 0)
				compareKVCache(t, batched, sequential, 0)
			})
		}
	}
}

// TestPrefillTokensQuantLastRow covers the prefill shape (last row only),
// which takes the fused output head rather than the batched one.
func TestPrefillTokensQuantLastRow(t *testing.T) {
	if !cpu.HasAVX2 {
		t.Skip("the cached int8 quant kernels need AVX2")
	}
	tokens := []int{3, 9, 1, 14, 6, 2, 11}
	batched := newQuantParityInstance(t, mcf.DTypeQ8)
	sequential := newQuantParityInstance(t, mcf.DTypeQ8)
	want := sequentialLogits(t, sequential, tokens)
	got, err := batched.PrefillTokens(tokens)
	if err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	compareLogitRows(t, "last row", [][]float32{got}, want[len(want)-1:], 0)
}
