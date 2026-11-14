//go:build cuda

package cuda

import (
	"os"
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// TestBatchedPrefillRealHybridMCF is the opt-in real-model parity check for the
// hybrid (LFM3) batched prefill: batched CUDA prefill vs the sequential CUDA
// oracle on the same container. bf16 accumulation order differs between the
// batched GEMM and the per-token matvec, so the tolerance is bf16-scale
// (relative), not exact; the strict structural check is the synthetic
// TestBatchedPrefillParityHybrid.
func TestBatchedPrefillRealHybridMCF(t *testing.T) {
	path := os.Getenv("MANTLE_CUDA_LFM3_MCF")
	if path == "" {
		t.Skip("set MANTLE_CUDA_LFM3_MCF")
	}
	if !cudaAvailable(t) {
		return
	}
	tokens := []int{124894, 124899, 5922, 207, 2992, 355, 278, 5205, 302, 3980, 39, 41774, 296, 734, 12683, 22, 3077}
	load := func() *GraphRuntime {
		file, err := mcfstore.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = file.Close() })
		b := &Backend{}
		rt, err := b.LoadModel(file, file.SectionData(mcf.SectionHFConfigJSON), 64, core.LoadModelOptions{GpuLayers: -1})
		if err != nil {
			t.Fatal(err)
		}
		return rt.(*GraphRuntime)
	}
	batched := load()
	// Quantized models are opt-in (quantBatchedPrefillEnv): the batched quant
	// GEMM quantizes activations and the per-token path does not, so their
	// logits differ. Assert the gate here, on a real container, then skip the
	// parity comparison that only has a defined answer for dense weights.
	if batched.hasQuantWeights() && !envEnabled(quantBatchedPrefillEnv) {
		if batched.batchedPrefillEligible(tokens) {
			t.Fatalf("quantized model must not be eligible without %s", quantBatchedPrefillEnv)
		}
		t.Skipf("quantized model: batched prefill is opt-in via %s", quantBatchedPrefillEnv)
	}
	if !batched.batchedPrefillEligible(tokens) {
		t.Fatal("not eligible")
	}
	got, err := batched.PrefillTokens(tokens)
	if err != nil {
		t.Fatal(err)
	}
	seq := load()
	os.Setenv("MANTLE_CUDA_BATCHED_PREFILL", "0")
	want, err := seq.PrefillTokens(tokens)
	os.Unsetenv("MANTLE_CUDA_BATCHED_PREFILL")
	if err != nil {
		t.Fatal(err)
	}
	worst := float32(0)
	for i := range got {
		d := got[i] - want[i]
		if d < 0 {
			d = -d
		}
		if d > worst {
			worst = d
		}
	}
	t.Logf("worst abs logit diff = %v", worst)

	// bf16-scale tolerance: accumulation order differs between batched GEMM
	// and per-token matvec, and the error compounds over 24 layers; near-zero
	// logits carry the largest relative noise. argmax must agree.
	const atol, rtol = 1e-1, 1e-1
	amG, amW := 0, 0
	for i := range got {
		if got[i] > got[amG] {
			amG = i
		}
		if want[i] > want[amW] {
			amW = i
		}
	}
	if amG != amW {
		t.Fatalf("argmax differs: batched %d want %d", amG, amW)
	}
	for i := range got {
		d := got[i] - want[i]
		if d < 0 {
			d = -d
		}
		wa := want[i]
		if wa < 0 {
			wa = -wa
		}
		if d > atol+rtol*wa {
			t.Fatalf("batched != sequential at logit %d: got %v want %v (diff %v)", i, got[i], want[i], d)
		}
	}
	// Decode continuity: the conv state rotated on device during the batched
	// prefill must serve the per-token decode path exactly as the sequential
	// prefill's host state does. Compare 8 greedy tokens.
	tokG, tokW := amG, amW
	for i := 0; i < 8; i++ {
		lg, err := batched.ForwardToken(tokG)
		if err != nil {
			t.Fatal(err)
		}
		lw, err := seq.ForwardToken(tokW)
		if err != nil {
			t.Fatal(err)
		}
		ng, nw := 0, 0
		for j := range lg {
			if lg[j] > lg[ng] {
				ng = j
			}
			if lw[j] > lw[nw] {
				nw = j
			}
		}
		if ng != nw {
			t.Fatalf("greedy token %d differs after batched prefill: got %d want %d", i, ng, nw)
		}
		tokG, tokW = ng, nw
	}
}
