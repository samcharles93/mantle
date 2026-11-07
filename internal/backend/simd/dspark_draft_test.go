package simd

import (
	"os"
	"testing"

	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/internal/model"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// TestForwardBlockNonCausal checks the non-causal block mode on a synthetic
// model. The last row attends to the same keys causally and non-causally, so
// its logits must match ForwardTokens; earlier rows see later rows only in
// the non-causal mode, so they must differ. Pos must not advance.
func TestForwardBlockNonCausal(t *testing.T) {
	tokens := []int{1, 4, 2, 7}
	causal := newBatchParityInstance(t, batchParityContext, mcf.DTypeBF16)
	block := newBatchParityInstance(t, batchParityContext, mcf.DTypeBF16)

	want, err := causal.ForwardTokens(tokens)
	if err != nil {
		t.Fatalf("ForwardTokens: %v", err)
	}
	got := make([][]float32, len(tokens))
	err = block.ForwardBlockNonCausal(tokens, func(row int, logits, hidden []float32) {
		got[row] = append([]float32(nil), logits...)
		if len(hidden) != block.Config.Config.EmbeddingLength {
			t.Fatalf("hidden row width %d", len(hidden))
		}
	})
	if err != nil {
		t.Fatalf("ForwardBlockNonCausal: %v", err)
	}
	if block.Pos != 0 {
		t.Fatalf("Pos advanced to %d", block.Pos)
	}
	last := len(tokens) - 1
	compareLogitRows(t, "last row", got[last:], want[last:], batchParityTol)
	if floatsEqual(got[0], want[0], batchParityTol) {
		t.Fatal("row 0 matches the causal result; non-causal attention had no effect")
	}
	if err := block.ForwardBlockNonCausal(tokens[:1], func(int, []float32, []float32) {}); err == nil {
		t.Fatal("a one-token block must be rejected")
	}
}

// TestDSparkDraftAcceptance pairs the real MiniCPM5-2B target with its DSpark
// draft and measures how many drafted tokens match the target's own greedy
// continuation. There is no reference implementation on this host, so this is
// the end-to-end correctness signal: a wrong projector, cache layout, RoPE or
// Markov chain collapses acceptance to about zero.
func TestDSparkDraftAcceptance(t *testing.T) {
	targetPath, draftPath := os.Getenv("MANTLE_DSPARK_TARGET"), os.Getenv("MANTLE_DSPARK_DRAFT")
	if targetPath == "" || draftPath == "" {
		t.Skip("set MANTLE_DSPARK_TARGET and MANTLE_DSPARK_DRAFT to MiniCPM5-2B and its DSpark draft MCFs")
	}
	const maxContext = 512
	tf, err := mcfstore.Open(targetPath)
	if err != nil {
		t.Fatalf("open target: %v", err)
	}
	t.Cleanup(func() { _ = tf.Close() })
	df, err := mcfstore.Open(draftPath)
	if err != nil {
		t.Fatalf("open draft: %v", err)
	}
	t.Cleanup(func() { _ = df.Close() })

	draftCfg := df.SectionData(mcf.SectionHFConfigJSON)
	cfg, err := model.ParseDSparkConfig(draftCfg)
	if err != nil {
		t.Fatalf("draft config: %v", err)
	}
	target, err := LoadModelMCF(tf, tf.SectionData(mcf.SectionHFConfigJSON), maxContext,
		LoadModelOptions{HiddenTapLayers: cfg.TargetLayerIDs})
	if err != nil {
		t.Fatalf("load target: %v", err)
	}
	draft, err := LoadDSparkDraftMCF(df, draftCfg, target.asCore(), maxContext, LoadModelOptions{})
	if err != nil {
		t.Fatalf("load draft: %v", err)
	}

	// "<s><|im_start|>user\nWhat is the capital of France? Answer in one
	// sentence.<|im_end|>\n<|im_start|>assistant\n", tokenized by MiniCPM5.
	prompt := []int{0, 130072, 8448, 220, 2928, 357, 285, 4894, 304, 6918, 52, 10893, 310, 678, 9622, 35, 130073, 220, 130072, 130071, 220}

	var logits []float32
	for start := 0; start < len(prompt); start += target.HiddenTaps.Capacity {
		end := min(start+target.HiddenTaps.Capacity, len(prompt))
		if logits, err = target.PrefillTokens(prompt[start:end]); err != nil {
			t.Fatalf("prefill: %v", err)
		}
		if target.HiddenTaps.Rows != end-start {
			t.Fatalf("captured %d tap rows for %d prompt tokens", target.HiddenTaps.Rows, end-start)
		}
		if err := draft.Inject(start, end-start); err != nil {
			t.Fatalf("inject: %v", err)
		}
	}
	anchor := argmaxHost(logits)

	const rounds = 8
	accepted := 0
	for round := range rounds {
		drafted, conf, err := draft.DraftBlock(anchor, cfg.BlockSize)
		if err != nil {
			t.Fatalf("draft: %v", err)
		}
		// Step the target greedily from the anchor, injecting each new
		// position, to obtain its own continuation.
		want := make([]int, len(drafted))
		tok := anchor
		for i := range want {
			pos := target.Pos
			l, err := target.ForwardToken(tok)
			if err != nil {
				t.Fatalf("target step: %v", err)
			}
			if err := draft.Inject(pos, 1); err != nil {
				t.Fatalf("inject step: %v", err)
			}
			want[i] = argmaxHost(l)
			tok = want[i]
		}
		n := 0
		for n < len(drafted) && drafted[n] == want[n] {
			n++
		}
		accepted += n
		t.Logf("round %d: accepted %d/%d draft=%v target=%v conf=%.2f", round, n, len(drafted), drafted, want, conf)
		anchor = tok
	}
	mean := float64(accepted) / rounds
	t.Logf("mean accepted draft tokens per block: %.2f", mean)
	if mean < 1.5 {
		t.Fatalf("mean acceptance %.2f is far below the checkpoint's reported 4 to 5 tokens", mean)
	}
}
