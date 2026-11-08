package simd

import (
	"os"
	"slices"
	"testing"

	"github.com/samcharles93/mantle/internal/logits"
	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/internal/model"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// TestDSparkSpeculateRealPairGreedy runs the speculative loop on the real
// MiniCPM5-2B + DSpark pair and compares it against the target's plain greedy
// continuation, with and without the CLI's repetition penalty. The
// verification batch and the per-token path round differently (GEMM vs
// matvec), so a flip at a near-tied argmax is expected rounding noise, not a
// logic error; the test reports the top-2 logit margin at any divergence.
func TestDSparkSpeculateRealPairGreedy(t *testing.T) {
	targetPath, draftPath := os.Getenv("MANTLE_DSPARK_TARGET"), os.Getenv("MANTLE_DSPARK_DRAFT")
	if targetPath == "" || draftPath == "" {
		t.Skip("set MANTLE_DSPARK_TARGET and MANTLE_DSPARK_DRAFT")
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
	targetCfg := tf.SectionData(mcf.SectionHFConfigJSON)
	target, err := LoadModelMCF(tf, targetCfg, maxContext, LoadModelOptions{HiddenTapLayers: cfg.TargetLayerIDs})
	if err != nil {
		t.Fatalf("load target: %v", err)
	}
	draft, err := LoadDSparkDraftMCF(df, draftCfg, target.asCore(), maxContext, LoadModelOptions{})
	if err != nil {
		t.Fatalf("load draft: %v", err)
	}
	ref, err := LoadModelMCF(tf, targetCfg, maxContext, LoadModelOptions{HiddenTapLayers: cfg.TargetLayerIDs})
	if err != nil {
		t.Fatalf("load ref: %v", err)
	}

	prompts := map[string][]int{
		"france": {0, 130072, 8448, 220, 2928, 357, 285, 4894, 304, 6918, 52, 10893, 310, 678, 9622, 35, 130073, 220, 130072, 130071, 220},
		"sky":    {0, 130072, 8448, 220, 77524, 3212, 285, 9467, 357, 5863, 310, 947, 15046, 35, 130073, 220, 130072, 130071, 220, 8, 130063, 9, 130063},
	}
	const steps = 48
	for name, prompt := range prompts {
		// CLI-style samplers: greedy, with and without the default repetition
		// penalty 1.1. A fresh sampler per run, as the engine builds one per
		// Generate call.
		newSampler := func(repeatPenalty float32) *logits.Sampler {
			return logits.NewSampler(logits.SamplerConfig{Temperature: 0, TopK: 40, TopP: 1, RepeatPenalty: repeatPenalty, RepeatLastN: 64, Seed: 1})
		}

		for _, mode := range []struct {
			name string
			rp   float32
		}{
			{"argmax", 1.0},
			{"greedy+repeat1.1", 1.1},
		} {
			// Fresh state for every run: reset, re-prefill, snapshot the
			// model-owned logits slice before any forward overwrites it.
			target.Reset()
			draft.Truncate(0)
			ref.Reset()
			prefillLogits, err := target.PrefillTokens(prompt)
			if err != nil {
				t.Fatalf("%s: prefill: %v", name, err)
			}
			promptLogits := append([]float32(nil), prefillLogits...)
			if err := draft.Inject(0, len(prompt)); err != nil {
				t.Fatalf("%s: inject: %v", name, err)
			}
			refPrefill, err := ref.PrefillTokens(prompt)
			if err != nil {
				t.Fatalf("%s: ref prefill: %v", name, err)
			}
			refBase := append([]float32(nil), refPrefill...)

			s := newSampler(mode.rp)
			var got []int
			q := append([]float32(nil), promptLogits...)
			toks := append([]int(nil), prompt...)
			for len(got) < steps {
				anchor := s.Sample(q, toks, nil)
				runToks := append(append([]int(nil), toks...), anchor)
				disagree := func(row []float32, drafted int, _ float32) bool {
					if s.Sample(row, runToks, nil) == drafted {
						runToks = append(runToks, drafted)
						return false
					}
					return true
				}
				accepted, next, err := draft.Speculate(anchor, cfg.BlockSize, disagree)
				if err != nil {
					t.Fatalf("%s/%s: speculate: %v", name, mode.name, err)
				}
				got = append(got, accepted...)
				toks = append(toks, accepted...)
				q = next
			}
			got = got[:steps]

			s2 := newSampler(mode.rp)
			var want []int
			var refRows [][]float32
			rq := append([]float32(nil), refBase...)
			rtoks := append([]int(nil), prompt...)
			for len(want) < steps {
				refRows = append(refRows, append([]float32(nil), rq...))
				tok := s2.Sample(rq, rtoks, nil)
				want = append(want, tok)
				rtoks = append(rtoks, tok)
				rl, err := ref.ForwardToken(tok)
				if err != nil {
					t.Fatalf("%s/%s: ref step: %v", name, mode.name, err)
				}
				rq = append([]float32(nil), rl...)
			}

			if slices.Equal(got, want) {
				t.Logf("%s/%s: identical for %d tokens", name, mode.name, steps)
				continue
			}
			for i := range got {
				if got[i] != want[i] {
					// Top-2 margin of the plain reference row at the
					// diverging position: a tiny margin means the batch and
					// per-token paths rounded a near-tie differently.
					row := append([]float32(nil), refRows[i]...)
					// Apply the same repetition penalty the sampler applies
					// before argmax, then measure the effective top-2 margin.
					if mode.rp > 1.0 {
						start := max(len(prompt)+i-64, 0)
						window := append(append([]int(nil), prompt...), want[:i]...)[start:]
						seen := make(map[int]bool)
						for _, id := range window {
							if id >= 0 && id < len(row) && !seen[id] {
								seen[id] = true
								if row[id] > 0 {
									row[id] /= mode.rp
								} else {
									row[id] *= mode.rp
								}
							}
						}
					}
					top1, top2 := float32(-1e30), float32(-1e30)
					for _, v := range row {
						if v > top1 {
							top2, top1 = top1, v
						} else if v > top2 {
							top2 = v
						}
					}
					t.Logf("%s/%s: diverged at %d: spec=%d plain=%d, post-penalty top-2 margin %.4f", name, mode.name, i, got[i], want[i], top1-top2)
					t.Fatalf("%s/%s: diverged at %d: spec=%d plain=%d\nspec=%v\nplain=%v", name, mode.name, i, got[i], want[i], got, want)
				}
			}
		}

		target.Reset()
		draft.Truncate(0)
		ref.Reset()
	}
}
