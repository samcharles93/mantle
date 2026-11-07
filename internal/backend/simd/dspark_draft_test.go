package simd

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/pkg/mcf"
)

const dsparkTestConfig = `{
  "architectures": ["Qwen3DSparkModel"],
  "hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 5,
  "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 128,
  "vocab_size": 130560, "draft_vocab_size": 130560, "rms_norm_eps": 1e-6,
  "block_size": 7, "mask_token_id": 75982, "target_layer_ids": [1, 10, 20, 30, 39],
  "num_target_layers": 42, "markov_rank": 256, "markov_head_type": "vanilla",
  "projector_type": "dspark", "enable_confidence_head": true, "hidden_act": "silu",
  "rope_parameters": {"rope_theta": 5000000, "rope_type": "default"}
}`

func TestParseDSparkConfig(t *testing.T) {
	c, err := ParseDSparkConfig([]byte(dsparkTestConfig))
	if err != nil {
		t.Fatalf("ParseDSparkConfig: %v", err)
	}
	if c.BlockSize != 7 || c.MaskTokenID != 75982 || c.MarkovRank != 256 || !c.HasConfidence ||
		c.RopeParameters.RopeTheta != 5e6 || len(c.TargetLayerIDs) != 5 {
		t.Fatalf("unexpected config: %+v", c)
	}

	bad := map[string]string{
		"not dspark":     `{"architectures":["LlamaForCausalLM"]}`,
		"rnn markov":     replaceJSON(dsparkTestConfig, `"vanilla"`, `"rnn"`),
		"reduced vocab":  replaceJSON(dsparkTestConfig, `"draft_vocab_size": 130560`, `"draft_vocab_size": 32000`),
		"unsorted taps":  replaceJSON(dsparkTestConfig, `[1, 10, 20, 30, 39]`, `[1, 20, 10, 30, 39]`),
		"oversize block": replaceJSON(dsparkTestConfig, `"block_size": 7`, `"block_size": 64`),
		"bonus anchor":   replaceJSON(dsparkTestConfig, `"hidden_act": "silu"`, `"hidden_act": "silu", "sample_from_anchor": false`),
	}
	for name, raw := range bad {
		if _, err := ParseDSparkConfig([]byte(raw)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func replaceJSON(s, old, new string) string {
	for i := 0; i+len(old) <= len(s); i++ {
		if s[i:i+len(old)] == old {
			return s[:i] + new + s[i+len(old):]
		}
	}
	panic("replaceJSON: " + old + " not found")
}

// TestDSparkBlockAttentionMatchesReference checks the draft layer's
// non-causal attention over injected context plus block rows against a
// direct float64 evaluation of the same softmax.
func TestDSparkBlockAttentionMatchesReference(t *testing.T) {
	const heads, kvHeads, hd, ctx, n = 4, 2, 16, 5, 3
	rng := rand.New(rand.NewSource(3))
	fill := func(k int) []float32 {
		v := make([]float32, k)
		for i := range v {
			v[i] = rng.Float32() - 0.5
		}
		return v
	}
	q := fill(n * heads * hd)
	cacheK, cacheV := fill(ctx*kvHeads*hd), fill(ctx*kvHeads*hd)
	k, v := fill(n*kvHeads*hd), fill(n*kvHeads*hd)

	got := make([]float32, n*heads*hd)
	dsparkAttend(got, q, k, v, cacheK, cacheV, make([]float32, ctx+n), n, ctx, heads, kvHeads, hd)

	kvDim := kvHeads * hd
	for i := range n {
		for h := range heads {
			kh := h / (heads / kvHeads)
			qv := q[i*heads*hd+h*hd : i*heads*hd+(h+1)*hd]
			keys := make([][]float32, 0, ctx+n)
			vals := make([][]float32, 0, ctx+n)
			for p := range ctx {
				keys = append(keys, cacheK[p*kvDim+kh*hd:p*kvDim+(kh+1)*hd])
				vals = append(vals, cacheV[p*kvDim+kh*hd:p*kvDim+(kh+1)*hd])
			}
			for j := range n {
				keys = append(keys, k[j*kvDim+kh*hd:j*kvDim+(kh+1)*hd])
				vals = append(vals, v[j*kvDim+kh*hd:j*kvDim+(kh+1)*hd])
			}
			w := make([]float64, len(keys))
			var sum, mx float64 = 0, math.Inf(-1)
			for p, key := range keys {
				var s float64
				for d := range hd {
					s += float64(qv[d]) * float64(key[d])
				}
				w[p] = s / math.Sqrt(hd)
				mx = math.Max(mx, w[p])
			}
			for p := range w {
				w[p] = math.Exp(w[p] - mx)
				sum += w[p]
			}
			for d := range hd {
				var want float64
				for p, val := range vals {
					want += w[p] / sum * float64(val[d])
				}
				g := got[i*heads*hd+h*hd+d]
				if math.Abs(float64(g)-want) > 1e-3 {
					t.Fatalf("row %d head %d dim %d: got %g want %g", i, h, d, g, want)
				}
			}
		}
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
	cfg, err := ParseDSparkConfig(draftCfg)
	if err != nil {
		t.Fatalf("draft config: %v", err)
	}
	target, err := LoadModelMCF(tf, tf.SectionData(mcf.SectionHFConfigJSON), maxContext,
		LoadModelOptions{HiddenTapLayers: cfg.TargetLayerIDs})
	if err != nil {
		t.Fatalf("load target: %v", err)
	}
	draft, err := LoadDSparkDraftMCF(df, draftCfg, target, maxContext)
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
		if err := draft.Inject(start); err != nil {
			t.Fatalf("inject: %v", err)
		}
	}
	anchor := argmaxF32(logits)

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
			if err := draft.Inject(pos); err != nil {
				t.Fatalf("inject step: %v", err)
			}
			want[i] = argmaxF32(l)
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
