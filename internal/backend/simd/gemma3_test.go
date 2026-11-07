package simd

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// TestGemma3ReferenceLogits compares gemma-3-270m-it against logits from HF
// transformers (testdata/gemma3_reference.json): the last-position top-10 for
// a short prompt and for a 608-token prompt that runs past the 512-token
// sliding window (exercising the local-RoPE sliding layers), plus greedy
// continuations.
func TestGemma3ReferenceLogits(t *testing.T) {
	path := os.Getenv("MANTLE_GEMMA3_MCF")
	if path == "" {
		t.Skip("set MANTLE_GEMMA3_MCF to the gemma-3-270m-it container to run the reference comparison")
	}
	raw, err := os.ReadFile("testdata/gemma3_reference.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fixture struct {
		Cases []lfm3ReferenceCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	file, err := mcfstore.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	t.Cleanup(func() { _ = file.Close() })

	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			m, err := LoadModelMCF(file, file.SectionData(mcf.SectionHFConfigJSON), len(tc.Tokens)+len(tc.Greedy)+1, LoadModelOptions{})
			if err != nil {
				t.Fatalf("LoadModelMCF: %v", err)
			}
			logits, err := m.PrefillTokens(tc.Tokens)
			if err != nil {
				t.Fatalf("PrefillTokens: %v", err)
			}
			for k, id := range tc.TopIDs {
				if d := math.Abs(float64(logits[id] - tc.TopLogits[k])); d > 2e-3 {
					t.Errorf("token %d: logit %g, reference %g", id, logits[id], tc.TopLogits[k])
				}
			}
			if got := argmaxHost(logits); got != tc.TopIDs[0] {
				t.Fatalf("argmax %d, reference %d", got, tc.TopIDs[0])
			}
			tok := tc.TopIDs[0]
			for i, want := range tc.Greedy {
				if tok != want {
					t.Fatalf("greedy token %d: got %d, reference %d", i, tok, want)
				}
				if i+1 == len(tc.Greedy) {
					break
				}
				l, err := m.ForwardToken(tok)
				if err != nil {
					t.Fatalf("ForwardToken: %v", err)
				}
				tok = argmaxHost(l)
			}
		})
	}
}
