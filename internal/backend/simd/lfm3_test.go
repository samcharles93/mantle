package simd

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	core "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// TestApplyAttnOutputGatePerHead checks the LFM3 gate shape: a [heads, hidden]
// projection whose sigmoid scales every element of its head.
func TestApplyAttnOutputGatePerHead(t *testing.T) {
	const heads, headDim, hidden = 3, 4, 5
	gateW := core.NewMat(heads, hidden)
	for i := range gateW.Data {
		gateW.Data[i] = float32(i%7)*0.1 - 0.3
	}
	x := []float32{0.5, -1, 0.25, 2, -0.75}
	m := &Instance{}
	m.Scratch.AttnGate = make([]float32, 2*heads*headDim)
	layer := &Layer{AttnGate: &gateW}
	attnOut := make([]float32, heads*headDim)
	for i := range attnOut {
		attnOut[i] = float32(i) - 5
	}
	want := append([]float32(nil), attnOut...)
	for h := range heads {
		var z float64
		for c := range hidden {
			z += float64(gateW.Data[h*hidden+c]) * float64(x[c])
		}
		s := 1 / (1 + math.Exp(-z))
		for d := range headDim {
			want[h*headDim+d] = float32(float64(want[h*headDim+d]) * s)
		}
	}

	applyAttnOutputGate(m, &DefaultOps{}, layer, x, nil, attnOut, heads, headDim)
	for i := range want {
		if math.Abs(float64(attnOut[i]-want[i])) > 1e-5 {
			t.Fatalf("element %d: got %g, want %g", i, attnOut[i], want[i])
		}
	}
}

type lfm3ReferenceCase struct {
	Name      string    `json:"name"`
	Tokens    []int     `json:"tokens"`
	TopIDs    []int     `json:"top_ids"`
	TopLogits []float32 `json:"top_logits"`
	Greedy    []int     `json:"greedy"`
}

// TestLFM3ReferenceLogits compares LFM3-1.7B against logits from the official
// modeling_lfm3.py (testdata/lfm3_reference.json): the last-position top-10
// for a chat prompt and for a 459-token prompt that runs past the 256-token
// sliding window, plus 40 greedy tokens for the chat prompt.
func TestLFM3ReferenceLogits(t *testing.T) {
	path := os.Getenv("MANTLE_LFM3_MCF")
	if path == "" {
		t.Skip("set MANTLE_LFM3_MCF to the LFM3-1.7B container to run the reference comparison")
	}
	raw, err := os.ReadFile("testdata/lfm3_reference.json")
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
