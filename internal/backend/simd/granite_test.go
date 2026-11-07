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

// TestSplitFusedGateUp checks the fused gate+up split used for
// GraniteMoeHybrid's shared_mlp input_linear: gate is the first row half, up
// the second, and both views read the fused backing storage correctly for f32
// and bf16-raw matrices.
func TestSplitFusedGateUp(t *testing.T) {
	const rows, cols = 8, 4 // fused [8,4] -> gate/up [4,4]
	f32 := core.NewMat(rows, cols)
	for i := range f32.Data {
		f32.Data[i] = float32(i)
	}
	gate, up := splitFusedGateUp(&f32)
	for i := 0; i < 4; i++ {
		g := make([]float32, cols)
		u := make([]float32, cols)
		gate.RowTo(g, i)
		up.RowTo(u, i)
		for c := 0; c < cols; c++ {
			if g[c] != float32((i*cols + c)) {
				t.Fatalf("gate row %d col %d = %v, want %v", i, c, g[c], float32(i*cols+c))
			}
			if u[c] != float32(((4+i)*cols + c)) {
				t.Fatalf("up row %d col %d = %v, want %v", i, c, u[c], float32((4+i)*cols+c))
			}
		}
	}

	// bf16-raw variant: same split over Raw bytes.
	raw := make([]byte, rows*cols*2)
	for i := range raw {
		raw[i] = byte(i)
	}
	bf := &core.Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeBF16, Raw: raw}
	gate, up = splitFusedGateUp(bf)
	if len(gate.Raw) != 4*cols*2 || len(up.Raw) != 4*cols*2 {
		t.Fatalf("raw split sizes: gate %d up %d", len(gate.Raw), len(up.Raw))
	}
	g := make([]float32, cols)
	gate.RowTo(g, 0)
	if g[0] != core.BF16ToFloat32(uint16(raw[0])|uint16(raw[1])<<8) {
		t.Fatalf("bf16 gate row0 col0 = %v", g[0])
	}
}

// TestGraniteReferenceLogits compares granite-4.0-350M (granitemoehybrid) against
// HF transformers logits (testdata/granite_reference.json): last-position top-10
// plus a greedy continuation.
func TestGraniteReferenceLogits(t *testing.T) {
	path := os.Getenv("MANTLE_GRANITE_MCF")
	if path == "" {
		t.Skip("set MANTLE_GRANITE_MCF to the granite-4.0-350M container to run the reference comparison")
	}
	raw, err := os.ReadFile("testdata/granite_reference.json")
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
