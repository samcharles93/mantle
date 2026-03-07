package simd

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/samcharles93/mantle/internal/model"
	"github.com/samcharles93/mantle/pkg/mcf"
)

type stubTensorSource struct {
	tensors map[string]tensorPayload
}

func (s stubTensorSource) ReadTensor(name string) (tensorPayload, error) {
	p, ok := s.tensors[name]
	if !ok {
		return tensorPayload{}, errors.New("missing tensor")
	}
	return p, nil
}

func (s stubTensorSource) TensorShape(name string) ([]int, bool) {
	p, ok := s.tensors[name]
	if !ok {
		return nil, false
	}
	shape := make([]int, len(p.Shape))
	copy(shape, p.Shape)
	return shape, true
}

func f32Raw(vals ...float32) []byte {
	raw := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	return raw
}

func TestLoadAttentionQAndGateSplitsFusedQProj(t *testing.T) {
	src := stubTensorSource{
		tensors: map[string]tensorPayload{
			"q_proj": {
				DType: mcf.DTypeF32,
				Shape: []int{8, 2},
				Raw: f32Raw(
					1, 2,
					3, 4,
					5, 6,
					7, 8,
					9, 10,
					11, 12,
					13, 14,
					15, 16,
				),
			},
		},
	}

	wq, gate, err := loadAttentionQAndGate(src, &model.HFConfig{AttnOutputGate: true}, "q_proj", 4, 2, 2)
	if err != nil {
		t.Fatalf("loadAttentionQAndGate: %v", err)
	}
	if gate == nil {
		t.Fatalf("expected fused attention gate to be split out")
	}
	if wq.R != 4 || wq.C != 2 {
		t.Fatalf("wq shape=%dx%d want 4x2", wq.R, wq.C)
	}
	if gate.R != 4 || gate.C != 2 {
		t.Fatalf("gate shape=%dx%d want 4x2", gate.R, gate.C)
	}
	if got := wq.Data; !equalFloat32s(got, []float32{1, 2, 3, 4, 9, 10, 11, 12}) {
		t.Fatalf("wq data=%v", got)
	}
	if got := gate.Data; !equalFloat32s(got, []float32{5, 6, 7, 8, 13, 14, 15, 16}) {
		t.Fatalf("gate data=%v", got)
	}
}

func equalFloat32s(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
