package simd

import (
	"errors"
	"strings"
	"testing"
)

type failingFastPathOps struct {
	err error
}

func (o *failingFastPathOps) MatVec(dst []float32, w *Mat, x []float32) {
	for r := range w.R {
		if r >= len(dst) {
			break
		}
		row := w.Data[r*w.Stride : r*w.Stride+w.C]
		var sum float32
		for c := range w.C {
			sum += row[c] * x[c]
		}
		dst[r] = sum
	}
}

func (o *failingFastPathOps) MatVecWithQuant(dst []float32, w *Mat, x []float32, _ *QuantVec) {
	o.MatVec(dst, w, x)
}

func (o *failingFastPathOps) RMSNorm(dst, src, _ []float32, _ float32) {
	copy(dst, src)
}

func (o *failingFastPathOps) Softmax(_ []float32) {}

func (o *failingFastPathOps) ApplyRoPE(_ []float32, _, _, _ int, _ []float64, _ float32) {}

func (o *failingFastPathOps) StoreKV(_ int, _ int, _ int, _ []float32, _ []float32, _ []uint16, _ []uint16, _ []int8, _ []int8, _ []float32, _ []float32, _ []float32, _ []float32) {
}

func (o *failingFastPathOps) BeginToken(_ []float32) {}

func (o *failingFastPathOps) EndToken(_ []float32) {}

func (o *failingFastPathOps) HostStateDirty(_ []float32) {}

func (o *failingFastPathOps) SyncHostState(_ []float32) {}

func (o *failingFastPathOps) DeviceAdd(_, _ []float32) bool { return false }

func (o *failingFastPathOps) DeviceRMSNorm(dst, src, _ []float32, _ float32) bool {
	copy(dst, src)
	return true
}

func (o *failingFastPathOps) DeviceMatVec(_ []float32, _ *Mat, _ []float32) bool { return false }

func (o *failingFastPathOps) DeviceMatVecNoCopy(_ *Mat, _ []float32) bool { return false }

func (o *failingFastPathOps) DeviceArgMaxLastResult() (idx int, ok bool) { return 0, false }

func (o *failingFastPathOps) ConsumeFastPathError() error {
	err := o.err
	o.err = nil
	return err
}

func tinyRuntimeInstance(ops Ops) *Instance {
	emb := NewMatFromData(3, 2, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	out := NewMatFromData(3, 2, []float32{
		1, 0,
		0, 1,
		1, 1,
	})

	return &Instance{
		Config: &ModelConfig{
			Config: Config{
				VocabSize:        3,
				EmbeddingLength:  2,
				LMHeadMultiplier: 1,
			},
		},
		Embeddings: &emb,
		OutputNorm: []float32{1, 1},
		Output:     &out,
		MaxContext: 16,
		RMSEpsilon: 1e-5,
		Scratch: ScratchBuffers{
			X:      make([]float32, 2),
			Tmp:    make([]float32, 2),
			Tmp2:   make([]float32, 2),
			Logits: make([]float32, 3),
		},
		ops: ops,
	}
}

func TestForwardTokenFailsOnFastPathError(t *testing.T) {
	sentinel := errors.New("device fast path failed")
	ops := &failingFastPathOps{err: sentinel}
	m := tinyRuntimeInstance(ops)

	_, err := m.ForwardToken(0)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if !errors.Is(err, sentinel) {
		t.Fatalf("expected wrapped sentinel error, got %v", err)
	}
	if !strings.Contains(err.Error(), "output head fast path failed") {
		t.Fatalf("unexpected error context: %v", err)
	}
}

func TestForwardTokenGreedyFailsOnFastPathError(t *testing.T) {
	sentinel := errors.New("greedy fast path failed")
	ops := &failingFastPathOps{err: sentinel}
	m := tinyRuntimeInstance(ops)

	_, err := m.ForwardTokenGreedy(0)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if !errors.Is(err, sentinel) {
		t.Fatalf("expected wrapped sentinel error, got %v", err)
	}
	if !strings.Contains(err.Error(), "output head fast path failed") {
		t.Fatalf("unexpected error context: %v", err)
	}
}

func TestForwardTokenFallsBackWhenNoFastPathError(t *testing.T) {
	ops := &failingFastPathOps{}
	m := tinyRuntimeInstance(ops)

	logits, err := m.ForwardToken(0)
	if err != nil {
		t.Fatalf("expected fallback success, got error: %v", err)
	}
	if len(logits) != 3 {
		t.Fatalf("unexpected logits length: %d", len(logits))
	}
}
