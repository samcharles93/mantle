package model

import (
	"math"
	"testing"
)

func TestRopeScalingLinear(t *testing.T) {
	cfg := &hfConfig{
		MaxPosition: 4096,
		RopeScaling: &ropeScaling{
			Type:   "linear",
			Factor: 2,
		},
	}
	rs := ropeScalingForConfig(cfg)
	if rs == nil {
		t.Fatalf("expected rope scaling")
	}
	if rs.Type != "linear" {
		t.Fatalf("unexpected rope scaling type: %q", rs.Type)
	}
	inv := []float64{1, 0.5, 0.25}
	applyRopeScaling(inv, cfg.MaxPosition, rs)
	want := []float64{0.5, 0.25, 0.125}
	for i := range inv {
		if math.Abs(inv[i]-want[i]) > 1e-9 {
			t.Fatalf("inv[%d]=%g want %g", i, inv[i], want[i])
		}
	}
}

func TestRopeScalingLlama3Band(t *testing.T) {
	rs := &RopeScaling{
		Type:       "llama3",
		Factor:     4,
		OrigMaxCtx: 8192,
		LowFactor:  1,
		HighFactor: 4,
	}
	inv := []float64{2 * math.Pi / 9000, 2 * math.Pi / 1024}
	applyRopeScaling(inv, 131072, rs)
	// Long wavelength stays close to original; short wavelength is scaled down.
	if math.Abs(inv[0]-(2*math.Pi/9000)) > 1e-9 {
		t.Fatalf("long wavelength changed unexpectedly: %g", inv[0])
	}
	if inv[1] >= 2*math.Pi/1024 {
		t.Fatalf("short wavelength not scaled: %g", inv[1])
	}
}
