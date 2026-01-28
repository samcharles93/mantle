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
	attn := applyRopeScaling(inv, 10_000, cfg.MaxPosition, rs)
	if math.Abs(attn-1) > 1e-9 {
		t.Fatalf("unexpected attention factor: %g", attn)
	}
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
	long := 2 * math.Pi / 9000
	short := 2 * math.Pi / 1024
	inv := []float64{long, short}
	_ = applyRopeScaling(inv, 10_000, 131072, rs)
	// Llama3 scaling reduces very long wavelengths and preserves short ones.
	if inv[0] >= long {
		t.Fatalf("long wavelength not scaled down: %g >= %g", inv[0], long)
	}
	if math.Abs(inv[1]-short) > 1e-9 {
		t.Fatalf("short wavelength changed unexpectedly: %g", inv[1])
	}
}

func TestRopeScalingYarnAttentionFactor(t *testing.T) {
	rs := &RopeScaling{
		Type:       "yarn",
		Factor:     16,
		OrigMaxCtx: 8192,
		BetaFast:   32,
		BetaSlow:   1,
	}
	orig := []float64{1, 0.5, 0.25, 0.125}
	inv := append([]float64(nil), orig...)
	attn := applyRopeScaling(inv, 10_000, 131072, rs)
	wantAttn := 1 + 0.1*math.Log(rs.Factor)
	if math.Abs(attn-wantAttn) > 1e-9 {
		t.Fatalf("attention_factor=%g want %g", attn, wantAttn)
	}
	changed := false
	for i, v := range inv {
		if math.Abs(v-orig[i]) > 1e-12 {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("expected yarn scaling to modify at least one frequency")
	}
}
