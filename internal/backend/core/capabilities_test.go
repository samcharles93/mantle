package core

import "testing"

func TestCapabilitiesSupports(t *testing.T) {
	c := &Capabilities{}
	if c.Supports(CapFusedFFN) {
		t.Fatalf("empty Capabilities should not support CapFusedFFN")
	}
	c.Set(CapFusedFFN, true)
	if !c.Supports(CapFusedFFN) {
		t.Fatalf("capability set was not observed")
	}
	c.Set(CapFusedFFN, false)
	if c.Supports(CapFusedFFN) {
		t.Fatalf("capability should have been cleared")
	}
}

func TestDefaultCapabilities(t *testing.T) {
	d := DefaultCapabilities()
	if !d.Supports(CapGraphCompute) {
		t.Fatalf("DefaultCapabilities must support CapGraphCompute")
	}
	if d.Supports(CapFusedFFN) || d.Supports(CapFusedAttention) || d.Supports(CapFusedMoE) {
		t.Fatalf("DefaultCapabilities must not include fused ops")
	}
}

func TestSIMDCapabilities(t *testing.T) {
	s := SIMDCapabilities()
	if !s.Supports(CapGraphCompute) {
		t.Fatalf("SIMDCapabilities must support CapGraphCompute")
	}
	if !s.Supports(CapFusedFFN) || !s.Supports(CapFusedAttention) || !s.Supports(CapFusedNormResidual) {
		t.Fatalf("SIMDCapabilities must include SIMD fused ops")
	}
	if s.Supports(CapFusedMoE) {
		t.Fatalf("SIMDCapabilities should not include MoE by default")
	}
}

func TestCUDACapabilities(t *testing.T) {
	c := CUDACapabilities()
	if !c.Supports(CapGraphCompute) || !c.Supports(CapGraphCapture) {
		t.Fatalf("CUDACapabilities must support graph compute and capture")
	}
	if !c.Supports(CapFusedFFN) || !c.Supports(CapFusedAttention) || !c.Supports(CapFusedNormResidual) || !c.Supports(CapFusedMoE) {
		t.Fatalf("CUDACapabilities must include all fused ops")
	}
}
