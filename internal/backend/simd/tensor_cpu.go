package simd

import "simd/archsimd"

// CPUFeatures holds detected CPU capabilities, checked once at init.
type CPUFeatures struct {
	HasAVX2 bool
}

var cpu CPUFeatures

func init() {
	cpu.HasAVX2 = archsimd.X86.AVX2()
}
