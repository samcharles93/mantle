package simd

import "simd/archsimd"

// CPUFeatures holds detected CPU capabilities, checked once at init.
type CPUFeatures struct {
	HasAVX2       bool
	HasFMA        bool
	HasAVXVNNI    bool
	HasAVX512     bool
	HasAVX512VNNI bool
}

var cpu CPUFeatures

func init() {
	cpu.HasAVX2 = archsimd.X86.AVX2()
	cpu.HasFMA = archsimd.X86.FMA()
	cpu.HasAVXVNNI = archsimd.X86.AVXVNNI()
	cpu.HasAVX512 = archsimd.X86.AVX512()
	cpu.HasAVX512VNNI = archsimd.X86.AVX512VNNI()
}
