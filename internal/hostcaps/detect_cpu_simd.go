package hostcaps

import "simd/archsimd"

func detectCPUFeatures() CPUFeatures {
	return CPUFeatures{
		HasAVX2:       archsimd.X86.AVX2(),
		HasAVX512:     archsimd.X86.AVX512(),
		HasFMA:        archsimd.X86.FMA(),
		HasAVXVNNI:    archsimd.X86.AVXVNNI(),
		HasAVX512VNNI: archsimd.X86.AVX512VNNI(),
	}
}
