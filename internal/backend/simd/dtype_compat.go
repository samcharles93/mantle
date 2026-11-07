package simd

import instance "github.com/samcharles93/mantle/internal/backend/core"

func bf16ToF32Table(u uint16) float32 {
	return instance.BF16ToFloat32(u)
}

func fp16ToF32Table(u uint16) float32 {
	return instance.FP16ToFloat32(u)
}
