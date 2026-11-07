package simd

import instance "github.com/samcharles93/mantle/internal/backend/core"

func Float32ToFloat16Slice(src []float32, dst []uint16) {
	for i, v := range src {
		dst[i] = instance.Float32ToFloat16(v)
	}
}
