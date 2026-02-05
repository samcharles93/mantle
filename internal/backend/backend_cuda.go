//go:build cuda

package backend

import (
	"github.com/samcharles93/mantle/internal/backend/cuda"
	"github.com/samcharles93/mantle/internal/backend/simd"
)

const cudaEnabled = true

func newCPU() (Backend, error) {
	return simd.New(), nil
}

func newCUDA() (Backend, error) {
	return cuda.New()
}
