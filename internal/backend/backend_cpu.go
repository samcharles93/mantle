//go:build !cuda

package backend

import (
	"errors"

	"github.com/samcharles93/mantle/internal/backend/simd"
)

const cudaEnabled = false

var errCUDAUnavailable = errors.New("cuda backend not implemented in this build")

func newCPU() (Backend, error) {
	return simd.New(), nil
}

func newCUDA() (Backend, error) {
	return nil, errCUDAUnavailable
}
