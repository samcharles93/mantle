//go:build !cuda

package backend

import (
	"errors"

	"github.com/samcharles93/mantle/internal/backend/simd"
)

var errCUDAUnavailable = errors.New("cuda backend unavailable in this binary (built without cuda support)")

func newCPU() (Backend, error) {
	return simd.New(), nil
}

func newCUDA() (Backend, error) {
	return nil, errCUDAUnavailable
}
