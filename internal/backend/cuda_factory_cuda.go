//go:build cuda

package backend

import "github.com/samcharles93/mantle/internal/backend/cuda"

func NewCUDA() (Backend, error) {
	return cuda.New()
}
