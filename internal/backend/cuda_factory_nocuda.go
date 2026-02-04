//go:build !cuda

package backend

import "fmt"

func NewCUDA() (Backend, error) {
	return nil, fmt.Errorf("cuda backend is not available in this build")
}
