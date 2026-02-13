package backend

import (
	"fmt"
	"strings"

	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/internal/mcfstore"
)

const (
	CPU  = "cpu"
	CUDA = "cuda"
	Auto = "auto"
)

type Backend interface {
	Name() string
	LoadModel(mcfFile *mcfstore.File, cfgBytes []byte, maxContext int, opts simd.LoadModelOptions) (simd.Runtime, error)
}

func Has(name string) bool {
	switch name {
	case CPU:
		return true
	case CUDA:
		return cudaEnabled
	default:
		return false
	}
}

func New(name string) (Backend, error) {
	normalized, err := Normalize(name)
	if err != nil {
		return nil, err
	}

	switch normalized {
	case CPU:
		return newCPU()
	case CUDA:
		return newCUDA()
	default:
		return nil, fmt.Errorf("unknown backend %q", normalized)
	}
}

// Available returns a comma-separated list of available backends.
func Available() string {
	entries := []string{CPU}
	if Has(CUDA) {
		entries = append(entries, CUDA)
	}
	return strings.Join(entries, ",")
}

func Normalize(name string) (string, error) {
	backend := strings.ToLower(strings.TrimSpace(name))
	if backend == "" {
		return Auto, nil
	}
	switch backend {
	case CPU, CUDA, Auto:
		return backend, nil
	default:
		return "", fmt.Errorf("unknown backend %q (expected auto, cpu, or cuda)", backend)
	}
}
