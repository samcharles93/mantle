package backend

import (
	"fmt"
	"strings"

	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/internal/model"
)

const (
	CPU  = "cpu"
	CUDA = "cuda"
	Auto = "auto"
)

type Backend interface {
	Name() string
	LoadModel(mcfFile *mcfstore.File, cfgBytes []byte, maxContext int) (model.Runtime, error)
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
