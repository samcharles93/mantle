package quant

import "infer/pkg/mcf"

type QuantScheme interface {
	Name() string
	Quantise(t mcf.TensorEntry) (QuantTensor, error)
}

type QuantTensor struct {
	BlockSize int
	Scales    []float32
	Zeroes    []int8
	Data      []byte
}
