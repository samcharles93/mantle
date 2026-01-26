package tensor

import (
	"fmt"
	"infer/internal/gguf"
	"infer/internal/safetensors"
)

// LoadGGUFMat loads a 2D matrix from a GGUF file.
func LoadGGUFMat(f *gguf.File, name string) (*Mat, error) {
	data, dims, err := gguf.ReadTensorF32(f, name)
	if err != nil {
		return nil, err
	}
	if len(dims) != 2 {
		return nil, fmt.Errorf("%s: expected 2D tensor", name)
	}
	r := int(dims[1])
	c := int(dims[0])
	if r*c != len(data) {
		return nil, fmt.Errorf("%s: size mismatch", name)
	}
	return &Mat{R: r, C: c, Stride: c, Data: data}, nil
}

// LoadGGUFVec loads a 1D vector from a GGUF file.
func LoadGGUFVec(f *gguf.File, name string) ([]float32, error) {
	data, dims, err := gguf.ReadTensorF32(f, name)
	if err != nil {
		return nil, err
	}
	if len(dims) != 1 {
		return nil, fmt.Errorf("%s: expected 1D tensor", name)
	}
	return data, nil
}

// LoadSafetensorsMat loads a 2D matrix from a Safetensors file.
func LoadSafetensorsMat(st *safetensors.File, name string) (*Mat, error) {
	data, info, err := st.ReadTensorF32(name)
	if err != nil {
		return nil, err
	}
	if len(info.Shape) != 2 {
		return nil, fmt.Errorf("%s: expected 2D tensor", name)
	}
	r := info.Shape[0]
	c := info.Shape[1]
	if r*c != len(data) {
		return nil, fmt.Errorf("%s: size mismatch", name)
	}
	return &Mat{R: r, C: c, Stride: c, Data: data}, nil
}

// LoadSafetensorsVec loads a 1D vector from a Safetensors file.
func LoadSafetensorsVec(st *safetensors.File, name string) ([]float32, error) {
	data, info, err := st.ReadTensorF32(name)
	if err != nil {
		return nil, err
	}
	if len(info.Shape) != 1 {
		return nil, fmt.Errorf("%s: expected 1D tensor", name)
	}
	return data, nil
}

// LoadSafetensorsConvKernel loads a generic convolution kernel (typically 3D [out, 1, k])
// and flattens it to a 2D matrix [out, k].
func LoadSafetensorsConvKernel(st *safetensors.File, name string) (*Mat, error) {
	data, info, err := st.ReadTensorF32(name)
	if err != nil {
		return nil, err
	}
	if len(info.Shape) != 3 {
		return nil, fmt.Errorf("%s: expected 3D tensor", name)
	}
	out := info.Shape[0]
	in := info.Shape[1]
	k := info.Shape[2]
	if in != 1 {
		return nil, fmt.Errorf("%s: expected in=1, got %d", name, in)
	}
	if out*k != len(data) {
		return nil, fmt.Errorf("%s: size mismatch", name)
	}
	return &Mat{R: out, C: k, Stride: k, Data: data}, nil
}
