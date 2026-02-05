//go:build cuda

package cuda

import (
	"fmt"

	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/pkg/mcf"
)

type Backend struct{}

func New() (*Backend, error) {
	count, err := native.DeviceCount()
	if err != nil {
		return nil, fmt.Errorf("cuda device query failed: %w", err)
	}
	if count < 1 {
		return nil, fmt.Errorf("no cuda devices detected")
	}
	return &Backend{}, nil
}

func (b *Backend) Name() string {
	return "cuda"
}

func (b *Backend) LoadModel(mcfFile *mcfstore.File, cfgBytes []byte, maxContext int) (simd.Runtime, error) {
	stream, err := native.NewStream()
	if err != nil {
		return nil, fmt.Errorf("cuda stream create failed: %w", err)
	}
	blas, err := native.NewBlasHandle(stream)
	if err != nil {
		_ = stream.Destroy()
		return nil, fmt.Errorf("cublas init failed: %w", err)
	}

	m, err := simd.LoadModelMCF(mcfFile, cfgBytes, maxContext)
	if err != nil {
		_ = blas.Destroy()
		_ = stream.Destroy()
		return nil, err
	}
	if err := validateNoQuant(m); err != nil {
		_ = blas.Destroy()
		_ = stream.Destroy()
		return nil, err
	}

	ops := NewOps(stream, blas)
	m.SetOps(ops)

	return &cudaRuntime{model: m, ops: ops, stream: stream, blas: blas}, nil
}

type cudaRuntime struct {
	model  *simd.Instance
	ops    *Ops
	stream native.Stream
	blas   native.BlasHandle
}

func (r *cudaRuntime) ForwardToken(id int) ([]float32, error) {
	return r.model.ForwardToken(id)
}

func (r *cudaRuntime) Reset() {
	r.model.Reset()
}

func (r *cudaRuntime) ModelConfig() *simd.ModelConfig {
	return r.model.ModelConfig()
}

func (r *cudaRuntime) UpdateRoPE() {
	r.model.UpdateRoPE()
}

func (r *cudaRuntime) Close() error {
	var err error
	if r.ops != nil {
		if e := r.ops.Close(); e != nil {
			err = e
		}
	}
	if e := r.blas.Destroy(); e != nil && err == nil {
		err = e
	}
	if e := r.stream.Destroy(); e != nil && err == nil {
		err = e
	}
	return err
}

func validateNoQuant(m *simd.Instance) error {
	check := func(name string, mat *simd.Mat) error {
		if mat == nil {
			return nil
		}
		if mcf.DTypeRequiresAligned64(mat.DType) {
			return fmt.Errorf("cuda backend does not support quantized weights (%s dtype=0x%04x)", name, uint16(mat.DType))
		}
		return nil
	}

	if err := check("embeddings", m.Embeddings); err != nil {
		return err
	}
	if err := check("output", m.Output); err != nil {
		return err
	}
	for i := range m.Layers {
		layer := &m.Layers[i]
		if err := check("layer.wq", layer.Wq); err != nil {
			return err
		}
		if err := check("layer.wk", layer.Wk); err != nil {
			return err
		}
		if err := check("layer.wv", layer.Wv); err != nil {
			return err
		}
		if err := check("layer.wo", layer.Wo); err != nil {
			return err
		}
		if err := check("layer.attn_gate", layer.AttnGate); err != nil {
			return err
		}
		if err := check("layer.shortconv_kernel", layer.ShortConvKernel); err != nil {
			return err
		}
		if err := check("layer.shortconv_in", layer.ShortConvInProj); err != nil {
			return err
		}
		if err := check("layer.shortconv_out", layer.ShortConvOutProj); err != nil {
			return err
		}
		if err := check("layer.ffn_up", layer.FfnUp); err != nil {
			return err
		}
		if err := check("layer.ffn_gate", layer.FfnGate); err != nil {
			return err
		}
		if err := check("layer.ffn_down", layer.FfnDown); err != nil {
			return err
		}
		if layer.MoE != nil {
			if err := check("layer.moe.router", layer.MoE.Router); err != nil {
				return err
			}
			if err := check("layer.moe.shared_up", layer.MoE.Shared.Up); err != nil {
				return err
			}
			if err := check("layer.moe.shared_gate", layer.MoE.Shared.Gate); err != nil {
				return err
			}
			if err := check("layer.moe.shared_down", layer.MoE.Shared.Down); err != nil {
				return err
			}
			for j := range layer.MoE.Experts {
				ex := &layer.MoE.Experts[j]
				if err := check("layer.moe.expert_up", ex.Up); err != nil {
					return err
				}
				if err := check("layer.moe.expert_gate", ex.Gate); err != nil {
					return err
				}
				if err := check("layer.moe.expert_down", ex.Down); err != nil {
					return err
				}
			}
		}
	}
	return nil
}
