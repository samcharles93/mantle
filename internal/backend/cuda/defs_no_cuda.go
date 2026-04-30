//go:build !cuda

package cuda

import (
	model "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/cuda/native"
)

// Minimal stubs to satisfy non-cuda analysis (LSP / tools). These are
// build-tagged !cuda so they are only used when CUDA is not enabled.

type Ops struct{}

type deviceAttnCache struct{}

type convDeviceState struct{}

type ropeInvFreqDevEntry struct {
	buf  native.DeviceBuffer
	half int
}

type cudaWeightMode int

const (
	cudaWeightModeAuto cudaWeightMode = iota
	cudaWeightModeQuant
	cudaWeightModeDequant
)

func useQuantKernel() bool                                         { return false }
func shouldPreferQuantWeights(_ *model.Mat, _ cudaWeightMode) bool { return false }
func currentCUDAWeightMode() cudaWeightMode                        { return cudaWeightModeAuto }
func mulInt(a, b int) (int, bool)                                  { return a * b, true }
func useAttentionInnerFastPath() bool                              { return false }
