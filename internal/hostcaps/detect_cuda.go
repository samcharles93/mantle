//go:build cuda

package hostcaps

import "github.com/samcharles93/mantle/internal/backend/cuda/native"

func detectCUDAFeatures() CUDAFeatures {
	count, err := native.DeviceCount()
	if err != nil {
		return CUDAFeatures{CompiledWithCUDA: true}
	}
	return CUDAFeatures{
		CompiledWithCUDA: true,
		HasCUDADevice:    count > 0,
		CUDADeviceCount:  count,
	}
}
