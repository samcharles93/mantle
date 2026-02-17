//go:build !cuda

package hostcaps

func detectCUDAFeatures() CUDAFeatures {
	return CUDAFeatures{}
}
