package hostcaps

import (
	"os"
	"strconv"
	"strings"
)

// Detect captures a host capability snapshot for runtime dispatch and policy.
func Detect() *Snapshot {
	weightMode := detectCUDAWeightMode()
	return &Snapshot{
		CPU:  detectCPUFeatures(),
		CUDA: detectCUDAFeatures(),
		Policy: CUDAPolicy{
			Trace:                    os.Getenv("MANTLE_CUDA_TRACE") != "",
			Fuse:                     envEnabled("MANTLE_CUDA_FUSE"),
			QuantKernel:              detectCUDAQuantKernel(weightMode),
			K4Raw:                    envEnabled("MANTLE_CUDA_K4_RAW"),
			LegacySync:               envEnabled("MANTLE_CUDA_LEGACY_SYNC"),
			Graphs:                   detectCUDAGraphsEnabled(),
			AttnSoftmax:              envEnabled("MANTLE_CUDA_ATTN_SOFTMAX"),
			DisableFFNFastPath:       envEnabled("MANTLE_CUDA_DISABLE_FFN_FASTPATH"),
			DisableQKVFastPath:       envEnabled("MANTLE_CUDA_DISABLE_QKV_FASTPATH"),
			DisableAttnInnerFastPath: envEnabled("MANTLE_CUDA_DISABLE_ATTN_INNER_FASTPATH"),
			TraceSync:                envEnabled("MANTLE_CUDA_TRACE_SYNC"),
			WeightMode:               weightMode,
		},
	}
}

func detectCUDAQuantKernel(weightMode string) bool {
	if _, ok := os.LookupEnv("MANTLE_CUDA_QUANT_KERNEL"); ok {
		return envEnabled("MANTLE_CUDA_QUANT_KERNEL")
	}
	return weightMode != "dequant"
}

func detectCUDAGraphsEnabled() bool {
	v, ok := os.LookupEnv("MANTLE_CUDA_GRAPHS")
	if !ok {
		return true
	}
	if b, err := strconv.ParseBool(v); err == nil {
		return b
	}
	switch v {
	case "0", "off", "OFF", "no", "NO", "n", "N":
		return false
	default:
		return true
	}
}

func detectCUDAWeightMode() string {
	switch strings.TrimSpace(strings.ToLower(os.Getenv("MANTLE_CUDA_WEIGHT_MODE"))) {
	case "quant":
		return "quant"
	case "dequant":
		return "dequant"
	default:
		return "auto"
	}
}

func envEnabled(name string) bool {
	v, ok := os.LookupEnv(name)
	if !ok {
		return false
	}
	if b, err := strconv.ParseBool(v); err == nil {
		return b
	}
	switch v {
	case "1", "on", "ON", "yes", "YES", "y", "Y":
		return true
	default:
		return false
	}
}
