//go:build goexperiment.simd

package simd

import (
	"os"
	"sync"
	"sync/atomic"
)

type HostPerfCounters struct {
	BF16RoundCalls       int64
	BF16RoundElems       int64
	RMSNormWeightedCalls int64
	RMSNormWeightedElems int64
	RMSNormUnitCalls     int64
	RMSNormUnitElems     int64
	SoftmaxCalls         int64
	SoftmaxElems         int64
	TopKCalls            int64
}

var hostPerfCounters HostPerfCounters
var hostPerfEnabledOnce sync.Once
var hostPerfEnabledCached bool

func hostPerfEnabled() bool {
	hostPerfEnabledOnce.Do(func() {
		hostPerfEnabledCached = os.Getenv("MANTLE_CUDA_TRACE") != ""
	})
	return hostPerfEnabledCached
}

func ResetHostPerfCounters() {
	hostPerfCounters = HostPerfCounters{}
}

func GetHostPerfCounters() HostPerfCounters {
	return HostPerfCounters{
		BF16RoundCalls:       atomic.LoadInt64(&hostPerfCounters.BF16RoundCalls),
		BF16RoundElems:       atomic.LoadInt64(&hostPerfCounters.BF16RoundElems),
		RMSNormWeightedCalls: atomic.LoadInt64(&hostPerfCounters.RMSNormWeightedCalls),
		RMSNormWeightedElems: atomic.LoadInt64(&hostPerfCounters.RMSNormWeightedElems),
		RMSNormUnitCalls:     atomic.LoadInt64(&hostPerfCounters.RMSNormUnitCalls),
		RMSNormUnitElems:     atomic.LoadInt64(&hostPerfCounters.RMSNormUnitElems),
		SoftmaxCalls:         atomic.LoadInt64(&hostPerfCounters.SoftmaxCalls),
		SoftmaxElems:         atomic.LoadInt64(&hostPerfCounters.SoftmaxElems),
		TopKCalls:            atomic.LoadInt64(&hostPerfCounters.TopKCalls),
	}
}

func recordHostBF16Round(n int) {
	if !hostPerfEnabled() {
		return
	}
	atomic.AddInt64(&hostPerfCounters.BF16RoundCalls, 1)
	atomic.AddInt64(&hostPerfCounters.BF16RoundElems, int64(n))
}

func recordHostRMSNormWeighted(n int) {
	if !hostPerfEnabled() {
		return
	}
	atomic.AddInt64(&hostPerfCounters.RMSNormWeightedCalls, 1)
	atomic.AddInt64(&hostPerfCounters.RMSNormWeightedElems, int64(n))
}

func recordHostRMSNormUnit(n int) {
	if !hostPerfEnabled() {
		return
	}
	atomic.AddInt64(&hostPerfCounters.RMSNormUnitCalls, 1)
	atomic.AddInt64(&hostPerfCounters.RMSNormUnitElems, int64(n))
}

func recordHostSoftmax(n int) {
	if !hostPerfEnabled() {
		return
	}
	atomic.AddInt64(&hostPerfCounters.SoftmaxCalls, 1)
	atomic.AddInt64(&hostPerfCounters.SoftmaxElems, int64(n))
}

func recordHostTopK() {
	if !hostPerfEnabled() {
		return
	}
	atomic.AddInt64(&hostPerfCounters.TopKCalls, 1)
}
