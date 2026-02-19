//go:build cuda

package native

import (
	"os"
	"sync"
	"sync/atomic"
)

// PerfCounters holds CUDA backend performance statistics.
type PerfCounters struct {
	MatVecCalls            int64
	MatVecCPUFallbackCalls int64
	RMSNormCalls           int64
	StoreKVCalls           int64
	StreamSyncs            int64
	GraphCaptures          int64
	GraphLaunches          int64
	GraphFailures          int64
	FlushIfPendingCalls    int64
	H2DBytes               int64
	D2HBytes               int64
	ManagedAllocs          int64
	ManagedBytes           int64
	DeviceAllocs           int64
	DeviceBytes            int64
}

var globalPerfCounters PerfCounters
var perfEnabledOnce sync.Once
var perfEnabledCached bool

// GetPerfCounters returns a copy of current counters.
func GetPerfCounters() PerfCounters {
	return PerfCounters{
		MatVecCalls:            globalPerfCounters.MatVecCalls,
		MatVecCPUFallbackCalls: globalPerfCounters.MatVecCPUFallbackCalls,
		RMSNormCalls:           globalPerfCounters.RMSNormCalls,
		StoreKVCalls:           globalPerfCounters.StoreKVCalls,
		StreamSyncs:            globalPerfCounters.StreamSyncs,
		GraphCaptures:          globalPerfCounters.GraphCaptures,
		GraphLaunches:          globalPerfCounters.GraphLaunches,
		GraphFailures:          globalPerfCounters.GraphFailures,
		FlushIfPendingCalls:    globalPerfCounters.FlushIfPendingCalls,
		H2DBytes:               globalPerfCounters.H2DBytes,
		D2HBytes:               globalPerfCounters.D2HBytes,
		ManagedAllocs:          globalPerfCounters.ManagedAllocs,
		ManagedBytes:           globalPerfCounters.ManagedBytes,
		DeviceAllocs:           globalPerfCounters.DeviceAllocs,
		DeviceBytes:            globalPerfCounters.DeviceBytes,
	}
}

// ResetPerfCounters resets all counters to zero.
func ResetPerfCounters() {
	globalPerfCounters = PerfCounters{}
}

// perfEnabled returns true if MANTLE_CUDA_TRACE env var is set.
func perfEnabled() bool {
	perfEnabledOnce.Do(func() {
		perfEnabledCached = os.Getenv("MANTLE_CUDA_TRACE") != ""
	})
	return perfEnabledCached
}

func recordMatVec() {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.MatVecCalls, 1)
	}
}

func recordMatVecCPUFallback() {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.MatVecCPUFallbackCalls, 1)
	}
}

func recordRMSNorm() {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.RMSNormCalls, 1)
	}
}

func recordStoreKV() {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.StoreKVCalls, 1)
	}
}

func recordStreamSync() {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.StreamSyncs, 1)
	}
}

func recordGraphCapture() {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.GraphCaptures, 1)
	}
}

func recordGraphLaunch() {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.GraphLaunches, 1)
	}
}

func recordGraphFailure() {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.GraphFailures, 1)
	}
}

func recordFlushIfPending() {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.FlushIfPendingCalls, 1)
	}
}

func recordH2D(bytes int64) {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.H2DBytes, bytes)
	}
}

func recordD2H(bytes int64) {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.D2HBytes, bytes)
	}
}

func recordDeviceAlloc(bytes int64, managed bool) {
	if !perfEnabled() {
		return
	}
	if managed {
		atomic.AddInt64(&globalPerfCounters.ManagedAllocs, 1)
		atomic.AddInt64(&globalPerfCounters.ManagedBytes, bytes)
	} else {
		atomic.AddInt64(&globalPerfCounters.DeviceAllocs, 1)
		atomic.AddInt64(&globalPerfCounters.DeviceBytes, bytes)
	}
}

// RecordMatVec records a MatVec operation.
func RecordMatVec() { recordMatVec() }

// RecordMatVecCPUFallback records a MatVec operation that fell back to CPU.
func RecordMatVecCPUFallback() { recordMatVecCPUFallback() }

// RecordRMSNorm records an RMSNorm operation.
func RecordRMSNorm() { recordRMSNorm() }

// RecordStoreKV records a StoreKV operation.
func RecordStoreKV() { recordStoreKV() }

// RecordGraphCapture records a CUDA Graph capture event.
func RecordGraphCapture() { recordGraphCapture() }

// RecordGraphLaunch records a CUDA Graph launch event.
func RecordGraphLaunch() { recordGraphLaunch() }

// RecordGraphFailure records a CUDA Graph capture/launch failure.
func RecordGraphFailure() { recordGraphFailure() }

// RecordFlushIfPending records a forced D2H flush triggered by buffer reuse.
func RecordFlushIfPending() { recordFlushIfPending() }
