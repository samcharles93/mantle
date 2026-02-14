//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"os"
	"sync/atomic"
	"time"

	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/internal/mcfstore"
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
	if os.Getenv("MANTLE_CUDA_TRACE") != "" {
		logUMCapabilities()
	}
	return &Backend{}, nil
}

func (b *Backend) Name() string {
	return "cuda"
}

func (b *Backend) LoadModel(mcfFile *mcfstore.File, cfgBytes []byte, maxContext int, opts simd.LoadModelOptions) (simd.Runtime, error) {
	native.ResetManagedFallbackFlag()
	native.ResetPerfCounters()
	stream, err := native.NewStream()
	if err != nil {
		return nil, fmt.Errorf("cuda stream create failed: %w", err)
	}
	blas, err := native.NewBlasHandle(stream)
	if err != nil {
		_ = stream.Destroy()
		return nil, fmt.Errorf("cublas init failed: %w", err)
	}

	restoreQuantCache := simd.SetQuantCacheBuildEnabledForLoad(false)
	m, err := func() (*simd.Instance, error) {
		defer restoreQuantCache()
		return simd.LoadModelMCF(mcfFile, cfgBytes, maxContext, opts)
	}()
	if err != nil {
		_ = blas.Destroy()
		_ = stream.Destroy()
		return nil, err
	}

	ops := NewOps(stream, blas)
	m.SetOps(ops)
	mode := currentCUDAWeightMode()
	estimatedWeights := estimateModelWeightBytes(m, mode)
	if os.Getenv("MANTLE_CUDA_TRACE") != "" {
		graphEnv, hasGraphEnv := os.LookupEnv("MANTLE_CUDA_GRAPHS")
		graphEnvVal := "<default>"
		if hasGraphEnv {
			graphEnvVal = graphEnv
		}
		fmt.Fprintf(
			os.Stderr,
			"CUDA graph replay: enabled=%t env=MANTLE_CUDA_GRAPHS(%s)\n",
			useCUDAGraphs(),
			graphEnvVal,
		)
	}
	if freeBytes, totalBytes, memErr := native.MemInfo(); memErr == nil {
		if os.Getenv("MANTLE_CUDA_TRACE") != "" {
			fmt.Fprintf(
				os.Stderr,
				"CUDA mem preflight: free=%d MiB total=%d MiB estimated_weight=%d MiB mode=%s\n",
				freeBytes/1024/1024,
				totalBytes/1024/1024,
				estimatedWeights/1024/1024,
				cudaWeightModeString(mode),
			)
		}
		// Keep a safety headroom for KV + scratch allocations.
		if mode == cudaWeightModeDequant && estimatedWeights > int64(float64(freeBytes)*0.90) {
			_ = ops.Close()
			_ = blas.Destroy()
			_ = stream.Destroy()
			return nil, fmt.Errorf(
				"cuda preload aborted: dequant mode likely exceeds free VRAM (need ~%d MiB weights, free %d MiB). try --cuda-weight-mode quant",
				estimatedWeights/1024/1024,
				freeBytes/1024/1024,
			)
		}
	}
	preloadStart := time.Now()
	if err := ops.PreloadModelWeights(m); err != nil {
		_ = ops.Close()
		_ = blas.Destroy()
		_ = stream.Destroy()
		return nil, fmt.Errorf("cuda weight preload failed: %w", err)
	}
	if os.Getenv("MANTLE_CUDA_TRACE") != "" {
		fmt.Fprintf(os.Stderr, "CUDA preload complete in %s\n", time.Since(preloadStart).Round(time.Millisecond))
	}

	managedLog := false
	if native.ManagedFallbackUsed() {
		fmt.Fprintln(os.Stderr, "CUDA Unified Memory fallback active (device memory pressure detected)")
		managedLog = true
	}

	preloadCounters := native.GetPerfCounters()
	native.ResetPerfCounters()

	return &cudaRuntime{
		model:              m,
		ops:                ops,
		stream:             stream,
		blas:               blas,
		managedFallbackLog: managedLog,
		preloadCounters:    preloadCounters,
	}, nil
}

type cudaRuntime struct {
	model              *simd.Instance
	ops                *Ops
	stream             native.Stream
	blas               native.BlasHandle
	managedFallbackLog bool
	preloadCounters    native.PerfCounters
	tokenCount         atomic.Int64
}

func (r *cudaRuntime) ForwardToken(id int) (logits []float32, err error) {
	if r.model == nil {
		return nil, fmt.Errorf("cuda runtime is closed")
	}
	defer func() {
		if rec := recover(); rec != nil {
			err = cudaExecutionError(rec)
		}
	}()
	logits, err = r.model.ForwardToken(id)
	if err != nil {
		return nil, err
	}
	if !r.managedFallbackLog && native.ManagedFallbackUsed() {
		fmt.Fprintln(os.Stderr, "CUDA Unified Memory fallback active (device memory pressure detected)")
		r.managedFallbackLog = true
	}
	r.tokenCount.Add(1)
	return logits, nil
}

func (r *cudaRuntime) Reset() {
	if r.model == nil {
		return
	}
	r.model.Reset()
}

func (r *cudaRuntime) ModelConfig() *simd.ModelConfig {
	return r.model.ModelConfig()
}

func (r *cudaRuntime) UpdateRoPE() {
	r.model.UpdateRoPE()
}

func (r *cudaRuntime) Close() error {
	var errs []error
	if r.ops != nil {
		if e := r.ops.Close(); e != nil {
			errs = append(errs, e)
		}
		r.ops = nil
	}
	if e := r.blas.Destroy(); e != nil {
		errs = append(errs, e)
	}
	r.blas = native.BlasHandle{}
	if e := r.stream.Destroy(); e != nil {
		errs = append(errs, e)
	}
	r.stream = native.Stream{}
	r.model = nil

	if os.Getenv("MANTLE_CUDA_TRACE") != "" {
		r.emitPerfSummary()
	}

	return errors.Join(errs...)
}

func (r *cudaRuntime) emitPerfSummary() {
	decode := native.GetPerfCounters()
	toks := r.tokenCount.Load()
	fmt.Fprintf(os.Stderr, "\n=== CUDA Performance Summary ===\n")
	fmt.Fprintf(os.Stderr, "Tokens processed: %d\n", toks)
	fmt.Fprintf(os.Stderr, "-- Decode Counters --\n")
	fmt.Fprintf(os.Stderr, "MatVec calls: %d (%.1f per token)\n", decode.MatVecCalls, float64(decode.MatVecCalls)/float64(max(toks, 1)))
	fmt.Fprintf(os.Stderr, "RMSNorm calls: %d (%.1f per token)\n", decode.RMSNormCalls, float64(decode.RMSNormCalls)/float64(max(toks, 1)))
	fmt.Fprintf(os.Stderr, "StoreKV calls: %d (%.1f per token)\n", decode.StoreKVCalls, float64(decode.StoreKVCalls)/float64(max(toks, 1)))
	fmt.Fprintf(os.Stderr, "Stream syncs: %d\n", decode.StreamSyncs)
	fmt.Fprintf(os.Stderr, "Graph captures: %d\n", decode.GraphCaptures)
	fmt.Fprintf(os.Stderr, "Graph launches: %d\n", decode.GraphLaunches)
	fmt.Fprintf(os.Stderr, "Graph failures: %d\n", decode.GraphFailures)
	fmt.Fprintf(os.Stderr, "H2D bytes: %d MB\n", decode.H2DBytes/1024/1024)
	fmt.Fprintf(os.Stderr, "D2H bytes: %d MB\n", decode.D2HBytes/1024/1024)
	fmt.Fprintf(os.Stderr, "Device allocs: %d (%.1f MB)\n", decode.DeviceAllocs, float64(decode.DeviceBytes)/1024/1024)
	fmt.Fprintf(os.Stderr, "Managed allocs: %d (%.1f MB)\n", decode.ManagedAllocs, float64(decode.ManagedBytes)/1024/1024)
	fmt.Fprintf(os.Stderr, "-- Preload Counters --\n")
	fmt.Fprintf(os.Stderr, "MatVec calls: %d\n", r.preloadCounters.MatVecCalls)
	fmt.Fprintf(os.Stderr, "RMSNorm calls: %d\n", r.preloadCounters.RMSNormCalls)
	fmt.Fprintf(os.Stderr, "StoreKV calls: %d\n", r.preloadCounters.StoreKVCalls)
	fmt.Fprintf(os.Stderr, "Stream syncs: %d\n", r.preloadCounters.StreamSyncs)
	fmt.Fprintf(os.Stderr, "Graph captures: %d\n", r.preloadCounters.GraphCaptures)
	fmt.Fprintf(os.Stderr, "Graph launches: %d\n", r.preloadCounters.GraphLaunches)
	fmt.Fprintf(os.Stderr, "Graph failures: %d\n", r.preloadCounters.GraphFailures)
	fmt.Fprintf(os.Stderr, "H2D bytes: %d MB\n", r.preloadCounters.H2DBytes/1024/1024)
	fmt.Fprintf(os.Stderr, "D2H bytes: %d MB\n", r.preloadCounters.D2HBytes/1024/1024)
	fmt.Fprintf(os.Stderr, "Device allocs: %d (%.1f MB)\n", r.preloadCounters.DeviceAllocs, float64(r.preloadCounters.DeviceBytes)/1024/1024)
	fmt.Fprintf(os.Stderr, "Managed allocs: %d (%.1f MB)\n", r.preloadCounters.ManagedAllocs, float64(r.preloadCounters.ManagedBytes)/1024/1024)
	fmt.Fprintf(os.Stderr, "================================\n")
}

func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func logUMCapabilities() {
	concurrent, err1 := native.DeviceGetAttribute(native.DevAttrConcurrentManagedAccess, 0)
	pageable, err2 := native.DeviceGetAttribute(native.DevAttrPageableMemoryAccess, 0)
	coherent, err3 := native.DeviceGetAttribute(native.DevAttrPageableMemoryAccessUsesHostPageTables, 0)
	if err1 != nil || err2 != nil || err3 != nil {
		fmt.Fprintf(os.Stderr, "warning: failed to query CUDA UM device attributes (concurrent=%v pageable=%v coherent=%v)\n", err1, err2, err3)
		return
	}
	fmt.Fprintln(os.Stderr, "CUDA Device 0 UM Capabilities:")
	fmt.Fprintf(os.Stderr, "  ConcurrentManagedAccess: %d\n", concurrent)
	fmt.Fprintf(os.Stderr, "  PageableMemoryAccess: %d\n", pageable)
	fmt.Fprintf(os.Stderr, "  PageableMemoryAccessUsesHostPageTables: %d\n", coherent)
}
