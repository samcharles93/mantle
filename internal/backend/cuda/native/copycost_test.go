//go:build cuda

package native

import (
	"testing"
	"unsafe"
)

// Copy-cost probe. These benchmarks separate transfer cost from pipeline wait
// so the layer ping-pong can be attributed. Run with:
//
//	go test -tags cuda ./internal/backend/cuda/native -run XXX -bench CopyCost -benchtime 2000x

const copyCostN = 2048 // floats, roughly one hidden-size vector for E2B

func copyCostSetup(tb testing.TB) (DeviceBuffer, []float32, HostBuffer, Stream) {
	tb.Helper()
	dev, err := AllocDevice(copyCostN * 4)
	if err != nil {
		tb.Fatalf("alloc device: %v", err)
	}
	host := make([]float32, copyCostN)
	for i := range host {
		host[i] = float32(i)
	}
	pinned, err := AllocHostPinned(copyCostN * 4)
	if err != nil {
		tb.Fatalf("alloc pinned: %v", err)
	}
	stream, err := NewStream()
	if err != nil {
		tb.Fatalf("create stream: %v", err)
	}
	tb.Cleanup(func() {
		_ = stream.Destroy()
		_ = pinned.Free()
		_ = dev.Free()
	})
	return dev, host, pinned, stream
}

func BenchmarkCopyCostH2DSyncIdle(b *testing.B) {
	dev, host, _, _ := copyCostSetup(b)
	src := unsafe.Pointer(&host[0])
	bytes := int64(copyCostN * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := MemcpyH2D(dev, src, bytes); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCopyCostH2DAsyncIdle(b *testing.B) {
	dev, host, _, stream := copyCostSetup(b)
	src := unsafe.Pointer(&host[0])
	bytes := int64(copyCostN * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := MemcpyH2DAsync(dev, src, bytes, stream); err != nil {
			b.Fatal(err)
		}
		if err := stream.Synchronize(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCopyCostH2DSyncPinned(b *testing.B) {
	dev, host, pinned, _ := copyCostSetup(b)
	copyHostToPinned(pinned, host)
	src := pinned.Ptr()
	bytes := int64(copyCostN * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := MemcpyH2D(dev, src, bytes); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCopyCostD2HSyncIdle(b *testing.B) {
	dev, host, _, _ := copyCostSetup(b)
	dst := unsafe.Pointer(&host[0])
	bytes := int64(copyCostN * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := MemcpyD2H(dst, dev, bytes); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCopyCostD2HAsyncIdle(b *testing.B) {
	dev, host, _, stream := copyCostSetup(b)
	dst := unsafe.Pointer(&host[0])
	bytes := int64(copyCostN * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := MemcpyD2HAsync(dst, dev, bytes, stream); err != nil {
			b.Fatal(err)
		}
		if err := stream.Synchronize(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkCopyCostH2DSyncBusy queues work on the compute stream first, the way
// a layer does, then uploads. The extra time over the idle case is pipeline wait.
func BenchmarkCopyCostH2DSyncBusy(b *testing.B) {
	dev, host, _, stream := copyCostSetup(b)
	src := unsafe.Pointer(&host[0])
	bytes := int64(copyCostN * 4)
	dst, err := AllocDevice(256 * 1024 * 4)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = dst.Free() }()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := RoundBF16InPlaceF32(dst, 256*1024/8, stream); err != nil {
			b.Fatal(err)
		}
		if err := MemcpyH2D(dev, src, bytes); err != nil {
			b.Fatal(err)
		}
	}
}

func copyHostToPinned(dst HostBuffer, src []float32) {
	out := unsafe.Slice((*float32)(dst.Ptr()), len(src))
	copy(out, src)
}

// BenchmarkCopyCostH2DSyncBusyLayer queues a layer's worth of kernels before the
// upload, which is the shape the decoder's per-layer ping-pong has. The gap
// between this and the idle case is what one sync point costs in dispatch
// latency, so it bounds what batching launches can recover.
func BenchmarkCopyCostH2DSyncBusyLayer(b *testing.B) {
	dev, host, _, stream := copyCostSetup(b)
	src := unsafe.Pointer(&host[0])
	bytes := int64(copyCostN * 4)
	const kernels = 16
	dst, err := AllocDevice(256 * 1024 * 4)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = dst.Free() }()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for range kernels {
			if err := RoundBF16InPlaceF32(dst, 256*1024/8, stream); err != nil {
				b.Fatal(err)
			}
		}
		if err := MemcpyH2D(dev, src, bytes); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkCopyCostD2HSyncBusyLayer is the decode path's actual shape: kernels
// queued, then a device-to-host copy the host must wait for.
func BenchmarkCopyCostD2HSyncBusyLayer(b *testing.B) {
	dev, host, _, stream := copyCostSetup(b)
	dstHost := unsafe.Pointer(&host[0])
	bytes := int64(copyCostN * 4)
	const kernels = 16
	work, err := AllocDevice(256 * 1024 * 4)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = work.Free() }()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for range kernels {
			if err := RoundBF16InPlaceF32(work, 256*1024/8, stream); err != nil {
				b.Fatal(err)
			}
		}
		if err := MemcpyD2H(dstHost, dev, bytes); err != nil {
			b.Fatal(err)
		}
	}
}
