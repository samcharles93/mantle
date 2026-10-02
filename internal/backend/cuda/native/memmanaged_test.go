//go:build cuda

package native

import (
	"testing"
	"unsafe"
)

// CUDA 13 changed cudaMemAdvise and cudaMemPrefetchAsync to take a
// cudaMemLocation struct instead of an int device. Under the pre-13 ABI the
// driver read location.type = 0 (cudaMemLocationTypeInvalid) and every call
// returned cudaErrorInvalidValue. Because callers treat these as best-effort
// hints and discard the error, it stayed latched and was then reported by the
// next cudaGetLastError-based kernel-launch check, which surfaced as bogus
// failures in unrelated kernels (RoPE, RMSNorm, SiLU) and in weight upload.
func TestManagedHintsUseCuda13LocationABI(t *testing.T) {
	count, err := DeviceCount()
	if err != nil {
		t.Fatalf("DeviceCount: %v", err)
	}
	if count < 1 {
		t.Skip("no cuda device available")
	}

	stream, err := NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	const n = 4096
	managed, err := AllocManaged(int64(n) * 4)
	if err != nil {
		t.Fatalf("AllocManaged: %v", err)
	}
	defer func() { _ = managed.Free() }()

	if err := MemAdvise(managed, managed.Nbytes(), MemAdviseSetReadMostly, 0); err != nil {
		t.Fatalf("MemAdvise(SetReadMostly): %v", err)
	}
	if err := MemAdvise(managed, managed.Nbytes(), MemAdviseSetAccessedBy, 0); err != nil {
		t.Fatalf("MemAdvise(SetAccessedBy): %v", err)
	}
	if err := MemPrefetchAsync(managed, managed.Nbytes(), 0, stream); err != nil {
		t.Fatalf("MemPrefetchAsync: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream synchronize: %v", err)
	}

	// A kernel launched afterwards must not observe a stale error from the hints.
	gate, err := AllocDevice(int64(n) * 4)
	if err != nil {
		t.Fatalf("AllocDevice gate: %v", err)
	}
	defer func() { _ = gate.Free() }()
	up, err := AllocDevice(int64(n) * 4)
	if err != nil {
		t.Fatalf("AllocDevice up: %v", err)
	}
	defer func() { _ = up.Free() }()
	out, err := AllocDevice(int64(n) * 4)
	if err != nil {
		t.Fatalf("AllocDevice out: %v", err)
	}
	defer func() { _ = out.Free() }()

	host := make([]float32, n)
	for i := range host {
		host[i] = float32(i%7) - 3
	}
	for _, dst := range []DeviceBuffer{gate, up} {
		if err := MemcpyH2D(dst, unsafe.Pointer(&host[0]), int64(n)*4); err != nil {
			t.Fatalf("MemcpyH2D: %v", err)
		}
	}
	if err := SiluMulF32(gate, up, out, n, stream); err != nil {
		t.Fatalf("SiluMulF32 after managed hints: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("sync after kernel: %v", err)
	}
}
