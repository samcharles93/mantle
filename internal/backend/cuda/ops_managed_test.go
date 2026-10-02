//go:build cuda

package cuda

import (
	"testing"

	"github.com/samcharles93/mantle/internal/backend/cuda/native"
)

// cudaMemPrefetchAsync rejects a range that spans more than one managed
// allocation. recordManagedLayerBufs previously coalesced each layer's managed
// buffers into a single [lo,hi) span, so every prefetch failed with
// cudaErrorInvalidValue and the latched error was reported by unrelated kernel
// launches. Ranges must stay one-per-allocation.
func TestManagedRangesKeepsAllocationsSeparate(t *testing.T) {
	count, err := native.DeviceCount()
	if err != nil {
		t.Fatalf("DeviceCount: %v", err)
	}
	if count < 1 {
		t.Skip("no cuda device available")
	}

	a, err := native.AllocManaged(1 << 20)
	if err != nil {
		t.Fatalf("AllocManaged a: %v", err)
	}
	defer func() { _ = a.Free() }()
	b, err := native.AllocManaged(1 << 20)
	if err != nil {
		t.Fatalf("AllocManaged b: %v", err)
	}
	defer func() { _ = b.Free() }()

	ranges := managedRanges([]native.DeviceBuffer{a, b})
	if len(ranges) != 2 {
		t.Fatalf("managedRanges: got %d ranges, want one per allocation (2)", len(ranges))
	}
	if ranges[0].start != a.Ptr() || ranges[0].bytes != a.Nbytes() {
		t.Fatalf("range[0] = (%p, %d), want the first allocation (%p, %d)", ranges[0].start, ranges[0].bytes, a.Ptr(), a.Nbytes())
	}
	if ranges[1].start != b.Ptr() || ranges[1].bytes != b.Nbytes() {
		t.Fatalf("range[1] = (%p, %d), want the second allocation (%p, %d)", ranges[1].start, ranges[1].bytes, b.Ptr(), b.Nbytes())
	}

	stream, err := native.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()
	for i, r := range ranges {
		if err := native.MemPrefetchAsync(native.DeviceBufferFromRaw(r.start), r.bytes, 0, stream); err != nil {
			t.Fatalf("MemPrefetchAsync(range %d, %d bytes): %v", i, r.bytes, err)
		}
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream synchronize: %v", err)
	}
}
