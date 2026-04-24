//go:build cuda

package cuda

import (
	"testing"
	"unsafe"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/cuda/native"
)

func TestSyncAttentionCacheToHostF16(t *testing.T) {
	t.Helper()

	ops := NewOps(native.Stream{}, native.BlasHandle{})
	layer := &core.Layer{
		AttnCache: core.AttnCache{
			K16:      make([]uint16, 0),
			V16:      make([]uint16, 0),
			KvStride: 2,
			CacheLen: 3,
		},
	}

	kDev, err := native.AllocDevice(12)
	if err != nil {
		t.Fatalf("alloc K cache: %v", err)
	}
	defer func() { _ = kDev.Free() }()

	vDev, err := native.AllocDevice(12)
	if err != nil {
		t.Fatalf("alloc V cache: %v", err)
	}
	defer func() { _ = vDev.Free() }()

	kSrc := []uint16{11, 12, 21, 22, 31, 32}
	vSrc := []uint16{101, 102, 201, 202, 301, 302}
	if err := native.MemcpyH2D(kDev, unsafe.Pointer(&kSrc[0]), int64(len(kSrc))*2); err != nil {
		t.Fatalf("upload K cache: %v", err)
	}
	if err := native.MemcpyH2D(vDev, unsafe.Pointer(&vSrc[0]), int64(len(vSrc))*2); err != nil {
		t.Fatalf("upload V cache: %v", err)
	}

	ops.attnCaches[layer] = deviceAttnCache{
		kF16:     kDev,
		vF16:     vDev,
		kvStride: 2,
		cacheLen: 3,
	}
	ops.lastDevKVPos[layer] = 1

	ops.SyncAttentionCacheToHost(layer, 1)

	wantK := []uint16{11, 12, 21, 22}
	wantV := []uint16{101, 102, 201, 202}
	if got := layer.AttnCache.K16; len(got) != 6 {
		t.Fatalf("K16 len = %d, want 6", len(got))
	} else {
		for i := range wantK {
			if got[i] != wantK[i] {
				t.Fatalf("K16[%d] = %d, want %d", i, got[i], wantK[i])
			}
		}
	}
	if got := layer.AttnCache.V16; len(got) != 6 {
		t.Fatalf("V16 len = %d, want 6", len(got))
	} else {
		for i := range wantV {
			if got[i] != wantV[i] {
				t.Fatalf("V16[%d] = %d, want %d", i, got[i], wantV[i])
			}
		}
	}
	if got := ops.lastHostKVPos[layer]; got != 1 {
		t.Fatalf("lastHostKVPos = %d, want 1", got)
	}

	kNext := []uint16{500, 501, 600, 601, 700, 701}
	if err := native.MemcpyH2D(kDev, unsafe.Pointer(&kNext[0]), int64(len(kNext))*2); err != nil {
		t.Fatalf("rewrite K cache: %v", err)
	}
	ops.SyncAttentionCacheToHost(layer, 1)
	for i := range wantK {
		if layer.AttnCache.K16[i] != wantK[i] {
			t.Fatalf("K16 changed after cached sync at %d: got %d want %d", i, layer.AttnCache.K16[i], wantK[i])
		}
	}
}
