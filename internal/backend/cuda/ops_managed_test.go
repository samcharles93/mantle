//go:build cuda

package cuda

import (
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
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

func TestCUDALoadModelRejectsUnsupportedCacheTypes(t *testing.T) {
	b := &Backend{}
	for _, ct := range []string{"q4_0", "f32", "bf16", "invalid"} {
		if _, err := b.LoadModel(nil, nil, 4096, core.LoadModelOptions{CacheTypeK: ct}); err == nil {
			t.Fatalf("LoadModel accepted unsupported CacheTypeK=%q", ct)
		}
		if _, err := b.LoadModel(nil, nil, 4096, core.LoadModelOptions{CacheTypeV: ct}); err == nil {
			t.Fatalf("LoadModel accepted unsupported CacheTypeV=%q", ct)
		}
	}
}

func TestEstimateModelKVCacheBytesAndPreloadBudget(t *testing.T) {
	const (
		numLayers = 42
		kvStride  = 256
		cacheLen  = 4096
		freeBytes = int64(3363) << 20
	)
	buildInst := func(q8K, q8V bool) *core.Instance {
		layers := make([]core.Layer, numLayers)
		for i := range layers {
			c := core.AttnCache{
				KvStride: kvStride,
				CacheLen: cacheLen,
			}
			if q8K {
				c.KQ8 = make([]int8, 0)
				c.KQ8S = make([]float32, 0)
			} else {
				c.K16 = make([]uint16, 0)
			}
			if q8V {
				c.VQ8 = make([]int8, 0)
				c.VQ8S = make([]float32, 0)
			} else {
				c.V16 = make([]uint16, 0)
			}
			layers[i] = core.Layer{
				HeadKV:         2,
				HeadDim:        128,
				SharedKVSource: -1,
				AttnCache:      c,
			}
		}
		return &core.Instance{
			MaxContext: cacheLen,
			Layers:     layers,
		}
	}

	f16Inst := buildInst(false, false)
	q8Inst := buildInst(true, true)

	f16KV := estimateModelKVCacheBytes(f16Inst, 0)
	q8KV := estimateModelKVCacheBytes(q8Inst, 0)

	// 42 layers * 256 stride * 4096 tokens * 2 (K+V) * 2 bytes = 168 MiB
	wantF16 := int64(numLayers) * kvStride * cacheLen * 4
	if f16KV != wantF16 {
		t.Fatalf("f16 KV estimate = %d, want %d", f16KV, wantF16)
	}
	// 42 layers * 2 (K+V) * (256*4096 + (256/32)*4096*4) = 94.5 MiB
	wantQ8 := int64(numLayers) * 2 * (kvStride*cacheLen + (kvStride/32)*cacheLen*4)
	if q8KV != wantQ8 {
		t.Fatalf("q8_0 KV estimate = %d, want %d", q8KV, wantQ8)
	}
	if q8KV >= f16KV {
		t.Fatalf("expected q8_0 KV estimate (%d) < f16 KV estimate (%d)", q8KV, f16KV)
	}

	f16Budget, _ := computePreloadUploadBudget(freeBytes, f16KV)
	q8Budget, _ := computePreloadUploadBudget(freeBytes, q8KV)
	if q8Budget-f16Budget != f16KV-q8KV {
		t.Fatalf("q8 budget gain = %d, want exact KV savings %d", q8Budget-f16Budget, f16KV-q8KV)
	}
}

func TestOpsStoreKVQ8PerRowScale(t *testing.T) {
	const kvStride = 64
	cache := core.AttnCache{
		KvStride: kvStride,
		CacheLen: 64,
		KQ8:      make([]int8, 0),
		KQ8S:     make([]float32, 0),
		VQ8:      make([]int8, 0),
		VQ8S:     make([]float32, 0),
	}
	var ops Ops
	k := make([]float32, kvStride)
	v := make([]float32, kvStride)
	for i := range kvStride {
		k[i] = float32(i+1) * 0.25
		v[i] = -float32(i+1) * 0.5
	}
	for pos := range 40 {
		cache.EnsurePos(pos)
		ops.StoreKV(-1, pos, kvStride, nil, nil, nil, nil, cache.KQ8, cache.VQ8, cache.KQ8S, cache.VQ8S, k, v)
		if cache.KQ8S[pos] <= 0 || cache.VQ8S[pos] <= 0 {
			t.Fatalf("pos %d: expected positive Q8 scales, got k=%g v=%g", pos, cache.KQ8S[pos], cache.VQ8S[pos])
		}
	}
}
