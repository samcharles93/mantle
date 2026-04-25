//go:build cuda

package native

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

func sigmoidRef(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

func moeRouterRef(rawIn, bias []float32, k int, routeScale float32, numExperts int) (sigmoidOut []float32, idx []int32, weights []float32) {
	sigmoidOut = make([]float32, numExperts)
	for i := range numExperts {
		sigmoidOut[i] = sigmoidRef(rawIn[i])
	}
	sel := make([]float32, numExperts)
	for i := range numExperts {
		b := float32(0)
		if i < len(bias) {
			b = bias[i]
		}
		sel[i] = sigmoidOut[i] + b
	}

	idx = make([]int32, k)
	used := make([]bool, numExperts)
	for j := range k {
		bestI := -1
		bestV := float32(-math.MaxFloat32)
		for i := range numExperts {
			if used[i] {
				continue
			}
			if sel[i] > bestV || (sel[i] == bestV && bestI >= 0 && i < bestI) {
				bestV = sel[i]
				bestI = i
			}
		}
		idx[j] = int32(bestI)
		used[bestI] = true
	}

	var denom float32
	for j := range k {
		denom += sigmoidOut[idx[j]]
	}
	if denom == 0 {
		denom = 1
	}
	weights = make([]float32, k)
	for j := range k {
		weights[j] = (sigmoidOut[idx[j]] / denom) * routeScale
	}
	return
}

func TestMoERouterF32Parity(t *testing.T) {
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
	defer func() {
		if err := stream.Destroy(); err != nil {
			t.Fatalf("stream destroy: %v", err)
		}
	}()

	cases := []struct {
		numExperts int
		k          int
		hasBias    bool
		routeScale float32
	}{
		{8, 2, true, 1.0},
		{16, 4, true, 2.5},
		{32, 4, false, 1.0},
		{64, 8, true, 1.5},
		{128, 6, true, 1.0},
		{256, 8, true, 0.5},
	}

	rng := rand.New(rand.NewSource(0xC0FFEE7))
	for _, c := range cases {
		raw := make([]float32, c.numExperts)
		for i := range c.numExperts {
			raw[i] = (rng.Float32()*2 - 1) * 4
		}
		var bias []float32
		biasLen := 0
		if c.hasBias {
			bias = make([]float32, c.numExperts)
			for i := range c.numExperts {
				bias[i] = (rng.Float32()*2 - 1) * 0.1
			}
			biasLen = c.numExperts
		}

		wantSig, wantIdx, wantW := moeRouterRef(raw, bias, c.k, c.routeScale, c.numExperts)

		rawDev, err := AllocDevice(int64(c.numExperts) * 4)
		if err != nil {
			t.Fatalf("alloc raw: %v", err)
		}
		if err := MemcpyH2D(rawDev, unsafe.Pointer(&raw[0]), int64(c.numExperts)*4); err != nil {
			t.Fatalf("h2d raw: %v", err)
		}
		var biasDev DeviceBuffer
		if biasLen > 0 {
			biasDev, err = AllocDevice(int64(biasLen) * 4)
			if err != nil {
				t.Fatalf("alloc bias: %v", err)
			}
			if err := MemcpyH2D(biasDev, unsafe.Pointer(&bias[0]), int64(biasLen)*4); err != nil {
				t.Fatalf("h2d bias: %v", err)
			}
		}
		idxDev, err := AllocDevice(int64(c.k) * 4)
		if err != nil {
			t.Fatalf("alloc idx: %v", err)
		}
		wDev, err := AllocDevice(int64(c.k) * 4)
		if err != nil {
			t.Fatalf("alloc w: %v", err)
		}

		if err := MoERouterF32(rawDev, biasDev, biasLen, c.k, c.routeScale, c.numExperts, idxDev, wDev, stream); err != nil {
			t.Fatalf("MoERouterF32: %v", err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("sync: %v", err)
		}

		gotSig := make([]float32, c.numExperts)
		gotIdx := make([]int32, c.k)
		gotW := make([]float32, c.k)
		if err := MemcpyD2H(unsafe.Pointer(&gotSig[0]), rawDev, int64(c.numExperts)*4); err != nil {
			t.Fatalf("d2h sig: %v", err)
		}
		if err := MemcpyD2H(unsafe.Pointer(&gotIdx[0]), idxDev, int64(c.k)*4); err != nil {
			t.Fatalf("d2h idx: %v", err)
		}
		if err := MemcpyD2H(unsafe.Pointer(&gotW[0]), wDev, int64(c.k)*4); err != nil {
			t.Fatalf("d2h w: %v", err)
		}

		_ = rawDev.Free()
		if biasLen > 0 {
			_ = biasDev.Free()
		}
		_ = idxDev.Free()
		_ = wDev.Free()

		for i := range c.numExperts {
			if !approxEqual(gotSig[i], wantSig[i], 1e-6) {
				t.Fatalf("ne=%d k=%d sig[%d]: got=%g want=%g", c.numExperts, c.k, i, gotSig[i], wantSig[i])
			}
		}
		for j := range c.k {
			if gotIdx[j] != wantIdx[j] {
				t.Fatalf("ne=%d k=%d idx[%d]: got=%d want=%d (gotAll=%v wantAll=%v)", c.numExperts, c.k, j, gotIdx[j], wantIdx[j], gotIdx, wantIdx)
			}
			if !approxEqual(gotW[j], wantW[j], 5e-6) {
				t.Fatalf("ne=%d k=%d w[%d]: got=%g want=%g", c.numExperts, c.k, j, gotW[j], wantW[j])
			}
		}
	}
}

func TestMoERouterF32TieBreak(t *testing.T) {
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
	defer stream.Destroy()

	const numExperts = 32
	const k = 4
	raw := make([]float32, numExperts)
	for i := range numExperts {
		raw[i] = 0.0
	}
	bias := make([]float32, numExperts)
	rawDev, _ := AllocDevice(int64(numExperts) * 4)
	defer rawDev.Free()
	biasDev, _ := AllocDevice(int64(numExperts) * 4)
	defer biasDev.Free()
	idxDev, _ := AllocDevice(int64(k) * 4)
	defer idxDev.Free()
	wDev, _ := AllocDevice(int64(k) * 4)
	defer wDev.Free()

	if err := MemcpyH2D(rawDev, unsafe.Pointer(&raw[0]), int64(numExperts)*4); err != nil {
		t.Fatal(err)
	}
	if err := MemcpyH2D(biasDev, unsafe.Pointer(&bias[0]), int64(numExperts)*4); err != nil {
		t.Fatal(err)
	}
	if err := MoERouterF32(rawDev, biasDev, numExperts, k, 1.0, numExperts, idxDev, wDev, stream); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	gotIdx := make([]int32, k)
	if err := MemcpyD2H(unsafe.Pointer(&gotIdx[0]), idxDev, int64(k)*4); err != nil {
		t.Fatal(err)
	}
	for j := range k {
		if gotIdx[j] != int32(j) {
			t.Fatalf("tie-break: idx[%d]=%d want %d (all=%v)", j, gotIdx[j], j, gotIdx)
		}
	}
}

func TestMoERouterF32NilBuffer(t *testing.T) {
	var empty DeviceBuffer
	if err := MoERouterF32(empty, empty, 0, 2, 1.0, 8, empty, empty, Stream{}); err == nil {
		t.Fatal("expected error for nil buffers")
	}
}

func BenchmarkMoERouter(b *testing.B) {
	count, err := DeviceCount()
	if err != nil {
		b.Fatalf("DeviceCount: %v", err)
	}
	if count < 1 {
		b.Skip("no cuda device available")
	}

	cases := []struct {
		name       string
		numExperts int
		topK       int
	}{
		{name: "e8k2", numExperts: 8, topK: 2},
		{name: "e64k8", numExperts: 64, topK: 8},
	}

	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			numExperts := c.numExperts
			topK := c.topK

			rng := rand.New(rand.NewSource(0xC0FFEE7))
			raw := make([]float32, numExperts)
			for i := range raw {
				raw[i] = (rng.Float32()*2 - 1) * 3.0
			}
			bias := make([]float32, numExperts)
			for i := range bias {
				bias[i] = (rng.Float32()*2 - 1) * 0.5
			}
			const routeScale float32 = 2.5

			b.Run("cpu", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, idx, weights := moeRouterRef(raw, bias, topK, routeScale, numExperts)
					_ = idx
					_ = weights
				}
			})

			stream, err := NewStream()
			if err != nil {
				b.Fatalf("NewStream: %v", err)
			}
			defer func() { _ = stream.Destroy() }()

			rawDev, err := AllocDevice(int64(numExperts * 4))
			if err != nil {
				b.Fatalf("AllocDevice raw: %v", err)
			}
			defer func() { _ = rawDev.Free() }()
			biasDev, err := AllocDevice(int64(numExperts * 4))
			if err != nil {
				b.Fatalf("AllocDevice bias: %v", err)
			}
			defer func() { _ = biasDev.Free() }()
			idxDev, err := AllocDevice(int64(topK * 4))
			if err != nil {
				b.Fatalf("AllocDevice idx: %v", err)
			}
			defer func() { _ = idxDev.Free() }()
			wDev, err := AllocDevice(int64(topK * 4))
			if err != nil {
				b.Fatalf("AllocDevice w: %v", err)
			}
			defer func() { _ = wDev.Free() }()

			if err := MemcpyH2D(rawDev, unsafe.Pointer(&raw[0]), int64(numExperts*4)); err != nil {
				b.Fatalf("H2D raw: %v", err)
			}
			if err := MemcpyH2D(biasDev, unsafe.Pointer(&bias[0]), int64(numExperts*4)); err != nil {
				b.Fatalf("H2D bias: %v", err)
			}

			idxHost := make([]int32, topK)
			wHost := make([]float32, topK)

			b.Run("cuda", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if err := MoERouterF32(rawDev, biasDev, numExperts, topK, routeScale, numExperts, idxDev, wDev, stream); err != nil {
						b.Fatalf("MoERouterF32: %v", err)
					}
				}
				if err := stream.Synchronize(); err != nil {
					b.Fatalf("sync: %v", err)
				}
				b.StopTimer()
				if err := MemcpyD2H(unsafe.Pointer(&idxHost[0]), idxDev, int64(topK*4)); err != nil {
					b.Fatalf("D2H idx: %v", err)
				}
				if err := MemcpyD2H(unsafe.Pointer(&wHost[0]), wDev, int64(topK*4)); err != nil {
					b.Fatalf("D2H w: %v", err)
				}
			})

			_ = idxHost
			_ = wHost
		})
	}
}
