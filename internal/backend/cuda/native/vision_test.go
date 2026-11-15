//go:build cuda

package native

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// ---------- CPU references (mirror internal/backend/simd/vision_lfm3vl.go) ----------

func refVisionRoPE(x, cosT, sinT []float32, n, heads, headDim int) {
	half := headDim / 2
	for p := range n {
		cos := cosT[p*headDim : (p+1)*headDim]
		sin := sinT[p*headDim : (p+1)*headDim]
		for hi := range heads {
			base := p*heads*headDim + hi*headDim
			for d := range half {
				x1 := x[base+d]
				x2 := x[base+half+d]
				x[base+d] = x1*cos[d] - x2*sin[d]
				x[base+half+d] = x2*cos[half+d] + x1*sin[half+d]
			}
		}
	}
}

func refSoftmax(row []float32) {
	m := row[0]
	for _, v := range row {
		if v > m {
			m = v
		}
	}
	var sum float32
	for i := range row {
		row[i] = float32(math.Exp(float64(row[i] - m)))
		sum += row[i]
	}
	for i := range row {
		row[i] /= sum
	}
}

func refVisionAttn(q, k, v []float32, segStart, segLen []int, nSeg, heads, kvHeads, headDim int, scale float32) []float32 {
	n := len(q) / (heads * headDim)
	out := make([]float32, n*heads*headDim)
	groups := heads / kvHeads
	scores := make([]float32, 0)
	for s := range nSeg {
		start, length := segStart[s], segLen[s]
		if cap(scores) < length*length {
			scores = make([]float32, length*length)
		}
		for hi := range heads {
			kvh := hi / groups
			for qi := range length {
				row := scores[:length]
				qBase := (start + qi) * heads * headDim
				for ki := range length {
					kBase := (start + ki) * kvHeads * headDim
					var dot float32
					for d := range headDim {
						dot += q[qBase+hi*headDim+d] * k[kBase+kvh*headDim+d]
					}
					row[ki] = dot * scale
				}
				refSoftmax(row)
				outBase := (start + qi) * heads * headDim
				for ki := range length {
					vBase := (start + ki) * kvHeads * headDim
					w := row[ki]
					for d := range headDim {
						out[outBase+hi*headDim+d] += w * v[vBase+kvh*headDim+d]
					}
				}
			}
		}
	}
	return out
}

func blockMajorToRasterRef(i, H, W, m int) (y, x int) {
	bi := i / (m * m)
	wi := i % (m * m)
	bh := bi / (W / m)
	bw := bi % (W / m)
	mh := wi / m
	mw := wi % m
	return bh*m + mh, bw*m + mw
}

func rasterToBlockMajorRef(y, x, W, m int) int {
	bh := y / m
	bw := x / m
	mh := y % m
	mw := x % m
	return (bh*(W/m)+bw)*m*m + mh*m + mw
}

type refFrame struct {
	start, h, w int
}

func refVisionConv2dGated(in, convW []float32, frames []refFrame, n, C, kSize, merge int) []float32 {
	pad := kSize / 2
	out := make([]float32, n*C)
	for _, f := range frames {
		for i := 0; i < f.h*f.w; i++ {
			p := f.start + i
			y, x := blockMajorToRasterRef(i, f.h, f.w, merge)
			cBase := p*3*C + C
			for c := range C {
				kc := convW[c*kSize*kSize:]
				var acc float32
				for ky := range kSize {
					iy := y + ky - pad
					if iy < 0 || iy >= f.h {
						continue
					}
					for kx := range kSize {
						ix := x + kx - pad
						if ix < 0 || ix >= f.w {
							continue
						}
						pn := f.start + rasterToBlockMajorRef(iy, ix, f.w, merge)
						acc += kc[ky*kSize+kx] * (in[pn*3*C+c] * in[pn*3*C+2*C+c])
					}
				}
				out[p*C+c] = acc * in[cBase+c]
			}
		}
	}
	return out
}

func refLayerNormRows(src, weight, bias []float32, eps float32, rows, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := range rows {
		row := src[r*cols : (r+1)*cols]
		var mean float32
		for _, v := range row {
			mean += v
		}
		mean /= float32(cols)
		var varAcc float32
		for _, v := range row {
			d := v - mean
			varAcc += d * d
		}
		varAcc /= float32(cols)
		scale := 1.0 / float32(math.Sqrt(float64(varAcc+eps)))
		for i, v := range row {
			b := float32(0)
			if len(bias) > i {
				b = bias[i]
			}
			out[r*cols+i] = (v-mean)*scale*weight[i] + b
		}
	}
	return out
}

func refGeluExact(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
}

// ---------- helpers ----------

func mustAllocAndCopyInt32(t testing.TB, data []int32) DeviceBuffer {
	t.Helper()
	buf, err := AllocDevice(int64(len(data) * 4))
	if err != nil {
		t.Fatalf("AllocDevice: %v", err)
	}
	if err := MemcpyH2D(buf, unsafe.Pointer(&data[0]), int64(len(data)*4)); err != nil {
		_ = buf.Free()
		t.Fatalf("MemcpyH2D: %v", err)
	}
	return buf
}

func maxAbsDiff(a, b []float32) float64 {
	m := 0.0
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > m {
			m = d
		}
	}
	return m
}

// ---------- tests ----------

func TestVisionRoPEF32(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for _, tc := range []struct{ n, heads, headDim int }{
		{1, 1, 2}, {5, 4, 8}, {37, 16, 64}, {3, 2, 128},
	} {
		n, heads, hd := tc.n, tc.heads, tc.headDim
		x := make([]float32, n*heads*hd)
		cosT := make([]float32, n*hd)
		sinT := make([]float32, n*hd)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		for i := range cosT {
			cosT[i] = rng.Float32()*2 - 1
			sinT[i] = rng.Float32()*2 - 1
		}
		want := append([]float32(nil), x...)
		refVisionRoPE(want, cosT, sinT, n, heads, hd)

		stream, err := NewStream()
		if err != nil {
			t.Fatalf("NewStream: %v", err)
		}
		xBuf := mustAllocAndCopy(t, x)
		cosBuf := mustAllocAndCopy(t, cosT)
		sinBuf := mustAllocAndCopy(t, sinT)
		if err := VisionRoPEF32(xBuf, cosBuf, sinBuf, n, heads, hd, stream); err != nil {
			t.Fatalf("VisionRoPEF32: %v", err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("Synchronize: %v", err)
		}
		got := downloadF32(t, xBuf, len(x))
		if d := maxAbsDiff(got, want); d > 1e-6 {
			t.Fatalf("n=%d heads=%d hd=%d: max diff %g", n, heads, hd, d)
		}
		_ = xBuf.Free()
		_ = cosBuf.Free()
		_ = sinBuf.Free()
		_ = stream.Destroy()
	}
}

func TestVisionAttnSegmentsF32(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	type seg struct{ start, length int }
	for _, tc := range []struct {
		segs    []seg
		heads   int
		kvHeads int
		headDim int
	}{
		{[]seg{{0, 1}}, 4, 2, 8},
		{[]seg{{0, 37}}, 4, 2, 8},
		{[]seg{{0, 33}, {33, 31}, {64, 5}}, 16, 8, 64},
		{[]seg{{0, 256}}, 16, 8, 64},
	} {
		n := 0
		for _, s := range tc.segs {
			n += s.length
		}
		heads, kvHeads, hd := tc.heads, tc.kvHeads, tc.headDim
		scale := float32(1.0 / math.Sqrt(float64(hd)))
		q := make([]float32, n*heads*hd)
		k := make([]float32, n*kvHeads*hd)
		v := make([]float32, n*kvHeads*hd)
		for i := range q {
			q[i] = rng.Float32()*2 - 1
		}
		for i := range k {
			k[i] = rng.Float32()*2 - 1
			v[i] = rng.Float32()*2 - 1
		}
		segStart := make([]int32, len(tc.segs))
		segLen := make([]int32, len(tc.segs))
		for i, s := range tc.segs {
			segStart[i] = int32(s.start)
			segLen[i] = int32(s.length)
		}
		want := refVisionAttn(q, k, v, func() []int {
			out := make([]int, len(tc.segs))
			for i, s := range tc.segs {
				out[i] = s.start
			}
			return out
		}(), func() []int {
			out := make([]int, len(tc.segs))
			for i, s := range tc.segs {
				out[i] = s.length
			}
			return out
		}(), len(tc.segs), heads, kvHeads, hd, scale)

		stream, err := NewStream()
		if err != nil {
			t.Fatalf("NewStream: %v", err)
		}
		qBuf := mustAllocAndCopy(t, q)
		kBuf := mustAllocAndCopy(t, k)
		vBuf := mustAllocAndCopy(t, v)
		outBuf, err := AllocDevice(int64(len(q) * 4))
		if err != nil {
			t.Fatalf("AllocDevice: %v", err)
		}
		ssBuf := mustAllocAndCopyInt32(t, segStart)
		slBuf := mustAllocAndCopyInt32(t, segLen)
		if err := VisionAttnSegmentsF32(qBuf, kBuf, vBuf, outBuf, ssBuf, slBuf, n, len(tc.segs), heads, kvHeads, hd, scale, stream); err != nil {
			t.Fatalf("VisionAttnSegmentsF32: %v", err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("Synchronize: %v", err)
		}
		got := downloadF32(t, outBuf, len(want))
		if d := maxAbsDiff(got, want); d > 1e-5 {
			t.Fatalf("segs=%v heads=%d kvHeads=%d hd=%d: max diff %g", tc.segs, heads, kvHeads, hd, d)
		}
		_ = qBuf.Free()
		_ = kBuf.Free()
		_ = vBuf.Free()
		_ = outBuf.Free()
		_ = ssBuf.Free()
		_ = slBuf.Free()
		_ = stream.Destroy()
	}
}

func TestVisionConv2dGatedF32(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, tc := range []struct {
		frames     []refFrame
		n, C, k, m int
	}{
		{[]refFrame{{0, 14, 20}}, 280, 16, 3, 2},
		{[]refFrame{{0, 4, 6}, {24, 8, 8}}, 88, 8, 3, 2},
		{[]refFrame{{0, 5, 7}}, 35, 4, 5, 1},
	} {
		n, C, k, m := tc.n, tc.C, tc.k, tc.m
		in := make([]float32, n*3*C)
		convW := make([]float32, C*k*k)
		for i := range in {
			in[i] = rng.Float32()*2 - 1
		}
		for i := range convW {
			convW[i] = rng.Float32()*2 - 1
		}
		fs := make([]int32, len(tc.frames))
		fh := make([]int32, len(tc.frames))
		fw := make([]int32, len(tc.frames))
		for i, f := range tc.frames {
			fs[i], fh[i], fw[i] = int32(f.start), int32(f.h), int32(f.w)
		}
		want := refVisionConv2dGated(in, convW, tc.frames, n, C, k, m)

		stream, err := NewStream()
		if err != nil {
			t.Fatalf("NewStream: %v", err)
		}
		inBuf := mustAllocAndCopy(t, in)
		wBuf := mustAllocAndCopy(t, convW)
		outBuf, err := AllocDevice(int64(n * C * 4))
		if err != nil {
			t.Fatalf("AllocDevice: %v", err)
		}
		fsBuf := mustAllocAndCopyInt32(t, fs)
		fhBuf := mustAllocAndCopyInt32(t, fh)
		fwBuf := mustAllocAndCopyInt32(t, fw)
		maxTok := 0
		for _, f := range tc.frames {
			if f.h*f.w > maxTok {
				maxTok = f.h * f.w
			}
		}
		if err := VisionConv2dGatedF32(inBuf, wBuf, outBuf, fsBuf, fhBuf, fwBuf, len(tc.frames), n, maxTok, C, k, m, stream); err != nil {
			t.Fatalf("VisionConv2dGatedF32: %v", err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("Synchronize: %v", err)
		}
		got := downloadF32(t, outBuf, len(want))
		if d := maxAbsDiff(got, want); d > 1e-5 {
			t.Fatalf("frames=%v n=%d C=%d k=%d m=%d: max diff %g", tc.frames, n, C, k, m, d)
		}
		_ = inBuf.Free()
		_ = wBuf.Free()
		_ = outBuf.Free()
		_ = fsBuf.Free()
		_ = fhBuf.Free()
		_ = fwBuf.Free()
		_ = stream.Destroy()
	}
}

func TestLayerNormRowsF32(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	for _, withBias := range []bool{false, true} {
		rows, cols := 7, 64
		x := make([]float32, rows*cols)
		w := make([]float32, cols)
		bias := make([]float32, cols)
		for i := range x {
			x[i] = rng.Float32()*4 - 2
		}
		for i := range w {
			w[i] = rng.Float32()*0.5 + 0.75
		}
		for i := range bias {
			bias[i] = rng.Float32()*0.2 - 0.1
		}
		var biasArg []float32
		if withBias {
			biasArg = bias
		}
		eps := float32(1e-5)
		want := refLayerNormRows(x, w, biasArg, eps, rows, cols)

		stream, err := NewStream()
		if err != nil {
			t.Fatalf("NewStream: %v", err)
		}
		xBuf := mustAllocAndCopy(t, x)
		wBuf := mustAllocAndCopy(t, w)
		var bBuf DeviceBuffer
		if withBias {
			bBuf = mustAllocAndCopy(t, bias)
		}
		if err := LayerNormRowsF32(xBuf, xBuf, wBuf, bBuf, eps, rows, cols, stream); err != nil {
			t.Fatalf("LayerNormRowsF32: %v", err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("Synchronize: %v", err)
		}
		got := downloadF32(t, xBuf, len(want))
		if d := maxAbsDiff(got, want); d > 1e-6 {
			t.Fatalf("withBias=%v: max diff %g", withBias, d)
		}
		_ = xBuf.Free()
		_ = wBuf.Free()
		if withBias {
			_ = bBuf.Free()
		}
		_ = stream.Destroy()
	}
}

func TestGeluExactBiasF32(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	for _, withBias := range []bool{false, true} {
		rows, cols := 5, 33
		x := make([]float32, rows*cols)
		bias := make([]float32, cols)
		for i := range x {
			x[i] = rng.Float32()*8 - 4
		}
		for i := range bias {
			bias[i] = rng.Float32()*0.5 - 0.25
		}
		want := append([]float32(nil), x...)
		for i := range want {
			v := want[i]
			if withBias {
				v += bias[i%cols]
			}
			want[i] = refGeluExact(v)
		}

		stream, err := NewStream()
		if err != nil {
			t.Fatalf("NewStream: %v", err)
		}
		xBuf := mustAllocAndCopy(t, x)
		var bBuf DeviceBuffer
		if withBias {
			bBuf = mustAllocAndCopy(t, bias)
		}
		if err := GeluExactBiasF32(xBuf, bBuf, rows, cols, stream); err != nil {
			t.Fatalf("GeluExactBiasF32: %v", err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("Synchronize: %v", err)
		}
		got := downloadF32(t, xBuf, len(want))
		if d := maxAbsDiff(got, want); d > 1e-6 {
			t.Fatalf("withBias=%v: max diff %g", withBias, d)
		}
		_ = xBuf.Free()
		if withBias {
			_ = bBuf.Free()
		}
		_ = stream.Destroy()
	}
}

func TestAddBiasRowsF32(t *testing.T) {
	rng := rand.New(rand.NewSource(19))
	rows, cols := 9, 17
	x := make([]float32, rows*cols)
	bias := make([]float32, cols)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	for i := range bias {
		bias[i] = rng.Float32()*0.3 - 0.15
	}
	want := append([]float32(nil), x...)
	for i := range want {
		want[i] += bias[i%cols]
	}
	stream, err := NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	xBuf := mustAllocAndCopy(t, x)
	bBuf := mustAllocAndCopy(t, bias)
	if err := AddBiasRowsF32(xBuf, bBuf, rows, cols, stream); err != nil {
		t.Fatalf("AddBiasRowsF32: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	got := downloadF32(t, xBuf, len(want))
	if d := maxAbsDiff(got, want); d > 1e-7 {
		t.Fatalf("max diff %g", d)
	}
	_ = xBuf.Free()
	_ = bBuf.Free()
	_ = stream.Destroy()
}

// ---------- benchmarks ----------

func BenchmarkVisionRoPEF32(b *testing.B) {
	n, heads, hd := 16384, 16, 64
	rng := rand.New(rand.NewSource(23))
	x := make([]float32, n*heads*hd)
	cosT := make([]float32, n*hd)
	sinT := make([]float32, n*hd)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	for i := range cosT {
		cosT[i] = rng.Float32()*2 - 1
		sinT[i] = rng.Float32()*2 - 1
	}
	xBuf, err := AllocDevice(int64(len(x) * 4))
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = xBuf.Free() }()
	cosBuf := mustAllocAndCopyBench(b, cosT)
	defer func() { _ = cosBuf.Free() }()
	sinBuf := mustAllocAndCopyBench(b, sinT)
	defer func() { _ = sinBuf.Free() }()
	if err := MemcpyH2D(xBuf, unsafe.Pointer(&x[0]), int64(len(x)*4)); err != nil {
		b.Fatal(err)
	}
	stream, err := NewStream()
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = stream.Destroy() }()
	if err := VisionRoPEF32(xBuf, cosBuf, sinBuf, n, heads, hd, stream); err != nil {
		b.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := VisionRoPEF32(xBuf, cosBuf, sinBuf, n, heads, hd, stream); err != nil {
			b.Fatal(err)
		}
	}
	if err := stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
}

func mustAllocAndCopyBench(b *testing.B, data []float32) DeviceBuffer {
	b.Helper()
	buf, err := AllocDevice(int64(len(data) * 4))
	if err != nil {
		b.Fatalf("AllocDevice: %v", err)
	}
	if err := MemcpyH2D(buf, unsafe.Pointer(&data[0]), int64(len(data)*4)); err != nil {
		_ = buf.Free()
		b.Fatalf("MemcpyH2D: %v", err)
	}
	return buf
}

func BenchmarkVisionAttnSegmentsF32(b *testing.B) {
	heads, kvHeads, hd := 16, 8, 64
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	for _, L := range []int{256, 4096} {
		b.Run(itoa(L), func(b *testing.B) {
			rng := rand.New(rand.NewSource(int64(L)))
			q := make([]float32, L*heads*hd)
			k := make([]float32, L*kvHeads*hd)
			v := make([]float32, L*kvHeads*hd)
			for i := range q {
				q[i] = rng.Float32()*2 - 1
			}
			for i := range k {
				k[i] = rng.Float32()*2 - 1
				v[i] = rng.Float32()*2 - 1
			}
			qBuf := mustAllocAndCopyBench(b, q)
			defer func() { _ = qBuf.Free() }()
			kBuf := mustAllocAndCopyBench(b, k)
			defer func() { _ = kBuf.Free() }()
			vBuf := mustAllocAndCopyBench(b, v)
			defer func() { _ = vBuf.Free() }()
			outBuf, err := AllocDevice(int64(len(q) * 4))
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = outBuf.Free() }()
			ssBuf := mustAllocAndCopyInt32Bench(b, []int32{0})
			defer func() { _ = ssBuf.Free() }()
			slBuf := mustAllocAndCopyInt32Bench(b, []int32{int32(L)})
			defer func() { _ = slBuf.Free() }()
			stream, err := NewStream()
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = stream.Destroy() }()
			launch := func() {
				if err := VisionAttnSegmentsF32(qBuf, kBuf, vBuf, outBuf, ssBuf, slBuf, L, 1, heads, kvHeads, hd, scale, stream); err != nil {
					b.Fatal(err)
				}
			}
			launch()
			if err := stream.Synchronize(); err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				launch()
			}
			if err := stream.Synchronize(); err != nil {
				b.Fatal(err)
			}
		})
	}
}

func mustAllocAndCopyInt32Bench(b *testing.B, data []int32) DeviceBuffer {
	b.Helper()
	buf, err := AllocDevice(int64(len(data) * 4))
	if err != nil {
		b.Fatalf("AllocDevice: %v", err)
	}
	if err := MemcpyH2D(buf, unsafe.Pointer(&data[0]), int64(len(data)*4)); err != nil {
		_ = buf.Free()
		b.Fatalf("MemcpyH2D: %v", err)
	}
	return buf
}

func BenchmarkVisionConv2dGatedF32(b *testing.B) {
	n, C, k, m := 4096, 1024, 3, 2
	rng := rand.New(rand.NewSource(29))
	in := make([]float32, n*3*C)
	convW := make([]float32, C*k*k)
	for i := range in {
		in[i] = rng.Float32()*2 - 1
	}
	for i := range convW {
		convW[i] = rng.Float32()*0.1 - 0.05
	}
	inBuf := mustAllocAndCopyBench(b, in)
	defer func() { _ = inBuf.Free() }()
	wBuf := mustAllocAndCopyBench(b, convW)
	defer func() { _ = wBuf.Free() }()
	outBuf, err := AllocDevice(int64(n * C * 4))
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = outBuf.Free() }()
	fsBuf := mustAllocAndCopyInt32Bench(b, []int32{0})
	defer func() { _ = fsBuf.Free() }()
	fhBuf := mustAllocAndCopyInt32Bench(b, []int32{64})
	defer func() { _ = fhBuf.Free() }()
	fwBuf := mustAllocAndCopyInt32Bench(b, []int32{64})
	defer func() { _ = fwBuf.Free() }()
	stream, err := NewStream()
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = stream.Destroy() }()
	launch := func() {
		if err := VisionConv2dGatedF32(inBuf, wBuf, outBuf, fsBuf, fhBuf, fwBuf, 1, n, n, C, k, m, stream); err != nil {
			b.Fatal(err)
		}
	}
	launch()
	if err := stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		launch()
	}
	if err := stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
}

func BenchmarkLayerNormRowsF32(b *testing.B) {
	rows, cols := 4096, 1024
	rng := rand.New(rand.NewSource(31))
	x := make([]float32, rows*cols)
	w := make([]float32, cols)
	bias := make([]float32, cols)
	for i := range x {
		x[i] = rng.Float32()*4 - 2
	}
	for i := range w {
		w[i] = rng.Float32()*0.5 + 0.75
	}
	for i := range bias {
		bias[i] = rng.Float32() * 0.1
	}
	xBuf := mustAllocAndCopyBench(b, x)
	defer func() { _ = xBuf.Free() }()
	wBuf := mustAllocAndCopyBench(b, w)
	defer func() { _ = wBuf.Free() }()
	bBuf := mustAllocAndCopyBench(b, bias)
	defer func() { _ = bBuf.Free() }()
	stream, err := NewStream()
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = stream.Destroy() }()
	launch := func() {
		if err := LayerNormRowsF32(xBuf, xBuf, wBuf, bBuf, 1e-5, rows, cols, stream); err != nil {
			b.Fatal(err)
		}
	}
	launch()
	if err := stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		launch()
	}
	if err := stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
}

func BenchmarkGeluExactBiasF32(b *testing.B) {
	rows, cols := 1024, 4096
	rng := rand.New(rand.NewSource(37))
	x := make([]float32, rows*cols)
	bias := make([]float32, cols)
	for i := range x {
		x[i] = rng.Float32()*8 - 4
	}
	for i := range bias {
		bias[i] = rng.Float32()*0.5 - 0.25
	}
	xBuf := mustAllocAndCopyBench(b, x)
	defer func() { _ = xBuf.Free() }()
	bBuf := mustAllocAndCopyBench(b, bias)
	defer func() { _ = bBuf.Free() }()
	stream, err := NewStream()
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = stream.Destroy() }()
	launch := func() {
		if err := GeluExactBiasF32(xBuf, bBuf, rows, cols, stream); err != nil {
			b.Fatal(err)
		}
	}
	launch()
	if err := stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		launch()
	}
	if err := stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
}

func BenchmarkAddBiasRowsF32(b *testing.B) {
	rows, cols := 1024, 2048
	rng := rand.New(rand.NewSource(41))
	x := make([]float32, rows*cols)
	bias := make([]float32, cols)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	for i := range bias {
		bias[i] = rng.Float32() * 0.1
	}
	xBuf := mustAllocAndCopyBench(b, x)
	defer func() { _ = xBuf.Free() }()
	bBuf := mustAllocAndCopyBench(b, bias)
	defer func() { _ = bBuf.Free() }()
	stream, err := NewStream()
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = stream.Destroy() }()
	launch := func() {
		if err := AddBiasRowsF32(xBuf, bBuf, rows, cols, stream); err != nil {
			b.Fatal(err)
		}
	}
	launch()
	if err := stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		launch()
	}
	if err := stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
}
