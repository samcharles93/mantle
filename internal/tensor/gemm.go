package tensor

import (
	"runtime"

	"simd/archsimd"
)

// Tuned for the benchmark shape (256^3). Tile sizes are variables to allow
// test-time sweeps without recompilation.
const (
	defaultTileM = 32
	defaultTileN = 32
	defaultTileK = 16

	maxTileM = 64
	maxTileN = 64
	maxTileK = 64
)

var (
	tileM = defaultTileM
	tileN = defaultTileN
	tileK = defaultTileK
)

func selectGemmTiles(m, k, n int) (int, int, int) {
	if tileM != defaultTileM || tileN != defaultTileN || tileK != defaultTileK {
		return clampTile(tileM, maxTileM), clampTile(tileN, maxTileN), clampTile(tileK, maxTileK)
	}

	tm := defaultTileM
	tn := defaultTileN
	tk := defaultTileK

	switch {
	case k >= 192:
		tk = 32
	case k >= 96:
		tk = 24
	}

	return clampTile(tm, maxTileM), clampTile(tn, maxTileN), clampTile(tk, maxTileK)
}

func clampTile(value, max int) int {
	if value < 1 {
		return 1
	}
	if value > max {
		return max
	}
	return value
}

type gemmTask struct {
	C, A, B     *Mat
	alpha, beta float32
	rs, re      int
	tm, tn, tk  int
	done        chan struct{}
}

type gemmPool struct {
	size      int
	tasks     chan gemmTask
	doneSlots chan chan struct{}
}

func newGemmPool() *gemmPool {
	size := runtime.GOMAXPROCS(0)
	if size < 1 {
		size = 1
	}
	p := &gemmPool{
		size:      size,
		tasks:     make(chan gemmTask, size*2),
		doneSlots: make(chan chan struct{}, size),
	}
	for i := 0; i < size; i++ {
		p.doneSlots <- make(chan struct{}, 1)
	}
	for w := 0; w < size; w++ {
		packB := make([]float32, maxTileK*maxTileN)
		go func(packB []float32) {
			for task := range p.tasks {
				gemmRangeRows(task.C, task.A, task.B, task.alpha, task.beta, task.rs, task.re, packB, task.tm, task.tn, task.tk)
				task.done <- struct{}{}
			}
		}(packB)
	}
	return p
}

var gemmWorkPool = newGemmPool()

// GemmPar computes the matrix product C = alpha*A*B + beta*C using a
// blocked algorithm and parallelising across ranges of output rows.
func GemmPar(C, A, B *Mat, alpha, beta float32, workers int) {
	if A.C != B.R || C.R != A.R || C.C != B.C {
		panic("gemm: dimension mismatch")
	}
	if C.R == 0 || C.C == 0 {
		return
	}

	tm, tn, tk := selectGemmTiles(C.R, A.C, B.C)

	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers > C.R {
		workers = C.R
	}
	if workers <= 1 {
		gemmRangeRows(C, A, B, alpha, beta, 0, C.R, nil, tm, tn, tk)
		return
	}
	if workers > gemmWorkPool.size {
		workers = gemmWorkPool.size
	}

	chunk := (C.R + workers - 1) / workers

	done := <-gemmWorkPool.doneSlots
	for w := 0; w < workers; w++ {
		rs := w * chunk
		re := rs + chunk
		if re > C.R {
			re = C.R
		}
		gemmWorkPool.tasks <- gemmTask{
			C:     C,
			A:     A,
			B:     B,
			alpha: alpha,
			beta:  beta,
			rs:    rs,
			re:    re,
			tm:    tm,
			tn:    tn,
			tk:    tk,
			done:  done,
		}
	}
	for i := 0; i < workers; i++ {
		<-done
	}
	gemmWorkPool.doneSlots <- done
}

// gemmRangeRows performs a blocked GEMM on a contiguous range of rows of C.
func gemmRangeRows(C, A, B *Mat, alpha, beta float32, rs, re int, packB []float32, tm, tn, tk int) {
	if alpha == 1 && beta == 0 {
		gemmRangeRowsAlpha1Beta0(C, A, B, rs, re, packB, tm, tn, tk)
		return
	}

	if beta == 0 {
		cStride := C.Stride
		n := C.C
		for i := rs; i < re; i++ {
			base := i * cStride
			clear(C.Data[base : base+n])
		}
	} else if beta != 1 {
		cStride := C.Stride
		n := C.C
		for i := rs; i < re; i++ {
			base := i * cStride
			for j := 0; j < n; j++ {
				C.Data[base+j] *= beta
			}
		}
	}

	n := B.C
	k := A.C
	aStride := A.Stride
	bStride := B.Stride
	cStride := C.Stride

	for i0 := rs; i0 < re; i0 += tm {
		iMax := min(i0+tm, re)
		for k0 := 0; k0 < k; k0 += tk {
			kMax := min(k0+tk, k)
			for j0 := 0; j0 < n; j0 += tn {
				jMax := min(j0+tn, n)
				blockUpdateGeneric(C.Data, A.Data, B.Data, cStride, aStride, bStride, alpha, i0, iMax, j0, jMax, k0, kMax)
			}
		}
	}
}

func gemmRangeRowsAlpha1Beta0(C, A, B *Mat, rs, re int, packB []float32, tm, tn, tk int) {
	if cpu.HasAVX2 && len(packB) >= tk*tn {
		gemmRangeRowsAlpha1Beta0Packed(C, A, B, rs, re, packB, tm, tn, tk)
		return
	}

	cStride := C.Stride
	n := C.C
	cData := C.Data

	for i := rs; i < re; i++ {
		base := i * cStride
		clear(cData[base : base+n])
	}

	k := A.C
	aStride := A.Stride
	bStride := B.Stride
	aData := A.Data
	bData := B.Data

	for i0 := rs; i0 < re; i0 += tm {
		iMax := min(i0+tm, re)
		for k0 := 0; k0 < k; k0 += tk {
			kMax := min(k0+tk, k)
			for j0 := 0; j0 < n; j0 += tn {
				jMax := min(j0+tn, n)
				blockUpdateAlpha1(cData, aData, bData, cStride, aStride, bStride, i0, iMax, j0, jMax, k0, kMax)
			}
		}
	}
}

func gemmRangeRowsAlpha1Beta0Packed(C, A, B *Mat, rs, re int, packB []float32, tm, tn, tk int) {
	cStride := C.Stride
	n := C.C
	cData := C.Data

	for i := rs; i < re; i++ {
		base := i * cStride
		clear(cData[base : base+n])
	}

	k := A.C
	aStride := A.Stride
	bStride := B.Stride
	aData := A.Data
	bData := B.Data

	for k0 := 0; k0 < k; k0 += tileK {
		kMax := min(k0+tileK, k)
		kInner := kMax - k0
		for j0 := 0; j0 < n; j0 += tileN {
			jMax := min(j0+tileN, n)
			width := jMax - j0

			packBTile(packB, bData, bStride, k0, kMax, j0, jMax)

			for i0 := rs; i0 < re; i0 += tileM {
				iMax := min(i0+tileM, re)
				blockUpdateAlpha1SIMDPacked(cData, aData, packB, cStride, aStride, i0, iMax, j0, width, k0, kInner)
			}
		}
	}
}

func packBTile(dst []float32, bData []float32, bStride int, k0, kMax, j0, jMax int) {
	width := jMax - j0
	kInner := kMax - k0
	if width <= 0 || kInner <= 0 {
		return
	}
	if width > maxTileN || kInner > maxTileK {
		panic("packBTile exceeds max tile size")
	}
	for kk := 0; kk < kInner; kk++ {
		srcOff := (k0+kk)*bStride + j0
		copy(dst[kk*width:(kk+1)*width], bData[srcOff:srcOff+width])
	}
}

func blockUpdateGeneric(cData, aData, bData []float32, cStride, aStride, bStride int, alpha float32, i0, iMax, j0, jMax, k0, kMax int) {
	if cpu.HasAVX2 {
		blockUpdateGenericSIMD(cData, aData, bData, cStride, aStride, bStride, alpha, i0, iMax, j0, jMax, k0, kMax)
		return
	}
	blockUpdateGenericScalar(cData, aData, bData, cStride, aStride, bStride, alpha, i0, iMax, j0, jMax, k0, kMax)
}

func blockUpdateGenericScalar(cData, aData, bData []float32, cStride, aStride, bStride int, alpha float32, i0, iMax, j0, jMax, k0, kMax int) {
	width := jMax - j0
	for i := i0; i < iMax; i++ {
		aRow := aData[i*aStride:]
		cOff := i*cStride + j0
		cRow := cData[cOff : cOff+width]

		for kk := k0; kk < kMax; kk++ {
			aik := aRow[kk] * alpha
			bOff := kk*bStride + j0
			bRow := bData[bOff : bOff+width]

			j := 0
			for ; j+7 < width; j += 8 {
				cRow[j+0] += aik * bRow[j+0]
				cRow[j+1] += aik * bRow[j+1]
				cRow[j+2] += aik * bRow[j+2]
				cRow[j+3] += aik * bRow[j+3]
				cRow[j+4] += aik * bRow[j+4]
				cRow[j+5] += aik * bRow[j+5]
				cRow[j+6] += aik * bRow[j+6]
				cRow[j+7] += aik * bRow[j+7]
			}
			for ; j < width; j++ {
				cRow[j] += aik * bRow[j]
			}
		}
	}
}

// blockUpdateGenericSIMD uses a micro-kernel approach.
// Accumulates across multiple kk iterations before storing to reduce memory traffic.
func blockUpdateGenericSIMD(cData, aData, bData []float32, cStride, aStride, bStride int, alpha float32, i0, iMax, j0, jMax, k0, kMax int) {
	width := jMax - j0
	kInner := kMax - k0

	for i := i0; i < iMax; i++ {
		aRow := aData[i*aStride:]
		cOff := i*cStride + j0
		cRow := cData[cOff : cOff+width]

		// For small k, use direct approach
		if kInner <= 4 {
			for kk := k0; kk < kMax; kk++ {
				aik := aRow[kk] * alpha
				vaik := archsimd.BroadcastFloat32x8(aik)
				bOff := kk*bStride + j0
				bRow := bData[bOff : bOff+width]
				j := 0
				for ; j+16 <= width; j += 16 {
					vc0 := archsimd.LoadFloat32x8Slice(cRow[j:])
					vb0 := archsimd.LoadFloat32x8Slice(bRow[j:])
					vc0 = vc0.Add(vb0.Mul(vaik))
					vc0.StoreSlice(cRow[j:])
					vc1 := archsimd.LoadFloat32x8Slice(cRow[j+8:])
					vb1 := archsimd.LoadFloat32x8Slice(bRow[j+8:])
					vc1 = vc1.Add(vb1.Mul(vaik))
					vc1.StoreSlice(cRow[j+8:])
				}
				for ; j+8 <= width; j += 8 {
					vc := archsimd.LoadFloat32x8Slice(cRow[j:])
					vb := archsimd.LoadFloat32x8Slice(bRow[j:])
					vc = vc.Add(vb.Mul(vaik))
					vc.StoreSlice(cRow[j:])
				}
				for ; j < width; j++ {
					cRow[j] += aik * bRow[j]
				}
			}
			continue
		}

		// For larger k, accumulate in registers across multiple kk iterations
		// We process 32 elements at a time (4 SIMD vectors)
		j := 0
		for ; j+32 <= width; j += 32 {
			// Initialize accumulators from C
			var acc0, acc1, acc2, acc3 archsimd.Float32x8
			acc0 = archsimd.LoadFloat32x8Slice(cRow[j:])
			acc1 = archsimd.LoadFloat32x8Slice(cRow[j+8:])
			acc2 = archsimd.LoadFloat32x8Slice(cRow[j+16:])
			acc3 = archsimd.LoadFloat32x8Slice(cRow[j+24:])

			// Accumulate across all kk iterations
			for kk := k0; kk < kMax; kk++ {
				aik := aRow[kk] * alpha
				vaik := archsimd.BroadcastFloat32x8(aik)
				bOff := kk*bStride + j
				bRow := bData[bOff : bOff+32]

				vb0 := archsimd.LoadFloat32x8Slice(bRow[0:])
				acc0 = acc0.Add(vb0.Mul(vaik))
				vb1 := archsimd.LoadFloat32x8Slice(bRow[8:])
				acc1 = acc1.Add(vb1.Mul(vaik))
				vb2 := archsimd.LoadFloat32x8Slice(bRow[16:])
				acc2 = acc2.Add(vb2.Mul(vaik))
				vb3 := archsimd.LoadFloat32x8Slice(bRow[24:])
				acc3 = acc3.Add(vb3.Mul(vaik))
			}

			// Store back to C
			acc0.StoreSlice(cRow[j:])
			acc1.StoreSlice(cRow[j+8:])
			acc2.StoreSlice(cRow[j+16:])
			acc3.StoreSlice(cRow[j+24:])
		}

		// Handle remaining elements with direct approach
		for ; j+8 <= width; j += 8 {
			var acc archsimd.Float32x8
			acc = archsimd.LoadFloat32x8Slice(cRow[j:])
			for kk := k0; kk < kMax; kk++ {
				aik := aRow[kk] * alpha
				vaik := archsimd.BroadcastFloat32x8(aik)
				bOff := kk*bStride + j0 + j
				vb := archsimd.LoadFloat32x8Slice(bData[bOff : bOff+8])
				acc = acc.Add(vb.Mul(vaik))
			}
			acc.StoreSlice(cRow[j:])
		}
		// Handle remaining elements with scalar
		for ; j < width; j++ {
			for kk := k0; kk < kMax; kk++ {
				aik := aRow[kk] * alpha
				bOff := kk*bStride + j0 + j
				cRow[j] += aik * bData[bOff]
			}
		}
	}
}

func blockUpdateAlpha1(cData, aData, bData []float32, cStride, aStride, bStride int, i0, iMax, j0, jMax, k0, kMax int) {
	if cpu.HasAVX2 {
		blockUpdateAlpha1SIMD(cData, aData, bData, cStride, aStride, bStride, i0, iMax, j0, jMax, k0, kMax)
		return
	}
	blockUpdateAlpha1Scalar(cData, aData, bData, cStride, aStride, bStride, i0, iMax, j0, jMax, k0, kMax)
}

func blockUpdateAlpha1Scalar(cData, aData, bData []float32, cStride, aStride, bStride int, i0, iMax, j0, jMax, k0, kMax int) {
	width := jMax - j0
	for i := i0; i < iMax; i++ {
		aRow := aData[i*aStride:]
		cOff := i*cStride + j0
		cRow := cData[cOff : cOff+width]

		for kk := k0; kk < kMax; kk++ {
			aik := aRow[kk]
			bOff := kk*bStride + j0
			bRow := bData[bOff : bOff+width]

			j := 0
			for ; j+3 < width; j += 4 {
				cRow[j+0] += aik * bRow[j+0]
				cRow[j+1] += aik * bRow[j+1]
				cRow[j+2] += aik * bRow[j+2]
				cRow[j+3] += aik * bRow[j+3]
			}
			for ; j < width; j++ {
				cRow[j] += aik * bRow[j]
			}
		}
	}
}

// blockUpdateAlpha1SIMD uses a micro-kernel approach.
// Accumulates across multiple kk iterations before storing to reduce memory traffic.
func blockUpdateAlpha1SIMD(cData, aData, bData []float32, cStride, aStride, bStride int, i0, iMax, j0, jMax, k0, kMax int) {
	width := jMax - j0
	kInner := kMax - k0

	for i := i0; i < iMax; i++ {
		aRow := aData[i*aStride:]
		cOff := i*cStride + j0
		cRow := cData[cOff : cOff+width]

		// For small k, use direct approach
		if kInner <= 4 {
			for kk := k0; kk < kMax; kk++ {
				aik := aRow[kk]
				vaik := archsimd.BroadcastFloat32x8(aik)
				bOff := kk*bStride + j0
				bRow := bData[bOff : bOff+width]
				j := 0
				for ; j+16 <= width; j += 16 {
					vc0 := archsimd.LoadFloat32x8Slice(cRow[j:])
					vb0 := archsimd.LoadFloat32x8Slice(bRow[j:])
					vc0 = vc0.Add(vb0.Mul(vaik))
					vc0.StoreSlice(cRow[j:])
					vc1 := archsimd.LoadFloat32x8Slice(cRow[j+8:])
					vb1 := archsimd.LoadFloat32x8Slice(bRow[j+8:])
					vc1 = vc1.Add(vb1.Mul(vaik))
					vc1.StoreSlice(cRow[j+8:])
				}
				for ; j+8 <= width; j += 8 {
					vc := archsimd.LoadFloat32x8Slice(cRow[j:])
					vb := archsimd.LoadFloat32x8Slice(bRow[j:])
					vc = vc.Add(vb.Mul(vaik))
					vc.StoreSlice(cRow[j:])
				}
				for ; j < width; j++ {
					cRow[j] += aik * bRow[j]
				}
			}
			continue
		}

		// For larger k, accumulate in registers across multiple kk iterations
		j := 0
		for ; j+32 <= width; j += 32 {
			var acc0, acc1, acc2, acc3 archsimd.Float32x8
			acc0 = archsimd.LoadFloat32x8Slice(cRow[j:])
			acc1 = archsimd.LoadFloat32x8Slice(cRow[j+8:])
			acc2 = archsimd.LoadFloat32x8Slice(cRow[j+16:])
			acc3 = archsimd.LoadFloat32x8Slice(cRow[j+24:])

			for kk := k0; kk < kMax; kk++ {
				aik := aRow[kk]
				vaik := archsimd.BroadcastFloat32x8(aik)
				bOff := kk*bStride + j
				bRow := bData[bOff : bOff+32]

				vb0 := archsimd.LoadFloat32x8Slice(bRow[0:])
				acc0 = acc0.Add(vb0.Mul(vaik))
				vb1 := archsimd.LoadFloat32x8Slice(bRow[8:])
				acc1 = acc1.Add(vb1.Mul(vaik))
				vb2 := archsimd.LoadFloat32x8Slice(bRow[16:])
				acc2 = acc2.Add(vb2.Mul(vaik))
				vb3 := archsimd.LoadFloat32x8Slice(bRow[24:])
				acc3 = acc3.Add(vb3.Mul(vaik))
			}

			acc0.StoreSlice(cRow[j:])
			acc1.StoreSlice(cRow[j+8:])
			acc2.StoreSlice(cRow[j+16:])
			acc3.StoreSlice(cRow[j+24:])
		}

		// Handle remaining elements
		for ; j+8 <= width; j += 8 {
			var acc archsimd.Float32x8
			acc = archsimd.LoadFloat32x8Slice(cRow[j:])
			for kk := k0; kk < kMax; kk++ {
				aik := aRow[kk]
				vaik := archsimd.BroadcastFloat32x8(aik)
				bOff := kk*bStride + j0 + j
				vb := archsimd.LoadFloat32x8Slice(bData[bOff : bOff+8])
				acc = acc.Add(vb.Mul(vaik))
			}
			acc.StoreSlice(cRow[j:])
		}
		// Handle remaining elements with scalar
		for ; j < width; j++ {
			for kk := k0; kk < kMax; kk++ {
				aik := aRow[kk]
				bOff := kk*bStride + j0 + j
				cRow[j] += aik * bData[bOff]
			}
		}
	}
}

// blockUpdateAlpha1SIMDPacked uses a packed B tile with contiguous rows.
// packB is arranged as kInner rows of length width.
func blockUpdateAlpha1SIMDPacked(cData, aData, packB []float32, cStride, aStride int, i0, iMax, j0, width, k0, kInner int) {
	if width <= 0 || kInner <= 0 {
		return
	}
	for i := i0; i < iMax; i++ {
		aRow := aData[i*aStride:]
		cOff := i*cStride + j0
		cRow := cData[cOff : cOff+width]

		if kInner <= 4 {
			for kk := 0; kk < kInner; kk++ {
				aik := aRow[k0+kk]
				vaik := archsimd.BroadcastFloat32x8(aik)
				bRow := packB[kk*width : kk*width+width]
				j := 0
				for ; j+16 <= width; j += 16 {
					vc0 := archsimd.LoadFloat32x8Slice(cRow[j:])
					vb0 := archsimd.LoadFloat32x8Slice(bRow[j:])
					vc0 = vc0.Add(vb0.Mul(vaik))
					vc0.StoreSlice(cRow[j:])
					vc1 := archsimd.LoadFloat32x8Slice(cRow[j+8:])
					vb1 := archsimd.LoadFloat32x8Slice(bRow[j+8:])
					vc1 = vc1.Add(vb1.Mul(vaik))
					vc1.StoreSlice(cRow[j+8:])
				}
				for ; j+8 <= width; j += 8 {
					vc := archsimd.LoadFloat32x8Slice(cRow[j:])
					vb := archsimd.LoadFloat32x8Slice(bRow[j:])
					vc = vc.Add(vb.Mul(vaik))
					vc.StoreSlice(cRow[j:])
				}
				for ; j < width; j++ {
					cRow[j] += aik * bRow[j]
				}
			}
			continue
		}

		j := 0
		for ; j+32 <= width; j += 32 {
			var acc0, acc1, acc2, acc3 archsimd.Float32x8
			acc0 = archsimd.LoadFloat32x8Slice(cRow[j:])
			acc1 = archsimd.LoadFloat32x8Slice(cRow[j+8:])
			acc2 = archsimd.LoadFloat32x8Slice(cRow[j+16:])
			acc3 = archsimd.LoadFloat32x8Slice(cRow[j+24:])

			for kk := 0; kk < kInner; kk++ {
				aik := aRow[k0+kk]
				vaik := archsimd.BroadcastFloat32x8(aik)
				bRow := packB[kk*width : kk*width+width]

				vb0 := archsimd.LoadFloat32x8Slice(bRow[j:])
				acc0 = acc0.Add(vb0.Mul(vaik))
				vb1 := archsimd.LoadFloat32x8Slice(bRow[j+8:])
				acc1 = acc1.Add(vb1.Mul(vaik))
				vb2 := archsimd.LoadFloat32x8Slice(bRow[j+16:])
				acc2 = acc2.Add(vb2.Mul(vaik))
				vb3 := archsimd.LoadFloat32x8Slice(bRow[j+24:])
				acc3 = acc3.Add(vb3.Mul(vaik))
			}

			acc0.StoreSlice(cRow[j:])
			acc1.StoreSlice(cRow[j+8:])
			acc2.StoreSlice(cRow[j+16:])
			acc3.StoreSlice(cRow[j+24:])
		}

		for ; j+8 <= width; j += 8 {
			var acc archsimd.Float32x8
			acc = archsimd.LoadFloat32x8Slice(cRow[j:])
			for kk := 0; kk < kInner; kk++ {
				aik := aRow[k0+kk]
				vaik := archsimd.BroadcastFloat32x8(aik)
				bRow := packB[kk*width : kk*width+width]
				vb := archsimd.LoadFloat32x8Slice(bRow[j:])
				acc = acc.Add(vb.Mul(vaik))
			}
			acc.StoreSlice(cRow[j:])
		}

		for ; j < width; j++ {
			for kk := 0; kk < kInner; kk++ {
				aik := aRow[k0+kk]
				cRow[j] += aik * packB[kk*width+j]
			}
		}
	}
}
