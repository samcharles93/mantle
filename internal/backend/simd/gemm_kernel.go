package simd

import "simd/archsimd"

// gemmRangeRows performs a blocked GEMM on a contiguous range of rows of C.
func gemmRangeRows(
	cfg GemmConfig, C, A, B *Mat, alpha, beta float32, rs, re int, packB []float32,
) {
	tm := cfg.TileM
	tn := cfg.TileN
	tk := cfg.TileK

	if alpha == 1 && beta == 0 {
		gemmRangeRowsAlpha1Beta0(cfg, C, A, B, rs, re, packB)
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

func gemmRangeRowsAlpha1Beta0(cfg GemmConfig, C, A, B *Mat, rs, re int, packB []float32) {
	tm := cfg.TileM
	tn := cfg.TileN
	tk := cfg.TileK

	if cpu.HasAVX2 && len(packB) >= tk*tn {
		gemmRangeRowsAlpha1Beta0Packed(cfg, C, A, B, rs, re, packB)
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

func gemmRangeRowsAlpha1Beta0Packed(cfg GemmConfig, C, A, B *Mat, rs, re int, packB []float32) {
	tm := cfg.TileM
	tn := cfg.TileN
	tk := cfg.TileK

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

	for k0 := 0; k0 < k; k0 += tk {
		kMax := min(k0+tk, k)
		kInner := kMax - k0
		for j0 := 0; j0 < n; j0 += tn {
			jMax := min(j0+tn, n)
			width := jMax - j0

			packBTile(packB, bData, bStride, k0, kMax, j0, jMax)

			for i0 := rs; i0 < re; i0 += tm {
				iMax := min(i0+tm, re)
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
					vc0 = vb0.MulAdd(vaik, vc0)
					vc0.StoreSlice(cRow[j:])
					vc1 := archsimd.LoadFloat32x8Slice(cRow[j+8:])
					vb1 := archsimd.LoadFloat32x8Slice(bRow[j+8:])
					vc1 = vb1.MulAdd(vaik, vc1)
					vc1.StoreSlice(cRow[j+8:])
				}
				for ; j+8 <= width; j += 8 {
					vc := archsimd.LoadFloat32x8Slice(cRow[j:])
					vb := archsimd.LoadFloat32x8Slice(bRow[j:])
					vc = vb.MulAdd(vaik, vc)
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
				acc0 = vb0.MulAdd(vaik, acc0)
				vb1 := archsimd.LoadFloat32x8Slice(bRow[8:])
				acc1 = vb1.MulAdd(vaik, acc1)
				vb2 := archsimd.LoadFloat32x8Slice(bRow[16:])
				acc2 = vb2.MulAdd(vaik, acc2)
				vb3 := archsimd.LoadFloat32x8Slice(bRow[24:])
				acc3 = vb3.MulAdd(vaik, acc3)
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
				acc = vb.MulAdd(vaik, acc)
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
					vc0 = vb0.MulAdd(vaik, vc0)
					vc0.StoreSlice(cRow[j:])
					vc1 := archsimd.LoadFloat32x8Slice(cRow[j+8:])
					vb1 := archsimd.LoadFloat32x8Slice(bRow[j+8:])
					vc1 = vb1.MulAdd(vaik, vc1)
					vc1.StoreSlice(cRow[j+8:])
				}
				for ; j+8 <= width; j += 8 {
					vc := archsimd.LoadFloat32x8Slice(cRow[j:])
					vb := archsimd.LoadFloat32x8Slice(bRow[j:])
					vc = vb.MulAdd(vaik, vc)
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
				acc0 = vb0.MulAdd(vaik, acc0)
				vb1 := archsimd.LoadFloat32x8Slice(bRow[8:])
				acc1 = vb1.MulAdd(vaik, acc1)
				vb2 := archsimd.LoadFloat32x8Slice(bRow[16:])
				acc2 = vb2.MulAdd(vaik, acc2)
				vb3 := archsimd.LoadFloat32x8Slice(bRow[24:])
				acc3 = vb3.MulAdd(vaik, acc3)
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
				acc = vb.MulAdd(vaik, acc)
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
			for kk := range kInner {
				aik := aRow[k0+kk]
				vaik := archsimd.BroadcastFloat32x8(aik)
				bRow := packB[kk*width : kk*width+width]
				j := 0
				for ; j+16 <= width; j += 16 {
					vc0 := archsimd.LoadFloat32x8Slice(cRow[j:])
					vb0 := archsimd.LoadFloat32x8Slice(bRow[j:])
					vc0 = vb0.MulAdd(vaik, vc0)
					vc0.StoreSlice(cRow[j:])
					vc1 := archsimd.LoadFloat32x8Slice(cRow[j+8:])
					vb1 := archsimd.LoadFloat32x8Slice(bRow[j+8:])
					vc1 = vb1.MulAdd(vaik, vc1)
					vc1.StoreSlice(cRow[j+8:])
				}
				for ; j+8 <= width; j += 8 {
					vc := archsimd.LoadFloat32x8Slice(cRow[j:])
					vb := archsimd.LoadFloat32x8Slice(bRow[j:])
					vc = vb.MulAdd(vaik, vc)
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

			for kk := range kInner {
				aik := aRow[k0+kk]
				vaik := archsimd.BroadcastFloat32x8(aik)
				bRow := packB[kk*width : kk*width+width]

				vb0 := archsimd.LoadFloat32x8Slice(bRow[j:])
				acc0 = vb0.MulAdd(vaik, acc0)
				vb1 := archsimd.LoadFloat32x8Slice(bRow[j+8:])
				acc1 = vb1.MulAdd(vaik, acc1)
				vb2 := archsimd.LoadFloat32x8Slice(bRow[j+16:])
				acc2 = vb2.MulAdd(vaik, acc2)
				vb3 := archsimd.LoadFloat32x8Slice(bRow[j+24:])
				acc3 = vb3.MulAdd(vaik, acc3)
			}

			acc0.StoreSlice(cRow[j:])
			acc1.StoreSlice(cRow[j+8:])
			acc2.StoreSlice(cRow[j+16:])
			acc3.StoreSlice(cRow[j+24:])
		}

		for ; j+8 <= width; j += 8 {
			var acc archsimd.Float32x8
			acc = archsimd.LoadFloat32x8Slice(cRow[j:])
			for kk := range kInner {
				aik := aRow[k0+kk]
				vaik := archsimd.BroadcastFloat32x8(aik)
				bRow := packB[kk*width : kk*width+width]
				vb := archsimd.LoadFloat32x8Slice(bRow[j:])
				acc = vb.MulAdd(vaik, acc)
			}
			acc.StoreSlice(cRow[j:])
		}

		for ; j < width; j++ {
			for kk := range kInner {
				aik := aRow[k0+kk]
				cRow[j] += aik * packB[kk*width+j]
			}
		}
	}
}
