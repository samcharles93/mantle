package gguf

import (
	"errors"
	"fmt"
	"math"
)

const (
	QK_K         = 256
	q4kBlockSize = 2 + 2 + 12 + 128
	q6kBlockSize = 2 + 128 + 64 + 16
)

func DequantizeQ4K(data []byte, n int) ([]float32, error) {
	if n%QK_K != 0 {
		return nil, fmt.Errorf("q4_k: n must be multiple of %d", QK_K)
	}
	blocks := n / QK_K
	if len(data) != blocks*q4kBlockSize {
		return nil, fmt.Errorf("q4_k: invalid data length %d for n=%d", len(data), n)
	}
	out := make([]float32, n)
	off := 0
	for b := range blocks {
		d := fp16ToFloat32(data[off : off+2])
		dmin := fp16ToFloat32(data[off+2 : off+4])
		scales := data[off+4 : off+4+12]
		qs := data[off+4+12 : off+q4kBlockSize]

		y := out[b*QK_K:]
		is := 0
		q := qs
		yi := 0
		for j := 0; j < QK_K; j += 64 {
			sc1, m1 := getScaleMinK4(is+0, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			d1 := d * float32(sc1)
			d2 := d * float32(sc2)
			mm1 := dmin * float32(m1)
			mm2 := dmin * float32(m2)
			for l := range 32 {
				v := q[l]
				y[yi] = d1*float32(v&0x0F) - mm1
				yi++
			}
			for l := range 32 {
				v := q[l]
				y[yi] = d2*float32(v>>4) - mm2
				yi++
			}
			q = q[32:]
			is += 2
		}

		off += q4kBlockSize
	}
	return out, nil
}

func DequantizeQ6K(data []byte, n int) ([]float32, error) {
	if n%QK_K != 0 {
		return nil, fmt.Errorf("q6_k: n must be multiple of %d", QK_K)
	}
	blocks := n / QK_K
	if len(data) != blocks*q6kBlockSize {
		return nil, fmt.Errorf("q6_k: invalid data length %d for n=%d", len(data), n)
	}
	out := make([]float32, n)
	off := 0
	for b := range blocks {
		d := fp16ToFloat32(data[off : off+2])
		ql := data[off+2 : off+2+128]
		qh := data[off+2+128 : off+2+128+64]
		scales := data[off+2+128+64 : off+q6kBlockSize]

		y := out[b*QK_K:]
		yi := 0
		qlp := ql
		qhp := qh
		scp := scales
		for n := 0; n < QK_K; n += 128 {
			for l := range 32 {
				is := l / 16
				q1 := int8((qlp[l+0]&0x0F)|(((qhp[l]>>0)&3)<<4)) - 32
				q2 := int8((qlp[l+32]&0x0F)|(((qhp[l]>>2)&3)<<4)) - 32
				q3 := int8((qlp[l+0]>>4)|(((qhp[l]>>4)&3)<<4)) - 32
				q4 := int8((qlp[l+32]>>4)|(((qhp[l]>>6)&3)<<4)) - 32
				y[yi+0] = d * float32(int8(scp[is+0])) * float32(q1)
				y[yi+32] = d * float32(int8(scp[is+2])) * float32(q2)
				y[yi+64] = d * float32(int8(scp[is+4])) * float32(q3)
				y[yi+96] = d * float32(int8(scp[is+6])) * float32(q4)
				yi++
			}
			yi += 96
			qlp = qlp[64:]
			qhp = qhp[32:]
			scp = scp[8:]
		}

		off += q6kBlockSize
	}
	return out, nil
}

func getScaleMinK4(j int, scales []byte) (uint8, uint8) {
	if j < 4 {
		return scales[j] & 63, scales[j+4] & 63
	}
	d := (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
	m := (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
	return d, m
}

func fp16ToFloat32(b []byte) float32 {
	if len(b) < 2 {
		return float32(math.NaN())
	}
	h := uint16(b[0]) | uint16(b[1])<<8
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1F
	frac := uint32(h & 0x3FF)
	var f uint32
	switch exp {
	case 0:
		if frac == 0 {
			f = sign << 31
		} else {
			e := uint32(127 - 15 + 1)
			for (frac & 0x400) == 0 {
				frac <<= 1
				e--
			}
			frac &= 0x3FF
			f = (sign << 31) | (e << 23) | (frac << 13)
		}
	case 0x1F:
		f = (sign << 31) | 0x7F800000 | (frac << 13)
	default:
		e := exp + (127 - 15)
		f = (sign << 31) | (e << 23) | (frac << 13)
	}
	return math.Float32frombits(f)
}

var ErrUnsupportedType = errors.New("unsupported tensor type")
