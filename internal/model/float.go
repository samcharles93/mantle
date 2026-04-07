package model

import "math"

// RoundFloat32ToBF16 rounds f to the nearest value representable in BF16.
func RoundFloat32ToBF16(f float32) float32 {
	bits := math.Float32bits(f)
	lsb := (bits >> 16) & 1
	bias := uint32(0x7FFF + lsb)
	return math.Float32frombits((bits + bias) & 0xFFFF0000)
}
