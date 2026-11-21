package main

import "fmt"

// dotU8I8 computes sum over 32 bytes of (u8 * i8) as int32 in lanes, using
// the VEX-encoded VPDPBUSD that this CL adds to the assembler.
func dotU8I8(a, b *[32]uint8) [8]int32

func encil()

func main() {
	encil()
	var a, b [32]uint8
	for i := range a {
		a[i] = uint8(i + 1)
		b[i] = uint8(i) // signed values 0..31, low nibble test
	}
	got := dotU8I8(&a, &b)
	fmt.Println("lanes:", got)
	var acc [8]int32
	for lane := range 8 {
		var sum int32
		for k := range 4 {
			i := lane*4 + k
			sum += int32(a[i]) * int32(int8(b[i]))
		}
		acc[lane] = sum
	}
	fmt.Println("want: ", acc)
	ok := got == acc
	fmt.Println("MATCH:", ok)
}
