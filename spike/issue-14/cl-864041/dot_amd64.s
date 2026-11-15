#include "textflag.h"

// func dotU8I8(a, b *[32]uint8) [8]int32
TEXT ·dotU8I8(SB), NOSPLIT, $0-40
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	VMOVDQU (AX), Y0
	VMOVDQU (BX), Y1
	VPXOR Y2, Y2, Y2
	VPDPBUSD Y1, Y0, Y2
	VMOVDQU Y2, ret+16(FP)
	VZEROUPPER
	RET
