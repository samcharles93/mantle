package simd

type shortConvFastPath interface {
	ShortConvBlock(layer *Layer, x []float32, out []float32, embd int) bool
}

// ShortConv implements recurrent attention with a short convolution kernel.
// This is used in architectures like Gemma3 that use hybrid attention mechanisms.
func ShortConv(m *Instance, layer *Layer, x []float32) []float32 {
	embd := m.Config.Config.EmbeddingLength

	// Try CUDA fast path: runs InProj, conv, OutProj entirely on device
	if fp, ok := m.Ops().(shortConvFastPath); ok && fp.ShortConvBlock(layer, x, m.Scratch.Tmp, embd) {
		return m.Scratch.Tmp
	}
	syncDeviceSlice(m.Ops(), x)

	m.Ops().MatVec(m.Scratch.ScProj, layer.ShortConvInProj, x)
	b := m.Scratch.ScProj[:embd]
	c := m.Scratch.ScProj[embd : 2*embd]
	xg := m.Scratch.ScProj[2*embd:]

	bx := m.Scratch.ScBx
	for i := range embd {
		bx[i] = b[i] * xg[i]
	}

	kernel := layer.ShortConvKernel
	convOut := m.Scratch.ScConv
	kernelLen := kernel.C
	state := layer.ShortConvState.Buf
	for i := range embd {
		row := kernel.Row(i)
		var sum float32
		for k := 0; k < kernelLen-1; k++ {
			sum += row[k] * state[k*embd+i]
		}
		sum += row[kernelLen-1] * bx[i]
		convOut[i] = sum
	}

	// update state: shift left and append current bx
	if kernelLen > 1 {
		if kernelLen == 2 {
			copy(state, bx)
		} else {
			copy(state, state[embd:])
			copy(state[(kernelLen-2)*embd:], bx)
		}
	}

	for i := range embd {
		m.Scratch.Tmp2[i] = c[i] * convOut[i]
	}
	m.Ops().MatVec(m.Scratch.Tmp, layer.ShortConvOutProj, m.Scratch.Tmp2)
	return m.Scratch.Tmp
}
