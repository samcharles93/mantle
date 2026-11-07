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

// shortConvBlock runs the recurrent (short-conv) block for every row of a
// batched chunk and leaves the out-projection output in p.proj. It mirrors
// ShortConv above for a whole [n,hidden] block: the in-projection is one GEMM
// over the chunk, the gated conv input is formed per row, the causal
// convolution reads the chunk's earlier rows plus the layer's carried state,
// and the state is rotated forward by the chunk's rows at the end, so a chunk
// boundary is invisible to the next chunk and to later single tokens.
func (p *batchPrefillChunk) shortConvBlock(layer *Layer) {
	n := p.n
	hidden := p.hidden

	p.gemmRows(p.scProj, p.norm, layer.ShortConvInProj) // [n, 3*hidden]
	for r := range n {
		proj := p.scProj[r*3*hidden : (r+1)*3*hidden]
		b, xg := proj[:hidden], proj[2*hidden:]
		bx := p.scBx[r*hidden : (r+1)*hidden]
		for i := range hidden {
			bx[i] = b[i] * xg[i]
		}
	}

	kernel := layer.ShortConvKernel
	kernelLen := kernel.C
	state := layer.ShortConvState.Buf
	// The state is oldest-first: state[k] holds the gated conv input of global
	// position pos-(kernelLen-2)+k, so tap k of the causal conv (weight
	// row[k], input at pos-(kernelLen-1)+k) reads state slot r+k while the
	// position is still before the chunk, and chunk row r-kernelLen+1+k once
	// it is inside it. The final tap row[kernelLen-1] reads the current row.
	for r := range n {
		conv := p.scConv[r*hidden : (r+1)*hidden]
		for i := range hidden {
			row := kernel.Row(i)
			var sum float32
			for k := 0; k < kernelLen-1; k++ {
				if j := r - kernelLen + 1 + k; j >= 0 {
					sum += row[k] * p.scBx[j*hidden+i]
				} else {
					sum += row[k] * state[(r+k)*hidden+i]
				}
			}
			sum += row[kernelLen-1] * p.scBx[r*hidden+i]
			conv[i] = sum
		}
		// c gates the conv output; conv is then the out-projection input.
		c := p.scProj[r*3*hidden+hidden : r*3*hidden+2*hidden]
		for i := range hidden {
			conv[i] *= c[i]
		}
	}
	p.gemmRows(p.proj, p.scConv, layer.ShortConvOutProj)

	// Rotate the state forward, matching the per-token shift-append: afterwards
	// state[k] holds the gated conv input of global position
	// startPos+n-kernelLen+1+k, i.e. chunk row n-kernelLen+1+k while that is
	// inside the chunk and old state row n+k beyond it. The staging copy keeps
	// the old rows readable while the new ones are written.
	if kernelLen > 1 {
		old := p.scState[:(kernelLen-1)*hidden]
		copy(old, state)
		for k := 0; k < kernelLen-1; k++ {
			dst := state[k*hidden : (k+1)*hidden]
			if row := n - kernelLen + 1 + k; row >= 0 {
				copy(dst, p.scBx[row*hidden:(row+1)*hidden])
			} else {
				copy(dst, old[(n+k)*hidden:(n+k+1)*hidden])
			}
		}
	}
}
