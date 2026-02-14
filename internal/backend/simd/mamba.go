package simd

import (
	"math"

	"simd/archsimd"
)

func Mamba(m *Instance, layer *Layer, x []float32) []float32 {
	ml := layer.Mamba
	if ml == nil {
		return nil
	}
	if ml.InProj == nil || ml.OutProj == nil || ml.Conv == nil {
		return nil
	}
	syncDeviceSlice(m.Ops(), x)

	in := x
	if scale := m.Config.Config.SSMInMultiplier; scale != 0 && scale != 1 {
		buf := m.Scratch.MambaIn
		s := float32(scale)
		for i := range x {
			buf[i] = x[i] * s
		}
		in = buf
	}

	dInProj := 2*ml.Inner + 2*ml.Groups*ml.DState + ml.HeadCount
	proj := m.Scratch.MambaProj[:dInProj]
	m.Ops().MatVec(proj, ml.InProj, in)

	copy(m.Scratch.MambaZ[:ml.Inner], proj[:ml.Inner])
	copy(m.Scratch.MambaX[:ml.ConvChannels], proj[ml.Inner:ml.Inner+ml.ConvChannels])
	copy(m.Scratch.MambaDT[:ml.HeadCount], proj[ml.Inner+ml.ConvChannels:ml.Inner+ml.ConvChannels+ml.HeadCount])

	mambaDepthwiseConv(
		m.Scratch.MambaConv[:ml.ConvChannels],
		m.Scratch.MambaX[:ml.ConvChannels],
		ml.Conv,
		ml.ConvBias,
		ml.ConvState,
	)
	// Vectorized Silu activation
	n := ml.ConvChannels
	i := 0
	if cpu.HasAVX2 {
		for ; i+8 <= n; i += 8 {
			v := archsimd.LoadFloat32x8Slice(m.Scratch.MambaConv[i:])
			v = fastSiluVec(v)
			v.StoreSlice(m.Scratch.MambaConv[i:])
		}
	}
	for ; i < n; i++ {
		m.Scratch.MambaConv[i] = Silu(m.Scratch.MambaConv[i])
	}

	copy(m.Scratch.MambaX[:ml.Inner], m.Scratch.MambaConv[:ml.Inner])
	copy(m.Scratch.MambaB[:ml.Groups*ml.DState], m.Scratch.MambaConv[ml.Inner:ml.Inner+ml.Groups*ml.DState])
	copy(m.Scratch.MambaC[:ml.Groups*ml.DState], m.Scratch.MambaConv[ml.Inner+ml.Groups*ml.DState:ml.Inner+2*ml.Groups*ml.DState])

	dt := m.Scratch.MambaDT[:ml.HeadCount]
	for i := range dt {
		dt[i] = Softplus(dt[i] + ml.DTBias[i])
		dt[i] = clampTimeStep(dt[i], m.Config.Config.TimeStepMin, m.Config.Config.TimeStepMax, m.Config.Config.TimeStepFloor)
	}

	mambaScan(
		m.Scratch.MambaY[:ml.Inner],
		ml,
		m.Scratch.MambaX[:ml.Inner],
		dt,
		m.Scratch.MambaB[:ml.Groups*ml.DState],
		m.Scratch.MambaC[:ml.Groups*ml.DState],
	)

	y := m.Scratch.MambaY[:ml.Inner]
	if m.Config.Config.MambaRMSNorm && ml.Norm != nil {
		RMSNormGated(y, y, m.Scratch.MambaZ[:ml.Inner], ml.Norm, m.RMSEpsilon, m.Config.Config.MambaNormBeforeGate)
	} else {
		// Vectorized Silu with multiplication
		n := len(y)
		i := 0
		if cpu.HasAVX2 {
			for ; i+8 <= n; i += 8 {
				vy := archsimd.LoadFloat32x8Slice(y[i:])
				vz := archsimd.LoadFloat32x8Slice(m.Scratch.MambaZ[i:])
				vy = vy.Mul(fastSiluVec(vz))
				vy.StoreSlice(y[i:])
			}
		}
		for ; i < n; i++ {
			y[i] *= Silu(m.Scratch.MambaZ[i])
		}
	}

	m.Ops().MatVec(m.Scratch.MambaOut, ml.OutProj, y)
	if scale := m.Config.Config.SSMOutMultiplier; scale != 0 && scale != 1 {
		s := float32(scale)
		for i := range m.Scratch.MambaOut {
			m.Scratch.MambaOut[i] *= s
		}
	}
	return m.Scratch.MambaOut
}

func mambaDepthwiseConv(out, in []float32, kernel *Mat, bias []float32, state []float32) {
	kernelLen := kernel.C
	channels := kernel.R
	for c := range channels {
		row := kernel.Row(c)
		sum := float32(0)
		if len(bias) == channels {
			sum = bias[c]
		}
		for k := 0; k < kernelLen-1; k++ {
			sum += row[k] * state[k*channels+c]
		}
		sum += row[kernelLen-1] * in[c]
		out[c] = sum
	}
	if kernelLen > 1 {
		if kernelLen == 2 {
			copy(state, in)
		} else {
			copy(state, state[channels:])
			copy(state[(kernelLen-2)*channels:], in)
		}
	}
}

func mambaScan(out []float32, ml *MambaLayer, x []float32, dt []float32, b []float32, c []float32) {
	headDim := ml.HeadDim
	dState := ml.DState
	groupSize := ml.GroupSize
	for h := 0; h < ml.HeadCount; h++ {
		group := h / groupSize
		a := -float32(math.Exp(float64(ml.ALog[h])))
		dtH := dt[h]
		dA := float32(math.Exp(float64(a * dtH)))
		bGroup := b[group*dState : (group+1)*dState]
		cGroup := c[group*dState : (group+1)*dState]
		for p := range headDim {
			xhp := x[h*headDim+p]
			stateBase := (h*headDim + p) * dState
			var sum float32
			for n := range dState {
				idx := stateBase + n
				ml.SSMState[idx] = ml.SSMState[idx]*dA + dtH*bGroup[n]*xhp
				sum += cGroup[n] * ml.SSMState[idx]
			}
			out[h*headDim+p] = sum + ml.D[h]*xhp
		}
	}
}

func clampTimeStep(v float32, min, max, floor float64) float32 {
	if floor > 0 && v < float32(floor) {
		v = float32(floor)
	}
	if min > 0 && v < float32(min) {
		v = float32(min)
	}
	if max > 0 && v > float32(max) {
		v = float32(max)
	}
	return v
}
