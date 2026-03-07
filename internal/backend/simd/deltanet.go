package simd

import "math"

// DeltaNet implements the Qwen3.5 linear-attention/Gated DeltaNet block for a
// single token step. It follows the upstream recurrent formulation used for
// decoding and keeps the recurrent state on the host for now.
func DeltaNet(m *Instance, layer *Layer, x []float32) []float32 {
	dl := layer.DeltaNet
	if dl == nil {
		return nil
	}
	syncDeviceSlice(m.Ops(), x)

	ops := m.Ops()
	ops.MatVec(m.Scratch.DeltaQKV[:dl.Conv.R], dl.QKVProj, x)
	ops.MatVec(m.Scratch.DeltaA[:dl.NumValueHeads], dl.AProj, x)
	ops.MatVec(m.Scratch.DeltaB[:dl.NumValueHeads], dl.BProj, x)
	ops.MatVec(m.Scratch.DeltaZ[:dl.ValueDim], dl.ZProj, x)

	mambaDepthwiseConv(
		m.Scratch.DeltaQKV[:dl.Conv.R],
		m.Scratch.DeltaQKV[:dl.Conv.R],
		dl.Conv,
		nil,
		dl.ConvState,
	)
	for i := 0; i < dl.Conv.R; i++ {
		m.Scratch.DeltaQKV[i] = Silu(m.Scratch.DeltaQKV[i])
	}

	deltaNetSplitQKV(
		m.Scratch.DeltaQ[:dl.KeyDim],
		m.Scratch.DeltaK[:dl.KeyDim],
		m.Scratch.DeltaV[:dl.ValueDim],
		m.Scratch.DeltaQKV[:dl.Conv.R],
		dl.NumKeyHeads,
		dl.NumValueHeads,
		dl.HeadKeyDim,
		dl.HeadValueDim,
	)

	out := m.Scratch.DeltaOut[:dl.ValueDim]
	for i := range out {
		out[i] = 0
	}

	groupSize := 1
	if dl.NumKeyHeads > 0 && dl.NumValueHeads > dl.NumKeyHeads {
		groupSize = dl.NumValueHeads / dl.NumKeyHeads
	}
	scale := float32(1.0 / math.Sqrt(float64(dl.HeadKeyDim)))
	eps := float32(1e-6)

	for hv := 0; hv < dl.NumValueHeads; hv++ {
		hk := hv
		if groupSize > 1 {
			hk = hv / groupSize
		}
		qHead := m.Scratch.DeltaQ[hk*dl.HeadKeyDim : (hk+1)*dl.HeadKeyDim]
		kHead := m.Scratch.DeltaK[hk*dl.HeadKeyDim : (hk+1)*dl.HeadKeyDim]
		vHead := m.Scratch.DeltaV[hv*dl.HeadValueDim : (hv+1)*dl.HeadValueDim]
		zHead := m.Scratch.DeltaZ[hv*dl.HeadValueDim : (hv+1)*dl.HeadValueDim]
		outHead := out[hv*dl.HeadValueDim : (hv+1)*dl.HeadValueDim]
		state := dl.RecurrentState[hv*dl.HeadKeyDim*dl.HeadValueDim : (hv+1)*dl.HeadKeyDim*dl.HeadValueDim]

		deltaNetL2NormInPlace(qHead, eps)
		deltaNetL2NormInPlace(kHead, eps)
		g := -fastExp(dl.ALog[hv]) * Softplus(m.Scratch.DeltaA[hv]+dl.DTBias[hv])
		beta := Sigmoid(m.Scratch.DeltaB[hv])
		decay := fastExp(g)

		for i := range state {
			state[i] *= decay
		}

		delta := m.Scratch.AttnOut[:dl.HeadValueDim]
		for v := 0; v < dl.HeadValueDim; v++ {
			var kvMem float32
			for k := 0; k < dl.HeadKeyDim; k++ {
				kvMem += state[k*dl.HeadValueDim+v] * kHead[k]
			}
			delta[v] = (vHead[v] - kvMem) * beta
		}

		for k := 0; k < dl.HeadKeyDim; k++ {
			base := k * dl.HeadValueDim
			kk := kHead[k]
			for v := 0; v < dl.HeadValueDim; v++ {
				state[base+v] += kk * delta[v]
			}
		}

		for v := 0; v < dl.HeadValueDim; v++ {
			var sum float32
			for k := 0; k < dl.HeadKeyDim; k++ {
				sum += state[k*dl.HeadValueDim+v] * (qHead[k] * scale)
			}
			outHead[v] = sum
		}

		RMSNormGated(outHead, outHead, zHead, dl.Norm, m.RMSEpsilon, true)
	}

	ops.MatVec(m.Scratch.Tmp, dl.OutProj, out)
	return m.Scratch.Tmp
}

func deltaNetL2NormInPlace(x []float32, eps float32) {
	var sum float32
	for _, v := range x {
		sum += v * v
	}
	scale := float32(1.0 / math.Sqrt(float64(sum+eps)))
	for i, v := range x {
		x[i] = v * scale
	}
}

func deltaNetSplitQKV(qDst, kDst, vDst, mixed []float32, numKeyHeads, numValueHeads, headKeyDim, headValueDim int) {
	if numKeyHeads <= 0 || numValueHeads <= 0 || headKeyDim <= 0 || headValueDim <= 0 {
		panic("invalid DeltaNet grouped QKV shape")
	}
	if numValueHeads%numKeyHeads != 0 {
		panic("delta value heads must be divisible by key heads")
	}
	valuesPerKey := numValueHeads / numKeyHeads
	valueGroupDim := valuesPerKey * headValueDim
	groupDim := 2*headKeyDim + valueGroupDim
	if len(mixed) < numKeyHeads*groupDim {
		panic("delta mixed QKV buffer too small")
	}
	if len(qDst) < numKeyHeads*headKeyDim || len(kDst) < numKeyHeads*headKeyDim || len(vDst) < numValueHeads*headValueDim {
		panic("delta output buffers too small")
	}
	for hk := range numKeyHeads {
		srcBase := hk * groupDim
		copy(qDst[hk*headKeyDim:(hk+1)*headKeyDim], mixed[srcBase:srcBase+headKeyDim])
		srcBase += headKeyDim
		copy(kDst[hk*headKeyDim:(hk+1)*headKeyDim], mixed[srcBase:srcBase+headKeyDim])
		srcBase += headKeyDim
		vBase := hk * valueGroupDim
		copy(vDst[vBase:vBase+valueGroupDim], mixed[srcBase:srcBase+valueGroupDim])
	}
}
