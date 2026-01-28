package model

import (
	"math"
	"strings"
)

func ropeScalingForConfig(cfg *hfConfig) *RopeScaling {
	if cfg == nil {
		return nil
	}

	if cfg.RopeScaling != nil {
		out := ropeScalingFromValues(
			cfg.MaxPosition,
			cfg.RopeScaling.RopeType,
			cfg.RopeScaling.Type,
			cfg.RopeScaling.Factor,
			cfg.RopeScaling.OriginalMaxPositionEmbeddings,
			cfg.RopeScaling.LowFreqFactor,
			cfg.RopeScaling.HighFreqFactor,
			cfg.RopeScaling.AttentionFactor,
			cfg.RopeScaling.BetaFast,
			cfg.RopeScaling.BetaSlow,
			cfg.RopeScaling.MScale,
			cfg.RopeScaling.MScaleAllDim,
			cfg.RopeScaling.Truncate,
		)
		if out != nil {
			return out
		}
	}

	if cfg.RopeParameters == nil {
		return nil
	}
	rp := cfg.RopeParameters
	return ropeScalingFromValues(
		cfg.MaxPosition,
		rp.RopeType,
		rp.Type,
		rp.Factor,
		rp.OriginalMaxPositionEmbeddings,
		rp.LowFreqFactor,
		rp.HighFreqFactor,
		rp.AttentionFactor,
		rp.BetaFast,
		rp.BetaSlow,
		rp.MScale,
		rp.MScaleAllDim,
		rp.Truncate,
	)
}

func ropeScalingFromValues(
	maxPosition int,
	ropeTypeField string,
	typeField string,
	factor float64,
	origMaxCtx int,
	lowFactor float64,
	highFactor float64,
	attentionFactor float64,
	betaFast float64,
	betaSlow float64,
	mscale float64,
	mscaleAllDim float64,
	truncate *bool,
) *RopeScaling {
	ropeType := strings.TrimSpace(ropeTypeField)
	if ropeType == "" {
		ropeType = strings.TrimSpace(typeField)
	}
	ropeType = strings.ToLower(ropeType)

	if ropeType == "" || ropeType == "default" {
		if factor > 0 {
			ropeType = "linear"
		} else {
			return nil
		}
	}

	switch ropeType {
	case "linear", "llama3", "yarn":
		// supported
	default:
		return nil
	}

	out := &RopeScaling{
		Type:            ropeType,
		Factor:          factor,
		OrigMaxCtx:      origMaxCtx,
		LowFactor:       lowFactor,
		HighFactor:      highFactor,
		AttentionFactor: attentionFactor,
		BetaFast:        betaFast,
		BetaSlow:        betaSlow,
		MScale:          mscale,
		MScaleAllDim:    mscaleAllDim,
	}
	if truncate != nil {
		out.Truncate = *truncate
		out.HasTruncate = true
	}

	if out.OrigMaxCtx <= 0 {
		out.OrigMaxCtx = maxPosition
	}
	if out.LowFactor <= 0 {
		out.LowFactor = 1
	}
	if out.HighFactor <= 0 {
		out.HighFactor = out.LowFactor
	}
	if out.BetaFast <= 0 {
		out.BetaFast = 32
	}
	if out.BetaSlow <= 0 {
		out.BetaSlow = 1
	}

	if out.Factor <= 0 && out.OrigMaxCtx > 0 && maxPosition > 0 && maxPosition != out.OrigMaxCtx {
		out.Factor = float64(maxPosition) / float64(out.OrigMaxCtx)
	}
	if out.Factor <= 0 {
		out.Factor = 1
	}
	if out.Type == "yarn" && out.AttentionFactor <= 0 {
		out.AttentionFactor = yarnAttentionFactor(out.Factor, out.MScale, out.MScaleAllDim)
	} else if out.AttentionFactor <= 0 {
		out.AttentionFactor = 1
	}

	return out
}

func applyRopeScaling(invFreq []float64, base float64, ctxLen int, rs *RopeScaling) float64 {
	if len(invFreq) == 0 || rs == nil {
		return 1
	}
	if base <= 0 {
		base = 10_000
	}
	origCtx := rs.OrigMaxCtx
	if origCtx <= 0 {
		origCtx = ctxLen
	}
	if origCtx <= 0 {
		origCtx = 1
	}

	factor := rs.Factor
	if factor <= 0 && ctxLen > 0 && origCtx > 0 {
		factor = float64(ctxLen) / float64(origCtx)
	}
	if factor <= 0 {
		factor = 1
	}

	attnFactor := rs.AttentionFactor
	if attnFactor <= 0 {
		attnFactor = 1
	}

	switch rs.Type {
	case "llama3":
		applyLlama3Scaling(invFreq, factor, float64(origCtx), rs.LowFactor, rs.HighFactor)
	case "yarn":
		if rs.AttentionFactor <= 0 {
			attnFactor = yarnAttentionFactor(factor, rs.MScale, rs.MScaleAllDim)
		}
		truncate := true
		if rs.HasTruncate {
			truncate = rs.Truncate
		}
		applyYarnScaling(invFreq, base, factor, float64(origCtx), rs.BetaFast, rs.BetaSlow, truncate)
	default:
		if factor != 1 {
			for i, f := range invFreq {
				invFreq[i] = f / factor
			}
		}
	}

	return attnFactor
}

func applyLlama3Scaling(invFreq []float64, factor float64, origCtx float64, lowFactor float64, highFactor float64) {
	if factor == 0 || factor == 1 || len(invFreq) == 0 {
		return
	}
	if origCtx <= 0 {
		return
	}
	if lowFactor <= 0 {
		lowFactor = 1
	}
	if highFactor <= 0 {
		highFactor = lowFactor
	}
	if highFactor <= lowFactor {
		for i, f := range invFreq {
			invFreq[i] = f / factor
		}
		return
	}

	lowFreqWavelen := origCtx / lowFactor
	highFreqWavelen := origCtx / highFactor

	for i, f := range invFreq {
		if f == 0 {
			continue
		}
		waveLen := (2 * math.Pi) / f

		if waveLen > lowFreqWavelen {
			invFreq[i] = f / factor
			continue
		}
		if waveLen < highFreqWavelen {
			invFreq[i] = f
			continue
		}

		smoothDenom := highFactor - lowFactor
		if smoothDenom == 0 {
			invFreq[i] = f / factor
			continue
		}
		smoothFactor := (origCtx/waveLen - lowFactor) / smoothDenom
		invScaled := f / factor
		invFreq[i] = (1-smoothFactor)*invScaled + smoothFactor*f
	}
}

func yarnAttentionFactor(factor float64, mscale float64, mscaleAllDim float64) float64 {
	getMScale := func(scale float64, mul float64) float64 {
		if scale <= 1 {
			return 1
		}
		if mul <= 0 {
			mul = 1
		}
		return 0.1*mul*math.Log(scale) + 1
	}

	if mscale > 0 && mscaleAllDim > 0 {
		num := getMScale(factor, mscale)
		den := getMScale(factor, mscaleAllDim)
		if den == 0 {
			return 1
		}
		return num / den
	}

	mul := mscale
	if mul <= 0 {
		mul = 1
	}
	return getMScale(factor, mul)
}

func applyYarnScaling(invFreq []float64, base float64, factor float64, origCtx float64, betaFast float64, betaSlow float64, truncate bool) {
	if len(invFreq) == 0 || factor == 0 || factor == 1 {
		return
	}
	if base <= 1 || origCtx <= 0 {
		for i, f := range invFreq {
			invFreq[i] = f / factor
		}
		return
	}
	if betaFast <= 0 {
		betaFast = 32
	}
	if betaSlow <= 0 {
		betaSlow = 1
	}

	dimHalf := len(invFreq)
	dim := float64(dimHalf * 2)

	findCorrectionDim := func(numRotations float64) float64 {
		numer := origCtx / (numRotations * 2 * math.Pi)
		if numer <= 0 {
			return 0
		}
		denom := 2 * math.Log(base)
		if denom == 0 {
			return 0
		}
		return (dim * math.Log(numer)) / denom
	}
	findCorrectionRange := func(lowRot float64, highRot float64) (float64, float64) {
		low := findCorrectionDim(lowRot)
		high := findCorrectionDim(highRot)
		if truncate {
			low = math.Floor(low)
			high = math.Ceil(high)
		}
		if low < 0 {
			low = 0
		}
		maxDim := dim - 1
		if high > maxDim {
			high = maxDim
		}
		return low, high
	}
	linearRamp := func(min float64, max float64, i int) float64 {
		if min == max {
			max += 0.001
		}
		v := (float64(i) - min) / (max - min)
		if v < 0 {
			return 0
		}
		if v > 1 {
			return 1
		}
		return v
	}

	low, high := findCorrectionRange(betaFast, betaSlow)

	for i, f := range invFreq {
		ramp := linearRamp(low, high, i)
		invExtrap := f
		invInterp := f / factor
		invFreq[i] = invInterp*ramp + invExtrap*(1-ramp)
	}
}
