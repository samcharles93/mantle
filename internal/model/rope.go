package model

import (
	"math"
	"strings"
)

func ropeScalingForConfig(cfg *hfConfig) *RopeScaling {
	if cfg == nil {
		return nil
	}

	// Prefer rope_scaling when present.
	if cfg.RopeScaling != nil {
		rs := cfg.RopeScaling
		ropeType := strings.TrimSpace(rs.RopeType)
		if ropeType == "" {
			ropeType = strings.TrimSpace(rs.Type)
		}
		ropeType = strings.ToLower(ropeType)
		switch ropeType {
		case "":
			ropeType = "linear"
		case "linear", "llama3":
			// supported
		default:
			// Unsupported scaling types (for example yarn) are intentionally
			// ignored rather than approximated incorrectly.
			return nil
		}
		out := &RopeScaling{
			Type:       ropeType,
			Factor:     rs.Factor,
			OrigMaxCtx: rs.OriginalMaxPositionEmbeddings,
			LowFactor:  rs.LowFreqFactor,
			HighFactor: rs.HighFreqFactor,
		}
		if out.Factor <= 0 {
			return nil
		}
		if out.OrigMaxCtx <= 0 {
			out.OrigMaxCtx = cfg.MaxPosition
		}
		if out.LowFactor <= 0 {
			out.LowFactor = 1
		}
		if out.HighFactor <= 0 {
			out.HighFactor = out.LowFactor
		}
		if out.Type == "" {
			out.Type = "linear"
		}
		return out
	}

	// Fall back to rope_parameters when present (used by mistral3 configs).
	if cfg.RopeParameters == nil {
		return nil
	}
	rp := cfg.RopeParameters
	ropeType := strings.TrimSpace(rp.RopeType)
	if ropeType == "" {
		ropeType = strings.TrimSpace(rp.Type)
	}
	ropeType = strings.ToLower(ropeType)
	switch ropeType {
	case "":
		ropeType = "linear"
	case "linear", "llama3":
		// supported
	default:
		// Unsupported rope scaling types (for example yarn) are skipped.
		return nil
	}
	out := &RopeScaling{
		Type:       ropeType,
		Factor:     rp.Factor,
		OrigMaxCtx: rp.OriginalMaxPositionEmbeddings,
		LowFactor:  rp.LowFreqFactor,
		HighFactor: rp.HighFreqFactor,
	}
	if out.Factor <= 0 {
		return nil
	}
	if out.OrigMaxCtx <= 0 {
		out.OrigMaxCtx = cfg.MaxPosition
	}
	if out.LowFactor <= 0 {
		out.LowFactor = 1
	}
	if out.HighFactor <= 0 {
		out.HighFactor = out.LowFactor
	}
	if out.Type == "" {
		out.Type = "linear"
	}
	return out
}

func applyRopeScaling(invFreq []float64, ctxLen int, rs *RopeScaling) {
	if len(invFreq) == 0 || rs == nil {
		return
	}
	factor := rs.Factor
	if factor == 0 || factor == 1 {
		return
	}

	switch rs.Type {
	case "llama3":
		orig := float64(rs.OrigMaxCtx)
		if orig <= 0 {
			orig = float64(ctxLen)
		}
		lowThresh := orig / rs.LowFactor
		highThresh := orig / rs.HighFactor
		if highThresh > lowThresh {
			highThresh, lowThresh = lowThresh, highThresh
		}
		if lowThresh == highThresh {
			for i, f := range invFreq {
				invFreq[i] = f / factor
			}
			return
		}
		for i, f := range invFreq {
			if f == 0 {
				continue
			}
			waveLen := (2 * math.Pi) / f
			var scale float64
			switch {
			case waveLen >= lowThresh:
				scale = 1.0
			case waveLen <= highThresh:
				scale = factor
			default:
				// Interpolate scaling in the transition band.
				t := (lowThresh - waveLen) / (lowThresh - highThresh)
				scale = 1.0 + t*(factor-1.0)
			}
			invFreq[i] = f / scale
		}
	default:
		for i, f := range invFreq {
			invFreq[i] = f / factor
		}
	}
}
