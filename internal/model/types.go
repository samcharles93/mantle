package model

// RoPEScaling describes rotary positional embedding scaling parameters.
// This is config-only and does not include runtime state.
type RopeScaling struct {
	Type            string
	Factor          float64
	OrigMaxCtx      int
	LowFactor       float64
	HighFactor      float64
	AttentionFactor float64
	BetaFast        float64
	BetaSlow        float64
	MScale          float64
	MScaleAllDim    float64
	Truncate        bool
	HasTruncate     bool
}
