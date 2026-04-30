package simd

import core "github.com/samcharles93/mantle/internal/backend/core"

type Instance core.Instance

type (
	Layer                    = core.Layer
	Mat                      = core.Mat
	ModelConfig              = core.ModelConfig
	Config                   = core.Config
	RopeScaling              = core.RopeScaling
	MoELayer                 = core.MoELayer
	MoEExpert                = core.MoEExpert
	MoEShared                = core.MoEShared
	Gemma4PerLayerInputModel = core.Gemma4PerLayerInputModel
	Gemma4PLELayer           = core.Gemma4PLELayer
	Gemma4MoELayer           = core.Gemma4MoELayer
	Gemma4MoEExpert          = core.Gemma4MoEExpert
	MambaLayer               = core.MambaLayer
	ScratchBuffers           = core.ScratchBuffers
	ShortConvState           = core.ShortConvState
	AttnCache                = core.AttnCache
)
