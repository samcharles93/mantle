package core

import "github.com/samcharles93/mantle/internal/hostcaps"

// LoadModelOptions controls backend/runtime behavior during model load.
type LoadModelOptions struct {
	CacheTypeK   string
	CacheTypeV   string
	HostCaps     *hostcaps.Snapshot
	TilingConfig TilingConfig
	GpuLayers    int  // -1 auto, 0 all layers on CPU, N first N layers on GPU
	UseGraph     bool // experimental: use graph-based execution

	// HiddenTapLayers is the caller-supplied list of decoder layer indices whose
	// post-layer residual outputs are captured for the DSpark draft model's
	// context taps. -1 selects the embedding output. Empty (the default) disables
	// capture. It is never read from the MCF.
	HiddenTapLayers []int

	// SharedEmbeddings and SharedOutput, when set, are used as the token
	// embedding and LM head instead of loading them from the container. A
	// DSpark draft uses them to share its paired target's tensors; the draft
	// container carries neither. Both must be [vocab, hidden].
	SharedEmbeddings *Mat
	SharedOutput     *Mat

	// ProcessorConfigJSON optionally carries the packed processor_config.json
	// section bytes for multimodal models (image preprocessing parameters and,
	// for audio models, the feature_extractor block).
	ProcessorConfigJSON []byte

	// AudioTokenizerConfigJSON optionally carries the config.json half of the
	// packed section 0x0107 (audio_tokenizer/config.json) for audio models.
	AudioTokenizerConfigJSON []byte
}
