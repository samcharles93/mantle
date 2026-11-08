package core

import (
	"github.com/samcharles93/mantle/internal/audio"
)

// AudioConfig carries the audio-encoder and audio-projector parameters of an
// audio-language model (LFM3-Omni's audio_encoder_config plus the packed
// processor_config.json feature_extractor block and the placeholder token ids).
// The text decoder parameters stay in Config.
type AudioConfig struct {
	// FeatureExtractor is the log-mel front end configuration
	// (processor_config.json feature_extractor): 16 kHz, 128 mel bins,
	// n_fft/win 400, hop 160, encoder_stride 8 -> ceil(frames/8) tokens at
	// 12.5 tokens per second of audio.
	FeatureExtractor audio.FeatureExtractorConfig

	// Audio encoder tower (model.audio_encoder.*): a stride-8 (mel, time)
	// downsampler feeding an LFM2 stack.
	ConvChannels   int // downsample conv channels (256)
	HiddenSize     int // tower + projector input dim (768)
	NumMelBins     int
	SlidingWindow  int // causal attention window of full_attention layers (32)
	NumLayers      int
	NumHeads       int
	NumKVHeads     int
	HeadDim        int
	FFNDim         int      // effective feed-forward dim after block auto-adjust
	LayerTypes     []string // "full_attention" or "conv" per layer
	NormEps        float64
	ConvKernelSize int // short-conv kernel of conv layers (conv_L_cache)
	RopeTheta      float64

	// Audio projector (model.audio_projector.*): LayerNorm -> linear_1 ->
	// GELU -> linear_2 mapping tower hidden to the text embedding dim.
	TextHiddenSize int

	// Special token ids used to mark audio placeholders in the prompt.
	AudioTokenID       int // <|audio|>, one per audio-encoder token
	AudioOutTokenID    int // <|audio_out|> (speech output; input side reserve)
	VisionStartTokenID int // <|vision_start|> wraps expanded audio placeholders
	VisionEndTokenID   int // <|vision_end|>
}

// NumEncoderTokens returns the number of <|audio|> placeholder tokens one
// clip of numFrames log-mel frames expands to: ceil(numFrames / stride).
func (c *AudioConfig) NumEncoderTokens(numFrames int) int {
	if c == nil {
		return 0
	}
	return c.FeatureExtractor.NumEncoderTokens(numFrames)
}

// AudioEncoder runs the audio tower + projector over log-mel features and
// returns, per clip, one projected embedding row per audio token
// (NumEncoderTokens rows of TextHiddenSize each). It implements the audio
// analogue of VisionEncoder and runs on the host even when the text decoder
// runs on a device.
type AudioEncoder interface {
	Config() *AudioConfig
	// Encode consumes one log-mel matrix per clip, logMels[clip][melBin][frame],
	// as produced by the feature extractor (audio.FeatureExtractor.LogMel), and
	// returns per clip one embedding row per audio token:
	// [clip][token][TextHiddenSize].
	Encode(logMels [][][]float64) ([][][]float32, error)
}

// AudioEncoderProvider is implemented by model runtimes that carry an audio
// encoder (nil when the model is text-only or vision-only).
type AudioEncoderProvider interface {
	AudioEncoder() AudioEncoder
}
