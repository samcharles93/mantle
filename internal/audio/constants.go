package audio

// Canonical LFM3-Omni-2B processor constants.
//
// Input speech front end (processor_config.json feature_extractor):
// sampling_rate 16000, n_fft 400, win_length 400, hop_length 160,
// feature_size 128 mel bins, log_floor -12, normalization (x+4)/4,
// encoder_stride 8 -> ceil(frames/8) audio tokens per clip, nominally
// 16000/160/8 = 12.5 tokens per second of audio.
const (
	SampleRate           = 16000 // input speech sampling rate (Hz)
	NFFT                 = 400   // STFT size of the mel front end (2^4 * 5^2)
	WinLength            = 400   // analysis window length (== n_fft)
	HopLength            = 160   // STFT hop (100 frames per second)
	NumMelBins           = 128   // mel filterbank bins
	LogFloor             = -12.0 // clamp(min=log_floor) on log10(mel)
	NormalizationOffset  = 4.0   // (x + offset) / divisor
	NormalizationDivisor = 4.0
	EncoderStride        = 8 // mel frames per audio-encoder token
)

// Token ids from config.json / tokenizer.json added_tokens.
const (
	AudioTokenID      = 125031 // <|audio|> — one per audio-encoder token
	AudioOutTokenID   = 125032 // <|audio_out|> — one per codec frame on output
	AudioStartTokenID = 125012 // <|audio_start|>
)

// LiquidCodec (audio_tokenizer/config.json) constants: 24 kHz speech codec with
// STFT 1280/320 and frame_stack 6, i.e. 6*320 = 1920 samples per codec frame
// (24000/1920 = 12.5 codec frames per second, matching the input token rate).
const (
	CodecSampleRate   = 24000                       // codec sampling rate (Hz)
	CodecNFFT         = 1280                        // codec STFT size (2^8 * 5)
	CodecWinLength    = 1280                        // codec STFT window (== n_fft)
	CodecHopLength    = 320                         // codec STFT hop
	FrameStack        = 6                           // STFT frames stacked per codec frame
	CodecFrameSamples = FrameStack * CodecHopLength // 1920 samples per codec frame
)

// LiquidCodec FSQ quantizer constants (audio_tokenizer/config.json + top-level
// config.json): n_q 12 codebooks of dimension 4 over levels [8,7,6,6]; the code
// of one codebook is sum(idx_k * prod(levels[:k])) in [0, 2015]; 2016 is the
// end-of-speech code and 2017 the codebook size.
const (
	NumCodebooks   = 12
	FSQCodebookDim = 4
	QuantizerDim   = 256
	CodebookSize   = 2017
	AudioEOSCode   = 2016
	MaxFSQCode     = CodebookSize - 2 // 2015 = prod(fsq_levels) - 1
)

// DefaultFSQLevels is the fsq_levels array of audio_tokenizer/config.json.
var DefaultFSQLevels = [FSQCodebookDim]int{8, 7, 6, 6}

// FeatureExtractorConfig mirrors the feature_extractor block of
// processor_config.json (Lfm3OmniFeatureExtractor).
type FeatureExtractorConfig struct {
	FeatureSize          int     `json:"feature_size"`
	SamplingRate         int     `json:"sampling_rate"`
	NFFT                 int     `json:"n_fft"`
	HopLength            int     `json:"hop_length"`
	WinLength            int     `json:"win_length"`
	LogFloor             float64 `json:"log_floor"`
	NormalizationOffset  float64 `json:"normalization_offset"`
	NormalizationDivisor float64 `json:"normalization_divisor"`
	EncoderStride        int     `json:"encoder_stride"`
	PaddingValue         float64 `json:"padding_value"`
}

// DefaultFeatureExtractorConfig returns the LFM3-Omni-2B feature extractor
// configuration.
func DefaultFeatureExtractorConfig() FeatureExtractorConfig {
	return FeatureExtractorConfig{
		FeatureSize:          NumMelBins,
		SamplingRate:         SampleRate,
		NFFT:                 NFFT,
		HopLength:            HopLength,
		WinLength:            WinLength,
		LogFloor:             LogFloor,
		NormalizationOffset:  NormalizationOffset,
		NormalizationDivisor: NormalizationDivisor,
		EncoderStride:        EncoderStride,
		PaddingValue:         0,
	}
}

// PadLength is feature_extraction_lfm3_omni.py:43-45: right-pad a clip so its
// length covers a whole number of frames (at least one n_fft window). The
// Python expression `num_samples + (-(num_samples - n_fft) % hop_length)` uses
// floor modulo, i.e. the excess above n_fft is rounded UP to a whole hop.
func (c FeatureExtractorConfig) PadLength(numSamples int) int {
	if numSamples < c.NFFT {
		return c.NFFT
	}
	excess := numSamples - c.NFFT
	return numSamples + (c.HopLength-excess%c.HopLength)%c.HopLength
}

// NumFrames is feature_extraction_lfm3_omni.py:47-49: number of STFT frames of
// the right-padded clip, 1 + (padded - n_fft) / hop.
func (c FeatureExtractorConfig) NumFrames(numSamples int) int {
	padded := c.PadLength(numSamples)
	return 1 + (padded-c.NFFT)/c.HopLength
}

// NumEncoderTokens is feature_extraction_lfm3_omni.py:78-79:
// ceil(numFrames / encoder_stride).
func (c FeatureExtractorConfig) NumEncoderTokens(numFrames int) int {
	if numFrames <= 0 {
		return 0
	}
	return (numFrames + c.EncoderStride - 1) / c.EncoderStride
}

// CodecConfig mirrors audio_tokenizer/config.json (LiquidCodecConfig).
type CodecConfig struct {
	SampleRate     int   `json:"sample_rate"`
	FrameSamples   int   `json:"frame_samples"`
	STFTNFFT       int   `json:"stft_n_fft"`
	STFTWinLength  int   `json:"stft_win_length"`
	STFTHopLength  int   `json:"stft_hop_length"`
	STFTFrameStack int   `json:"stft_frame_stack"`
	EncoderConvDim int   `json:"encoder_conv_dim"`
	EncoderDim     int   `json:"encoder_dim"`
	QuantizerDim   int   `json:"quantizer_dim"`
	NQ             int   `json:"n_q"`
	FSQLevels      []int `json:"fsq_levels"`
	DecoderWindow  int   `json:"decoder_window"`
}

// DefaultCodecConfig returns the LFM3-Omni-2B audio_tokenizer configuration.
func DefaultCodecConfig() CodecConfig {
	levels := make([]int, len(DefaultFSQLevels))
	copy(levels, DefaultFSQLevels[:])
	return CodecConfig{
		SampleRate:     CodecSampleRate,
		FrameSamples:   CodecFrameSamples,
		STFTNFFT:       CodecNFFT,
		STFTWinLength:  CodecWinLength,
		STFTHopLength:  CodecHopLength,
		STFTFrameStack: FrameStack,
		EncoderConvDim: 256,
		EncoderDim:     512,
		QuantizerDim:   QuantizerDim,
		NQ:             NumCodebooks,
		FSQLevels:      levels,
		DecoderWindow:  30,
	}
}

// NumCodecFrames is the number of codec frames for numSamples of 24 kHz audio:
// encode pads the tail to a whole frame, ceil(numSamples / frameSamples).
func (c CodecConfig) NumCodecFrames(numSamples int) int {
	if numSamples <= 0 {
		return 0
	}
	return (numSamples + c.FrameSamples - 1) / c.FrameSamples
}
