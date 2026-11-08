package audio

import "testing"

func TestCanonicalConstants(t *testing.T) {
	// feature extractor (processor_config.json feature_extractor)
	if SampleRate != 16000 {
		t.Fatalf("SampleRate = %d, want 16000", SampleRate)
	}
	if NFFT != 400 {
		t.Fatalf("NFFT = %d, want 400", NFFT)
	}
	if WinLength != 400 {
		t.Fatalf("WinLength = %d, want 400", WinLength)
	}
	if HopLength != 160 {
		t.Fatalf("HopLength = %d, want 160", HopLength)
	}
	if NumMelBins != 128 {
		t.Fatalf("NumMelBins = %d, want 128", NumMelBins)
	}
	if LogFloor != -12.0 {
		t.Fatalf("LogFloor = %v, want -12", LogFloor)
	}
	if NormalizationOffset != 4.0 {
		t.Fatalf("NormalizationOffset = %v, want 4", NormalizationOffset)
	}
	if NormalizationDivisor != 4.0 {
		t.Fatalf("NormalizationDivisor = %v, want 4", NormalizationDivisor)
	}
	if EncoderStride != 8 {
		t.Fatalf("EncoderStride = %d, want 8", EncoderStride)
	}
	// codec (audio_tokenizer/config.json)
	if CodecSampleRate != 24000 {
		t.Fatalf("CodecSampleRate = %d, want 24000", CodecSampleRate)
	}
	if CodecNFFT != 1280 {
		t.Fatalf("CodecNFFT = %d, want 1280", CodecNFFT)
	}
	if CodecWinLength != 1280 {
		t.Fatalf("CodecWinLength = %d, want 1280", CodecWinLength)
	}
	if CodecHopLength != 320 {
		t.Fatalf("CodecHopLength = %d, want 320", CodecHopLength)
	}
	if FrameStack != 6 {
		t.Fatalf("FrameStack = %d, want 6", FrameStack)
	}
	if CodecFrameSamples != 1920 {
		t.Fatalf("CodecFrameSamples = %d, want 1920", CodecFrameSamples)
	}
	if CodecFrameSamples != FrameStack*CodecHopLength {
		t.Fatalf("CodecFrameSamples = %d, want FrameStack*CodecHopLength = %d", CodecFrameSamples, FrameStack*CodecHopLength)
	}
	// FSQ quantizer (audio_tokenizer/config.json + config.json)
	if NumCodebooks != 12 {
		t.Fatalf("NumCodebooks = %d, want 12", NumCodebooks)
	}
	if FSQCodebookDim != 4 {
		t.Fatalf("FSQCodebookDim = %d, want 4", FSQCodebookDim)
	}
	if QuantizerDim != 256 {
		t.Fatalf("QuantizerDim = %d, want 256", QuantizerDim)
	}
	if CodebookSize != 2017 {
		t.Fatalf("CodebookSize = %d, want 2017", CodebookSize)
	}
	if AudioEOSCode != 2016 {
		t.Fatalf("AudioEOSCode = %d, want 2016", AudioEOSCode)
	}
	if MaxFSQCode != 2015 {
		t.Fatalf("MaxFSQCode = %d, want 2015", MaxFSQCode)
	}
	if DefaultFSQLevels != [4]int{8, 7, 6, 6} {
		t.Fatalf("DefaultFSQLevels = %v, want [8 7 6 6]", DefaultFSQLevels)
	}
	// token ids (config.json / tokenizer.json added_tokens)
	if AudioTokenID != 125031 {
		t.Fatalf("AudioTokenID = %d, want 125031 (<|audio|>)", AudioTokenID)
	}
	if AudioOutTokenID != 125032 {
		t.Fatalf("AudioOutTokenID = %d, want 125032 (<|audio_out|>)", AudioOutTokenID)
	}
	if AudioStartTokenID != 125012 {
		t.Fatalf("AudioStartTokenID = %d, want 125012 (<|audio_start|>)", AudioStartTokenID)
	}
}

func TestDefaultFeatureExtractorConfig(t *testing.T) {
	cfg := DefaultFeatureExtractorConfig()
	if cfg.SamplingRate != SampleRate {
		t.Fatalf("SamplingRate = %d, want %d", cfg.SamplingRate, SampleRate)
	}
	if cfg.NFFT != NFFT {
		t.Fatalf("NFFT = %d, want %d", cfg.NFFT, NFFT)
	}
	if cfg.WinLength != WinLength {
		t.Fatalf("WinLength = %d, want %d", cfg.WinLength, WinLength)
	}
	if cfg.HopLength != HopLength {
		t.Fatalf("HopLength = %d, want %d", cfg.HopLength, HopLength)
	}
	if cfg.FeatureSize != NumMelBins {
		t.Fatalf("FeatureSize = %d, want %d", cfg.FeatureSize, NumMelBins)
	}
	if cfg.LogFloor != LogFloor {
		t.Fatalf("LogFloor = %v, want %v", cfg.LogFloor, LogFloor)
	}
	if cfg.NormalizationOffset != NormalizationOffset {
		t.Fatalf("NormalizationOffset = %v, want %v", cfg.NormalizationOffset, NormalizationOffset)
	}
	if cfg.NormalizationDivisor != NormalizationDivisor {
		t.Fatalf("NormalizationDivisor = %v, want %v", cfg.NormalizationDivisor, NormalizationDivisor)
	}
	if cfg.EncoderStride != EncoderStride {
		t.Fatalf("EncoderStride = %d, want %d", cfg.EncoderStride, EncoderStride)
	}
}

func TestDefaultCodecConfig(t *testing.T) {
	cfg := DefaultCodecConfig()
	if cfg.SampleRate != CodecSampleRate {
		t.Fatalf("SampleRate = %d, want %d", cfg.SampleRate, CodecSampleRate)
	}
	if cfg.STFTNFFT != CodecNFFT {
		t.Fatalf("STFTNFFT = %d, want %d", cfg.STFTNFFT, CodecNFFT)
	}
	if cfg.STFTWinLength != CodecWinLength {
		t.Fatalf("STFTWinLength = %d, want %d", cfg.STFTWinLength, CodecWinLength)
	}
	if cfg.STFTHopLength != CodecHopLength {
		t.Fatalf("STFTHopLength = %d, want %d", cfg.STFTHopLength, CodecHopLength)
	}
	if cfg.STFTFrameStack != FrameStack {
		t.Fatalf("STFTFrameStack = %d, want %d", cfg.STFTFrameStack, FrameStack)
	}
	if cfg.FrameSamples != CodecFrameSamples {
		t.Fatalf("FrameSamples = %d, want %d", cfg.FrameSamples, CodecFrameSamples)
	}
	if cfg.NQ != NumCodebooks {
		t.Fatalf("NQ = %d, want %d", cfg.NQ, NumCodebooks)
	}
	if cfg.QuantizerDim != QuantizerDim {
		t.Fatalf("QuantizerDim = %d, want %d", cfg.QuantizerDim, QuantizerDim)
	}
	if len(cfg.FSQLevels) != FSQCodebookDim {
		t.Fatalf("len(FSQLevels) = %d, want %d", len(cfg.FSQLevels), FSQCodebookDim)
	}
	for i, l := range cfg.FSQLevels {
		if l != DefaultFSQLevels[i] {
			t.Fatalf("FSQLevels[%d] = %d, want %d", i, l, DefaultFSQLevels[i])
		}
	}
}

func TestFrameAndTokenMath(t *testing.T) {
	cfg := DefaultFeatureExtractorConfig()
	// pad_length / num_frames semantics of feature_extraction_lfm3_omni.py:43-49.
	if got := cfg.PadLength(399); got != 400 {
		t.Fatalf("PadLength(399) = %d, want 400", got)
	}
	if got := cfg.PadLength(400); got != 400 {
		t.Fatalf("PadLength(400) = %d, want 400", got)
	}
	// 16000 samples: Python floor-mod pads the excess 15600 up by 80 to 16080;
	// frames = 1 + (16080-400)/160 = 99.
	if got := cfg.PadLength(16000); got != 16080 {
		t.Fatalf("PadLength(16000) = %d, want 16080", got)
	}
	if got := cfg.NumFrames(16000); got != 99 {
		t.Fatalf("NumFrames(16000) = %d, want 99", got)
	}
	if got := cfg.NumFrames(400); got != 1 {
		t.Fatalf("NumFrames(400) = %d, want 1", got)
	}
	// num_encoder_tokens = ceil(frames / encoder_stride) (line 78-79): 100 frames/s
	// at hop 160 over stride 8 = 12.5 audio tokens per second.
	cases := map[int]int{1: 1, 8: 1, 9: 2, 97: 13, 98: 13, 99: 13, 100: 13, 800: 100}
	for frames, want := range cases {
		if got := cfg.NumEncoderTokens(frames); got != want {
			t.Fatalf("NumEncoderTokens(%d) = %d, want %d", frames, got, want)
		}
	}
}

func TestCodecFrames(t *testing.T) {
	cfg := DefaultCodecConfig()
	// encode pads to a whole codec frame: ceil(numSamples / frameSamples).
	cases := map[int]int{1: 1, 1920: 1, 1921: 2, 24000: 13}
	for samples, want := range cases {
		if got := cfg.NumCodecFrames(samples); got != want {
			t.Fatalf("NumCodecFrames(%d) = %d, want %d", samples, got, want)
		}
	}
}
