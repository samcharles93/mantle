package simd

import (
	"fmt"

	"github.com/samcharles93/mantle/internal/audio"
	"github.com/samcharles93/mantle/internal/backend/core"
)

// ProcessAudioClips decodes WAV clips and runs the log-mel front end for an
// audio model: each clip must be mono/stereo 16-bit PCM or 32-bit float WAV at
// exactly the feature extractor's sampling rate (HF's Lfm3OmniFeatureExtractor
// rejects any other rate), and yields logMels[clip][melBin][frame] exactly as
// audio.FeatureExtractor.LogMel produces (processor_config.json
// feature_extractor: 16 kHz, 128 mel bins, n_fft/win 400, hop 160, stride 8).
func ProcessAudioClips(wavs [][]byte, ac *core.AudioConfig) ([][][]float64, error) {
	if ac == nil {
		return nil, fmt.Errorf("audio: nil audio config")
	}
	fe := audio.NewFeatureExtractor(ac.FeatureExtractor)
	out := make([][][]float64, len(wavs))
	for i, wav := range wavs {
		samples, sr, err := audio.DecodeWAV(wav)
		if err != nil {
			return nil, fmt.Errorf("audio clip %d: %w", i, err)
		}
		if sr != ac.FeatureExtractor.SamplingRate {
			return nil, fmt.Errorf("audio clip %d: Lfm3OmniFeatureExtractor expects audio sampled at %d Hz, got %d Hz",
				i, ac.FeatureExtractor.SamplingRate, sr)
		}
		mel := fe.LogMel(samples)
		if len(mel) != ac.NumMelBins {
			return nil, fmt.Errorf("audio clip %d: log-mel has %d bins, want %d", i, len(mel), ac.NumMelBins)
		}
		out[i] = mel
	}
	return out, nil
}

// AudioTokenCount returns the number of <|audio|> placeholder tokens one clip
// of numFrames log-mel frames expands to (ceil(frames / encoder_stride)).
func AudioTokenCount(ac *core.AudioConfig, numFrames int) int {
	return ac.NumEncoderTokens(numFrames)
}
