package audio

import "math"

// Slaney mel scale (auditory toolbox / librosa htk=False), the mel_scale="slaney"
// used by transformers.audio_utils.mel_filter_bank: linear below 1 kHz with
// f_sp = 200/3 mel per Hz, logarithmic above with logstep = ln(6.4)/27.
const (
	slaneyFSp       = 200.0 / 3.0
	slaneyMinLogHz  = 1000.0
	slaneyMinLogMel = slaneyMinLogHz / slaneyFSp // 15
)

// slaneyLogstep is ln(6.4)/27 (math.Log is not a constant expression).
var slaneyLogstep = math.Log(6.4) / 27.0

// HzToMel converts a frequency in Hz to the Slaney mel scale.
func HzToMel(f float64) float64 {
	mel := f / slaneyFSp
	if f >= slaneyMinLogHz {
		mel = slaneyMinLogMel + math.Log(f/slaneyMinLogHz)/slaneyLogstep
	}
	return mel
}

// MelToHz converts a Slaney mel value back to Hz (exact inverse of HzToMel).
func MelToHz(m float64) float64 {
	freq := slaneyFSp * m
	if m >= slaneyMinLogMel {
		freq = slaneyMinLogHz * math.Exp(slaneyLogstep*(m-slaneyMinLogMel))
	}
	return freq
}

// MelFrequencies returns n mel-spaced band edge frequencies in Hz between fmin
// and fmax (librosa.mel_frequencies with htk=False: linspace in mel, inclusive).
func MelFrequencies(n int, fmin, fmax float64) []float64 {
	lo, hi := HzToMel(fmin), HzToMel(fmax)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = MelToHz(lo + (hi-lo)*float64(i)/float64(n-1))
	}
	return out
}

// FFTFrequencies returns the center frequencies of the nFFT/2+1 rfft bins:
// i * sr / nFFT for i = 0..nFFT/2 (numpy.fft.rfftfreq).
func FFTFrequencies(sr float64, nFFT int) []float64 {
	bins := nFFT/2 + 1
	out := make([]float64, bins)
	for i := 0; i < bins; i++ {
		out[i] = float64(i) * sr / float64(nFFT)
	}
	return out
}

// MelFilterBank builds the Slaney-scale, Slaney-normalized mel filterbank with
// numMels triangular filters over numBins = nFFT/2+1 FFT bins spanning
// [fmin, fmax] Hz at sampling rate sr. The result is [numMels][numBins],
// matching transformers.audio_utils.mel_filter_bank(..., norm="slaney",
// mel_scale="slaney") used by Lfm3OmniFeatureExtractor:
//
//	lower = (fft - f_i) / (f_{i+1} - f_i)
//	upper = (f_{i+2} - fft) / (f_{i+2} - f_{i+1})
//	w     = max(0, min(lower, upper)) * 2 / (f_{i+2} - f_i)   // slaney norm
//
// with all band frequencies f_* in Hz.
func MelFilterBank(sr float64, nFFT, numMels int, fmin, fmax float64) [][]float64 {
	numBins := nFFT/2 + 1
	fftFreqs := FFTFrequencies(sr, nFFT)
	melFreqs := MelFrequencies(numMels+2, fmin, fmax)
	fdiff := make([]float64, numMels+1)
	for i := 0; i <= numMels; i++ {
		fdiff[i] = melFreqs[i+1] - melFreqs[i]
	}
	W := make([][]float64, numMels)
	for i := 0; i < numMels; i++ {
		row := make([]float64, numBins)
		for j := 0; j < numBins; j++ {
			lower := (fftFreqs[j] - melFreqs[i]) / fdiff[i]
			upper := (melFreqs[i+2] - fftFreqs[j]) / fdiff[i+1]
			w := math.Min(lower, upper)
			if w < 0 {
				w = 0
			}
			row[j] = w
		}
		enorm := 2.0 / (melFreqs[i+2] - melFreqs[i])
		for j := range row {
			row[j] *= enorm
		}
		W[i] = row
	}
	return W
}

// FeatureExtractor is the LFM3-Omni log-mel front end
// (feature_extraction_lfm3_omni.py): periodic Hann window, torch.stft with
// center=False, power spectrum, Slaney mel filterbank, log10 clamped at the
// log floor, then (x + offset) / divisor.
type FeatureExtractor struct {
	Config     FeatureExtractorConfig
	MelFilters [][]float64 // [numMelBins][nFFT/2+1]
	Window     []float64   // periodic Hann, length WinLength
}

// NewFeatureExtractor builds the feature extractor for cfg, precomputing the
// Slaney mel filterbank and the analysis window.
func NewFeatureExtractor(cfg FeatureExtractorConfig) *FeatureExtractor {
	sr := float64(cfg.SamplingRate)
	return &FeatureExtractor{
		Config:     cfg,
		MelFilters: MelFilterBank(sr, cfg.NFFT, cfg.FeatureSize, 0, sr/2),
		Window:     HannWindow(cfg.WinLength),
	}
}

// LogMel runs the full HF front end on a mono clip (feature_extraction_
// lfm3_omni.py __call__ + _log_mel, lines 65-95 and 154-160): the clip is
// right-padded with zeros to PadLength, then
//
//	spec  = stft(clip, n_fft, hop, win, hann, center=false)
//	power = |spec|^2                                  // [nFFT/2+1][frames]
//	mel   = melFilters @ power                        // [numBins][frames]
//	logmel = clamp(log10(mel), min=logFloor)
//	out   = (logmel + offset) / divisor
//
// The result is [FeatureSize][NumFrames(len(waveform))].
func (f *FeatureExtractor) LogMel(waveform []float32) [][]float64 {
	cfg := f.Config
	padded := cfg.PadLength(len(waveform))
	x := make([]float64, padded)
	for i, v := range waveform {
		x[i] = float64(v)
	}
	spec := STFT(x, cfg.NFFT, cfg.HopLength, cfg.WinLength, f.Window)
	bins := cfg.NFFT/2 + 1
	frames := 0
	if len(spec) > 0 {
		frames = len(spec[0])
	}
	out := make([][]float64, cfg.FeatureSize)
	for m := 0; m < cfg.FeatureSize; m++ {
		row := make([]float64, frames)
		filt := f.MelFilters[m]
		for t := 0; t < frames; t++ {
			var energy float64
			for k := 0; k < bins; k++ {
				c := spec[k][t]
				power := real(c)*real(c) + imag(c)*imag(c)
				energy += filt[k] * power
			}
			v := math.Log10(energy) // -Inf for 0, clamped below
			if v < cfg.LogFloor {
				v = cfg.LogFloor
			}
			row[t] = (v + cfg.NormalizationOffset) / cfg.NormalizationDivisor
		}
		out[m] = row
	}
	return out
}
