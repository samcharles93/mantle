// Package audio implements the LFM3-Omni-2B audio foundation: a RIFF/WAVE PCM
// reader, a mixed-radix FFT with torch.stft (center=False) semantics, the Slaney
// mel filterbank and log-mel front end of Lfm3OmniFeatureExtractor, and the
// LiquidCodec FSQ quantizer.
//
// The package is stdlib-only and self-contained: model weights are passed in as
// plain float32 slices by the caller, so nothing here imports pkg/mcf or any
// backend package.
//
// Reference points (HuggingFace LiquidAI/LFM3-Omni-2B):
//   - processor_config.json feature_extractor block (16 kHz, n_fft/win 400,
//     hop 160, 128 mel bins, log floor -12, (x+4)/4, encoder_stride 8)
//   - feature_extraction_lfm3_omni.py:65-95 (_log_mel)
//   - audio_tokenizer/config.json (24 kHz LiquidCodec: stft 1280/320,
//     frame_stack 6 = 1920 samples per codec frame, FSQ levels [8,7,6,6], n_q 12)
//   - modeling_liquid_codec.py:141-190 (LiquidCodecQuantizer)
package audio
