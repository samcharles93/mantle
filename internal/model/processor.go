package model

import (
	"encoding/json"
	"fmt"

	"github.com/samcharles93/mantle/internal/audio"
)

// ProcessorConfig mirrors the subset of a HF processor_config.json that the
// runtime needs for multimodal inputs (image/video preprocessing parameters
// and, for audio models such as LFM3-Omni, the log-mel feature extractor and
// the placeholder marker strings).
type ProcessorConfig struct {
	ImageProcessor   *ImageProcessorConfig         `json:"image_processor"`
	VideoProcessor   *VideoProcessorConfig         `json:"video_processor"`
	FeatureExtractor *audio.FeatureExtractorConfig `json:"feature_extractor"`

	// Placeholder marker strings (resolved to token ids by the tokenizer).
	AudioToken       string `json:"audio_token"`        // "<|audio|>"
	AudioOutToken    string `json:"audio_out_token"`    // "<|audio_out|>"
	ImageToken       string `json:"image_token"`        // "<|image|>"
	VideoToken       string `json:"video_token"`        // "<|video|>"
	VisionStartToken string `json:"vision_start_token"` // "<|vision_start|>"
	VisionEndToken   string `json:"vision_end_token"`   // "<|vision_end|>"
}

// VideoProcessorConfig mirrors the video_processor half of processor_config.json.
type VideoProcessorConfig struct {
	FPS           float64   `json:"fps"`
	ImageMean     []float64 `json:"image_mean"`
	ImageStd      []float64 `json:"image_std"`
	RescaleFactor float64   `json:"rescale_factor"`
	PatchSize     int       `json:"patch_size"`
	MergeSize     int       `json:"merge_size"`
	TemporalPatch int       `json:"temporal_patch_size"`
	Size          *struct {
		ShortestEdge int `json:"shortest_edge"`
		LongestEdge  int `json:"longest_edge"`
	} `json:"size"`
}

type ImageProcessorConfig struct {
	ImageMean     []float64 `json:"image_mean"`
	ImageStd      []float64 `json:"image_std"`
	RescaleFactor float64   `json:"rescale_factor"`
	DoResize      bool      `json:"do_resize"`
	DoRescale     bool      `json:"do_rescale"`
	DoNormalize   bool      `json:"do_normalize"`
	DoConvertRGB  bool      `json:"do_convert_rgb"`
	PatchSize     int       `json:"patch_size"`
	MergeSize     int       `json:"merge_size"`
	TemporalPatch int       `json:"temporal_patch_size"`
	Size          *struct {
		ShortestEdge int `json:"shortest_edge"`
		LongestEdge  int `json:"longest_edge"`
	} `json:"size"`
}

// LoadProcessorConfigBytes parses a processor_config.json. An empty input
// yields (nil, nil): the model is text-only or the section is absent.
func LoadProcessorConfigBytes(raw []byte) (*ProcessorConfig, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var pc ProcessorConfig
	if err := json.Unmarshal(raw, &pc); err != nil {
		return nil, fmt.Errorf("processor_config: %w", err)
	}
	return &pc, nil
}
