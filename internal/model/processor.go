package model

import (
	"encoding/json"
	"fmt"
)

// ProcessorConfig mirrors the subset of a HF processor_config.json that the
// runtime needs for multimodal inputs (image preprocessing parameters).
type ProcessorConfig struct {
	ImageProcessor *ImageProcessorConfig `json:"image_processor"`
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
