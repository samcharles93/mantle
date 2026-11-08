package core

// VisionConfig carries the vision-encoder and multimodal-projector parameters
// of a vision-language model (e.g. LFM3-VL's vision_config plus the top-level
// projector/token fields). The text decoder parameters stay in Config.
type VisionConfig struct {
	// Vision tower.
	HiddenSize       int
	InChannels       int
	PatchSize        int
	ConvKernelSize   int
	IntermediateSize int
	NumLayers        int
	NumHeads         int
	NumKVHeads       int
	HeadDim          int
	LayerTypes       []string // "full_attention" or "conv" per layer
	NormEps          float64
	MergeSize        int // spatial_merge_size
	TemporalPatch    int // temporal_patch_size
	RopeTheta        float64

	// Multimodal projector (vision hidden -> text hidden).
	DownsampleFactor      int // projector merges DownsampleFactor^2 vision tokens
	ProjectorHiddenSize   int
	ProjectorHiddenAct    string
	ProjectorLayerNormEps float64
	ProjectorBias         bool
	TextHiddenSize        int

	// Special token ids used to mark image placeholders in the prompt.
	ImageTokenID       int
	VisionStartTokenID int
	VisionEndTokenID   int

	// Image processor parameters (from processor_config.json).
	ImageMean     []float64
	ImageStd      []float64
	RescaleFactor float64
	MinPixels     int
	MaxPixels     int
}

// PatchDim is the flattened per-patch input size of the patchifier:
// in_channels * temporal_patch_size * patch_size^2.
func (c *VisionConfig) PatchDim() int {
	if c == nil {
		return 0
	}
	return c.InChannels * c.TemporalPatch * c.PatchSize * c.PatchSize
}

// TokensPerImage returns the number of language-model tokens one image with
// the given (t, h, w) patch grid expands to: t*h*w patches grouped into
// MergeSize^2 merge blocks.
func (c *VisionConfig) TokensPerImage(grid [3]int) int {
	m := c.MergeSize
	if m <= 0 {
		m = 1
	}
	return grid[0] * grid[1] * grid[2] / (m * m)
}

// VisionEncoder runs the vision tower + projector over preprocessed image
// patches and returns, per image, one projected embedding row per image
// token (TokensPerImage rows of TextHiddenSize each).
type VisionEncoder interface {
	Config() *VisionConfig
	// Encode consumes pixelValues laid out as numPatches rows of PatchDim()
	// floats (merge-block-major order, see the image processor) and one
	// (t, h, w) grid per image. It returns per image one embedding row per
	// image token: [image][token][TextHiddenSize].
	Encode(pixelValues []float32, grids [][3]int) ([][][]float32, error)
}

// VisionEncoderProvider is implemented by model runtimes that carry a vision
// encoder (nil when the model is text-only).
type VisionEncoderProvider interface {
	VisionEncoder() VisionEncoder
}
