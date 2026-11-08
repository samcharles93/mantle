package inference

import (
	"fmt"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/simd"
)

// visionConfigOf returns the model's vision config, or nil for text-only models.
func visionConfigOf(e *EngineImpl) *core.VisionConfig {
	rt, ok := e.model.(core.Runtime)
	if !ok || rt == nil {
		return nil
	}
	cfg := rt.ModelConfig()
	if cfg == nil {
		return nil
	}
	return cfg.Vision
}

// visionEncoderOf returns the model runtime's vision encoder, or nil.
func visionEncoderOf(e *EngineImpl) core.VisionEncoder {
	p, ok := e.model.(core.VisionEncoderProvider)
	if !ok || p == nil {
		return nil
	}
	return p.VisionEncoder()
}

// prepareMultimodal encodes the request's images through the vision tower and
// expands the prompt's image placeholder tokens into one token per projected
// image embedding. It returns the expanded token ids plus a parallel
// embeddings slice (nil except at image-token positions) suitable for
// PrefillTokensEmbeds.
func prepareMultimodal(e *EngineImpl, req *Request, ids []int) ([]int, [][]float32, error) {
	vc := visionConfigOf(e)
	if vc == nil {
		if len(req.Images) > 0 {
			return nil, nil, fmt.Errorf("model has no vision tower but %d image(s) were provided", len(req.Images))
		}
		return ids, nil, nil
	}
	if len(req.Images) == 0 {
		return ids, nil, nil
	}
	ve := visionEncoderOf(e)
	if ve == nil {
		return nil, nil, fmt.Errorf("model config declares a vision tower but the runtime has no encoder")
	}
	if vc.ImageTokenID <= 0 {
		return nil, nil, fmt.Errorf("vision config missing image_token_id")
	}

	pixelValues, grids, err := simd.ProcessImages(req.Images, vc)
	if err != nil {
		return nil, nil, fmt.Errorf("process images: %w", err)
	}
	feats, err := ve.Encode(pixelValues, grids)
	if err != nil {
		return nil, nil, fmt.Errorf("vision encode: %w", err)
	}
	if len(feats) != len(grids) {
		return nil, nil, fmt.Errorf("vision encode returned %d images, want %d", len(feats), len(grids))
	}

	// Count image placeholder tokens in the prompt; each one consumes one
	// image's features in order.
	counts := make([]int, len(grids))
	nPlaceholders := 0
	for _, id := range ids {
		if id == vc.ImageTokenID {
			nPlaceholders++
		}
	}
	if nPlaceholders != len(req.Images) {
		return nil, nil, fmt.Errorf("prompt has %d image placeholder token(s) but %d image(s) were provided", nPlaceholders, len(req.Images))
	}
	for i, g := range grids {
		counts[i] = vc.TokensPerImage(g)
		if len(feats[i]) != counts[i] {
			return nil, nil, fmt.Errorf("image %d: vision produced %d token rows, grid implies %d", i, len(feats[i]), counts[i])
		}
	}

	// Expand: each image placeholder token becomes counts[img] copies of the
	// image token id, carrying that many embedding rows.
	var newIDs []int
	embeds := make([][]float32, 0, len(ids))
	img := 0
	for _, id := range ids {
		if id == vc.ImageTokenID {
			for k := range counts[img] {
				newIDs = append(newIDs, id)
				embeds = append(embeds, feats[img][k])
			}
			img++
			continue
		}
		newIDs = append(newIDs, id)
		embeds = append(embeds, nil)
	}
	return newIDs, embeds, nil
}
