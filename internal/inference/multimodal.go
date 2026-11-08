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

// prepareMultimodal encodes the request's images/videos through the vision
// tower and expands the prompt's placeholder tokens: each image placeholder
// becomes one token per projected image embedding; each video placeholder
// becomes per-temporal-segment timestamp text plus vision_start, one video
// token per projected token, and vision_end. It returns the expanded token
// ids plus a parallel embeddings slice (nil except at media-token positions)
// suitable for PrefillTokensEmbeds.
func prepareMultimodal(e *EngineImpl, req *Request, ids []int) ([]int, [][]float32, error) {
	vc := visionConfigOf(e)
	if vc == nil {
		if len(req.Images) > 0 || len(req.Videos) > 0 {
			return nil, nil, fmt.Errorf("model has no vision tower but media was provided")
		}
		return ids, nil, nil
	}
	if len(req.Images) == 0 && len(req.Videos) == 0 {
		return ids, nil, nil
	}
	ve := visionEncoderOf(e)
	if ve == nil {
		return nil, nil, fmt.Errorf("model config declares a vision tower but the runtime has no encoder")
	}
	if vc.ImageTokenID <= 0 {
		return nil, nil, fmt.Errorf("vision config missing image_token_id")
	}

	// Build the combined patch payload: images first, then videos.
	var pixelValues []float32
	var grids [][3]int
	for _, img := range req.Images {
		pv, g, err := simd.ProcessImages([][]byte{img}, vc)
		if err != nil {
			return nil, nil, fmt.Errorf("process image: %w", err)
		}
		pixelValues = append(pixelValues, pv...)
		grids = append(grids, g[0])
	}
	for _, frames := range req.Videos {
		pv, g, err := simd.ProcessVideoFrames(frames, vc)
		if err != nil {
			return nil, nil, fmt.Errorf("process video: %w", err)
		}
		pixelValues = append(pixelValues, pv...)
		grids = append(grids, g)
	}
	feats, err := ve.Encode(pixelValues, grids)
	if err != nil {
		return nil, nil, fmt.Errorf("vision encode: %w", err)
	}
	if len(feats) != len(grids) {
		return nil, nil, fmt.Errorf("vision encode returned %d media items, want %d", len(feats), len(grids))
	}

	// Placeholder counts must match the provided media.
	nImg, nVid := 0, 0
	for _, id := range ids {
		switch id {
		case vc.ImageTokenID:
			nImg++
		case vc.VideoTokenID:
			nVid++
		}
	}
	if nImg != len(req.Images) {
		return nil, nil, fmt.Errorf("prompt has %d image placeholder token(s) but %d image(s) were provided", nImg, len(req.Images))
	}
	if nVid != len(req.Videos) {
		return nil, nil, fmt.Errorf("prompt has %d video placeholder token(s) but %d video(s) were provided", nVid, len(req.Videos))
	}

	// Expand placeholders in prompt order, consuming media of each kind in order.
	var newIDs []int
	embeds := make([][]float32, 0, len(ids))
	imgIdx, vidIdx := 0, 0
	appendTok := func(id int, emb []float32) {
		newIDs = append(newIDs, id)
		embeds = append(embeds, emb)
	}
	imgGridBase, vidGridBase := 0, len(req.Images)
	for _, id := range ids {
		switch id {
		case vc.ImageTokenID:
			gi := imgGridBase + imgIdx
			n := len(feats[gi])
			for k := 0; k < n; k++ {
				appendTok(id, feats[gi][k])
			}
			imgIdx++
		case vc.VideoTokenID:
			gi := vidGridBase + vidIdx
			g := grids[gi]
			segLen := vc.TokensPerImage([3]int{1, g[1], g[2]})
			fps := vc.VideoFPS
			if fps <= 0 {
				fps = 2
			}
			for k := 0; k < g[0]; k++ {
				// timestamp of segment k: average of its two frames' times,
				// frame j at j/fps (HF _calculate_timestamps)
				ts := float64(4*k+1) / (2 * fps)
				tsIDs, err := safeEncode(e.tokenizer, fmt.Sprintf("<%.1f seconds>", ts))
				if err != nil {
					return nil, nil, fmt.Errorf("encode video timestamp: %w", err)
				}
				for _, tid := range tsIDs {
					appendTok(tid, nil)
				}
				appendTok(vc.VisionStartTokenID, nil)
				rowBase := k * segLen
				for j := 0; j < segLen; j++ {
					appendTok(id, feats[gi][rowBase+j])
				}
				appendTok(vc.VisionEndTokenID, nil)
			}
			vidIdx++
		default:
			appendTok(id, nil)
		}
	}
	return newIDs, embeds, nil
}
