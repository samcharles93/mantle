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

// audioConfigOf returns the model's audio config, or nil for models without an
// audio tower.
func audioConfigOf(e *EngineImpl) *core.AudioConfig {
	rt, ok := e.model.(core.Runtime)
	if !ok || rt == nil {
		return nil
	}
	cfg := rt.ModelConfig()
	if cfg == nil {
		return nil
	}
	return cfg.Audio
}

// audioEncoderOf returns the model runtime's audio encoder, or nil.
func audioEncoderOf(e *EngineImpl) core.AudioEncoder {
	p, ok := e.model.(core.AudioEncoderProvider)
	if !ok || p == nil {
		return nil
	}
	return p.AudioEncoder()
}

// prepareMultimodal encodes the request's images/videos through the vision
// tower and its audio clips through the audio tower, then expands the prompt's
// placeholder tokens: each image placeholder becomes one token per projected
// image embedding; each video placeholder becomes per-temporal-segment
// timestamp text plus vision_start, one video token per projected token, and
// vision_end; each audio placeholder becomes ceil(melFrames/encoderStride)
// <|audio|> tokens, one per projected audio embedding (processing_lfm3_omni.py
// replace_audio_token semantics). It returns the expanded token ids plus a
// parallel embeddings slice (nil except at media-token positions) suitable for
// PrefillTokensEmbeds.
func prepareMultimodal(e *EngineImpl, req *Request, ids []int) ([]int, [][]float32, error) {
	vc := visionConfigOf(e)
	ac := audioConfigOf(e)
	hasMedia := len(req.Images) > 0 || len(req.Videos) > 0 || len(req.Audio) > 0
	if vc == nil && ac == nil {
		if hasMedia {
			return nil, nil, fmt.Errorf("model has no vision or audio tower but media was provided")
		}
		return ids, nil, nil
	}
	if !hasMedia {
		return ids, nil, nil
	}

	// Vision payload: build the combined patch payload (images, then videos).
	var pixelValues []float32
	var grids [][3]int
	var feats [][][]float32
	if vc != nil && (len(req.Images) > 0 || len(req.Videos) > 0) {
		ve := visionEncoderOf(e)
		if ve == nil {
			return nil, nil, fmt.Errorf("model config declares a vision tower but the runtime has no encoder")
		}
		if vc.ImageTokenID <= 0 {
			return nil, nil, fmt.Errorf("vision config missing image_token_id")
		}
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
	}

	// Audio payload: WAV -> log-mel -> audio tower -> projected embeddings.
	var audioFeats [][][]float32
	if ac != nil && len(req.Audio) > 0 {
		ae := audioEncoderOf(e)
		if ae == nil {
			return nil, nil, fmt.Errorf("model config declares an audio tower but the runtime has no encoder")
		}
		if ac.AudioTokenID <= 0 {
			return nil, nil, fmt.Errorf("audio config missing audio_token_id")
		}
		mels, err := simd.ProcessAudioClips(req.Audio, ac)
		if err != nil {
			return nil, nil, fmt.Errorf("process audio: %w", err)
		}
		audioFeats, err = ae.Encode(mels)
		if err != nil {
			return nil, nil, fmt.Errorf("audio encode: %w", err)
		}
		if len(audioFeats) != len(req.Audio) {
			return nil, nil, fmt.Errorf("audio encode returned %d clips, want %d", len(audioFeats), len(req.Audio))
		}
	}

	// Placeholder counts must match the provided media.
	nImg, nVid, nAud := 0, 0, 0
	for _, id := range ids {
		switch {
		case vc != nil && id == vc.ImageTokenID:
			nImg++
		case vc != nil && id == vc.VideoTokenID:
			nVid++
		case ac != nil && id == ac.AudioTokenID:
			nAud++
		}
	}
	if vc != nil {
		if nImg != len(req.Images) {
			return nil, nil, fmt.Errorf("prompt has %d image placeholder token(s) but %d image(s) were provided", nImg, len(req.Images))
		}
		if nVid != len(req.Videos) {
			return nil, nil, fmt.Errorf("prompt has %d video placeholder token(s) but %d video(s) were provided", nVid, len(req.Videos))
		}
	}
	if ac != nil && nAud != len(req.Audio) {
		return nil, nil, fmt.Errorf("prompt has %d audio placeholder token(s) but %d audio clip(s) were provided", nAud, len(req.Audio))
	}

	// Expand placeholders in prompt order, consuming media of each kind in order.
	var newIDs []int
	embeds := make([][]float32, 0, len(ids))
	imgIdx, vidIdx, audIdx := 0, 0, 0
	appendTok := func(id int, emb []float32) {
		newIDs = append(newIDs, id)
		embeds = append(embeds, emb)
	}
	imgGridBase, vidGridBase := 0, len(req.Images)
	for _, id := range ids {
		switch {
		case vc != nil && id == vc.ImageTokenID:
			gi := imgGridBase + imgIdx
			n := len(feats[gi])
			for k := range n {
				appendTok(id, feats[gi][k])
			}
			imgIdx++
		case vc != nil && id == vc.VideoTokenID:
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
				for j := range segLen {
					appendTok(id, feats[gi][rowBase+j])
				}
				appendTok(vc.VisionEndTokenID, nil)
			}
			vidIdx++
		case ac != nil && id == ac.AudioTokenID:
			// Each <|audio|> placeholder expands to one token per projected
			// audio embedding of the next clip (ceil(melFrames/stride) rows).
			rows := audioFeats[audIdx]
			for k := range rows {
				appendTok(id, rows[k])
			}
			audIdx++
		default:
			appendTok(id, nil)
		}
	}
	return newIDs, embeds, nil
}
