package simd

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"

	"github.com/samcharles93/mantle/internal/backend/core"
)

// ProcessImages decodes raw image files (PNG/JPEG), runs the LFM3-VL image
// preprocessing (smart resize, rescale, normalize, patchify) and returns the
// flattened patch payload plus one (t, h, w) patch grid per image, ready for
// VisionTower.Encode.
//
// The resize replicates torchvision's bicubic antialias=True semantics on
// uint8 data (cubic kernel a=-0.5, support 2*max(in/out,1), weights normalized
// over the in-range taps, out-of-range taps dropped, per-pass rounding back to
// uint8). HF processes uint8 tensors through torchvision's fixed-point uint8
// kernel, which differs from this float path by at most a couple of LSBs per
// pixel; parity tests use a tolerance for the pixel payload and argmax/greedy
// agreement for end-to-end logits.
func ProcessImages(images [][]byte, cfg *core.VisionConfig) (pixelValues []float32, grids [][3]int, err error) {
	if cfg == nil {
		return nil, nil, fmt.Errorf("vision: nil vision config")
	}
	patch := cfg.PatchSize
	merge := cfg.MergeSize
	if patch <= 0 || merge <= 0 {
		return nil, nil, fmt.Errorf("vision: invalid patch/merge size")
	}
	factor := patch * merge
	for _, raw := range images {
		pix, h, w, err := decodeImageRGB(raw)
		if err != nil {
			return nil, nil, err
		}
		rh, rw := smartResize(h, w, factor, cfg.MinPixels, cfg.MaxPixels)
		if rh != h || rw != w {
			pix = resizeBicubicUInt8(pix, h, w, rh, rw)
			h, w = rh, rw
		}
		gh, gw := h/patch, w/patch
		if gh*patch != h || gw*patch != w {
			return nil, nil, fmt.Errorf("vision: resized image %dx%d not divisible by patch %d", h, w, patch)
		}
		grid := [3]int{1, gh, gw}
		patches := patchify(pix, h, w, cfg)
		pixelValues = append(pixelValues, patches...)
		grids = append(grids, grid)
	}
	return pixelValues, grids, nil
}

// decodeImageRGB decodes PNG/JPEG bytes to a channel-first uint8 RGB image.
// Alpha (if present) is composited over white, matching HF convert_to_rgb.
func decodeImageRGB(raw []byte) (pix []uint8, h, w int, err error) {
	img, format, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("vision: decode image: %w", err)
	}
	_ = format
	b := img.Bounds()
	h, w = b.Dy(), b.Dx()
	pix = make([]uint8, 3*h*w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bl, a := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			// composite over white in 0..1 float space, then back to uint8
			af := float64(a) / 65535.0
			rf := (float64(r)/65535.0)*af + (1 - af)
			gf := (float64(g)/65535.0)*af + (1 - af)
			bf := (float64(bl)/65535.0)*af + (1 - af)
			pix[(y*w + x)] = uint8(math.Round(rf * 255))
			pix[h*w+(y*w+x)] = uint8(math.Round(gf * 255))
			pix[2*h*w+(y*w+x)] = uint8(math.Round(bf * 255))
		}
	}
	return pix, h, w, nil
}

// smartResize mirrors HF's smart_resize: round both dims to a multiple of
// factor, then scale to fit [minPixels, maxPixels] preserving aspect ratio.
func smartResize(h, w, factor, minPixels, maxPixels int) (int, int) {
	if factor <= 0 {
		factor = 1
	}
	if max(h, w) > 0 && float64(max(h, w))/float64(min(h, w)) > 200 {
		// HF raises; we clamp instead to keep the runtime total
		return h, w
	}
	hBar := int(math.Round(float64(h)/float64(factor))) * factor
	wBar := int(math.Round(float64(w)/float64(factor))) * factor
	if hBar*wBar > maxPixels {
		beta := math.Sqrt(float64(h*w) / float64(maxPixels))
		hBar = max(factor, int(math.Floor(float64(h)/beta/float64(factor)))*factor)
		wBar = max(factor, int(math.Floor(float64(w)/beta/float64(factor)))*factor)
	} else if hBar*wBar < minPixels {
		beta := math.Sqrt(float64(minPixels) / float64(h*w))
		hBar = int(math.Ceil(float64(h)*beta/float64(factor))) * factor
		wBar = int(math.Ceil(float64(w)*beta/float64(factor))) * factor
	}
	return hBar, wBar
}

// resizeAxisCoeffs replicates torchvision's uint8 antialias bicubic
// coefficient computation (aten HelperInterpCubic, aa path): center =
// r*(o+0.5), support 2*max(r,1), taps [xmin, xmin+xsize) with weights
// cubic_a-0.5((j + xmin - center + 0.5) * invscale) normalized over the taps,
// quantized to int16 fixed point at the smallest precision p in [0,22) that
// keeps wt_max*2^(p+1) < 2^15. out = clamp((sum(x_j*q_j) + 2^(p-1)) >> p).
type resizeAxisCoeffs struct {
	xmin []int
	q    []int32 // [O][maxInterp] quantized weights
	p    uint
}

func computeResizeAxis(I, O int) resizeAxisCoeffs {
	r := float64(I) / float64(O)
	support := 2.0
	if r >= 1 {
		support = 2 * r
	}
	maxInterp := int(math.Ceil(support))*2 + 1
	invscale := 1.0
	if r >= 1 {
		invscale = 1.0 / r
	}
	// pass 1: float weights, track max
	w := make([]float64, O*maxInterp)
	xmin := make([]int, O)
	xsize := make([]int, O)
	wtMax := 0.0
	for o := 0; o < O; o++ {
		center := r * (float64(o) + 0.5)
		xm := int64(center - support + 0.5) // C++ int64 cast truncates toward zero
		if xm < 0 {
			xm = 0
		}
		xs := int64(center + support + 0.5)
		if xs > int64(I) {
			xs = int64(I)
		}
		xs -= xm
		if xs < 0 {
			xs = 0
		}
		if xs > int64(maxInterp) {
			xs = int64(maxInterp)
		}
		xmin[o] = int(xm)
		xsize[o] = int(xs)
		total := 0.0
		for j := 0; j < int(xs); j++ {
			wv := cubicBicubicAA((float64(j)+float64(xm)-center+0.5)*invscale) / 1.0
			w[o*maxInterp+j] = wv
			total += wv
		}
		if total != 0 {
			mx := 0.0
			for j := 0; j < int(xs); j++ {
				w[o*maxInterp+j] /= total
				if w[o*maxInterp+j] > mx {
					mx = w[o*maxInterp+j]
				}
			}
			if mx > wtMax {
				wtMax = mx
			}
		}
	}
	// precision: smallest p in [0,22) with (int)(0.5 + wtMax*2^(p+1)) >= 2^15
	p := uint(0)
	for p < 22 {
		next := int(0.5 + wtMax*float64(int64(1)<<(p+1)))
		if next >= 1<<15 {
			break
		}
		p++
	}
	q := make([]int32, O*maxInterp)
	scale := float64(int64(1) << p)
	for i, v := range w {
		s := v * scale
		if s < 0 {
			q[i] = int32(-0.5 + s)
		} else {
			q[i] = int32(0.5 + s)
		}
	}
	return resizeAxisCoeffs{xmin: xmin, q: q, p: p}
}

// cubicBicubicAA is the PIL-compatible bicubic kernel (a=-0.5) used on the
// antialias path.
func cubicBicubicAA(x float64) float64 {
	ax := math.Abs(x)
	if ax < 1 {
		return 1.5*ax*ax*ax - 2.5*ax*ax + 1
	}
	if ax < 2 {
		return -0.5*ax*ax*ax + 2.5*ax*ax - 4*ax + 2
	}
	return 0
}

// resizeAxisUint8 resizes one row/column of uint8 values with the quantized
// fixed-point kernel: out = clamp((sum(x*q) + 2^(p-1)) >> p, 0, 255).
func resizeAxisUint8(x []uint8, O int, c resizeAxisCoeffs) []uint8 {
	out := make([]uint8, O)
	half := int32(1) << (c.p - 1)
	maxInterp := len(c.q) / O
	for o := 0; o < O; o++ {
		acc := half
		base := o * maxInterp
		xm := c.xmin[o]
		for j := 0; j < maxInterp; j++ {
			q := c.q[base+j]
			if q == 0 {
				continue
			}
			i := xm + j
			if i < 0 || i >= len(x) {
				continue
			}
			acc += int32(x[i]) * q
		}
		v := acc >> c.p
		if v < 0 {
			v = 0
		}
		if v > 255 {
			v = 255
		}
		out[o] = uint8(v)
	}
	return out
}

// resizeBicubicUInt8 resizes a channel-first uint8 image (C,H,W) to (C,H',W')
// with torchvision's uint8 antialias bicubic: horizontal (W) pass first, then
// vertical (H), each rounding back to uint8 (the AVX kernel's intermediate
// buffer is uint8).
func resizeBicubicUInt8(pix []uint8, h, w, H, W int) []uint8 {
	C := len(pix) / (h * w)
	// horizontal pass: (C, h, w) -> (C, h, W)
	cw := computeResizeAxis(w, W)
	mid := make([]uint8, C*h*W)
	for c := 0; c < C; c++ {
		for y := 0; y < h; y++ {
			row := pix[c*h*w+y*w : c*h*w+(y+1)*w]
			out := resizeAxisUint8(row, W, cw)
			copy(mid[c*h*W+y*W:c*h*W+(y+1)*W], out)
		}
	}
	// vertical pass: (C, h, W) -> (C, H, W)
	ch := computeResizeAxis(h, H)
	out := make([]uint8, C*H*W)
	for c := 0; c < C; c++ {
		for x := 0; x < W; x++ {
			col := make([]uint8, h)
			for y := 0; y < h; y++ {
				col[y] = mid[c*h*W+y*W+x]
			}
			res := resizeAxisUint8(col, H, ch)
			for y := 0; y < H; y++ {
				out[c*H*W+y*W+x] = res[y]
			}
		}
	}
	return out
}

// patchify converts a channel-first uint8 image into the flattened patch
// payload: one row per patch of in_channels*temporal_patch*patch^2 floats,
// ordered merge-block-major (block row, block col, in-block row, in-block
// col), each row laid out [channel][temporal][patch row][patch col] with the
// temporal slots duplicated for still images. Values are rescaled and
// normalized per channel: (px*rescale - mean) / std.
func patchify(pix []uint8, h, w int, cfg *core.VisionConfig) []float32 {
	C := cfg.InChannels
	patch := cfg.PatchSize
	merge := cfg.MergeSize
	temporal := cfg.TemporalPatch
	gh, gw := h/patch, w/patch
	patchDim := C * temporal * patch * patch
	n := gh * gw
	out := make([]float32, n*patchDim)
	mean := cfg.ImageMean
	std := cfg.ImageStd
	rescale := cfg.RescaleFactor
	for bi := 0; bi < (gh/merge)*(gw/merge); bi++ {
		bh := bi / (gw / merge)
		bw := bi % (gw / merge)
		for wi := 0; wi < merge*merge; wi++ {
			mh := wi / merge
			mw := wi % merge
			py := (bh*merge + mh) * patch
			px := (bw*merge + mw) * patch
			row := (bi*merge*merge + wi) * patchDim
			off := 0
			for c := 0; c < C; c++ {
				m, s := 0.5, 0.5
				if len(mean) > c {
					m = mean[c]
				}
				if len(std) > c {
					s = std[c]
				}
				for t := 0; t < temporal; t++ {
					for pi := 0; pi < patch; pi++ {
						for pj := 0; pj < patch; pj++ {
							v := float64(pix[c*h*w+(py+pi)*w+(px+pj)]) * rescale
							out[row+off] = float32((v - m) / s)
							off++
						}
					}
				}
			}
		}
	}
	return out
}
