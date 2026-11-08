//go:build cuda

package cuda

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/samcharles93/mantle/internal/backend/bootstrap"
	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// simdTestdataPath resolves a fixture in internal/backend/simd/testdata from
// the cuda package's test working directory.
func simdTestdataPath(name string) string {
	return "../simd/testdata/" + name
}

type lfm3vlRefCase struct {
	Name        string   `json:"name"`
	PixelsFile  string   `json:"pixels_file,omitempty"`
	PixelsShape []int    `json:"pixels_shape,omitempty"`
	GridTHW     [][]int  `json:"grid_thw,omitempty"`
	FeatsFile   string   `json:"feats_file,omitempty"`
	Video       bool     `json:"video,omitempty"`
	Frames      []string `json:"frames,omitempty"`
}

func loadLFM3VLRefCases(t testing.TB) []lfm3vlRefCase {
	t.Helper()
	raw, err := os.ReadFile(simdTestdataPath("lfm3vl_reference.json"))
	if err != nil {
		t.Fatalf("read reference fixture: %v", err)
	}
	var fx struct {
		Cases []lfm3vlRefCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse reference fixture: %v", err)
	}
	return fx.Cases
}

func readF32File(t testing.TB, path string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if len(raw)%4 != 0 {
		t.Fatalf("%s: size %d not a multiple of 4", path, len(raw))
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// newVisionTestOps loads the LFM3-VL model's host runtime and builds only the
// device vision encoder (no text-weight preload), returning the Ops holding it
// and the core instance (whose Vision field is the host oracle).
func newVisionTestOps(t *testing.T) (*Ops, *core.Instance) {
	t.Helper()
	return newVisionTestOpsPath(t, "MANTLE_LFM3VL_MCF")
}

func newVisionTestOpsPath(t *testing.T, envName string) (*Ops, *core.Instance) {
	t.Helper()
	path := os.Getenv(envName)
	if path == "" {
		t.Skipf("set %s to a LFM3-VL container to run the device vision tests", envName)
	}
	if !cudaAvailable(t) {
		return nil, nil
	}
	file, err := mcfstore.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	t.Cleanup(func() { _ = file.Close() })
	_, coreInst, err := bootstrap.LoadSIMDRuntime(file, file.SectionData(mcf.SectionHFConfigJSON), 64, core.LoadModelOptions{}, true)
	if err != nil {
		t.Fatalf("LoadSIMDRuntime: %v", err)
	}
	if coreInst.Vision == nil {
		t.Fatalf("model has no vision tower loaded")
	}
	stream, err := native.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { _ = stream.Destroy() })
	blas, err := native.NewBlasHandle(stream)
	if err != nil {
		t.Fatalf("NewBlasHandle: %v", err)
	}
	t.Cleanup(func() { _ = blas.Destroy() })
	ops := NewOps(stream, blas)
	t.Cleanup(func() { _ = ops.Close() })
	if err := ops.preloadVisionWeights(coreInst); err != nil {
		t.Fatalf("preloadVisionWeights: %v", err)
	}
	if ops.vision == nil {
		t.Fatalf("device vision encoder was not built")
	}
	return ops, coreInst
}

// maxFeatsDiff compares got ([][][]float32 per image) against a flat f32
// reference and returns the max abs diff.
func maxFeatsDiff(got [][][]float32, ref []float32) (float64, int) {
	n := 0
	for _, img := range got {
		for _, row := range img {
			n += len(row)
		}
	}
	if n != len(ref) {
		return math.MaxFloat64, n
	}
	worst := 0.0
	i := 0
	for _, img := range got {
		for _, row := range img {
			for _, v := range row {
				if d := math.Abs(float64(v - ref[i])); d > worst {
					worst = d
				}
				i++
			}
		}
	}
	return worst, n
}

// TestCUDAVisionEncoderParityFixtures runs the device vision encoder on the
// HF-captured pixel fixtures (256x256 and 100x70 images, plus the 4-frame
// video case) and checks (a) the projected features against the HF reference
// feats at 1e-4 (mirroring simd/lfm3vl_test.go) and (b) device-vs-host tower
// agreement.
func TestCUDAVisionEncoderParityFixtures(t *testing.T) {
	ops, coreInst := newVisionTestOps(t)
	enc := ops.vision
	cases := loadLFM3VLRefCases(t)
	want := map[string]bool{"image_256": true, "image_100x70": true, "video_4frames": true}
	ran := 0
	for _, tc := range cases {
		if !want[tc.Name] || tc.PixelsFile == "" {
			continue
		}
		ran++
		t.Run(tc.Name, func(t *testing.T) {
			pix := readF32File(t, simdTestdataPath(tc.PixelsFile))
			if len(pix) != tc.PixelsShape[0]*tc.PixelsShape[1] {
				t.Fatalf("pixels size %d, want %d", len(pix), tc.PixelsShape[0]*tc.PixelsShape[1])
			}
			grids := [][3]int{{tc.GridTHW[0][0], tc.GridTHW[0][1], tc.GridTHW[0][2]}}

			got, err := enc.Encode(pix, grids)
			if err != nil {
				t.Fatalf("device Encode: %v", err)
			}
			hostGot, err := coreInst.Vision.Encode(pix, grids)
			if err != nil {
				t.Fatalf("host Encode: %v", err)
			}

			// (b) device vs host tower.
			hostFlat := flattenFeats(hostGot)
			devVsHost, n := maxFeatsDiff(got, hostFlat)
			if n != len(hostFlat) {
				t.Fatalf("device produced %d feats, host %d", n, len(hostFlat))
			}
			t.Logf("device-vs-host max abs diff = %g (%d feats)", devVsHost, n)
			if devVsHost > 1e-4 {
				t.Fatalf("device vs host tower diff %g exceeds 1e-4", devVsHost)
			}

			// (a) HF reference feats where available.
			if tc.FeatsFile != "" {
				ref := readF32File(t, simdTestdataPath(tc.FeatsFile))
				d, n := maxFeatsDiff(got, ref)
				if n != len(ref) {
					t.Fatalf("feats size mismatch: got %d, ref %d", n, len(ref))
				}
				t.Logf("device-vs-HF max abs diff = %g", d)
				if d > 1e-4 {
					t.Fatalf("device feats vs HF reference diff %g exceeds 1e-4", d)
				}
			}
		})
	}
	if ran != len(want) {
		t.Fatalf("ran %d cases, want %d", ran, len(want))
	}
}

// TestCUDAVisionEncoderQ8Parity exercises the quantized-vision-weights path:
// the q8 container stores the vision tower as q8 blocks, which the device
// encoder dequantizes to f32 at preload. Compares device vs host tower on the
// 256px fixture (device-vs-host agreement; the HF feats fixture is for the
// bf16 container).
func TestCUDAVisionEncoderQ8Parity(t *testing.T) {
	ops, coreInst := newVisionTestOpsPath(t, "MANTLE_LFM3VL_Q8_MCF")
	enc := ops.vision
	for _, tc := range loadLFM3VLRefCases(t) {
		if tc.Name != "image_256" || tc.PixelsFile == "" {
			continue
		}
		pix := readF32File(t, simdTestdataPath(tc.PixelsFile))
		grids := [][3]int{{tc.GridTHW[0][0], tc.GridTHW[0][1], tc.GridTHW[0][2]}}
		got, err := enc.Encode(pix, grids)
		if err != nil {
			t.Fatalf("device Encode: %v", err)
		}
		want, err := coreInst.Vision.Encode(pix, grids)
		if err != nil {
			t.Fatalf("host Encode: %v", err)
		}
		d, n := maxFeatsDiff(got, flattenFeats(want))
		if n != len(flattenFeats(want)) {
			t.Fatalf("device produced %d feats, host %d", n, len(flattenFeats(want)))
		}
		t.Logf("q8 image_256: device-vs-host max abs diff = %g", d)
		// The host oracle quantizes activations to int8 blocks for q8 weights
		// (int8 dot * scale), while the device path dequantizes the weights to
		// f32 and accumulates in f32 — strictly more precise, so the two agree
		// only up to q8 activation-quantization noise (~5e-2 here), not 1e-4.
		// A decode bug would show O(1) diffs; 0.1 catches that with headroom.
		if d > 0.1 {
			t.Fatalf("q8 device vs host diff %g exceeds q8-noise tolerance 0.1", d)
		}
	}
}

func flattenFeats(f [][][]float32) []float32 {
	n := 0
	for _, img := range f {
		for _, row := range img {
			n += len(row)
		}
	}
	out := make([]float32, 0, n)
	for _, img := range f {
		for _, row := range img {
			out = append(out, row...)
		}
	}
	return out
}

// TestCUDAVisionEncoderFallbackGate checks that an input above the
// MANTLE_CUDA_VISION_MAX_PATCHES gate is served by the host tower and matches
// it bit-for-bit.
func TestCUDAVisionEncoderFallbackGate(t *testing.T) {
	t.Setenv("MANTLE_CUDA_VISION_MAX_PATCHES", "64")
	ops, coreInst := newVisionTestOps(t)
	enc := ops.vision
	if enc.maxPatches != 64 {
		t.Fatalf("maxPatches = %d, want 64", enc.maxPatches)
	}
	cases := loadLFM3VLRefCases(t)
	for _, tc := range cases {
		if tc.Name != "image_256" {
			continue
		}
		pix := readF32File(t, simdTestdataPath(tc.PixelsFile))
		grids := [][3]int{{tc.GridTHW[0][0], tc.GridTHW[0][1], tc.GridTHW[0][2]}}
		got, err := enc.Encode(pix, grids) // 256 patches > 64 -> host fallback
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		want, err := coreInst.Vision.Encode(pix, grids)
		if err != nil {
			t.Fatalf("host Encode: %v", err)
		}
		gotFlat := flattenFeats(got)
		wantFlat := flattenFeats(want)
		if len(gotFlat) != len(wantFlat) {
			t.Fatalf("fallback produced %d feats, host %d", len(gotFlat), len(wantFlat))
		}
		for i := range gotFlat {
			if gotFlat[i] != wantFlat[i] {
				t.Fatalf("fallback not bit-identical to host at %d: %g vs %g", i, gotFlat[i], wantFlat[i])
			}
		}
	}
}

// TestCUDARuntimeVisionEncoderWiring loads the full CUDA backend (text weights
// included) and checks that VisionEncoderProvider returns the device encoder,
// then runs the 256px fixture through it against the HF feats.
func TestCUDARuntimeVisionEncoderWiring(t *testing.T) {
	path := os.Getenv("MANTLE_LFM3VL_MCF")
	if path == "" {
		t.Skip("set MANTLE_LFM3VL_MCF to the LFM3-VL-2B container to run the device vision tests")
	}
	if !cudaAvailable(t) {
		return
	}
	file, err := mcfstore.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	t.Cleanup(func() { _ = file.Close() })
	b := &Backend{}
	rt, err := b.LoadModel(file, file.SectionData(mcf.SectionHFConfigJSON), 64, core.LoadModelOptions{})
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	if c, ok := rt.(io.Closer); ok {
		t.Cleanup(func() { _ = c.Close() })
	}
	p, ok := rt.(core.VisionEncoderProvider)
	if !ok {
		t.Fatalf("cuda runtime does not implement core.VisionEncoderProvider")
	}
	ve := p.VisionEncoder()
	if ve == nil {
		t.Fatalf("VisionEncoder() returned nil")
	}
	if _, ok := ve.(*visionEncoder); !ok {
		t.Fatalf("VisionEncoder() = %T, want *visionEncoder (device tower)", ve)
	}
	for _, tc := range loadLFM3VLRefCases(t) {
		if tc.Name != "image_256" || tc.FeatsFile == "" {
			continue
		}
		pix := readF32File(t, simdTestdataPath(tc.PixelsFile))
		grids := [][3]int{{tc.GridTHW[0][0], tc.GridTHW[0][1], tc.GridTHW[0][2]}}
		got, err := ve.Encode(pix, grids)
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		ref := readF32File(t, simdTestdataPath(tc.FeatsFile))
		d, n := maxFeatsDiff(got, ref)
		if n != len(ref) {
			t.Fatalf("feats size mismatch: got %d, ref %d", n, len(ref))
		}
		t.Logf("full-backend device feats vs HF max abs diff = %g", d)
		if d > 1e-4 {
			t.Fatalf("full-backend device feats vs HF diff %g exceeds 1e-4", d)
		}
	}
}

// TestCUDAVisionEncoderLargeParity runs a 512x512 px image (32x32 = 1024
// patches -> 256 image tokens) through the device encoder and the host tower
// and compares the projected features, with timings for the report.
func TestCUDAVisionEncoderLargeParity(t *testing.T) {
	ops, coreInst := newVisionTestOps(t)
	cfg := coreInst.Vision.Config()
	pix, grids := benchVisionPayload(cfg.PatchDim(), 32, 32)

	t0 := time.Now()
	got, err := ops.vision.Encode(pix, grids)
	if err != nil {
		t.Fatalf("device Encode: %v", err)
	}
	devDur := time.Since(t0)

	t0 = time.Now()
	want, err := coreInst.Vision.Encode(pix, grids)
	if err != nil {
		t.Fatalf("host Encode: %v", err)
	}
	hostDur := time.Since(t0)

	d, n := maxFeatsDiff(got, flattenFeats(want))
	if n != len(flattenFeats(want)) {
		t.Fatalf("device produced %d feats, host %d", n, len(flattenFeats(want)))
	}
	t.Logf("1024 patches: device %s vs host %s; max abs diff = %g", devDur.Round(time.Millisecond), hostDur.Round(time.Millisecond), d)
	if d > 1e-4 {
		t.Fatalf("large-image device vs host diff %g exceeds 1e-4", d)
	}
}

// benchVisionPayload builds a deterministic synthetic pixel payload for a
// [1, h, w] grid.
func benchVisionPayload(patchDim, h, w int) ([]float32, [][3]int) {
	rng := rand.New(rand.NewSource(99))
	n := h * w
	pix := make([]float32, n*patchDim)
	for i := range pix {
		pix[i] = rng.Float32()*2 - 1
	}
	return pix, [][3]int{{1, h, w}}
}

// BenchmarkVisionEncodeDevice256 times the device vision encode of the 256px
// fixture image (256 patches -> 64 image tokens).
func BenchmarkVisionEncodeDevice256(b *testing.B) {
	path := os.Getenv("MANTLE_LFM3VL_MCF")
	if path == "" {
		b.Skip("set MANTLE_LFM3VL_MCF")
	}
	if !cudaAvailable(b) {
		return
	}
	file, err := mcfstore.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = file.Close() }()
	_, coreInst, err := bootstrap.LoadSIMDRuntime(file, file.SectionData(mcf.SectionHFConfigJSON), 64, core.LoadModelOptions{}, true)
	if err != nil {
		b.Fatal(err)
	}
	stream, err := native.NewStream()
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = stream.Destroy() }()
	blas, err := native.NewBlasHandle(stream)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = blas.Destroy() }()
	ops := NewOps(stream, blas)
	defer func() { _ = ops.Close() }()
	if err := ops.preloadVisionWeights(coreInst); err != nil {
		b.Fatal(err)
	}
	enc := ops.vision
	var pix []float32
	var grids [][3]int
	for _, tc := range loadLFM3VLRefCases(b) {
		if tc.Name == "image_256" {
			pix = readF32File(b, simdTestdataPath(tc.PixelsFile))
			grids = [][3]int{{tc.GridTHW[0][0], tc.GridTHW[0][1], tc.GridTHW[0][2]}}
		}
	}
	if pix == nil {
		b.Fatal("image_256 case missing")
	}
	if _, err := enc.Encode(pix, grids); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := enc.Encode(pix, grids); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkVisionEncodeHost256 times the host (simd) vision encode of the
// same 256px fixture for comparison.
func BenchmarkVisionEncodeHost256(b *testing.B) {
	path := os.Getenv("MANTLE_LFM3VL_MCF")
	if path == "" {
		b.Skip("set MANTLE_LFM3VL_MCF")
	}
	file, err := mcfstore.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = file.Close() }()
	rt, _, err := bootstrap.LoadSIMDRuntime(file, file.SectionData(mcf.SectionHFConfigJSON), 64, core.LoadModelOptions{}, true)
	if err != nil {
		b.Fatal(err)
	}
	ve := rt.(core.VisionEncoderProvider).VisionEncoder()
	var pix []float32
	var grids [][3]int
	for _, tc := range loadLFM3VLRefCases(b) {
		if tc.Name == "image_256" {
			pix = readF32File(b, simdTestdataPath(tc.PixelsFile))
			grids = [][3]int{{tc.GridTHW[0][0], tc.GridTHW[0][1], tc.GridTHW[0][2]}}
		}
	}
	if pix == nil {
		b.Fatal("image_256 case missing")
	}
	if _, err := ve.Encode(pix, grids); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ve.Encode(pix, grids); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkVisionEncodeDeviceLarge times the device vision encode of a large
// synthetic 1024x1024 px image (64x64 = 4096 patches -> 1024 tokens), and
// the host oracle on the same payload, for the device-vs-host speed report.
func BenchmarkVisionEncodeLarge(b *testing.B) {
	path := os.Getenv("MANTLE_LFM3VL_MCF")
	if path == "" {
		b.Skip("set MANTLE_LFM3VL_MCF")
	}
	if !cudaAvailable(b) {
		return
	}
	file, err := mcfstore.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = file.Close() }()
	rt, coreInst, err := bootstrap.LoadSIMDRuntime(file, file.SectionData(mcf.SectionHFConfigJSON), 64, core.LoadModelOptions{}, true)
	if err != nil {
		b.Fatal(err)
	}
	_ = rt
	stream, err := native.NewStream()
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = stream.Destroy() }()
	blas, err := native.NewBlasHandle(stream)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = blas.Destroy() }()
	ops := NewOps(stream, blas)
	defer func() { _ = ops.Close() }()
	if err := ops.preloadVisionWeights(coreInst); err != nil {
		b.Fatal(err)
	}
	cfg := coreInst.Vision.Config()
	pix, grids := benchVisionPayload(cfg.PatchDim(), 64, 64) // 1024x1024 px

	b.Run("device", func(b *testing.B) {
		benchVisionEncode(b, ops.vision, pix, grids)
	})
	b.Run("host", func(b *testing.B) {
		ve := coreInst.Vision
		benchVisionEncode(b, ve, pix, grids)
	})
	// 2048x2048 px = 16384 patches (the default device gate boundary); device
	// only — the host oracle needs hours at this size.
	pixBig, gridsBig := benchVisionPayload(cfg.PatchDim(), 128, 128)
	b.Run("device_16384", func(b *testing.B) {
		benchVisionEncode(b, ops.vision, pixBig, gridsBig)
	})
}

func benchVisionEncode(b *testing.B, ve core.VisionEncoder, pix []float32, grids [][3]int) {
	{
		if _, err := ve.Encode(pix, grids); err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := ve.Encode(pix, grids); err != nil {
				b.Fatal(err)
			}
		}
	}
}
