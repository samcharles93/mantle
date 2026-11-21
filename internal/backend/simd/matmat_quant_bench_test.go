package simd

import (
	"fmt"
	"runtime"
	"simd/archsimd"
	"sync"
	"testing"
	"time"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// lfm3QuantShapes are the batched projections of an LFM3-1.7B layer, as
// [rows, cols] of the weight. Every shape runs once per layer of its kind: the
// FFN in all 24 layers, conv in 13 and attention in 11. The 16-row attention
// gate is left out; it is a rounding error next to these.
var lfm3QuantShapes = []struct {
	name       string
	rows, cols int
}{
	{"attn_kv_512x2048", 512, 2048},
	{"attn_qo_conv_out_2048x2048", 2048, 2048},
	{"conv_in_6144x2048", 6144, 2048},
	{"ffn_up_7680x2048", 7680, 2048},
	{"ffn_down_2048x7680", 2048, 7680},
}

// quantBenchRows are the activation row counts measured: decode (1), the
// small chunks useBatchQuant accepts, and the default prefill chunk (64).
var quantBenchRows = []int{1, 2, 4, 8, 16, 32, 64}

// reportGEMMRates reports, for iters calls of an n-row product against a
// rows x cols weight of wBytes payload, the multiply-add rate (GOPS, counting a
// multiply and an add as two operations) and the weight bytes streamed per
// second (wGB/s), the two axes of a roofline.
func reportGEMMRates(b *testing.B, iters, rows, cols, n, wBytes int, elapsed time.Duration) {
	sec := elapsed.Seconds()
	if sec <= 0 || iters == 0 {
		return
	}
	b.ReportMetric(2*float64(rows)*float64(cols)*float64(n)*float64(iters)/sec/1e9, "GOPS")
	b.ReportMetric(float64(wBytes)*float64(iters)/sec/1e9, "wGB/s")
}

// BenchmarkMatMatQuantShapes measures MatMatQuant on packed k4 weights at the
// real LFM3-1.7B projection shapes, for every activation row count in
// quantBenchRows, split so each cost can be read on its own:
//
//   - pool: the public entry point, activation quantisation and dispatch
//     included, at the worker pool's width (GOMAXPROCS at first use).
//   - core: matMatRangeQuant over every row on the calling goroutine with the
//     activations already quantised: the kernel alone on one core.
//   - core_cached: core on the int8 cache instead of the packed payload, so
//     core minus core_cached is the per-call nibble decode.
//   - quantise: getQuantRows alone, the activation pass pool includes.
//   - matvec: n separate MatVec calls, the path a batch of n replaces
//     (n <= 8 only, where the comparison is close).
func BenchmarkMatMatQuantShapes(b *testing.B) {
	if !cpu.HasAVX2 {
		b.Skip("needs AVX2")
	}
	for _, shape := range lfm3QuantShapes {
		rows, cols := shape.rows, shape.cols
		w := packed4Mat(b, mcf.DTypeK4, rows, cols)
		cached := w
		qc, err := instance.BuildQuantCache(&cached)
		if err != nil {
			b.Fatal(err)
		}
		cached.Quant = qc
		wBytes := len(w.Raw)
		maxN := quantBenchRows[len(quantBenchRows)-1]
		x := quantTestActivations(maxN, cols)
		dst := make([]float32, maxN*rows)

		for _, n := range quantBenchRows {
			prefix := fmt.Sprintf("%s/n=%d", shape.name, n)
			b.Run(prefix+"/pool", func(b *testing.B) {
				iters := 0
				for b.Loop() {
					MatMatQuant(dst, &w, x, n)
					iters++
				}
				reportGEMMRates(b, iters, rows, cols, n, wBytes, b.Elapsed())
			})
			for _, mode := range []struct {
				name string
				w    *Mat
			}{{"core", &w}, {"core_cached", &cached}} {
				b.Run(prefix+"/"+mode.name, func(b *testing.B) {
					qr := getQuantRows(x, n, cols, batchWeightsPacked(mode.w))
					defer putQuantRows(qr)
					worker := &matVecWorker{}
					iters := 0
					for b.Loop() {
						matMatRangeQuant(dst, mode.w, qr, n, 0, rows, worker)
						iters++
					}
					reportGEMMRates(b, iters, rows, cols, n, wBytes, b.Elapsed())
				})
			}
			b.Run(prefix+"/quantise", func(b *testing.B) {
				for b.Loop() {
					putQuantRows(getQuantRows(x, n, cols, true))
				}
			})
			if n <= 8 {
				b.Run(prefix+"/matvec", func(b *testing.B) {
					iters := 0
					for b.Loop() {
						for r := range n {
							MatVec(dst[r*rows:(r+1)*rows], &w, x[r*cols:(r+1)*cols])
						}
						iters++
					}
					reportGEMMRates(b, iters, rows, cols, n, wBytes, b.Elapsed())
				})
			}
		}
	}
}

// readBandwidthBytes is the buffer BenchmarkReadBandwidth streams: large
// enough that no cache level holds a meaningful part of it.
const readBandwidthBytes = 512 << 20

// readStripe folds every byte of buf into a checksum with four independent
// 256-bit accumulators, so the loop is bound by loads rather than by the add
// latency chain.
func readStripe(buf []uint64) uint64 {
	var a0, a1, a2, a3 archsimd.Uint64x4
	i := 0
	for ; i+16 <= len(buf); i += 16 {
		a0 = a0.Add(archsimd.LoadUint64x4Array((*[4]uint64)(buf[i : i+4])))
		a1 = a1.Add(archsimd.LoadUint64x4Array((*[4]uint64)(buf[i+4 : i+8])))
		a2 = a2.Add(archsimd.LoadUint64x4Array((*[4]uint64)(buf[i+8 : i+12])))
		a3 = a3.Add(archsimd.LoadUint64x4Array((*[4]uint64)(buf[i+12 : i+16])))
	}
	var lanes [4]uint64
	a0.Add(a1).Add(a2.Add(a3)).StoreArray(&lanes)
	sum := lanes[0] + lanes[1] + lanes[2] + lanes[3]
	for ; i < len(buf); i++ {
		sum += buf[i]
	}
	return sum
}

// BenchmarkReadBandwidth is the memory read ceiling the GEMM roofline is
// judged against: threads goroutines each stream an equal stripe of a buffer
// far larger than the last-level cache. Thread counts above GOMAXPROCS are
// skipped; pin with taskset to choose which cores take part.
func BenchmarkReadBandwidth(b *testing.B) {
	buf := make([]uint64, readBandwidthBytes/8)
	for i := range buf {
		buf[i] = uint64(i)
	}
	procs := runtime.GOMAXPROCS(0)
	for _, threads := range []int{1, 2, 4, 8, 16} {
		if threads > procs {
			continue
		}
		b.Run(fmt.Sprintf("threads=%d", threads), func(b *testing.B) {
			sums := make([]uint64, threads)
			b.SetBytes(readBandwidthBytes)
			for b.Loop() {
				var wg sync.WaitGroup
				for t := range threads {
					lo := t * len(buf) / threads
					hi := (t + 1) * len(buf) / threads
					wg.Go(func() { sums[t] = readStripe(buf[lo:hi]) })
				}
				wg.Wait()
			}
			var total uint64
			for _, s := range sums {
				total += s
			}
			if total == 0 {
				b.Fatal("read checksum is zero")
			}
		})
	}
}
