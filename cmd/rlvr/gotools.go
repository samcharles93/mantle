package main

import (
	"fmt"
	"math"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

type BenchResult struct {
	Name        string
	NsPerOp     float64
	BytesPerOp  float64
	AllocsPerOp float64
}

type Baseline struct {
	BenchNs    map[string]float64    `json:"bench_ns,omitempty"`
	BenchStats map[string]BenchStats `json:"bench_stats,omitempty"`
}

type BenchStats struct {
	N      int     `json:"n"`
	Mean   float64 `json:"mean"`
	Median float64 `json:"median"`
	Stddev float64 `json:"stddev"`
}

func normalizeBaseline(b *Baseline) {
	if b == nil {
		return
	}
	if b.BenchStats == nil && len(b.BenchNs) > 0 {
		b.BenchStats = make(map[string]BenchStats, len(b.BenchNs))
		for name, ns := range b.BenchNs {
			b.BenchStats[name] = BenchStats{N: 1, Mean: ns, Median: ns, Stddev: 0}
		}
	}
	if b.BenchNs == nil && len(b.BenchStats) > 0 {
		b.BenchNs = make(map[string]float64, len(b.BenchStats))
		for name, st := range b.BenchStats {
			b.BenchNs[name] = st.Median
		}
	}
}

func runGoFmt(dir string, files []string) error {
	var args []string
	args = append(args, "-w")
	hasGo := false
	for _, f := range files {
		if strings.HasSuffix(f, ".go") {
			args = append(args, f)
			hasGo = true
		}
	}
	if !hasGo {
		return nil
	}
	return runCmd(dir, "gofmt", args...)
}

func runGoTestAll(dir string) error {
	return runCmd(dir, "go", "test", "./...")
}

func runBench(dir, benchSpec string, runs int) (map[string]BenchStats, error) {
	parts := strings.SplitN(benchSpec, ":", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid bench spec (expected ./path:BenchName): %s", benchSpec)
	}
	pkg, benchName := parts[0], parts[1]

	samples := make(map[string][]float64)

	for range runs {
		cmd := exec.Command("go", "test", pkg, "-run", "^$", "-bench", fmt.Sprintf("^%s$", benchName), "-benchmem", "-count", "1")
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err != nil {
			return nil, fmt.Errorf("bench run failed: %v\n%s", err, string(out))
		}

		res := parseGoBench(string(out))
		if len(res) == 0 {
			return nil, fmt.Errorf("no benchmarks found in output")
		}
		for _, r := range res {
			samples[r.Name] = append(samples[r.Name], r.NsPerOp)
		}
		time.Sleep(50 * time.Millisecond)
	}

	stats := make(map[string]BenchStats)
	for k, v := range samples {
		mean, stddev := calculateMeanStddev(v)
		stats[k] = BenchStats{
			N:      len(v),
			Mean:   mean,
			Median: calculateMedian(v),
			Stddev: stddev,
		}
	}
	return stats, nil
}

var benchRe = regexp.MustCompile(`^(Benchmark\S+)\s+(\d+)\s+([0-9.]+)\s+ns/op(?:\s+([0-9.]+)\s+B/op)?(?:\s+([0-9.]+)\s+allocs/op)?`)

func parseGoBench(output string) []BenchResult {
	var results []BenchResult
	lines := strings.SplitSeq(output, "\n")
	for line := range lines {
		line = strings.TrimSpace(line)
		m := benchRe.FindStringSubmatch(line)
		if m == nil {
			continue
		}

		ns, _ := strconv.ParseFloat(m[2], 64)
		bOp := 0.0
		if len(m) > 3 && m[3] != "" {
			bOp, _ = strconv.ParseFloat(m[3], 64)
		}
		allocs := 0.0
		if len(m) > 4 && m[4] != "" {
			allocs, _ = strconv.ParseFloat(m[4], 64)
		}

		results = append(results, BenchResult{
			Name:        m[1],
			NsPerOp:     ns,
			BytesPerOp:  bOp,
			AllocsPerOp: allocs,
		})
	}
	return results
}

func calculateMedian(vals []float64) float64 {
	sort.Float64s(vals)
	n := len(vals)
	if n == 0 {
		return 0
	}
	if n%2 == 1 {
		return vals[n/2]
	}
	return 0.5 * (vals[n/2-1] + vals[n/2])
}

func calculateMeanStddev(vals []float64) (float64, float64) {
	if len(vals) == 0 {
		return 0, 0
	}
	var sum float64
	for _, v := range vals {
		sum += v
	}
	mean := sum / float64(len(vals))
	if len(vals) < 2 {
		return mean, 0
	}
	var ss float64
	for _, v := range vals {
		d := v - mean
		ss += d * d
	}
	return mean, math.Sqrt(ss / float64(len(vals)-1))
}

func isImproved(cand, base map[string]BenchStats, minImprove, minSigma float64) (bool, string) {
	var msgs []string
	ok := true

	for name, baseStats := range base {
		candStats, exists := cand[name]
		if !exists {
			msgs = append(msgs, fmt.Sprintf("%s: missing in candidate", name))
			ok = false
			continue
		}

		baseMedian := baseStats.Median
		candMedian := candStats.Median
		delta := (baseMedian - candMedian) / baseMedian
		msgs = append(msgs, fmt.Sprintf("%s: %.1f -> %.1f ns/op (%.2f%%) [base %.1f±%.1f n=%d, cand %.1f±%.1f n=%d]",
			name,
			baseMedian,
			candMedian,
			delta*100,
			baseStats.Mean,
			baseStats.Stddev,
			baseStats.N,
			candStats.Mean,
			candStats.Stddev,
			candStats.N,
		))

		if delta < minImprove {
			ok = false
		}

		if minSigma > 0 && baseStats.N > 1 && candStats.N > 1 {
			stderr := math.Sqrt(
				(baseStats.Stddev*baseStats.Stddev)/float64(baseStats.N) +
					(candStats.Stddev*candStats.Stddev)/float64(candStats.N),
			)
			if stderr > 0 && (baseStats.Mean-candStats.Mean) < (minSigma*stderr) {
				ok = false
			}
		}
	}
	return ok, strings.Join(msgs, "\n")
}
