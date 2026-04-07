package inference

import (
	"fmt"
	"os"
	"sort"
)

type generationDebugLogit struct {
	id  int
	val float32
}

func logGenerationSampleDebug(next int, stopTokens []int, logitsVec []float32) {
	if os.Getenv("MANTLE_DEBUG_GEN") == "" {
		return
	}
	top := topGenerationDebugLogits(logitsVec, 5)
	fmt.Fprintf(os.Stderr, "  DEBUG gen: sampled=%d stop_tokens=%v logits_len=%d top5=%v\n", next, stopTokens, len(logitsVec), top)
}

func topGenerationDebugLogits(logitsVec []float32, limit int) []generationDebugLogit {
	if limit <= 0 || len(logitsVec) == 0 {
		return nil
	}
	if limit > len(logitsVec) {
		limit = len(logitsVec)
	}

	top := make([]generationDebugLogit, len(logitsVec))
	for i, v := range logitsVec {
		top[i] = generationDebugLogit{id: i, val: v}
	}
	sort.Slice(top, func(i, j int) bool {
		if top[i].val == top[j].val {
			return top[i].id < top[j].id
		}
		return top[i].val > top[j].val
	})
	return top[:limit]
}
