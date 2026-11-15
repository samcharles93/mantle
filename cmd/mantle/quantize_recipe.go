package main

import (
	"fmt"
	"strconv"
	"strings"
)

// Quantization recipes choose a method per tensor from the --method base.
//
// uniform gives every quantizable tensor the base method.
//
// mixed follows llama.cpp's _M rule: the output head (or the token embedding
// when the head is tied and absent), and the attention value and FFN down
// projections in the first eighth of layers, the last eighth, and every third
// layer between, take the upgraded method (k6 for k2-k4, q8 for q4).
const (
	recipeUniform = "uniform"
	recipeMixed   = "mixed"
)

func parseRecipe(s string) (string, error) {
	switch r := strings.ToLower(strings.TrimSpace(s)); r {
	case "", recipeUniform:
		return recipeUniform, nil
	case recipeMixed:
		return recipeMixed, nil
	default:
		return "", fmt.Errorf("unsupported recipe %q (use uniform|mixed)", s)
	}
}

// upgradedMethod returns the higher-precision method the mixed recipe uses
// for sensitive tensors, or false when the base is already 6 bits or more.
func upgradedMethod(base quantMethod) (quantMethod, bool) {
	var name string
	switch {
	case base.Family == "k" && base.Bits < 6:
		name = "k6"
	case base.Family == "q" && base.Bits < 8:
		name = "q8"
	default:
		return quantMethod{}, false
	}
	m, err := parseQuantMethod(name)
	if err != nil {
		return quantMethod{}, false
	}
	return m, true
}

// planMixedRecipe returns the tensors the mixed recipe upgrades, mapped to the
// upgraded method. Tensors not in the map keep the base method.
func planMixedRecipe(names []string, base quantMethod) map[string]quantMethod {
	up, ok := upgradedMethod(base)
	if !ok {
		return nil
	}
	plan := make(map[string]quantMethod)

	layerCount := make(map[string]int)
	for _, name := range names {
		if prefix, idx, ok := layerOf(name); ok {
			layerCount[prefix] = max(layerCount[prefix], idx+1)
		}
	}
	hasHead := false
	for _, name := range names {
		if name == "lm_head.weight" || strings.HasSuffix(name, ".lm_head.weight") {
			plan[name] = up
			hasHead = true
		}
	}
	for _, name := range names {
		if !hasHead && strings.HasSuffix(name, ".embed_tokens.weight") {
			plan[name] = up
			continue
		}
		if !isValueOrDownProjection(name) {
			continue
		}
		prefix, idx, ok := layerOf(name)
		if ok && useMoreBits(idx, layerCount[prefix]) {
			plan[name] = up
		}
	}
	return plan
}

func isValueOrDownProjection(name string) bool {
	return strings.HasSuffix(name, ".self_attn.v_proj.weight") ||
		strings.HasSuffix(name, ".down_proj.weight") ||
		strings.HasSuffix(name, ".feed_forward.w2.weight")
}

// layerOf splits "<prefix>.layers.<i>.<rest>" into prefix and i, so separate
// stacks (text decoder, vision tower) count their layers independently.
func layerOf(name string) (string, int, bool) {
	i := strings.LastIndex(name, ".layers.")
	start := i + len(".layers.")
	if i < 0 {
		if !strings.HasPrefix(name, "layers.") {
			return "", 0, false
		}
		i, start = 0, len("layers.")
	}
	end := strings.IndexByte(name[start:], '.')
	if end <= 0 {
		return "", 0, false
	}
	idx, err := strconv.Atoi(name[start : start+end])
	if err != nil || idx < 0 {
		return "", 0, false
	}
	return name[:i], idx, true
}

// useMoreBits is llama.cpp's layer selection for its _M recipes.
func useMoreBits(i, n int) bool {
	return i < n/8 || i >= 7*n/8 || (i-n/8)%3 == 2
}
