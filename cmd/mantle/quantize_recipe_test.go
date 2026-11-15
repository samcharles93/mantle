package main

import (
	"fmt"
	"maps"
	"slices"
	"testing"
)

func layerNames(prefix string, n int, suffixes ...string) []string {
	var out []string
	for i := range n {
		for _, s := range suffixes {
			out = append(out, fmt.Sprintf("%s.layers.%d.%s", prefix, i, s))
		}
	}
	return out
}

func TestPlanMixedRecipeSeparateHead(t *testing.T) {
	names := append([]string{"lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"},
		layerNames("model", 16, "self_attn.q_proj.weight", "self_attn.v_proj.weight", "mlp.down_proj.weight", "mlp.up_proj.weight")...)
	base, _ := parseQuantMethod("k4")
	plan := planMixedRecipe(names, base)

	// n=16: layers < 2, >= 14, and (i-2)%3 == 2 (4, 7, 10, 13).
	want := []string{"lm_head.weight"}
	for _, i := range []int{0, 1, 4, 7, 10, 13, 14, 15} {
		want = append(want,
			fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", i),
			fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", i))
	}
	slices.Sort(want)
	got := slices.Sorted(maps.Keys(plan))
	if !slices.Equal(got, want) {
		t.Fatalf("upgraded tensors:\n got %v\nwant %v", got, want)
	}
	for name, m := range plan {
		if m.Name != "k6" {
			t.Errorf("%s upgraded to %s, want k6", name, m.Name)
		}
	}
}

func TestPlanMixedRecipeTiedHeadAndStacks(t *testing.T) {
	names := []string{
		"model.language_model.embed_tokens.weight",
		"model.language_model.embed_tokens_per_layer.weight",
	}
	names = append(names, layerNames("model.language_model", 8, "self_attn.v_proj.weight", "feed_forward.w2.weight")...)
	names = append(names, layerNames("model.vision_tower.encoder", 3, "self_attn.v_proj.weight")...)
	base, _ := parseQuantMethod("q4")
	plan := planMixedRecipe(names, base)

	// Text stack n=8: layers 0, 7 and (i-1)%3 == 2 (3, 6). Vision stack n=3:
	// 3/8 == 0, so only (i-0)%3 == 2 (layer 2) and i >= 21/8 == 2.
	want := []string{"model.language_model.embed_tokens.weight"}
	for _, i := range []int{0, 3, 6, 7} {
		want = append(want,
			fmt.Sprintf("model.language_model.layers.%d.self_attn.v_proj.weight", i),
			fmt.Sprintf("model.language_model.layers.%d.feed_forward.w2.weight", i))
	}
	want = append(want, "model.vision_tower.encoder.layers.2.self_attn.v_proj.weight")
	slices.Sort(want)
	got := slices.Sorted(maps.Keys(plan))
	if !slices.Equal(got, want) {
		t.Fatalf("upgraded tensors:\n got %v\nwant %v", got, want)
	}
	for name, m := range plan {
		if m.Name != "q8" {
			t.Errorf("%s upgraded to %s, want q8", name, m.Name)
		}
	}
}

func TestPlanMixedRecipeNoUpgradeAtSixBits(t *testing.T) {
	base, _ := parseQuantMethod("k6")
	if plan := planMixedRecipe([]string{"lm_head.weight"}, base); len(plan) != 0 {
		t.Fatalf("k6 base upgraded %v", plan)
	}
}

func TestParseRecipe(t *testing.T) {
	for in, want := range map[string]string{"": recipeUniform, "uniform": recipeUniform, "Mixed": recipeMixed} {
		got, err := parseRecipe(in)
		if err != nil || got != want {
			t.Errorf("parseRecipe(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	if _, err := parseRecipe("large"); err == nil {
		t.Error("unknown recipe accepted")
	}
}
