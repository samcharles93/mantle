package inference

import (
	"reflect"
	"testing"

	"github.com/samcharles93/mantle/internal/tokenizer"
)

func TestBuildStopTokensBasic(t *testing.T) {
	cfg := tokenizer.TokenizerConfig{EOSTokenID: 1}
	got := BuildStopTokens(cfg, nil)
	want := []int{1}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestBuildStopTokensWithExtraIDs(t *testing.T) {
	// Simulates a model like Gemma3 where generation_config has [1, 106].
	cfg := tokenizer.TokenizerConfig{EOSTokenID: 1}
	got := BuildStopTokens(cfg, []int{1, 106})
	want := []int{1, 106}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestBuildStopTokensDeduplicates(t *testing.T) {
	cfg := tokenizer.TokenizerConfig{EOSTokenID: 1}
	got := BuildStopTokens(cfg, []int{1, 2, 1, 2})
	want := []int{1, 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestBuildStopTokensNoEOS(t *testing.T) {
	// EOSTokenID=-1 means no EOS; only extra IDs are used.
	cfg := tokenizer.TokenizerConfig{EOSTokenID: -1}
	got := BuildStopTokens(cfg, []int{106})
	want := []int{106}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestBuildStopTokensEmpty(t *testing.T) {
	cfg := tokenizer.TokenizerConfig{EOSTokenID: -1}
	got := BuildStopTokens(cfg, nil)
	want := []int{}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}
