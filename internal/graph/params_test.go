package graph

import (
	"reflect"
	"testing"
)

func TestParamsTypeSwitch(t *testing.T) {
	params := []NodeParams{
		AttentionParams{}, FFNParams{}, MoEParams{},
		MambaParams{}, DeltaNetParams{},
		EmbedParams{}, OutputParams{},
	}
	for _, p := range params {
		switch p.(type) {
		case AttentionParams, FFNParams, MoEParams, MambaParams, DeltaNetParams, EmbedParams, OutputParams:
		default:
			t.Fatalf("unexpected type for NodeParams: %T", p)
		}
	}
}

func TestParamsDistinct(t *testing.T) {
	params := []NodeParams{
		AttentionParams{}, FFNParams{}, MoEParams{},
		MambaParams{}, DeltaNetParams{},
		EmbedParams{}, OutputParams{},
	}
	seen := map[reflect.Type]bool{}
	for _, p := range params {
		typ := reflect.TypeOf(p)
		if seen[typ] {
			t.Fatalf("duplicate params type detected: %v", typ)
		}
		seen[typ] = true
	}
	if len(seen) != len(params) {
		t.Fatalf("expected %d distinct param types, got %d", len(params), len(seen))
	}
}
