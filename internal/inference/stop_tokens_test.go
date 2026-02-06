package inference

import (
	"reflect"
	"testing"

	"github.com/samcharles93/mantle/internal/tokenizer"
)

type fakeTokenizer struct {
	tokens map[int]string
}

func (f fakeTokenizer) Encode(text string) ([]int, error) {
	return nil, nil
}

func (f fakeTokenizer) Decode(ids []int) (string, error) {
	return "", nil
}

func (f fakeTokenizer) TokenString(id int) string {
	return f.tokens[id]
}

func TestBuildStopTokensLegacyID2(t *testing.T) {
	cases := []struct {
		name   string
		cfg    tokenizer.TokenizerConfig
		tokens map[int]string
		want   []int
	}{
		{
			name: "adds-legacy-2-when-eot",
			cfg:  tokenizer.TokenizerConfig{EOSTokenID: 1},
			tokens: map[int]string{
				2: "<|endoftext|>",
			},
			want: []int{1, 2},
		},
		{
			name: "no-legacy-2-when-not-eot",
			cfg:  tokenizer.TokenizerConfig{EOSTokenID: 1},
			tokens: map[int]string{
				2: "<|im_end|>",
			},
			want: []int{1},
		},
		{
			name: "legacy-2-only-when-eos-absent",
			cfg:  tokenizer.TokenizerConfig{EOSTokenID: -1},
			tokens: map[int]string{
				2: "</s>",
			},
			want: []int{2},
		},
		{
			name: "no-legacy-2-when-eos-absent-and-not-eot",
			cfg:  tokenizer.TokenizerConfig{EOSTokenID: -1},
			tokens: map[int]string{
				2: "<|im_end|>",
			},
			want: []int{},
		},
		{
			name: "no-duplicate-legacy-2-when-eos-is-2",
			cfg:  tokenizer.TokenizerConfig{EOSTokenID: 2},
			tokens: map[int]string{
				2: "<|end_of_text|>",
			},
			want: []int{2},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := BuildStopTokens(fakeTokenizer{tokens: tc.tokens}, tc.cfg)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestBuildStopTokensAddsImEnd(t *testing.T) {
	cfg := tokenizer.TokenizerConfig{
		EOSTokenID: 1,
		Tokens:     []string{"<pad>", "<eos>", "<|im_end|>"},
	}
	got := BuildStopTokens(fakeTokenizer{tokens: map[int]string{}}, cfg)
	want := []int{1, 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

type fakeDecoderTokenizer struct {
	decoder []string
}

func (f fakeDecoderTokenizer) Encode(text string) ([]int, error) { return nil, nil }
func (f fakeDecoderTokenizer) Decode(ids []int) (string, error)  { return "", nil }
func (f fakeDecoderTokenizer) Decoder() []string                 { return f.decoder }

func TestBuildStopTokensAddsImEndFromDecoder(t *testing.T) {
	cfg := tokenizer.TokenizerConfig{
		EOSTokenID: 5,
	}
	got := BuildStopTokens(fakeDecoderTokenizer{decoder: []string{"a", "b", "<|im_end|>"}}, cfg)
	want := []int{5, 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}
