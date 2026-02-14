package inference

import "testing"

func TestSanitizeAssistantForContext(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "removes closed think block",
			in:   "<think>internal</think>\nHello there",
			want: "Hello there",
		},
		{
			name: "removes unclosed think block tail",
			in:   "<think>internal only",
			want: "",
		},
		{
			name: "removes sentinel tokens",
			in:   "Answer<|im_end|><|endoftext|>",
			want: "Answer",
		},
		{
			name: "keeps plain text",
			in:   "All good.",
			want: "All good.",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := SanitizeAssistantForContext(tc.in)
			if got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}
