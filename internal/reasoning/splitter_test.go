package reasoning

import "testing"

func TestSplitRaw(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		in            string
		wantContent   string
		wantReasoning string
	}{
		{
			name:          "no thinking",
			in:            "Hello world",
			wantContent:   "Hello world",
			wantReasoning: "",
		},
		{
			name:          "closed thinking block",
			in:            "<think>internal</think>Hello",
			wantContent:   "Hello",
			wantReasoning: "internal",
		},
		{
			name:          "unclosed thinking block",
			in:            "<think>internal only",
			wantContent:   "",
			wantReasoning: "internal only",
		},
		{
			name:          "interleaved text",
			in:            "A<think>r1</think>B<think>r2</think>C",
			wantContent:   "ABC",
			wantReasoning: "r1r2",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := SplitRaw(tc.in)
			if got.Content != tc.wantContent {
				t.Fatalf("content got %q want %q", got.Content, tc.wantContent)
			}
			if got.Reasoning != tc.wantReasoning {
				t.Fatalf("reasoning got %q want %q", got.Reasoning, tc.wantReasoning)
			}
		})
	}
}

func TestSplitterPush(t *testing.T) {
	t.Parallel()

	var s Splitter

	c, r := s.Push("<think>abc")
	if c != "" || r != "abc" {
		t.Fatalf("first delta got content=%q reasoning=%q", c, r)
	}

	c, r = s.Push("</think>Hello")
	if c != "Hello" || r != "" {
		t.Fatalf("second delta got content=%q reasoning=%q", c, r)
	}
}
