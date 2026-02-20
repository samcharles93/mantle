package ux

import "testing"

func TestNormalizeStreamMode(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  StreamMode
		valid bool
	}{
		{name: "instant", input: "instant", want: StreamInstant, valid: true},
		{name: "smooth mixed case", input: "  SmOoTh  ", want: StreamSmooth, valid: true},
		{name: "typewriter", input: "typewriter", want: StreamTypewriter, valid: true},
		{name: "quiet", input: "quiet", want: StreamQuiet, valid: true},
		{name: "invalid", input: "fast", want: StreamSmooth, valid: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, valid := NormalizeStreamMode(tt.input)
			if got != tt.want {
				t.Fatalf("mode: got %q want %q", got, tt.want)
			}
			if valid != tt.valid {
				t.Fatalf("valid: got %v want %v", valid, tt.valid)
			}
		})
	}
}

func TestFormatModelSize(t *testing.T) {
	tests := []struct {
		name  string
		bytes int64
		want  string
	}{
		{name: "bytes", bytes: 999, want: "999 B"},
		{name: "kb", bytes: 1024, want: "1.0 KB"},
		{name: "mb", bytes: 1024 * 1024, want: "1.0 MB"},
		{name: "gb", bytes: 1024 * 1024 * 1024, want: "1.0 GB"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FormatModelSize(tt.bytes)
			if got != tt.want {
				t.Fatalf("got %q want %q", got, tt.want)
			}
		})
	}
}
