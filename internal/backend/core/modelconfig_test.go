package core

import "testing"

func TestResolveCacheType(t *testing.T) {
	cases := []struct {
		raw         string
		defaultType string
		want        string
		wantErr     bool
	}{
		{"", CacheTypeF32, CacheTypeF32, false},
		{"  ", CacheTypeF16, CacheTypeF16, false},
		{"f32", CacheTypeF16, CacheTypeF32, false},
		{"F16", CacheTypeF32, CacheTypeF16, false},
		{"Q8_0", CacheTypeF32, CacheTypeQ8_0, false},
		{"q4_0", CacheTypeF16, "", true},
		{"bf16", CacheTypeF16, "", true},
		{"bogus", CacheTypeF16, "", true},
	}
	for _, tc := range cases {
		got, err := ResolveCacheType("cache-type-k", tc.raw, tc.defaultType)
		if tc.wantErr {
			if err == nil {
				t.Fatalf("ResolveCacheType(%q) = %q, want error", tc.raw, got)
			}
			continue
		}
		if err != nil {
			t.Fatalf("ResolveCacheType(%q): %v", tc.raw, err)
		}
		if got != tc.want {
			t.Fatalf("ResolveCacheType(%q) = %q, want %q", tc.raw, got, tc.want)
		}
	}
}

func TestResolveCUDACacheType(t *testing.T) {
	cases := []struct {
		raw     string
		want    string
		wantErr bool
	}{
		{"", CacheTypeF16, false},
		{"  ", CacheTypeF16, false},
		{"f16", CacheTypeF16, false},
		{"F16", CacheTypeF16, false},
		{"q8_0", CacheTypeQ8_0, false},
		{"Q8_0", CacheTypeQ8_0, false},
		{"f32", "", true},
		{"q4_0", "", true},
		{"bf16", "", true},
		{"bogus", "", true},
	}
	for _, tc := range cases {
		got, err := ResolveCUDACacheType("cache-type-k", tc.raw)
		if tc.wantErr {
			if err == nil {
				t.Fatalf("ResolveCUDACacheType(%q) = %q, want error", tc.raw, got)
			}
			continue
		}
		if err != nil {
			t.Fatalf("ResolveCUDACacheType(%q): %v", tc.raw, err)
		}
		if got != tc.want {
			t.Fatalf("ResolveCUDACacheType(%q) = %q, want %q", tc.raw, got, tc.want)
		}
	}
}
