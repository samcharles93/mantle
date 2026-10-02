package main

import (
	"reflect"
	"testing"
)

func TestParseTapLayers(t *testing.T) {
	cases := []struct {
		name    string
		in      string
		want    []int
		wantErr bool
	}{
		{"empty disables", "", nil, false},
		{"blank disables", "   ", nil, false},
		{"single", "1", []int{1}, false},
		{"mini cpm5 dspark", "1,10,20,30,39", []int{1, 10, 20, 30, 39}, false},
		{"spaces", " 1 , 10 , 20 ", []int{1, 10, 20}, false},
		{"embedding", "-1,0", []int{-1, 0}, false},
		{"non integer", "1,x", nil, true},
		{"trailing comma", "1,", nil, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseTapLayers(tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("parseTapLayers(%q) = %v, want error", tc.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseTapLayers(%q): %v", tc.in, err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("parseTapLayers(%q) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}
