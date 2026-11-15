package main

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/urfave/cli/v3"
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

// TestTextOnlyFlagOnModelCommands checks the --text-only flag reaches both
// commands that load a model for inference, parses from the command line and
// picks up MANTLE_TEXT_ONLY.
func TestTextOnlyFlagOnModelCommands(t *testing.T) {
	for _, cmd := range []*cli.Command{runCmd(), serveCmd()} {
		var flag *cli.BoolFlag
		for _, f := range cmd.Flags {
			if bf, ok := f.(*cli.BoolFlag); ok && bf.Name == "text-only" {
				flag = bf
			}
		}
		if flag == nil {
			t.Fatalf("%s: missing --text-only flag", cmd.Name)
		}
		if flag.Destination != any(&textOnly) {
			t.Fatalf("%s: --text-only does not target the shared textOnly field", cmd.Name)
		}
		if !strings.Contains(flag.Usage, "image and audio requests are rejected") {
			t.Fatalf("%s: usage %q does not say media requests are rejected", cmd.Name, flag.Usage)
		}
	}

	parse := func(t *testing.T, args []string) bool {
		t.Helper()
		textOnly = false
		t.Cleanup(func() { textOnly = false })
		cmd := &cli.Command{
			Name:   "parse",
			Flags:  []cli.Flag{textOnlyFlag()},
			Action: func(context.Context, *cli.Command) error { return nil },
		}
		if err := cmd.Run(context.Background(), args); err != nil {
			t.Fatalf("parse %v: %v", args, err)
		}
		return textOnly
	}

	if !parse(t, []string{"parse", "--text-only"}) {
		t.Fatalf("--text-only did not set the load option")
	}
	if parse(t, []string{"parse"}) {
		t.Fatalf("text-only defaulted to on")
	}
	t.Setenv("MANTLE_TEXT_ONLY", "1")
	if !parse(t, []string{"parse"}) {
		t.Fatalf("MANTLE_TEXT_ONLY=1 did not set the load option")
	}
}
