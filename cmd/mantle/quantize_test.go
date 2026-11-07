package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadQuantAssignmentsAndMethodFor(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "assign.json")
	raw := `{
	  "default": "k4",
	  "budget_bytes": 123456,
	  "tensors": {
	    "model.layers.0.self_attn.q_proj.weight": "q8",
	    "model.layers.1.*": "k2",
	    "model.layers.*": "k3",
	    "model.embed_tokens.weight": "k6"
	  }
	}`
	if err := os.WriteFile(path, []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}
	a, err := loadQuantAssignments(path)
	if err != nil {
		t.Fatalf("loadQuantAssignments: %v", err)
	}
	if a.Budget != 123456 {
		t.Fatalf("budget = %d, want 123456", a.Budget)
	}
	cases := []struct {
		name string
		want string
	}{
		{"model.layers.0.self_attn.q_proj.weight", "q8"}, // exact beats glob
		{"model.layers.0.self_attn.k_proj.weight", "k3"}, // layer glob beats nothing... layers.* 
		{"model.layers.1.mlp.up_proj.weight", "k2"},      // longer glob (layers.1.*) beats layers.*
		{"model.layers.9.mlp.up_proj.weight", "k3"},      // layers.* 
		{"model.embed_tokens.weight", "k6"},              // exact
		{"model.norm.weight", "k4"},                      // default
	}
	for _, tc := range cases {
		if got := a.methodFor(tc.name).Name; got != tc.want {
			t.Errorf("methodFor(%q) = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestLoadQuantAssignmentsRejectsBadMethod(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(path, []byte(`{"default":"k4","tensors":{"a":"bogus"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadQuantAssignments(path); err == nil {
		t.Fatal("expected error for unknown method")
	}
}
