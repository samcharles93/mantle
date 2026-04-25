package simd

import (
	"fmt"
	"os"
	"testing"

	"github.com/samcharles93/mantle/internal/mcf"
	"github.com/samcharles93/mantle/internal/mcfstore"
)

func TestE2BBufferSizes(t *testing.T) {
	mcfFilePath := os.Getenv("E2B_MCF_PATH")
	if mcfFilePath == "" {
		mcfFilePath = os.ExpandEnv("$MANTLE_MODELS_DIR/mcf/gemma-4-E2B-it.k4.mcf")
	}
	f, err := os.Open(mcfFilePath)
	if err != nil {
		t.Skipf("model file not found: %v", err)
	}
	defer f.Close()

	mcfFile := mcf.NewFile(f)
	if err := mcfFile.ReadHeader(); err != nil {
		t.Fatalf("read header: %v", err)
	}

	store, err := mcfstore.NewMCFStore(mcfFile, nil)
	if err != nil {
		t.Fatalf("mcfstore: %v", err)
	}
	defer store.Close()

	m, err := Load(&LoadOptions{ModelFile: store})
	if err != nil {
		t.Fatalf("load model: %v", err)
	}

	fmt.Printf("model: %s  FFNLength: %d  layers: %d\n", m.Config.Config.Arch, m.Config.Config.FFNLength, len(m.Layers))
	fmt.Printf("Scratch.FfnGate: %d  Scratch.FfnUp: %d\n", len(m.Scratch.FfnGate), len(m.Scratch.FfnUp))

	for i, l := range m.Layers {
		if l.Gemma4MoE != nil {
			fmt.Printf("layer %d: Gemma4MoE.Intermediate=%d  Experts=%d\n", i, l.Gemma4MoE.Intermediate, len(l.Gemma4MoE.Experts))
			for j, e := range l.Gemma4MoE.Experts {
				if e.GateUp != nil {
					fmt.Printf("  expert %d: GateUp.R=%d\n", j, e.GateUp.R)
				}
			}
		}
		if l.FfnGate != nil {
			fmt.Printf("layer %d: FfnGate.R=%d\n", i, l.FfnGate.R)
		}
	}
}
