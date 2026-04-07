package paths

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const (
	// EnvMantlePackOutDir controls the default output directory used by pack.
	EnvMantlePackOutDir = "MANTLE_PACK_OUT_DIR"
	// EnvMantleModelsDir points to the directory that stores .mcf model files.
	EnvMantleModelsDir = "MANTLE_MODELS_DIR"
)

// stdinIsTTY is a small seam for tests.
var stdinIsTTY = isTTY

func ResolvePackOut(inDir, outFlag string) (string, bool, error) {
	outFlag = strings.TrimSpace(outFlag)
	if outFlag != "" {
		outPath := filepath.Clean(outFlag)
		if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
			return "", false, err
		}
		return outPath, false, nil
	}

	base := filepath.Base(filepath.Clean(inDir))
	if base == "" || base == "." || base == string(filepath.Separator) {
		return "", true, fmt.Errorf("invalid input directory: %q", inDir)
	}

	outDir := strings.TrimSpace(os.Getenv(EnvMantlePackOutDir))
	if outDir == "" {
		outDir = filepath.Join(".", "out")
	}

	outPath := filepath.Join(outDir, base+".mcf")
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return "", true, err
	}
	return outPath, true, nil
}

// ResolveRunModelPath selects the model path from flags, env, or interactive choice.
func ResolveRunModelPath(modelFlag string, modelsPath string, stdin io.Reader, stderr io.Writer) (string, error) {
	modelFlag = strings.TrimSpace(modelFlag)
	modelsDir := strings.TrimSpace(modelsPath)
	if modelsDir == "" {
		modelsDir = strings.TrimSpace(os.Getenv(EnvMantleModelsDir))
	}

	if modelFlag != "" {
		// If it's a direct path that exists, use it
		clean := filepath.Clean(modelFlag)
		if _, err := os.Stat(clean); err == nil {
			return clean, nil
		}

		// If it's just a filename and we have a modelsDir, try looking there
		if modelsDir != "" && !filepath.IsAbs(modelFlag) && !strings.Contains(modelFlag, string(filepath.Separator)) {
			joined := filepath.Join(modelsDir, modelFlag)
			if _, err := os.Stat(joined); err == nil {
				return joined, nil
			}
			// Also try appending .mcf if not present
			if !strings.HasSuffix(strings.ToLower(modelFlag), ".mcf") {
				joinedMCF := filepath.Join(modelsDir, modelFlag+".mcf")
				if _, err := os.Stat(joinedMCF); err == nil {
					return joinedMCF, nil
				}
			}
		}

		// Otherwise return as-is (Action will fail on Stat)
		return filepath.Clean(modelFlag), nil
	}

	if modelsDir == "" {
		return "", fmt.Errorf("--model or --models-path is required unless %s is set", EnvMantleModelsDir)
	}

	models, err := DiscoverMCFModels(modelsDir)
	if err != nil {
		return "", err
	}
	switch len(models) {
	case 0:
		return "", fmt.Errorf("no .mcf models found in %s", modelsDir)
	case 1:
		_, _ = fmt.Fprintf(stderr, "run: using model %s\n", models[0])
		return models[0], nil
	default:
		if !stdinIsTTY() {
			return "", fmt.Errorf(
				"multiple models found in %s but stdin is not interactive; set --model",
				modelsDir,
			)
		}
		return selectModelInteractively(modelsDir, models, stdin, stderr)
	}
}

// DiscoverMCFModels returns sorted .mcf model paths under dir.
func DiscoverMCFModels(dir string) ([]string, error) {
	if strings.TrimSpace(dir) == "" {
		return nil, errors.New("models directory is empty")
	}
	st, err := os.Stat(dir)
	if err != nil {
		return nil, err
	}
	if !st.IsDir() {
		return nil, fmt.Errorf("models path is not a directory: %s", dir)
	}

	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	models := make([]string, 0, len(ents))
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(strings.ToLower(name), ".mcf") {
			continue
		}
		models = append(models, filepath.Join(dir, name))
	}
	sort.Strings(models)
	return models, nil
}

func selectModelInteractively(modelsDir string, models []string, stdin io.Reader, stderr io.Writer) (string, error) {
	if len(models) == 0 {
		return "", fmt.Errorf("no models available in %s", modelsDir)
	}

	_, _ = fmt.Fprintf(stderr, "run: select a model from %s\n", modelsDir)
	for i, m := range models {
		_, _ = fmt.Fprintf(stderr, "%d. %s\n", i+1, modelDisplayName(modelsDir, m))
	}

	reader := bufio.NewReader(stdin)
	for {
		_, _ = fmt.Fprintf(stderr, "run: enter selection [1-%d]: ", len(models))
		line, err := reader.ReadString('\n')
		if err != nil && !errors.Is(err, io.EOF) {
			return "", err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			if errors.Is(err, io.EOF) {
				return "", errors.New("no selection provided on stdin; set --model")
			}
			continue
		}

		idx, convErr := strconv.Atoi(line)
		if convErr != nil || idx < 1 || idx > len(models) {
			_, _ = fmt.Fprintf(stderr, "run: invalid selection %q\n", line)
			if errors.Is(err, io.EOF) {
				return "", errors.New("invalid selection provided on stdin; set --model")
			}
			continue
		}
		return models[idx-1], nil
	}
}

func modelDisplayName(modelsDir, modelPath string) string {
	rel, err := filepath.Rel(modelsDir, modelPath)
	if err != nil || rel == "." {
		return filepath.Base(modelPath)
	}
	return rel
}

func isTTY() bool {
	st, err := os.Stdin.Stat()
	if err != nil {
		return false
	}
	return (st.Mode() & os.ModeCharDevice) != 0
}
