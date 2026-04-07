package hf

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

type ModelInfo struct {
	ID       string `json:"id"`
	Siblings []struct {
		Rfilename string `json:"rfilename"`
	} `json:"siblings"`
}

func GetModelInfo(repo string) (*ModelInfo, error) {
	url := fmt.Sprintf("https://huggingface.co/api/models/%s", repo)
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("hf: api error %d: %s", resp.StatusCode, url)
	}

	var info ModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, err
	}
	return &info, nil
}

func DownloadFile(repo, rpath, destDir string, progress func(current, total int64)) error {
	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, rpath)
	dest := filepath.Join(destDir, rpath)
	if err := os.MkdirAll(filepath.Dir(dest), 0755); err != nil {
		return err
	}

	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()

	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("hf: download error %d: %s", resp.StatusCode, url)
	}

	total := resp.ContentLength
	var current int64

	// Create a wrapper for progress reporting
	reader := &progressReader{
		Reader: resp.Body,
		onProgress: func(n int64) {
			current += n
			if progress != nil {
				progress(current, total)
			}
		},
	}

	if _, err := io.Copy(out, reader); err != nil {
		return err
	}
	return nil
}

type progressReader struct {
	io.Reader
	onProgress func(int64)
}

func (pr *progressReader) Read(p []byte) (n int, err error) {
	n, err = pr.Reader.Read(p)
	if n > 0 {
		pr.onProgress(int64(n))
	}
	return
}

func FilterFiles(info *ModelInfo) []string {
	var files []string
	interesting := []string{
		"config.json",
		"generation_config.json",
		"tokenizer.json",
		"tokenizer_config.json",
		"vocab.json",
		"merges.txt",
		"vocab.model",
	}

	for _, s := range info.Siblings {
		rpath := s.Rfilename
		lower := strings.ToLower(rpath)

		isInteresting := slices.Contains(interesting, rpath)

		if isInteresting || strings.HasSuffix(lower, ".safetensors") {
			files = append(files, rpath)
		}
	}
	return files
}
