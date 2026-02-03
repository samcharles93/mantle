package inference

import (
	"os"
	"strings"

	"github.com/samcharles93/mantle/internal/model"
	"github.com/samcharles93/mantle/internal/tokenizer"
)

func ResolveChatTemplate(override string, cfg tokenizer.TokenizerConfig, arch string, hfConfig []byte) (string, string) {
	template := strings.TrimSpace(override)
	source := ""
	switch {
	case template != "":
		source = "flag"
	case strings.TrimSpace(cfg.ChatTemplate) != "":
		template = cfg.ChatTemplate
		source = "tokenizer_config"
	default:
		if inferred, ok := model.InferChatTemplate(arch, hfConfig); ok {
			template = inferred
			source = "model-default"
		} else {
			return "", "none"
		}
	}

	if len(template) < 256 && fileExists(template) {
		if raw, err := os.ReadFile(template); err == nil && len(raw) > 0 {
			template = string(raw)
			source += ":file"
		}
	}
	return template, source
}

func fileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}
