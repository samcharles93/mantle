package ux

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/chzyer/readline"
)

var rlInstance *readline.Instance

func initReadline(prompt string) error {
	if rlInstance != nil {
		rlInstance.SetPrompt(prompt)
		return nil
	}

	home, _ := os.UserHomeDir()
	historyFile := filepath.Join(home, ".local", "share", "mantle", "history")
	if err := os.MkdirAll(filepath.Dir(historyFile), 0755); err != nil {
		// Fallback to no history file
		historyFile = ""
	}

	completer := readline.NewPrefixCompleter(
		readline.PcItem("/help"),
		readline.PcItem("/clear"),
		readline.PcItem("/stats"),
		readline.PcItem("/system"),
		readline.PcItem("/set",
			readline.PcItem("temp"),
			readline.PcItem("top_k"),
			readline.PcItem("top_p"),
			readline.PcItem("min_p"),
			readline.PcItem("penalty"),
		),
		readline.PcItem("/exit"),
		readline.PcItem("/quit"),
	)

	config := &readline.Config{
		Prompt:          prompt,
		HistoryFile:     historyFile,
		AutoComplete:    completer,
		InterruptPrompt: "^C",
		EOFPrompt:       "exit",
	}

	var err error
	rlInstance, err = readline.NewEx(config)
	if err != nil {
		return err
	}
	return nil
}

// ReadInteractiveLine reads one line from stdin with persistent history and editing.
func ReadInteractiveLine(prompt string) (string, error) {
	if !stdinIsTTY() {
		// Fallback for non-TTY
		fmt.Print(prompt)
		var s string
		_, err := fmt.Scanln(&s)
		if err != nil && err != io.EOF {
			return "", err
		}
		return s, err
	}

	if err := initReadline(prompt); err != nil {
		return "", err
	}

	line, err := rlInstance.Readline()
	if err != nil {
		if err == readline.ErrInterrupt {
			return "/exit", nil // Map Ctrl+C to exit in interactive mode if preferred
		}
		return "", err
	}
	return line, nil
}

// CloseInteractive closes the readline instance.
func CloseInteractive() {
	if rlInstance != nil {
		_ = rlInstance.Close()
	}
}
