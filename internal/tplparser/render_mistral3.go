package tplparser

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

func renderMistral3(opts RenderOptions) (string, bool, error) {
	var b strings.Builder

	if !opts.AddBOS && opts.BOSToken != "" {
		b.WriteString(opts.BOSToken)
	}

	today := time.Now().Format("2006-01-02")
	defaultSystem := fmt.Sprintf("You are a helpful AI assistant. Current date: %s", today)

	msgs := opts.Messages
	loopMessages := msgs
	if len(msgs) > 0 && msgs[0].Role == "system" {
		b.WriteString("[SYSTEM_PROMPT]")
		if err := writeMistralSystemContent(&b, msgs[0].Content); err != nil {
			return "", false, err
		}
		b.WriteString("[/SYSTEM_PROMPT]")
		loopMessages = msgs[1:]
	} else if defaultSystem != "" {
		b.WriteString("[SYSTEM_PROMPT]")
		b.WriteString(defaultSystem)
		b.WriteString("[/SYSTEM_PROMPT]")
	}

	if len(opts.Tools) > 0 {
		j, err := jsonString(opts.Tools)
		if err != nil {
			return "", false, fmt.Errorf("mistral3: tools tojson: %w", err)
		}
		b.WriteString("[AVAILABLE_TOOLS]")
		b.WriteString(j)
		b.WriteString("[/AVAILABLE_TOOLS]")
	}

	if err := validateMistralAlternation(loopMessages); err != nil {
		return "", false, err
	}

	for _, msg := range loopMessages {
		switch msg.Role {
		case "user":
			if err := writeMistralUser(&b, msg.Content); err != nil {
				return "", false, err
			}
		case "assistant":
			if err := writeMistralAssistant(&b, msg, opts.EOSToken); err != nil {
				return "", false, err
			}
		case "tool":
			b.WriteString("[TOOL_RESULTS]")
			b.WriteString(fmt.Sprint(msg.Content))
			b.WriteString("[/TOOL_RESULTS]")
		default:
			return "", false, fmt.Errorf("mistral3: unsupported role %q", msg.Role)
		}
	}

	return b.String(), true, nil
}

func writeMistralSystemContent(b *strings.Builder, content any) error {
	if s, ok := asString(content); ok {
		b.WriteString(s)
		return nil
	}
	if seq, ok := asSlice(content); ok {
		for _, item := range seq {
			m, ok := asMap(item)
			if !ok {
				return fmt.Errorf("mistral3: invalid system block")
			}
			if t, _ := asString(m["type"]); t != "text" {
				return fmt.Errorf("mistral3: only text blocks supported in system message")
			}
			if txt, ok := asString(m["text"]); ok {
				b.WriteString(txt)
			}
		}
		return nil
	}
	return fmt.Errorf("mistral3: invalid system content")
}

func validateMistralAlternation(msgs []Message) error {
	index := 0
	for _, msg := range msgs {
		if msg.Role == "user" || (msg.Role == "assistant" && len(msg.ToolCalls) == 0) {
			expectedUser := index%2 == 0
			if (msg.Role == "user") != expectedUser {
				return fmt.Errorf("mistral3: messages must alternate user/assistant roles")
			}
			index++
		}
	}
	return nil
}

func writeMistralUser(b *strings.Builder, content any) error {
	if s, ok := asString(content); ok {
		b.WriteString("[INST]")
		b.WriteString(s)
		b.WriteString("[/INST]")
		return nil
	}
	if seq, ok := asSlice(content); ok {
		b.WriteString("[INST]")
		blocks := seq
		if len(blocks) == 2 {
			sort.Slice(blocks, func(i, j int) bool {
				mi, _ := asMap(blocks[i])
				mj, _ := asMap(blocks[j])
				ti, _ := asString(mi["type"])
				tj, _ := asString(mj["type"])
				return ti < tj
			})
		}
		for _, item := range blocks {
			m, ok := asMap(item)
			if !ok {
				return fmt.Errorf("mistral3: invalid user block")
			}
			t, _ := asString(m["type"])
			switch t {
			case "text":
				if txt, ok := asString(m["text"]); ok {
					b.WriteString(txt)
				}
			case "image", "image_url":
				b.WriteString("[IMG]")
			default:
				return fmt.Errorf("mistral3: unsupported user block type %q", t)
			}
		}
		b.WriteString("[/INST]")
		return nil
	}
	return fmt.Errorf("mistral3: invalid user content")
}

func writeMistralAssistant(b *strings.Builder, msg Message, eosToken string) error {
	if (msg.Content == nil || msg.Content == "" || (isSliceLenZero(msg.Content))) && len(msg.ToolCalls) == 0 {
		return fmt.Errorf("mistral3: assistant message must have content or tool calls")
	}

	if s, ok := asString(msg.Content); ok {
		b.WriteString(s)
	} else if seq, ok := asSlice(msg.Content); ok {
		for _, item := range seq {
			m, ok := asMap(item)
			if !ok {
				return fmt.Errorf("mistral3: invalid assistant block")
			}
			t, _ := asString(m["type"])
			if t != "text" {
				return fmt.Errorf("mistral3: only text blocks supported in assistant message")
			}
			if txt, ok := asString(m["text"]); ok {
				b.WriteString(txt)
			}
		}
	}

	for _, call := range msg.ToolCalls {
		args := call.Function.Arguments
		var argStr string
		switch v := args.(type) {
		case string:
			if v == "" {
				argStr = "{}"
			} else {
				argStr = v
			}
		default:
			j, err := jsonString(v)
			if err != nil {
				return fmt.Errorf("mistral3: tool args tojson: %w", err)
			}
			argStr = j
		}
		b.WriteString("[TOOL_CALLS]")
		b.WriteString(call.Function.Name)
		b.WriteString("[ARGS]")
		b.WriteString(argStr)
	}

	if eosToken != "" {
		b.WriteString(eosToken)
	}
	return nil
}

func isSliceLenZero(v any) bool {
	if seq, ok := asSlice(v); ok {
		return len(seq) == 0
	}
	return false
}
