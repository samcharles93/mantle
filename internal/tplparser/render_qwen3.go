package tplparser

import (
	"fmt"
	"strings"
)

const qwen3ToolsPreamble = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"

const qwen3ToolsEpilogue = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"

func renderQwen3(opts RenderOptions) (string, bool, error) {
	var b strings.Builder

	msgs := opts.Messages
	if len(opts.Tools) > 0 {
		b.WriteString("<|im_start|>system\n")
		if len(msgs) > 0 && msgs[0].Role == "system" {
			if err := writeQwenSystemContent(&b, msgs[0].Content); err != nil {
				return "", false, err
			}
			b.WriteString("\n\n")
			msgs = msgs[1:]
		}
		b.WriteString(qwen3ToolsPreamble)
		for _, tool := range opts.Tools {
			b.WriteString("\n")
			j, err := jsonString(tool)
			if err != nil {
				return "", false, fmt.Errorf("qwen3: tool tojson: %w", err)
			}
			b.WriteString(j)
		}
		b.WriteString(qwen3ToolsEpilogue)
	} else if len(msgs) > 0 && msgs[0].Role == "system" {
		b.WriteString("<|im_start|>system\n")
		if err := writeQwenSystemContent(&b, msgs[0].Content); err != nil {
			return "", false, err
		}
		b.WriteString("<|im_end|>\n")
		msgs = msgs[1:]
	}

	imageCount := 0
	videoCount := 0

	for i, msg := range msgs {
		switch msg.Role {
		case "user":
			b.WriteString("<|im_start|>user\n")
			if err := writeQwenUserContent(&b, msg.Content, &imageCount, &videoCount, opts.AddVisionID); err != nil {
				return "", false, err
			}
			b.WriteString("<|im_end|>\n")
		case "assistant":
			b.WriteString("<|im_start|>assistant\n")
			if err := writeQwenAssistantContent(&b, msg.Content); err != nil {
				return "", false, err
			}
			if len(msg.ToolCalls) > 0 {
				for j, call := range msg.ToolCalls {
					if (j == 0 && msg.Content != nil) || j > 0 {
						b.WriteString("\n")
					}
					fn := call.Function
					b.WriteString("<tool_call>\n{\"name\": \"")
					b.WriteString(fn.Name)
					b.WriteString("\", \"arguments\": ")
					switch v := fn.Arguments.(type) {
					case string:
						b.WriteString(v)
					default:
						jv, err := jsonString(v)
						if err != nil {
							return "", false, fmt.Errorf("qwen3: tool arguments tojson: %w", err)
						}
						b.WriteString(jv)
					}
					b.WriteString("}\n</tool_call>")
				}
			}
			b.WriteString("<|im_end|>\n")
		case "tool":
			prevRole := ""
			if i > 0 {
				prevRole = msgs[i-1].Role
			}
			if i == 0 || prevRole != "tool" {
				b.WriteString("<|im_start|>user")
			}
			b.WriteString("\n<tool_response>\n")
			if err := writeQwenToolContent(&b, msg.Content, &imageCount, &videoCount, opts.AddVisionID); err != nil {
				return "", false, err
			}
			b.WriteString("\n</tool_response>")
			nextRole := ""
			if i < len(msgs)-1 {
				nextRole = msgs[i+1].Role
			}
			if i == len(msgs)-1 || nextRole != "tool" {
				b.WriteString("<|im_end|>\n")
			}
		}
	}

	if opts.AddGenerationPrompt {
		b.WriteString("<|im_start|>assistant\n")
	}
	return b.String(), true, nil
}

func writeQwenSystemContent(b *strings.Builder, content any) error {
	if s, ok := asString(content); ok {
		b.WriteString(s)
		return nil
	}
	if seq, ok := asSlice(content); ok {
		for _, item := range seq {
			m, ok := asMap(item)
			if !ok {
				continue
			}
			if txt, ok := m["text"]; ok {
				if s, ok := asString(txt); ok {
					b.WriteString(s)
				}
			}
		}
		return nil
	}
	return fmt.Errorf("qwen3: invalid system content")
}

func writeQwenUserContent(b *strings.Builder, content any, imageCount, videoCount *int, addVisionID bool) error {
	if s, ok := asString(content); ok {
		b.WriteString(s)
		return nil
	}
	if seq, ok := asSlice(content); ok {
		for _, item := range seq {
			m, ok := asMap(item)
			if !ok {
				continue
			}
			t, _ := asString(m["type"])
			if t == "image" || m["image"] != nil || m["image_url"] != nil {
				*imageCount++
				if addVisionID {
					b.WriteString(fmt.Sprintf("Picture %d: ", *imageCount))
				}
				b.WriteString("<|vision_start|><|image_pad|><|vision_end|>\n")
				continue
			}
			if t == "video" || m["video"] != nil {
				*videoCount++
				if addVisionID {
					b.WriteString(fmt.Sprintf("Video %d: ", *videoCount))
				}
				b.WriteString("<|vision_start|><|video_pad|><|vision_end|>\n")
				continue
			}
			if txt, ok := m["text"]; ok {
				if s, ok := asString(txt); ok {
					b.WriteString(s)
				}
			}
		}
		return nil
	}
	return fmt.Errorf("qwen3: invalid user content")
}

func writeQwenAssistantContent(b *strings.Builder, content any) error {
	if content == nil {
		return nil
	}
	if s, ok := asString(content); ok {
		b.WriteString(s)
		return nil
	}
	if seq, ok := asSlice(content); ok {
		for _, item := range seq {
			m, ok := asMap(item)
			if !ok {
				continue
			}
			if txt, ok := m["text"]; ok {
				if s, ok := asString(txt); ok {
					b.WriteString(s)
				}
			}
		}
		return nil
	}
	return fmt.Errorf("qwen3: invalid assistant content")
}

func writeQwenToolContent(b *strings.Builder, content any, imageCount, videoCount *int, addVisionID bool) error {
	if s, ok := asString(content); ok {
		b.WriteString(s)
		return nil
	}
	if seq, ok := asSlice(content); ok {
		for _, item := range seq {
			m, ok := asMap(item)
			if !ok {
				continue
			}
			t, _ := asString(m["type"])
			if t == "image" || m["image"] != nil || m["image_url"] != nil {
				*imageCount++
				if addVisionID {
					b.WriteString(fmt.Sprintf("Picture %d: ", *imageCount))
				}
				b.WriteString("<|vision_start|><|image_pad|><|vision_end|>\n")
				continue
			}
			if t == "video" || m["video"] != nil {
				*videoCount++
				if addVisionID {
					b.WriteString(fmt.Sprintf("Video %d: ", *videoCount))
				}
				b.WriteString("<|vision_start|><|video_pad|><|vision_end|>\n")
				continue
			}
			if txt, ok := m["text"]; ok {
				if s, ok := asString(txt); ok {
					b.WriteString(s)
				}
			}
		}
		return nil
	}
	return fmt.Errorf("qwen3: invalid tool content")
}
