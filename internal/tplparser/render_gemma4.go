package tplparser

import "strings"

func renderGemma4(opts RenderOptions) (string, bool, error) {
	var b strings.Builder

	if !opts.AddBOS && opts.BOSToken != "" {
		b.WriteString(opts.BOSToken)
	}

	msgs := opts.Messages
	loopMessages := msgs

	if len(opts.Tools) > 0 || (len(msgs) > 0 && isRole(msgs[0].Role, "system", "developer")) {
		b.WriteString("<|turn>system\n")
		if len(msgs) > 0 && isRole(msgs[0].Role, "system", "developer") {
			if err := writeGemmaSystemContent(&b, msgs[0].Content); err != nil {
				return "", false, err
			}
			loopMessages = msgs[1:]
		}
		if len(opts.Tools) > 0 {
			for _, tool := range opts.Tools {
				decl, err := formatFunctionDeclaration(tool)
				if err != nil {
					return "", false, err
				}
				b.WriteString("<|tool>")
				b.WriteString(strings.TrimSpace(decl))
				b.WriteString("<tool|>")
			}
		}
		b.WriteString("<turn|>\n")
	}

	for _, msg := range loopMessages {
		role := msg.Role
		if role == "assistant" {
			role = "model"
		}
		if role == "tool" {
			b.WriteString("<|turn>user\n")
			if err := writeGemma4ToolResponse(&b, msg); err != nil {
				return "", false, err
			}
			b.WriteString("<turn|>\n")
			continue
		}

		b.WriteString("<|turn>")
		b.WriteString(role)
		b.WriteString("\n")
		if msg.Content != nil {
			if err := writeGemmaMessageContent(&b, msg.Content); err != nil {
				return "", false, err
			}
		}
		for _, call := range msg.ToolCalls {
			fn := call.Function
			b.WriteString("<|tool_call>call:")
			b.WriteString(fn.Name)
			b.WriteString("{")
			if fn.Arguments != nil {
				switch v := fn.Arguments.(type) {
				case map[string]any:
					args, err := formatArgumentMap(v, false)
					if err != nil {
						return "", false, err
					}
					b.WriteString(args)
				case string:
					b.WriteString(v)
				default:
					args, err := formatArgument(v, false)
					if err != nil {
						return "", false, err
					}
					b.WriteString(args)
				}
			}
			b.WriteString("}<tool_call|>")
		}
		b.WriteString("<turn|>\n")
	}

	if opts.AddGenerationPrompt {
		b.WriteString("<|turn>model\n")
	}
	return b.String(), true, nil
}

func writeGemma4ToolResponse(b *strings.Builder, msg Message) error {
	content := msg.Content
	if content == nil {
		return nil
	}
	writeResponseMap := func(name any, response any) error {
		nameStr, ok := asString(name)
		if !ok {
			return nil
		}
		respMap, ok := asMap(response)
		if !ok {
			return nil
		}
		b.WriteString("<|tool_response>response:")
		b.WriteString(trimString(nameStr))
		b.WriteString("{")
		keys := sortedKeys(respMap)
		for i, key := range keys {
			if i > 0 {
				b.WriteString(",")
			}
			arg, err := formatArgument(respMap[key], false)
			if err != nil {
				return err
			}
			b.WriteString(key)
			b.WriteString(":")
			b.WriteString(arg)
		}
		b.WriteString("}<tool_response|>")
		return nil
	}
	if m, ok := asMap(content); ok {
		if name, ok := m["name"]; ok && m["response"] != nil {
			return writeResponseMap(name, m["response"])
		}
		if msg.Name != "" {
			return writeResponseMap(msg.Name, m)
		}
	}
	if s, ok := asString(content); ok && msg.Name != "" {
		b.WriteString("<|tool_response>response:")
		b.WriteString(trimString(msg.Name))
		b.WriteString("{value:")
		arg, err := formatArgument(s, false)
		if err != nil {
			return err
		}
		b.WriteString(arg)
		b.WriteString("}<tool_response|>")
		return nil
	}
	return nil
}
