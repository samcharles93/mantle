package tplparser

import (
	"fmt"
	"strings"
)

func renderGemma3(opts RenderOptions) (string, bool, error) {
	var b strings.Builder

	if !opts.AddBOS && opts.BOSToken != "" {
		b.WriteString(opts.BOSToken)
	}

	msgs := opts.Messages
	loopMessages := msgs

	if len(opts.Tools) > 0 || (len(msgs) > 0 && isRole(msgs[0].Role, "system", "developer")) {
		b.WriteString("<start_of_turn>developer\n")
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
				b.WriteString("<start_function_declaration>")
				b.WriteString(strings.TrimSpace(decl))
				b.WriteString("<end_function_declaration>")
			}
		}
		b.WriteString("<end_of_turn>\n")
	}

	prevType := ""
	for _, msg := range loopMessages {
		role := msg.Role
		if role == "assistant" {
			role = "model"
		}
		if role != "tool" {
			if prevType != "tool_response" {
				b.WriteString("<start_of_turn>")
				b.WriteString(role)
				b.WriteString("\n")
			}
			prevType = ""

			if msg.Content != nil {
				if err := writeGemmaMessageContent(&b, msg.Content); err != nil {
					return "", false, err
				}
				prevType = "content"
			}

			if len(msg.ToolCalls) > 0 {
				for j, call := range msg.ToolCalls {
					fn := call.Function
					b.WriteString("<start_function_call>call:")
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
					b.WriteString("}<end_function_call>")
					if j == len(msg.ToolCalls)-1 {
						b.WriteString("<start_function_response>")
					}
				}
				prevType = "tool_call"
			}
		} else {
			if err := writeGemmaToolResponse(&b, msg); err != nil {
				return "", false, err
			}
			prevType = "tool_response"
		}

		if prevType != "tool_call" && prevType != "tool_response" {
			b.WriteString("<end_of_turn>\n")
		}
	}

	if opts.AddGenerationPrompt {
		if prevType != "tool_response" {
			b.WriteString("<start_of_turn>model\n")
		}
	}

	return b.String(), true, nil
}

func isRole(role string, roles ...string) bool {
	for _, r := range roles {
		if role == r {
			return true
		}
	}
	return false
}

func writeGemmaSystemContent(b *strings.Builder, content any) error {
	if s, ok := asString(content); ok {
		b.WriteString(trimString(s))
		return nil
	}
	if seq, ok := asSlice(content); ok {
		for _, item := range seq {
			m, ok := asMap(item)
			if !ok {
				continue
			}
			t, _ := asString(m["type"])
			if t != "text" {
				continue
			}
			txt, _ := asString(m["text"])
			b.WriteString(trimString(txt))
		}
		return nil
	}
	return fmt.Errorf("gemma: invalid system content type")
}

func writeGemmaMessageContent(b *strings.Builder, content any) error {
	if s, ok := asString(content); ok {
		b.WriteString(trimString(s))
		return nil
	}
	if seq, ok := asSlice(content); ok {
		for _, item := range seq {
			m, ok := asMap(item)
			if !ok {
				return fmt.Errorf("gemma: invalid content item")
			}
			t, _ := asString(m["type"])
			switch t {
			case "image":
				b.WriteString("<start_of_image>")
			case "text":
				txt, _ := asString(m["text"])
				b.WriteString(trimString(txt))
			default:
				return fmt.Errorf("gemma: unsupported content type %q", t)
			}
		}
		return nil
	}
	return fmt.Errorf("gemma: invalid content type")
}

func writeGemmaToolResponse(b *strings.Builder, msg Message) error {
	content := msg.Content
	if content == nil {
		return nil
	}

	if m, ok := asMap(content); ok {
		if name, ok := m["name"]; ok && m["response"] != nil {
			return writeNamedResponseMap(b, name, m["response"])
		}
		if msg.Name != "" {
			return writeNamedResponseMap(b, msg.Name, m)
		}
		return fmt.Errorf("gemma: tool response mapping missing name")
	}

	if s, ok := asString(content); ok {
		if msg.Name == "" {
			return fmt.Errorf("gemma: tool response string missing name")
		}
		b.WriteString("<start_function_response>response:")
		b.WriteString(trimString(msg.Name))
		b.WriteString("{value:")
		arg, err := formatArgument(s, false)
		if err != nil {
			return err
		}
		b.WriteString(arg)
		b.WriteString("}<end_function_response>")
		return nil
	}

	if seq, ok := asSlice(content); ok {
		for _, item := range seq {
			m, ok := asMap(item)
			if !ok {
				return fmt.Errorf("gemma: invalid tool response item")
			}
			if name, ok := m["name"]; ok && m["response"] != nil {
				if err := writeNamedResponseMap(b, name, m["response"]); err != nil {
					return err
				}
				continue
			}
			if msg.Name != "" {
				if err := writeNamedResponseMap(b, msg.Name, m); err != nil {
					return err
				}
				continue
			}
			return fmt.Errorf("gemma: tool response item missing name")
		}
		return nil
	}
	return fmt.Errorf("gemma: invalid tool response content")
}

func writeNamedResponseMap(b *strings.Builder, name any, response any) error {
	nameStr, ok := asString(name)
	if !ok {
		return fmt.Errorf("gemma: tool response name must be string")
	}
	respMap, ok := asMap(response)
	if !ok {
		return fmt.Errorf("gemma: tool response must be mapping")
	}

	b.WriteString("<start_function_response>response:")
	b.WriteString(trimString(nameStr))
	b.WriteString("{")
	keys := sortedKeys(respMap)
	first := true
	for _, key := range keys {
		if !first {
			b.WriteString(",")
		}
		first = false
		val := respMap[key]
		arg, err := formatArgument(val, false)
		if err != nil {
			return err
		}
		b.WriteString(key)
		b.WriteString(":")
		b.WriteString(arg)
	}
	b.WriteString("}<end_function_response>")
	return nil
}

func formatFunctionDeclaration(tool any) (string, error) {
	toolMap, ok := asMap(tool)
	if !ok {
		return "", fmt.Errorf("gemma: tool must be mapping")
	}
	fnRaw, ok := toolMap["function"]
	if !ok {
		return "", fmt.Errorf("gemma: tool missing function")
	}
	fn, ok := asMap(fnRaw)
	if !ok {
		return "", fmt.Errorf("gemma: tool function must be mapping")
	}

	name, _ := asString(fn["name"])
	desc, _ := asString(fn["description"])

	var b strings.Builder
	b.WriteString("declaration:")
	b.WriteString(name)
	b.WriteString("\n")
	b.WriteString("{description:<escape>")
	b.WriteString(desc)
	b.WriteString("<escape>")

	if paramsRaw, ok := fn["parameters"]; ok && paramsRaw != nil {
		params, ok := asMap(paramsRaw)
		if !ok {
			return "", fmt.Errorf("gemma: function parameters must be mapping")
		}
		b.WriteString("\n,parameters:{")
		if propsRaw, ok := params["properties"]; ok && propsRaw != nil {
			props, ok := asMap(propsRaw)
			if !ok {
				return "", fmt.Errorf("gemma: parameters.properties must be mapping")
			}
			formatted, err := formatParameters(props)
			if err != nil {
				return "", err
			}
			b.WriteString("properties:{ ")
			b.WriteString(formatted)
			b.WriteString(" },")
		}
		if reqRaw, ok := params["required"]; ok && reqRaw != nil {
			reqList, ok := asSlice(reqRaw)
			if !ok {
				return "", fmt.Errorf("gemma: parameters.required must be array")
			}
			b.WriteString("required:[")
			for i, item := range reqList {
				itemStr, _ := asString(item)
				b.WriteString("<escape>")
				b.WriteString(itemStr)
				b.WriteString("<escape>")
				if i != len(reqList)-1 {
					b.WriteString(",")
				}
			}
			b.WriteString("],")
		}
		if typ, ok := params["type"]; ok && typ != nil {
			typeStr, _ := asString(typ)
			b.WriteString("type:<escape>")
			b.WriteString(strings.ToUpper(typeStr))
			b.WriteString("<escape>}")
		} else {
			b.WriteString("}")
		}
	}

	b.WriteString("\n}")
	return b.String(), nil
}

func formatParameters(properties map[string]any) (string, error) {
	standardKeys := map[string]struct{}{
		"description": {},
		"type":        {},
		"properties":  {},
		"required":    {},
		"nullable":    {},
	}

	var b strings.Builder
	keys := sortedKeys(properties)
	first := true
	for _, key := range keys {
		if _, skip := standardKeys[key]; skip {
			continue
		}
		val := properties[key]
		valMap, ok := asMap(val)
		if !ok {
			continue
		}
		if !first {
			b.WriteString(",")
		}
		first = false
		b.WriteString(key)
		b.WriteString(":{description:<escape>")
		desc, _ := asString(valMap["description"])
		b.WriteString(desc)
		b.WriteString("<escape>")

		typeStr, _ := asString(valMap["type"])
		typeUpper := strings.ToUpper(typeStr)
		switch typeUpper {
		case "STRING":
			if enumVal, ok := valMap["enum"]; ok && enumVal != nil {
				formatted, err := formatArgument(enumVal, true)
				if err != nil {
					return "", err
				}
				b.WriteString(",enum:")
				b.WriteString(formatted)
			}
		case "OBJECT":
			b.WriteString(",properties:{")
			if propsRaw, ok := valMap["properties"]; ok && propsRaw != nil {
				props, ok := asMap(propsRaw)
				if ok {
					formatted, err := formatParameters(props)
					if err != nil {
						return "", err
					}
					b.WriteString(formatted)
				}
			} else if props, ok := asMap(valMap); ok {
				formatted, err := formatParameters(props)
				if err != nil {
					return "", err
				}
				b.WriteString(formatted)
			}
			b.WriteString("}")
			if reqRaw, ok := valMap["required"]; ok && reqRaw != nil {
				reqList, ok := asSlice(reqRaw)
				if ok {
					b.WriteString(",required:[")
					for i, item := range reqList {
						itemStr, _ := asString(item)
						b.WriteString("<escape>")
						b.WriteString(itemStr)
						b.WriteString("<escape>")
						if i != len(reqList)-1 {
							b.WriteString(",")
						}
					}
					b.WriteString("]")
				}
			}
		case "ARRAY":
			if itemsRaw, ok := valMap["items"]; ok && itemsRaw != nil {
				itemsMap, ok := asMap(itemsRaw)
				if ok {
					b.WriteString(",items:{")
					itemKeys := sortedKeys(itemsMap)
					itemFirst := true
					for _, itemKey := range itemKeys {
						itemVal := itemsMap[itemKey]
						if itemVal == nil {
							continue
						}
						if !itemFirst {
							b.WriteString(",")
						}
						itemFirst = false
						switch itemKey {
						case "properties":
							b.WriteString("properties:{")
							if props, ok := asMap(itemVal); ok {
								formatted, err := formatParameters(props)
								if err != nil {
									return "", err
								}
								b.WriteString(formatted)
							}
							b.WriteString("}")
						case "required":
							if reqList, ok := asSlice(itemVal); ok {
								b.WriteString("required:[")
								for i, reqItem := range reqList {
									itemStr, _ := asString(reqItem)
									b.WriteString("<escape>")
									b.WriteString(itemStr)
									b.WriteString("<escape>")
									if i != len(reqList)-1 {
										b.WriteString(",")
									}
								}
								b.WriteString("]")
							}
						case "type":
							switch v := itemVal.(type) {
							case string:
								b.WriteString("type:")
								arg, err := formatArgument(strings.ToUpper(v), true)
								if err != nil {
									return "", err
								}
								b.WriteString(arg)
							case []any:
								b.WriteString("type:")
								upper := make([]any, 0, len(v))
								for _, item := range v {
									if s, ok := asString(item); ok {
										upper = append(upper, strings.ToUpper(s))
									}
								}
								arg, err := formatArgument(upper, true)
								if err != nil {
									return "", err
								}
								b.WriteString(arg)
							default:
								b.WriteString("type:")
								arg, err := formatArgument(itemVal, true)
								if err != nil {
									return "", err
								}
								b.WriteString(arg)
							}
						default:
							arg, err := formatArgument(itemVal, true)
							if err != nil {
								return "", err
							}
							b.WriteString(itemKey)
							b.WriteString(":")
							b.WriteString(arg)
						}
					}
					b.WriteString("}")
				}
			}
		}

		b.WriteString(",type:<escape>")
		b.WriteString(typeUpper)
		b.WriteString("<escape>}")
	}
	return b.String(), nil
}

func formatArgument(arg any, escapeKeys bool) (string, error) {
	switch v := arg.(type) {
	case string:
		return "<escape>" + v + "<escape>", nil
	case bool:
		if v {
			return "true", nil
		}
		return "false", nil
	case nil:
		return "null", nil
	case map[string]any:
		return formatArgumentMap(v, escapeKeys)
	case []any:
		var b strings.Builder
		b.WriteString("[")
		for i, item := range v {
			rendered, err := formatArgument(item, escapeKeys)
			if err != nil {
				return "", err
			}
			b.WriteString(rendered)
			if i != len(v)-1 {
				b.WriteString(",")
			}
		}
		b.WriteString("]")
		return b.String(), nil
	default:
		return fmt.Sprint(arg), nil
	}
}

func formatArgumentMap(m map[string]any, escapeKeys bool) (string, error) {
	var b strings.Builder
	b.WriteString("{")
	keys := sortedKeys(m)
	first := true
	for _, key := range keys {
		if !first {
			b.WriteString(",")
		}
		first = false
		if escapeKeys {
			b.WriteString("<escape>")
			b.WriteString(key)
			b.WriteString("<escape>")
		} else {
			b.WriteString(key)
		}
		b.WriteString(":")
		rendered, err := formatArgument(m[key], escapeKeys)
		if err != nil {
			return "", err
		}
		b.WriteString(rendered)
	}
	b.WriteString("}")
	return b.String(), nil
}
