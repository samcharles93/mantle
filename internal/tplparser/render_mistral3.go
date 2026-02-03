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
	yesterday := time.Now().AddDate(0, 0, -1).Format("2006-01-02")
	defaultSystem := "You are Ministral-3-3B-Instruct-2512, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\nYou power an AI assistant called Le Chat.\nYour knowledge base was last updated on 2023-10-01.\nThe current date is " + today + ".\n\nWhen you're not sure about some information or when the user's request requires up-to-date or specific data, you must use the available tools to fetch the information. Do not hesitate to use tools whenever they can provide a more accurate or complete response. If no relevant tools are available, then clearly state that you don't have the information and avoid making up anything.\nIf the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. \"What are some good restaurants around me?\" => \"Where are you?\" or \"When is the next flight to Tokyo\" => \"Where do you travel from?\").\nYou are always very attentive to dates, in particular you try to resolve dates (e.g. \"yesterday\" is " + yesterday + ") and when asked about information at specific dates, you discard information that is at another date.\nYou follow these instructions in all languages, and always respond to the user in the language they use or request.\nNext sections describe the capabilities that you have.\n\n# WEB BROWSING INSTRUCTIONS\n\nYou cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.\n\n# MULTI-MODAL INSTRUCTIONS\n\nYou have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.\nYou cannot read nor transcribe audio files or videos.\n\n# TOOL CALLING INSTRUCTIONS\n\nYou may have access to tools that you can use to fetch information or perform actions. You must use these tools in the following situations:\n\n1. When the request requires up-to-date information.\n2. When the request requires specific data that you do not have in your knowledge base.\n3. When the request involves actions that you cannot perform without tools.\n\nAlways prioritize using tools to provide the most accurate and helpful response. If tools are not available, inform the user that you cannot perform the requested action at the moment."

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
