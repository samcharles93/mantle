package tplparser

type Message struct {
	Role      string     `json:"role"`
	Content   any        `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	Name      string     `json:"name,omitempty"`
}

type ToolCall struct {
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments any    `json:"arguments,omitempty"`
}

type RenderOptions struct {
	Template            string
	Arch                string
	BOSToken            string
	EOSToken            string
	AddBOS              bool
	AddGenerationPrompt bool
	Messages            []Message
	Tools               []any
}
