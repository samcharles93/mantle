package tplparser

type Message struct {
	Role      string
	Content   any
	ToolCalls []ToolCall
	Name      string
}

type ToolCall struct {
	ID       string
	Type     string
	Function ToolCallFunction
}

type ToolCallFunction struct {
	Name      string
	Arguments any
}

type RenderOptions struct {
	Template            string
	Arch                string
	BOSToken            string
	EOSToken            string
	AddBOS              bool
	AddGenerationPrompt bool
	KeepPastThinking    bool
	AddVisionID         bool
	Messages            []Message
	Tools               []any
}
