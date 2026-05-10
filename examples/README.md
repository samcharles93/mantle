# Examples

These examples show how to use the `pkg/mantle` library to load MCF models
and run inference from your own Go programs.

**Pattern**: every example follows the same three steps:

1. `mantle.NewEngine(ctx, path, opts...)` — load an MCF model
2. `engine.Generate(ctx, request, streamFunc)` — run inference
3. `engine.Close()` — release resources

## simple

Minimal example — load a model, generate text, print the result.

```bash
go run ./examples/simple -m /models/Qwen3-0.6B.mcf -p "Hello world" -n 128
```

Flags control all sampling parameters. The `-backend` flag switches between
auto, cpu, and cuda.

## streaming

Token-by-token output via the `StreamFunc` callback. The callback receives
`StreamChunk` values with a `Type` (text_delta, reasoning_delta, prompt_echo)
and a `Delta` string.

Content appears as it's generated. Reasoning output (from models like
DeepSeek-R1) is dimmed when `-reasoning` is set.

```bash
# Streaming generation
go run ./examples/streaming -m model.mcf -p "Tell me a story" -n 256

# With reasoning model
go run ./examples/streaming -m model.mcf -p "Solve: 2+2" --reasoning
```

## chat

Multi-turn conversation using structured `Message` values with
role/content pairs. Mantle handles chat template rendering internally
so you don't need to worry about the model-specific prompt format.

Supports single-message mode (`-p`) and an interactive chat loop
(when `-p` is omitted).

```bash
# Single message
go run ./examples/chat -m model.mcf -p "What is Go?"

# With system prompt
go run ./examples/chat -m model.mcf -s "You are a poet" -p "Write a haiku"

# Interactive chat (Ctrl-C to exit)
go run ./examples/chat -m model.mcf
```

Interactive commands: `/exit`, `/clear`, `/stats`.
