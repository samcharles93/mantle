// Package mantle provides a public API for loading and running inference
// with Mantle models. It wraps the internal engine, tokenizer, sampler,
// and backend subsystems into a clean, minimal interface.
//
// Basic usage:
//
//	import "github.com/samcharles93/mantle/pkg/mantle"
//
//	ctx := context.Background()
//	engine, err := mantle.NewEngine(ctx, "/path/to/model.mcf")
//	if err != nil { /* handle */ }
//	defer engine.Close()
//
//	result, err := engine.Generate(ctx, mantle.Request{
//	    Prompt: "Hello!",
//	    Steps: 256,
//	}, nil)
//	if err != nil { /* handle */ }
//	fmt.Println(result.Text)
//
// Streaming output:
//
//	result, err := engine.Generate(ctx, req, func(chunk mantle.StreamChunk) {
//	    fmt.Print(chunk.Delta)
//	})
//
// Backend selection:
//
//	engine, err := mantle.NewEngine(ctx, "model.mcf", mantle.WithBackend("cpu"))
package mantle
