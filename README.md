# Mantle

Mantle is a model execution experiment in Pure Go.

This is just an experiment which will probably have heaps of changes and optimisations along the way. it's in no way a replacement for cpp-based apps. 
use Llama.cpp if you want speed, performance, and a working app.

Now that disclaimer is done.

There's 2 sides of this repo:
- Mantle: the runtime and tooling for model execution
- MCF: a single-file, random-access container format for model data (raw or quantised), designed to be efficient with mmap where available.

## It works, but...


Quantisation needs work to improve the speed.

```shell
bin/mantle run -m "$MANTLE_MODELS_DIR/Qwen3-0.6B.mcf" --steps 128 --show-config --show-tokens --system "You are a helpful assistant." --prompt "In 100 words, tell me about you."
Loading MCF model: /work/models/mcf/Qwen3-0.6B.mcf
Model loaded in 437.979357ms
MCF | arch=qwen3
blocks=28 embd=1024 ffn=3072 heads=16 head_dim=128 vocab=151936 ctx=40960
rope: base=1e+06 scaling=none
HeadCountKV: [8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8]
sampling: temp=0.6 (generation_config) top_k=20 (generation_config) top_p=0.95 (generation_config) repeat_penalty=1.1 (default)
chat_template: tokenizer_config

Input tokens (31): [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 641, 220, 16, 15, 15, 4244, 11, 3291, 752, 911, 498, 13, 151645, 198, 151644, 77091, 198]
<think>
Okay, the user wants me to describe myself in 100 words. Let me start by recalling my role as an AI assistant. I'm designed to provide information, answer questions, and support users. I should mention my purpose clearly but keep it concise.

I need to highlight that I'm here to help and assist. Also, emphasize that I don't have a physical form or personality, which is important for maintaining trust. Mentioning the variety of topics I can handle shows that I'm versatile. Finally, end with a friendly note to ensure the user feels comfortable using me.
</think>

I am an AI assistant designed
Stats: 8.20 TPS (128 tokens in 15.606648717s)
```

```shell
task run                                                                                                                                       (base)
task: [run] bin/mantle run -m "$MANTLE_MODELS_DIR/Qwen3-0.6B.k4.mcf" --steps 128 --show-config --show-tokens --system "You are a helpful assistant." --prompt "In 100 words, tell me about you."
Loading MCF model: /work/models/mcf/Qwen3-0.6B.k4.mcf
Model loaded in 701.666565ms
MCF | arch=qwen3
blocks=28 embd=1024 ffn=3072 heads=16 head_dim=128 vocab=151936 ctx=40960
rope: base=1e+06 scaling=none
HeadCountKV: [8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8]
sampling: temp=0.6 (generation_config) top_k=20 (generation_config) top_p=0.95 (generation_config) repeat_penalty=1.1 (default)
chat_template: tokenizer_config

Input tokens (31): [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 641, 220, 16, 15, 15, 4244, 11, 3291, 752, 911, 498, 13, 151645, 198, 151644, 77091, 198]
<think>
Okay, the user wants to know something in 100 words about me. Let me start by identifying what I need here. Since I'm an AI assistant, my main task is to process and analyze data efficiently.

First, I should mention that I am an AI system designed to assist with various tasks. Next, I can highlight how I process and understand information from different sources. It's important to note that I don't have personal experiences or feelings, which adds a professional touch. Also, emphasizing my ability to provide accurate information is key here.

I need to keep the response concise but comprehensive within the word limit.
Stats: 3.20 TPS (128 tokens in 40.039700112s)
```

## This repo is:

A big experiment to see what's possible in pure Go. and it was fun, it works!, and it's fairly decent for small models.
It started out just creating the kernels in pure go and trying to work with Go assembly '.s' files (to learn how to use them), but when I found out about `simd/archsimd` I was just too curious about seeing how much I could get from it.

Now, I didn't need the model container format (MCF). But, trying to decode GGUF was just stupidly difficult (tokenisers not being recognised, spewing stop tokens, model didn't produce cohesive responses), and safetensors, while possible and it worked, was slow.
therefore, I decided to embark on stuffing up another containerised model format. and here we are, MCF and Mantle...

Anyway, If you'd like to test this out now, You must have Go v1.26rc3 and enable the GOEXPERIMENT=simd env for all build/run/test commands or just infer the usage from the Taskfile (or use `task installGoRC`).

You will need to clone a safetensors model from HF and pack it using the CLI `bin/mantle pack -in /path/to/safetensors/ -out /path/to/model.mcf`,
You can also add the `--dedup` flag to remove duplicated tensors from the resulting model container. For most models, this does nothing, but in testing, Qwen3-0.6B has a duplicated tensor in attn and embd. and it reduced the model file by 300mb with no quality hit.

To run the model *only on CPU* you just need to use the `mantle run` command with `-m/--model`, or alternatively, pass in a directory of models to use with `MANTLE_MODELS_DIR=/path/to/models mantle run`.

### Build Requirements
- Go 1.26rc3
- AMD64 CPU (simd/archsimd will only compile for amd64)

### Build
```shell
GOEXPERIMENT=simd go build -o bin/mantle ./cmd/mantle
````

### CUDA Build Notes
The CUDA CGO integration expects the CUDA runtime and cuBLAS to be discoverable by the linker. If your toolkit libraries live outside the default search path, set `CGO_LDFLAGS` and `LD_LIBRARY_PATH` before building or testing with `-tags=cuda`:

```bash
export CGO_LDFLAGS="-L/path/to/cuda/lib64"
export LD_LIBRARY_PATH="/path/to/cuda/lib64:$LD_LIBRARY_PATH"
GOEXPERIMENT=simd go test -tags=cuda ./internal/backend/cuda/native
```

To use the CLI:
### Run

```bash
mantle run -m /models/mcf/Qwen3-3B-Instruct.mcf

# Or you can pass a directory to multiple models:
MANTLE_MODELS_DIR=/models mantle run
```

### Pack (create a model container)

By default the container will be packed with the same tensor dtype as the safetensors model.
Use the `--dedup` to deduplicate tensors or `--cast` to cast the tensors to another dtype.

```bash
mantle pack --in /path/to/model --out /work/models/MyModel.mcf
```

The flags will likely evolve, the `run` command will eventually be implied, the serve command will serve an OpenAI API endpoint. 
I plan to add `inspect` and `serve` only if the project continues, or, you're welcome to submit a PR.


## Architecture (high level)

### Mantle layers

* Model store: loads model containers from disk (MCF)
* Container reader: validates structure and exposes random-access views over sections
* Execution coordination: hands model bytes to the selected runtime backend and manages lifetimes and I/O modes

### MCF fundamentals

* Single-file container with absolute offsets
* Little-endian on-disk fields
* Optional sections: readers tolerate unknown sections and tolerate missing optional ones
* Runtime behavior remains explicit, but may consult explicit config fields. For example, sliding attention is only
  enabled when `layer_types` includes `sliding_attention` and `sliding_window > 0` in the model config.
* Designed for **random access**:

  * buffered I/O (`ReadAt`) as a baseline
  * mmap as an acceleration where supported
  * optional direct I/O is a runtime choice (not a format requirement)

### Current section types (indicative)

* Model info and metadata
* Quantisation info
* Tensor index (names, shapes, dtype, offsets)
* Tensor data

## Contributing

... Submit a PR.

## Developer Utilities

Developer investigation tools live under `work/` and are excluded from default builds/tests.
See `work/README.md` for usage with `-tags tools`.
