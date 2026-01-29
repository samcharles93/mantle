# Mantle

Mantle is a model execution experiment in Pure Go.

This is just an experiment which will probably have heaps of changes and optimisations along the way. it's in no way a replacement for cpp-based apps. 
use Llama.cpp if you want speed, performance, and a working app.

Now that disclaimer is done.

There's 2 sides of this repo:
- Mantle: the runtime and tooling for model execution
- MCF: a single-file, random-access container format for model data (raw or quantised), designed to be efficient with mmap where available.

## This repo is:

A big experiment to see what's possible in pure Go. and it was fun, it works!, and it's fairly decent for small models.
It started out just creating the kernels in pure go and trying to work with Go assembly '.s' files (to learn how to use them), but when I found out about `simd/archsimd` I was just too curious about seeing how much I could get from it.

Now, I didn't need the model container format (MCF). But, trying to decode GGUF was just stupidly difficult (tokenisers not being recognised, spewing stop tokens, model didn't produce cohesive responses), and safetensors, while possible and it worked, was slow.
therefore, I decided to embark on stuffing up another containerised model format. and here we are, MCF and Mantle...

Anyway, If you'd like to test this out now, You must have Go v1.26rc2 and enable the GOEXPERIMENT=simd env for all build/run/test commands or just infer the usage from the Taskfile (or use `task installGoRC`).

You will need to clone a safetensors model from HF and pack it using the CLI `bin/mantle pack -in /path/to/safetensors/ -out /path/to/model.mcf`,
You can also add the `--dedup` flag to remove duplicated tensors from the resulting model container. For most models, this does nothing, but in testing, Qwen3-0.6B has a duplicated tensor in attn and embd. and it reduced the model file by 300mb with no quality hit.

To run the model *only on CPU* you just need to use the `mantle run` command with `-m/--model`, or alternatively, pass in a directory of models to use with `MANTLE_MODELS_DIR=/path/to/models mantle run`.

### Build Requirements
- Go 1.26rc2
- AMD64 CPU (simd/archsimd will only compile for amd64)

### Build
```shell
GOEXPERIMENT=simd go1.26rc2 build -o bin/mantle ./cmd/mantle
````

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
