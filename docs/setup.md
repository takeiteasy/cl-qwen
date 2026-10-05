# Setup and models

Use SBCL, Quicklisp, Python 3, CMake and a C compiler on an ARM64 Mac. Keep
`cl-qwen` and `trivial-simd` as sibling checkouts; the scripts select those
checkouts before inherited ASDF configuration.

## Build

```sh
cmake -S ../trivial-simd -B ../trivial-simd/build -DCMAKE_BUILD_TYPE=Release
cmake --build ../trivial-simd/build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

`tools/bootstrap.lisp` loads CFFI, Bordeaux Threads, Babel, Unicode-aware
CL-PPCRE and trivial-simd through Quicklisp. FiveAM is a test dependency.
The cl-qwen numerical library is required for native Q8_0 sessions. Kernel and
Lisp sessions do not require it. For entirely Lisp numerical execution, skip
both builds and run with `TRIVIAL_SIMD_BACKEND=lisp` and `--engine lisp`.
See [execution engines](execution.md) for the alternatives and shared storage.

## Download reproducible models

The optional validation environment downloads the official Q8_0 artifact and
converts the original checkpoint to F32 GGUF. Models, reference builds and
Python packages stay under ignored `.cache/`.[^models]

```sh
python3 -m venv .cache/validation-env
.cache/validation-env/bin/pip install -r tools/reference-requirements.txt
.cache/validation-env/bin/python tools/prepare-reference.py
```

This prepares `.cache/models/Qwen3-0.6B-Q8_0.gguf`,
`.cache/models/Qwen3-0.6B-F32.gguf` and `.cache/reference`.
The reference revision and model hashes are in
[the manifest](../tools/reference-manifest.json).

## Run

```sh
sbcl --dynamic-space-size 4096 --script tools/cli.lisp \
  --model .cache/models/Qwen3-0.6B-Q8_0.gguf \
  --prompt 'The capital of France is' --tokens 8
```

The greedy continuation is ` Paris. The capital of France is also`.
For a Lisp example, run `examples/generate.lisp` with the model path as its
argument. See [the API](api.md) for session ownership and chat mode.

## Limitations

Other Lisps and platforms require validation and an appropriate array/mapping
bridge: [platform coverage](https://todo.sr.ht/~takeiteasy/trivial-simd/135).

[^models]: Official [Q8_0 repository](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)
    and [original checkpoint](https://huggingface.co/Qwen/Qwen3-0.6B). The model
    revisions are pinned; the code license does not replace upstream model
    licensing. Full reference preparation downloads several gigabytes.
