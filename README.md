# cl-qwen

A demo and test of [trivial-simd](https://git.sr.ht/~takeiteasy/trivial-simd):
Qwen3-0.6B inference in Common Lisp, with mapped GGUF weights, Lisp model
execution and native, `define-kernel` or Lisp numerical engines. Vector and
float operations use `trivial-simd` and `trivial-simd/blas`.

This repository is archived and not maintained as an inference library. Its
tensor layer continues in [cl-tensor](https://git.sr.ht/~takeiteasy/cl-tensor).

The tested target is SBCL on macOS ARM64. Raw completion and single-turn chat
with thinking disabled support F32 and Q8_0 models.

```sh
sbcl --dynamic-space-size 4096 --script tools/cli.lisp \
  --model .cache/models/Qwen3-0.6B-Q8_0.gguf \
  --mode chat --prompt 'What is the capital of France?' --tokens 16
```

- [Setup and models](docs/setup.md)
- [Lisp API and CLI](docs/api.md)
- [Execution engines](docs/execution.md)
- [Architecture and supported scope](docs/architecture.md)
- [Tests and reference validation](docs/validation.md)
- [Performance and memory](docs/performance.md)

## License

[MIT](LICENSE). Downloaded model artifacts retain their upstream licenses.
