# cl-qwen

Qwen3-0.6B inference in Common Lisp, with mapped GGUF weights, Lisp model
execution and a small C Q8_0 arithmetic library. Float operations use
`trivial-simd/blas`; vector operations use `trivial-simd`.

The tested target is SBCL on macOS ARM64. Raw completion and single-turn chat
with thinking disabled support F32 and Q8_0 models.

```sh
sbcl --dynamic-space-size 4096 --script tools/cli.lisp \
  --model .cache/models/Qwen3-0.6B-Q8_0.gguf \
  --mode chat --prompt 'What is the capital of France?' --tokens 16
```

- [Setup and models](docs/setup.md)
- [Lisp API and CLI](docs/api.md)
- [Architecture and supported scope](docs/architecture.md)
- [Tests and reference validation](docs/validation.md)
- [Performance and memory](docs/performance.md)

## License

[MIT](LICENSE). Downloaded model artifacts retain their upstream licenses.
