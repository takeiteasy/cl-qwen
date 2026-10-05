# Performance and memory

On Apple M1 with SBCL 2.6.8 and the native trivial-simd backend, Q8_0 reaches
about 30.5 decode tokens/s with three Lisp workers on the short measured prompt.
These are warm-cache measurements, not a whole-context throughput guarantee.

## Local measurements

| Weights | Workers | Load seconds | Prefill tokens/s | Decode tokens/s | Lisp MiB/decode step | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| F32 | 1 | 0.363 | 16.7 | 14.5 | 1.45 | 2392 |
| F32 | 3 | 0.394 | 17.6 | 16.7 | 1.52 | 2485 |
| Q8_0 | 1 | 0.362 | 24.5 | 22.0 | 1.44 | 849 |
| Q8_0 | 3 | 0.372 | 35.8 | 30.5 | 1.45 | 860 |

Each configuration uses one fresh process, one warm-up and three measured
trials. The prompt is `The capital of France is` (five tokens); each trial
measures sequential prefill and 16 greedy decode/consume steps with a
256-token F32 KV cache. Load excludes ASDF compilation and startup; decode
includes sampling. Configurations run sequentially.[^measurement]

```sh
/usr/bin/time -l sbcl --dynamic-space-size 4096 --script tools/benchmark.lisp \
  .cache/models/Qwen3-0.6B-Q8_0.gguf 3
```

Raw runs: [F32/one](benchmark-runs/2026-10-05-f32-1-workers.txt),
[F32/three](benchmark-runs/2026-10-05-f32-3-workers.txt),
[Q8_0/one](benchmark-runs/2026-10-05-q8-1-workers.txt) and
[Q8_0/three](benchmark-runs/2026-10-05-q8-3-workers.txt).

## Memory

The validated files map 3,012,480,736 bytes for F32 and 639,446,688 bytes for
Q8_0. Mapping size is file size, not private resident memory. F16/BF16 weights
need additional F32 storage when widened. Tokenizer tables, scratch and Lisp
runtime memory are additional to the mapping and cache.

For Qwen3-0.6B, the F32 K/V cache occupies 56 MiB at context 256 and 448 MiB
at the default context 2,048. Sessions allocate that capacity up front.[^cache]
Decode reuses numerical arrays; Lisp dispatch and scalar arithmetic still
allocate, as shown in the table.

## Limitations

Batch-size-one prefill and per-position attention calls limit throughput and
allocation efficiency. Blocked GEMM and lower-allocation attention are tracked
in [the performance follow-up](https://todo.sr.ht/~takeiteasy/trivial-simd/133).
The measurements cover a short context on one machine. CPU placement, thermal
state and background system load are unobserved. They establish no Lisp/C
runtime percentage or universal worker-count speedup.

[^measurement]: Wall time uses Lisp's internal real-time clock. Allocations
    use `sb-ext:get-bytes-consed`; macOS `/usr/bin/time -l` supplies process peak
    RSS. Reported throughput and allocation use trial medians. Raw files retain
    every trial and process counters.
[^cache]: Bytes = `2 * layers * context * KV-heads * head-size * 4`.
    The factor two accounts for keys and values. The cache uses F32 regardless
    of weight encoding.
