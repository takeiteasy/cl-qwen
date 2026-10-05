# Performance and memory

On Apple M1 with SBCL 2.6.8, three-worker Q8_0 decode measures
30.1 tokens/s for native, 1.06 for kernel and
0.77 for Lisp execution. Native is
28.3× faster than kernel and 38.9× faster than Lisp in this configuration.
These are warm-cache measurements, not a whole-context throughput guarantee.
See [execution engines](execution.md) for selection and numerical behavior.

## Local measurements

Native and kernel sessions select trivial-simd's native backend; Lisp sessions
force its Lisp backend. F32 native/kernel sessions use the same BLAS path.
Differences between those F32 timings reflect run variation.

### F32

| Engine | Workers | Load s | Prepare s | Prefill tokens/s | Decode tokens/s | Lisp MiB/step | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| [native](benchmark-runs/2026-10-05-engines-f32-native-1-workers.txt) | 1 | 0.361 | 0.002 | 16.72 | 15.26 | 1.45 | 2484 |
| [kernel](benchmark-runs/2026-10-05-engines-f32-kernel-1-workers.txt) | 1 | 0.369 | 0.002 | 16.67 | 15.36 | 1.45 | 2571 |
| [lisp](benchmark-runs/2026-10-05-engines-f32-lisp-1-workers.txt) | 1 | 0.374 | 0.002 | 0.74 | 0.74 | 2636.61 | 2548 |
| [native](benchmark-runs/2026-10-05-engines-f32-native-3-workers.txt) | 3 | 0.369 | 0.002 | 18.82 | 17.12 | 1.52 | 2575 |
| [kernel](benchmark-runs/2026-10-05-engines-f32-kernel-3-workers.txt) | 3 | 0.367 | 0.002 | 18.87 | 16.90 | 1.52 | 2575 |
| [lisp](benchmark-runs/2026-10-05-engines-f32-lisp-3-workers.txt) | 3 | 0.382 | 0.002 | 1.94 | 1.90 | 2635.12 | 2643 |

### Q8_0

| Engine | Workers | Load s | Prepare s | Prefill tokens/s | Decode tokens/s | Lisp MiB/step | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| [native](benchmark-runs/2026-10-05-engines-q8_0-native-1-workers.txt) | 1 | 0.352 | 0.003 | 23.44 | 20.58 | 1.44 | 742 |
| [kernel](benchmark-runs/2026-10-05-engines-q8_0-kernel-1-workers.txt) | 1 | 0.366 | 0.652 | 0.37 | 0.35 | 15.27 | 1579 |
| [lisp](benchmark-runs/2026-10-05-engines-q8_0-lisp-1-workers.txt) | 1 | 0.352 | 0.589 | 0.28 | 0.28 | 1145.39 | 1616 |
| [native](benchmark-runs/2026-10-05-engines-q8_0-native-3-workers.txt) | 3 | 0.349 | 0.004 | 35.71 | 30.05 | 1.45 | 901 |
| [kernel](benchmark-runs/2026-10-05-engines-q8_0-kernel-3-workers.txt) | 3 | 0.349 | 0.604 | 1.07 | 1.06 | 42.90 | 1593 |
| [lisp](benchmark-runs/2026-10-05-engines-q8_0-lisp-3-workers.txt) | 3 | 0.352 | 0.594 | 0.78 | 0.77 | 1162.10 | 1634 |

Each configuration uses one fresh process, one warm-up and three measured
trials. The prompt is `The capital of France is` (five tokens); each trial
measures sequential prefill and 16 greedy decode/consume steps with a
256-token F32 KV cache. Model load excludes ASDF compilation and startup.
Preparation includes session creation, workers and first-use split Q8 weights;
decode includes sampling. Configurations run sequentially.[^measurement]

```sh
/usr/bin/time -l sbcl --dynamic-space-size 4096 --script tools/benchmark.lisp \
  .cache/models/Qwen3-0.6B-Q8_0.gguf 3 lisp
```

The last argument selects `native` (default), `kernel` or `lisp`. Engine links
in the tables open raw runs; [machine-readable results](engine-benchmarks.json)
retain unrounded summaries and the actual numerical backend.

## Kernel spike comparison

The separate [Q8-like matvec spike](https://github.com/takeiteasy/trivial-simd/blob/f9ee3a2ea6033769abbdd3f4379db4d1c570aa9f/docs/q8-matvec.md)
measures generic kernels at 16.56–20.45% of direct C throughput, or
4.89–6.04× C latency. It uses a different arithmetic workload and measures
matvec alone. The tables above measure complete Qwen inference with quantized
activations, exact block dots and ordered scale/FMA accumulation; the spike
ratio does not predict whole-model engine throughput.

## Memory

The validated files map 3,012,480,736 bytes for F32 and 639,446,688 bytes for
Q8_0. Mapping size is file size, not private resident memory. F16/BF16 weights
need additional F32 storage when widened. Tokenizer tables, scratch and Lisp
runtime memory are additional to the mapping and cache.

For Qwen3-0.6B, the F32 K/V cache occupies 56 MiB at context 256 and 448 MiB
at the default context 2,048. Sessions allocate that capacity up front.[^cache]
Kernel/Lisp Q8 sessions add 639.4 MiB of shared signed-byte weights
and F32 block scales while retaining the mapping. The first such session pays
the preparation cost; later sessions reuse the cache. Worker block-result
scratch adds roughly one F32 value per row of the largest matvec, partitioned across workers.
Sessions reuse core numerical arrays; backend staging, dispatch and scalar
arithmetic still allocate, as shown in the tables.[^staging]

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

[^staging]: Lisp kernel rows stage foreign views and allocate dispatch data.
    These are temporary allocations, not additional retained weights or KV cache.
