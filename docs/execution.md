# Execution engines

Choose an engine per session. `native` is the default; `kernel` and `lisp`
provide Q8_0 inference without cl-qwen's custom numerical library.
See [performance and memory](performance.md) for measured costs.

| Engine | Q8_0 matvec | F32 matvec | Other vector operations |
|---|---|---|---|
| `native` | Packed-block C | trivial-simd BLAS | Selected trivial-simd backend |
| `kernel` | `define-kernel` block dots, Lisp accumulation | trivial-simd BLAS | Selected trivial-simd backend |
| `lisp` | Lisp-executed `define-kernel` block dots, Lisp accumulation | Lisp-executed row-dot kernel | Lisp backend |

`kernel` describes the implementation, not the execution language: its kernels
use C when trivial-simd selects its native backend. `lisp` executes numerical
work in Lisp, including in worker threads. Mapped-file access still uses CFFI
and operating-system services; SBCL can emit CPU FMA instructions.[^startup]

## Select an engine

```lisp
(let ((session (cl-qwen:make-session model :engine :lisp :context-size 256 :workers 3)))
  (cl-qwen:session-engine session)) ; => :LISP
```

```sh
sbcl --dynamic-space-size 4096 --script tools/cli.lisp \
  --model .cache/models/Qwen3-0.6B-Q8_0.gguf \
  --engine kernel --mode chat --prompt 'Hello' --tokens 16
```

Engines are fixed at session creation. Independent sessions can use different
engines on the same model. Worker jobs inherit their session's numerical
backend; selection does not change another session or the process default.
`--mode raw|chat` selects prompt formatting separately from `--engine`.
Unknown engines signal an error; there is no automatic engine fallback.

## Q8 arithmetic and storage

All engines quantize activations in 32-value blocks with nearest-even signed
bytes and F16 scales. Kernel engines compute exact block integer dots using
F32 values, then apply block scales and accumulate with ordered FMA in Lisp.
They retain the native path's activation policy and operation order.
Lisp-executed attention dots also preserve the ARM64 native dot's four-lane
summation order, including kernel sessions selecting the Lisp backend.

The first kernel/Lisp session prepares signed-byte weights and F32 block
scales under the model lock. Sessions share those arrays, including a shared
embedding/output projection. Preparation adds 36 bytes per 32 weights while
the original 34-byte mapped blocks remain available. Reset keeps the cache;
closing the model releases it. Sessions reuse activation and worker scratch.

Native Q8 sessions require `build/libcl_qwen_numeric.dylib` on macOS; model
loading alone does not require it. Native-library errors are reported when a
native Q8 session is created.

## Limitations

Engines can produce different floating-point reductions outside Q8 matvec.
Exact greedy-token validation covers the recorded corpus, not every prompt.
See [validation limitations](validation.md#limitations).

[^startup]: trivial-simd loads its numerical library at startup when present.
    To start without numerical libraries, omit both numerical builds and set
    `TRIVIAL_SIMD_BACKEND=lisp`. This also makes model-load conversions use Lisp.
    Without an in-process FMA implementation, Lisp uses exact portable FMA.
