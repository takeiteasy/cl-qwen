# Architecture

The model and executor are Lisp. Native calls operate on numerical buffers,
not model, tokenizer or generation state.

| Layer | Responsibility |
|---|---|
| `cl-qwen/gguf` | Header/metadata/tensor validation and read-only mapping |
| Lisp model | BPE, Qwen layers, attention, KV cache, RoPE preparation, sampling and scheduling |
| `trivial-simd/blas` | F32 matrix-vector arithmetic |
| `trivial-simd` | Vector arithmetic, copies, conversion and attention reductions |
| Small C library | Q8_0 activation packing and packed-weight matvec |

## Storage and arithmetic

Weights use private flat descriptors: storage, encoding, rows and columns.
F32 and Q8_0 weights remain mapped; F16/BF16 weights widen to F32 once at load.
Norm weights are small F32 vectors. A missing output projection shares token
embedding weights.[^qwen]

Internal `(matvec weights activation output)` passes a destination. F32
weights dispatch to `sgemv`; Q8_0 weights dispatch to packed integer dot
products with F32 accumulation. Q8_0 matvec packs each 32-element activation
block with an F16 scale and nearest-even signed bytes, matching the pinned
ARM64 reference policy.[^quantization]

Activations and KV cache use F32. RMSNorm accumulates F32 squares in F64 before
rounding the mean to F32. Softmax subtracts the maximum and normalizes an F64
sum of F32 exponentials. Typed Lisp supplies softmax, SiLU and RoPE stages.
RoPE tables use recurrent F32 angles and are reused per session. Rotations
use trivial-simd scalar FMA with the reference operation order.

## Sessions and workers

Each session preallocates KV cache, logits, numerical scratch, packed activation
scratch and RoPE tables. Model closure prevents new calls and waits for active
session calls before unmapping. Sessions close their persistent worker pools.

Worker count defaults to one. Additional Lisp workers own contiguous matvec
row ranges and disjoint output ranges; all finish before outputs are consumed.
Worker exceptions propagate to the caller. Native code creates no threads.

Prefill runs the decode path one token at a time. Attention is causal, uses
Q/K normalization and grouped-query heads, and stays in model code. Dimensions
come from validated metadata: Qwen3-0.6B's 128-wide heads are independent of its
1,024-wide hidden state.[^qwen]

## Limitations

- General tensors, dtype operation tables and shared attention abstraction
  follow a second dense architecture, starting with Llama before MoE:
  [library extraction](https://todo.sr.ht/~takeiteasy/trivial-simd/132).
- Batch-size-one prefill and per-position attention dispatch limit throughput
  and allocation: [GEMM and attention work](https://todo.sr.ht/~takeiteasy/trivial-simd/133).
- Supported input is single-file, little-endian GGUF v3 with F32, F16, BF16
  and Q8_0 payloads. Scaled RoPE and other quantizations are rejected.
  Split models, additional formats and reduced-precision cache/compute are
  [follow-ups](https://todo.sr.ht/~takeiteasy/trivial-simd/134).
- Tested execution is SBCL/macOS ARM64; the scalar C path has local numerical
  tests, not whole-model coverage on other platforms:
  [portability](https://todo.sr.ht/~takeiteasy/trivial-simd/135).

[^qwen]: [Pinned Qwen configuration](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json)
    and [reference model implementation](https://github.com/ggml-org/llama.cpp/blob/b809b886d94107349f4e2b1a0d4713d8566565aa/src/models/qwen3.cpp).
[^quantization]: The [reference CPU type table](https://github.com/ggml-org/llama.cpp/blob/b809b886d94107349f4e2b1a0d4713d8566565aa/ggml/src/ggml-cpu/ggml-cpu.c)
    selects Q8_0 activation dot products for Q8_0 weights. Packed blocks contain
    an F16 scale and 32 signed bytes. Native weight layout is retained; only
    activations use reusable packing scratch. Scalar C follows the same
    nearest-even packing policy as the ARM64 path.
