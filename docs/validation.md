# Tests and reference validation

Small fixtures test loading and complete forward passes without downloading a
model. Full validation compares F32 and Q8_0 inference with pinned CPU
llama.cpp using the identical GGUF files and prompts.

## Recorded result

All 16 format/prompt cases match input and greedy output IDs. One- and
three-worker runs agree. The Lisp suite passes 3,204 checks and both native C
builds pass. The pre-feature dependency baseline passes 440,971 SBCL checks
and five native checks.

Retained evidence (trailing whitespace removed): [Lisp suite](validation-runs/2026-10-05-sbcl.txt),
[native tests](validation-runs/2026-10-05-native.txt),
[reference corpus](validation-runs/2026-10-05-reference.txt) and
[dependency baseline](validation-runs/2026-10-05-dependency-baseline.txt).

## Regression tests

```sh
ctest --test-dir build --output-on-failure
sbcl --dynamic-space-size 4096 --script tests/run.lisp
```

The Lisp suite generates tiny deterministic GGUF models and independent
Python double-precision forward results. It covers conversions, normalization,
RoPE, grouped-query attention, tokenizer byte merges, Unicode, special tokens,
cache reset, context boundaries, worker errors and closure during active calls.
Malformed models cover truncated data, alignment, duplicate names, unsupported
encodings, architecture/head metadata and missing weights.

C tests exercise vector and forced-scalar builds: block boundaries, signed bytes,
output sentinels, empty dimensions, subnormal scales, activation rounding ties,
invalid widths and nonfinite activations.

## Reference corpus

```sh
.cache/validation-env/bin/python tools/prepare-reference.py
.cache/validation-env/bin/python tools/validate.py
```

The corpus contains four raw and four non-thinking chat prompts: English,
arithmetic, Chinese/emoji/accented text, whitespace and punctuation. Each format
compares input IDs and up to 16 greedy output IDs. A three-worker run must
produce the same IDs and numerically equivalent logits.

Reference execution uses CPU, one thread, batch size one, F32 K/V cache and
flash attention disabled. The reference keeps its default optimized weight
buffers. `tools/reference.cpp` is a test harness, not a runtime dependency.
Every decoded step writes logits into ignored `.cache/validation/`; stdout,
stderr, prompts and JSON results remain there for diagnosis.

F32 logits use `0.001 + 0.001 * abs(reference)` as the per-element bound. Q8_0
logit differences are recorded; quantization boundaries make small upstream
rounding differences nonlinear. Exact greedy IDs remain the acceptance gate.
Worker logits use absolute/relative tolerance `1e-6`.[^rounding]

For intermediate diagnostics, create two output directories and set
`CL_QWEN_TRACE` separately when invoking `.cache/reference` and
`tools/inspect.lisp`. Both dump named F32 intermediates at token position 1;
the Lisp dump includes normalization, RoPE and layer outputs.

Results and artifact hashes are retained in [validation-results.json](validation-results.json).
The run commands are pinned by [reference-manifest.json](../tools/reference-manifest.json).

## Limitations

The fixed corpus establishes this demo's coverage, not bitwise equality with
all llama.cpp builds or all prompts. Quantized activation rounding and backend
accumulation can affect near-tied token choices. Other formats and platforms
have [separate coverage work](https://todo.sr.ht/~takeiteasy/trivial-simd/134)
and [platform validation](https://todo.sr.ht/~takeiteasy/trivial-simd/135).

[^rounding]: Partitioning small F32 matrices can select different trivial-simd
    scalar/native size thresholds. Their results agree numerically, while
    floating-point accumulation can differ by a few ULPs. Greedy output is
    compared exactly rather than by decoded text alone.
