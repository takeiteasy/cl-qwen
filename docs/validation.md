# Tests and reference validation

Small fixtures test loading and complete forward passes without downloading a
model. Full validation compares F32 and Q8_0 inference with pinned CPU
llama.cpp using the identical GGUF files and prompts.

## Recorded result

All 16 format/prompt cases match input and greedy output IDs for each of the
native, kernel and Lisp engines: 48 engine/format/prompt cases. One- and
three-worker runs agree. The Lisp suite passes 14,924 checks and both native C
builds pass. The pre-feature dependency baseline passes 440,971 SBCL checks
and five native checks.

Retained evidence: [Lisp suite](validation-runs/2026-10-05-engines-sbcl.txt),
[native tests](validation-runs/2026-10-05-engines-native.txt),
[no numerical libraries](validation-runs/2026-10-05-engines-no-native.txt),
[native corpus](validation-runs/2026-10-05-engines-reference-native.txt),
[kernel corpus](validation-runs/2026-10-05-engines-reference-kernel.txt),
[Lisp corpus](validation-runs/2026-10-05-engines-reference-lisp.txt) and
[dependency baseline](validation-runs/2026-10-05-dependency-baseline.txt).

## Regression tests

```sh
ctest --test-dir build --output-on-failure
sbcl --dynamic-space-size 4096 --script tests/run.lisp
python3 tests/lisp-only.py
```

The Lisp suite generates tiny deterministic GGUF models and independent
Python double-precision forward results. It covers conversions, normalization,
RoPE, grouped-query attention, tokenizer byte merges, Unicode, special tokens,
cache reset, context boundaries, worker errors and closure during active calls.
Malformed models cover truncated data, alignment, duplicate names, unsupported
encodings, architecture/head metadata and missing weights.
Engine checks cover exact multi-block Q8 matvec, activation packing, shared
weight caches, mixed-engine sessions and repeated worker jobs with garbage
collections. Native packing masks C floating-point traps while retaining its
explicit nonfinite/overflow validation.

`tests/lisp-only.py` copies both sibling checkouts into a temporary directory
without numerical builds. It checks both alternative engines with one and
three workers, and rejects a native Q8 session when its library is absent.

C tests exercise vector and forced-scalar builds: block boundaries, signed bytes,
output sentinels, empty dimensions, subnormal scales, activation rounding ties,
invalid widths and nonfinite activations.

## Reference corpus

```sh
.cache/validation-env/bin/python tools/prepare-reference.py
.cache/validation-env/bin/python tools/validate.py --engine native --output .cache/validation-native-engines
.cache/validation-env/bin/python tools/validate.py --engine kernel --output .cache/validation-kernel
.cache/validation-env/bin/python tools/validate.py --engine lisp --output .cache/validation-lisp
```

The corpus contains four raw and four non-thinking chat prompts: English,
arithmetic, Chinese/emoji/accented text, whitespace and punctuation. Each format
compares input IDs and up to 16 greedy output IDs. A three-worker run must
produce the same IDs and numerically equivalent logits.

Reference execution uses CPU, one thread, batch size one, F32 K/V cache and
flash attention disabled. The reference keeps its default optimized weight
buffers. `tools/reference.cpp` is a test harness, not a runtime dependency.
Every decoded step writes logits into the selected ignored output directory;
stdout, stderr, prompts and JSON results remain there for diagnosis.

F32 logits use `0.001 + 0.001 * abs(reference)` as the per-element bound. Q8_0
logit differences are recorded; quantization boundaries make small upstream
rounding differences nonlinear. Exact greedy IDs remain the acceptance gate.
Worker logits use absolute/relative tolerance `1e-6`.[^rounding]

For intermediate diagnostics, create two output directories and set
`CL_QWEN_TRACE` separately when invoking `.cache/reference` and
`tools/inspect.lisp`. Both dump named F32 intermediates at token position 1;
the Lisp dump includes normalization, RoPE and layer outputs.

Results and artifact hashes are retained in [native results](validation-results.json),
[kernel results](validation-results-kernel.json) and [Lisp results](validation-results-lisp.json).
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
