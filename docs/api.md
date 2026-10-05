# Lisp API and CLI

Load a model once, create sessions with private scratch/cache, and close the
model when finished. Model closure waits for active session calls, joins their
workers and releases the mapping.

```lisp
(load "tools/bootstrap.lisp")
(let ((model (cl-qwen:load-model "model.gguf")))
  (unwind-protect
       (let ((session (cl-qwen:make-session model :context-size 256 :workers 1)))
         (multiple-value-bind (ids text)
             (cl-qwen:generate session "What is 2 + 2?" :chat t :max-tokens 16)
           (format t "~A~%~S~%" text ids)))
    (cl-qwen:close-model model)))
```

## Functions

| Call | Behavior |
|---|---|
| `(load-model path)` | Validates weights/metadata and owns a read-only mapping |
| `(close-model model)` | Waits for active calls, closes all sessions; repeated calls are harmless |
| `(model-metadata model)` | Returns the parsed metadata table; treat it as read-only |
| `(make-session model &key context-size workers engine)` | Defaults to 2,048 tokens, one worker and `:native` |
| `(close-session session)` | Joins workers and releases cache; repeated calls are harmless |
| `(reset-session session)` | Resets position to zero and returns the session |
| `(session-engine session)` | Returns `:native`, `:kernel` or `:lisp`; fixed at creation |
| `(session-position session)` | Returns the number of tokens consumed |
| `(tokenize model text &key special-tokens)` | Returns a simple unsigned-byte-32 token vector |
| `(detokenize model ids &key special-tokens)` | Returns UTF-8 text; hides special tokens by default |
| `(step! session token)` | Consumes one token and returns reusable F32 logits storage |
| `(generate session prompt &key max-tokens chat system)` | Resets, consumes prompt, greedily generates; returns token IDs and text |
| `(chat-prompt user &key system)` | Formats single-turn Qwen chat with thinking disabled |

`generate` accepts a string or token sequence; chat mode requires a string.
The default output limit is 32. End-of-generation tokens stop generation and
are omitted from returned IDs. Ties select the lowest ID.

Copy `step!`'s result if it must survive another step. Generation consumes each
sampled token needed to obtain the next logits; the last returned token is not
consumed. A zero output limit still consumes the prompt. Calls on one session
serialize; independent sessions share read-only weights.

An empty prompt, invalid token ID, invalid worker/context size or requested
context overflow signals an error. There is no implicit truncation or context
shift. Resetting hides old cache contents without clearing their allocation.

## Command line

```sh
sbcl --dynamic-space-size 4096 --script tools/cli.lisp \
  --model model.gguf --mode chat --prompt 'Hello' \
  --system 'Answer briefly.' --tokens 16 --context 256 --workers 3
```

`--engine` accepts `native` (default), `kernel` or `lisp`. See
[execution engines](execution.md) for numerical behavior and library requirements.

Use exactly one of `--prompt` and `--prompt-file`. `--mode` defaults to `raw`;
`--system` requires `chat`. `--help` lists options. The CLI prints completion
text and returns status 2 for invalid input or load/execution errors.

## Standalone GGUF reader

Load ASDF system `cl-qwen/gguf` without the model executor. `open-gguf` returns
metadata and tensor descriptors through `gguf-metadata` and `gguf-tensors`.
Descriptors expose name, dimensions, encoding, byte size and pointer. GGUF's
first dimension is contiguous. Pointers remain valid until `close-gguf`.[^reader]

## Limitations

Full templates, thinking modes, tools and multi-turn sessions are tracked in
[chat extensions](https://todo.sr.ht/~takeiteasy/trivial-simd/136).
Invalid UTF-8 token fragments decode with replacement characters; concatenate
IDs before detokenizing a sequence split across byte tokens.

[^reader]: The standalone reader is POSIX-based. Its callers own mapping
    lifetime and serialize closure themselves. The model API provides that
    synchronization for inference sessions.
