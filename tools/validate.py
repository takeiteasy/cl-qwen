"""Compare token IDs and per-step logits against the pinned CPU reference."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    ("raw-capital", "The capital of France is", False),
    ("raw-arithmetic", "2 + 2 =", False),
    ("raw-unicode", "你好！café 👋", False),
    ("raw-whitespace", "One\n\nTwo\t  three  ", False),
    ("chat-capital", "What is the capital of France?", True),
    ("chat-arithmetic", "What is 2 + 2? Answer briefly.", True),
    ("chat-unicode", "请用中文说你好。", True),
    ("chat-punctuation", "Finish this sequence: 1, 2, 3,", True),
]


def chat(text):
    return ("<|im_start|>user\n" + text + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def ids(stdout, label):
    return next([int(x) for x in line.split()[1:]] for line in stdout.splitlines() if line.startswith(label + " ") or line == label)


def run(command, prefix):
    result = subprocess.run([str(x) for x in command], text=True, capture_output=True, timeout=300)
    prefix.with_suffix(".out").write_text(result.stdout)
    prefix.with_suffix(".err").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed; see {prefix}.err")
    return {label.lower(): ids(result.stdout, label) for label in ["INPUT", "OUTPUT"]}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / ".cache/reference")
    parser.add_argument("--models", type=Path, default=ROOT / ".cache/models")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/validation")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--only", help="Run case names containing this substring")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"reference_revision": "b809b886d94107349f4e2b1a0d4713d8566565aa", "tokens": args.tokens,
              "reference_cache": "f32", "reference_threads": 1, "batch_size": 1, "models": {}, "cases": []}
    failures = []
    for kind in ["F32", "Q8_0"]:
        model = args.models / f"Qwen3-0.6B-{kind}.gguf"
        report["models"][kind] = {"sha256": digest(model), "bytes": model.stat().st_size}
        for name, text, is_chat in PROMPTS:
            name = kind.lower() + "-" + name
            if args.only and args.only not in name:
                continue
            prompt = args.output / (name + ".txt")
            prompt.write_text(chat(text) if is_chat else text)
            ref_prefix = args.output / (name + "-reference")
            lisp_prefix = args.output / (name + "-lisp")
            ref_logits = ref_prefix.with_suffix(".logits")
            lisp_logits = lisp_prefix.with_suffix(".logits")
            reference = run([args.reference, model, prompt, int(is_chat), args.tokens, ref_logits], ref_prefix)
            actual = run(["sbcl", "--dynamic-space-size", "4096", "--script", ROOT / "tools/inspect.lisp",
                          model, prompt, int(is_chat), args.tokens, lisp_logits], lisp_prefix)
            a = np.fromfile(lisp_logits, dtype="<f4").reshape(-1, 151936)
            b = np.fromfile(ref_logits, dtype="<f4").reshape(-1, 151936)
            same_tokens = actual == reference
            case = {"name": name, "model": kind, "prompt": text, "chat": is_chat,
                    "input_ids": actual["input"], "output_ids": actual["output"],
                    "reference_output_ids": reference["output"], "token_ids_match": same_tokens}
            common = min(len(a), len(b))
            delta = np.abs(a[:common].astype(np.float64) - b[:common])
            case.update(max_logit_error=float(np.max(delta)), mean_logit_error=float(np.mean(delta)),
                        lisp_steps=len(a), reference_steps=len(b))
            if kind == "F32":
                case["logits_within_tolerance"] = bool(np.all(delta <= 0.001 + 0.001 * np.abs(b[:common])))
            worker_prefix = args.output / (name + "-workers")
            worker_logits = worker_prefix.with_suffix(".logits")
            parallel = run(["sbcl", "--dynamic-space-size", "4096", "--script", ROOT / "tools/inspect.lisp",
                            model, prompt, int(is_chat), args.tokens, worker_logits, 3], worker_prefix)
            c = np.fromfile(worker_logits, dtype="<f4")
            case["workers_match"] = parallel == actual and c.shape == a.ravel().shape and bool(np.allclose(c, a.ravel(), atol=1e-6, rtol=1e-6))
            if not same_tokens or not case["workers_match"] or not case.get("logits_within_tolerance", True):
                failures.append(name)
            report["cases"].append(case)
            print(name, "PASS" if name not in failures else "FAIL", "max logits", case["max_logit_error"], flush=True)
    report["failures"] = failures
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if failures:
        raise SystemExit("Reference failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
