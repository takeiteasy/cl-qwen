"""Prepare pinned official models and the CPU-only reference harness."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
MANIFEST = json.loads((ROOT / "tools/reference-manifest.json").read_text())


def run(*args):
    subprocess.run([str(x) for x in args], check=True)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=CACHE / "llama.cpp")
    args = parser.parse_args()
    CACHE.mkdir(exist_ok=True)
    reference = args.reference_dir
    pin = MANIFEST["reference"]
    if not reference.exists():
        run("git", "clone", "--no-checkout", "--filter=blob:none", pin["url"], reference)
        run("git", "-C", reference, "checkout", "--detach", pin["revision"])
    revision = subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != pin["revision"]:
        raise RuntimeError("Reference checkout differs from manifest; use a clean pinned checkout")
    q8 = MANIFEST["models"]["q8"]
    hf_hub_download(q8["repository"], q8["filename"], revision=q8["revision"], local_dir=CACHE / "models")
    original = MANIFEST["models"]["original"]
    for name in original["files"]:
        hf_hub_download(original["repository"], name, revision=original["revision"], local_dir=CACHE / "original")
    model = CACHE / "models/Qwen3-0.6B-F32.gguf"
    if not model.exists():
        import sys
        run(sys.executable, reference / "convert_hf_to_gguf.py", CACHE / "original",
            "--outfile", model, "--outtype", "f32")
    for name, expected in MANIFEST.get("checksums", {}).items():
        if digest(CACHE / "models" / name) != expected:
            raise RuntimeError(f"Artifact checksum mismatch: {name}")
    build = reference / "build"
    run("cmake", "-S", reference, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_METAL=OFF", "-DGGML_BLAS=OFF", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_SERVER=OFF")
    run("cmake", "--build", build, "-j", "4", "--target", "llama-completion", "llama-tokenize", "llama-quantize")
    run("clang++", "-std=c++17", "-O2", "-I", reference / "include", "-I", reference / "ggml/include",
        ROOT / "tools/reference.cpp", "-L", build / "bin", "-lllama", "-lggml-base",
        "-Wl,-rpath," + str(build / "bin"), "-o", CACHE / "reference")


if __name__ == "__main__":
    main()
