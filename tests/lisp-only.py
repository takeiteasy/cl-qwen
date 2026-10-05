"""Exercise both kernel engines with numerical libraries absent."""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

with tempfile.TemporaryDirectory(prefix="cl-qwen-lisp-") as temporary:
    directory = Path(temporary)
    ignore = shutil.ignore_patterns(".git", ".cache", "build", "*.fasl", "__pycache__")
    for name in ["cl-qwen", "trivial-simd"]:
        shutil.copytree(ROOT.parent / name, directory / name, ignore=ignore)
    project = directory / "cl-qwen"
    subprocess.run(["python3", project / "tests/fixtures.py", project / ".cache/fixtures"], check=True)
    environment = dict(os.environ, TRIVIAL_SIMD_BACKEND="lisp")
    subprocess.run(["sbcl", "--script", project / "tests/lisp-only.lisp"],
                   cwd=project, env=environment, check=True)
