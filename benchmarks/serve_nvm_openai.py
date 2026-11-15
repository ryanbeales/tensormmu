r"""CLI Wrapper for the TensorMMU OpenAI + Anthropic + Kubernetes Server (Backwards-Compatible Entrypoint).

Prefer the unified `tmmu` CLI (`python -m tensormmu`):
  tmmu check checkpoints/gemma-4-26B-A4B-it-nvm-moe.nvm
  tmmu compile google/gemma-4-26B-A4B-it
  tmmu serve --model checkpoints/gemma-4-26B-A4B-it-nvm-moe.nvm --port 8000
"""

from __future__ import annotations
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

import sys

from tensormmu.cli import main as tmmu_main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("serve", "compile", "check", "-h", "--help", "--version"):
        argv = ["serve", "--fail-fast", *argv]
    raise SystemExit(tmmu_main(argv))
