r"""CLI Wrapper for the TensorMMU OpenAI + Anthropic + Kubernetes Server (Backwards-Compatible Entrypoint).

Prefer the unified `tmmu` CLI (`python -m tensormmu`):
  tmmu check checkpoints/gemma-4-26B-A4B-it-nvm-moe.nvm
  tmmu compile google/gemma-4-26B-A4B-it
  tmmu serve --model checkpoints/gemma-4-26B-A4B-it-nvm-moe.nvm --port 8000
"""

from __future__ import annotations

import sys

from tensormmu.cli import main as tmmu_main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("serve", "compile", "check", "-h", "--help", "--version"):
        argv = ["serve", "--fail-fast", *argv]
    raise SystemExit(tmmu_main(argv))
