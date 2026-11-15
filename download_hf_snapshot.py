"""Helper CLI to download a HuggingFace model snapshot cleanly (excluding legacy .bin/.pth duplicates)."""

import os
import sys
import time
from huggingface_hub import snapshot_download


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python download_hf_snapshot.py <repo_id>")
    repo_id = sys.argv[1]
    os.environ["HF_HUB_OFFLINE"] = "0"
    t0 = time.perf_counter()
    print(f"[HF Download] Starting snapshot_download for '{repo_id}'...", flush=True)
    snap_path = snapshot_download(
        repo_id=repo_id,
        local_files_only=False,
        ignore_patterns=["*.bin", "*.pth", "original/*", ".eval_results/*"],
    )
    st_files = [fn for fn in os.listdir(snap_path) if fn.endswith(".safetensors")]
    if not st_files:
        raise RuntimeError(
            f"Incomplete snapshot at '{snap_path}': 0 .safetensors files present (network blocked?)."
        )
    dt = time.perf_counter() - t0
    print(
        f"[HF Download] Completed '{repo_id}' ({len(st_files)} .safetensors shards) in {dt:.1f}s -> {snap_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
