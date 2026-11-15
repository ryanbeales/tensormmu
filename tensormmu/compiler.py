"""Unified Model Downloader & Streaming `.tmmu` / `.nvm` Compiler for TensorMMU.

Supports:
  1. Pre-flight GPU/RAM/Disk safety verification (`assess_compile_fit`) BEFORE downloading multi-GB weights.
  2. Automatic Hugging Face Hub download (`snapshot_download`) with progress callbacks for `/readyz`.
  3. Zero-Skeleton Sparse-MoE compilation (`nvm_moe_v1`, `< 5 GB` peak VRAM even on 80B-120B MoE models)
     and Dense Contrastive SVD+INT4/UINT4 compilation (`nvm_v1`).
  4. Optional post-compile `EagleLiteDraftHead` and `LookaheadRouterBank` calibration.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Any, Callable, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from data.calibration import build_contrastive_datasets
from decompiler.nvm_serializer import (
    _is_complete_hf_snapshot,
    compile_streaming_moe_to_nvm,
    compile_streaming_to_nvm,
    resolve_hf_snapshot_dir,
)
from tensormmu.hardware_safety import FitAssessment, _extract_text_cfg_dict, assess_compile_fit, fetch_or_load_hf_config

ProgressCallback = Callable[[str, str, Optional[float]], None]


def is_compiled_cartridge(path: str) -> bool:
    return bool(path) and os.path.isdir(path) and os.path.isfile(os.path.join(path, "manifest.json"))


def default_cartridge_dir_for_model(model_id_or_path: str, checkpoints_dir: str = "checkpoints") -> str:
    """Computes deterministic cartridge output path in `checkpoints_dir` for a HF model ID or local folder."""
    clean = model_id_or_path.rstrip("/\\")
    base = os.path.basename(clean) if os.path.isdir(clean) else clean.split("/")[-1]
    for suf in (".tmmu", ".nvm"):
        if base.endswith(suf):
            return os.path.abspath(os.path.join(checkpoints_dir, base))
    return os.path.abspath(os.path.join(checkpoints_dir, f"{base}.tmmu"))


def find_existing_cartridge(model_or_path: str, checkpoints_dir: str = "checkpoints") -> Optional[str]:
    """Checks if `model_or_path` already points to (or matches) a compiled `.tmmu` / `.nvm` cartridge."""
    if is_compiled_cartridge(model_or_path):
        return os.path.abspath(model_or_path)

    ckpt_abs = os.path.abspath(checkpoints_dir)
    if not os.path.isdir(ckpt_abs):
        return None

    short_name = model_or_path.rstrip("/\\").split("/")[-1]
    candidates = [
        os.path.join(ckpt_abs, model_or_path),
        os.path.join(ckpt_abs, f"{model_or_path}.tmmu"),
        os.path.join(ckpt_abs, f"{model_or_path}.nvm"),
        os.path.join(ckpt_abs, short_name),
        os.path.join(ckpt_abs, f"{short_name}.tmmu"),
        os.path.join(ckpt_abs, f"{short_name}.nvm"),
        os.path.join(ckpt_abs, f"{short_name}-nvm-moe.nvm"),
        os.path.join(ckpt_abs, f"{short_name}-tmmu.tmmu"),
    ]
    for cand in candidates:
        if is_compiled_cartridge(cand):
            return os.path.abspath(cand)
    return None


def _emit_progress(cb: Optional[ProgressCallback], stage: str, detail: str, pct: Optional[float] = None) -> None:
    print(f"[TensorMMU:{stage}] {detail}", flush=True)
    if cb is not None:
        cb(stage, detail, pct)


def download_hf_checkpoint_if_needed(
    model_id_or_path: str,
    on_progress: Optional[ProgressCallback] = None,
) -> str:
    """Resolves a local HF snapshot directory or downloads `.safetensors` + tokenizer files from Hugging Face Hub."""
    if os.path.isdir(model_id_or_path) and _is_complete_hf_snapshot(model_id_or_path):
        return os.path.abspath(model_id_or_path)

    # Check local HuggingFace cache first before hitting network
    try:
        cached = resolve_hf_snapshot_dir(model_id_or_path)
        if os.path.isdir(cached) and _is_complete_hf_snapshot(cached):
            _emit_progress(on_progress, "downloading", f"Found cached Hugging Face snapshot at `{cached}`.", 100.0)
            return os.path.abspath(cached)
    except Exception:
        pass

    from huggingface_hub import snapshot_download

    _emit_progress(
        on_progress,
        "downloading",
        f"Downloading `{model_id_or_path}` weights & tokenizer from Hugging Face Hub...",
        5.0,
    )
    ignore_patterns = ["*.bin", "*.pth", "*.pt", "*.gguf", "original/*", ".eval_results/*"]
    snap_dir = snapshot_download(
        repo_id=model_id_or_path,
        local_files_only=False,
        ignore_patterns=ignore_patterns,
    )
    _emit_progress(on_progress, "downloading", f"Download complete: `{snap_dir}`.", 100.0)
    return os.path.abspath(snap_dir)


def download_and_compile_model(
    model_id_or_path: str,
    out_dir: Optional[str] = None,
    checkpoints_dir: str = "checkpoints",
    device: str = "cuda:0",
    tlb_slots: Optional[int] = None,
    attn_quant: str = "int8",
    shared_mlp_quant: str = "int8",
    lm_head_quant: str = "auto",
    expert_group_size: int = 64,
    calibrate_drafter: bool = False,
    calibrate_lookahead: bool = False,
    force: bool = False,
    recompile: bool = False,
    on_progress: Optional[ProgressCallback] = None,
) -> str:
    """Downloads (if needed) and stream-compiles a Hugging Face model into a `.tmmu`/`.nvm` cartridge.

    Performs hardware & disk pre-flight checks *before* downloading weights unless `force=True`.
    """
    if not recompile:
        existing = find_existing_cartridge(out_dir or model_id_or_path, checkpoints_dir=checkpoints_dir)
        if existing is not None:
            _emit_progress(on_progress, "compiling", f"Using existing compiled cartridge `{existing}`.", 100.0)
            return existing

    target_out = os.path.abspath(out_dir) if out_dir else default_cartridge_dir_for_model(model_id_or_path, checkpoints_dir)

    _emit_progress(on_progress, "checking", f"Running GPU/RAM/Disk pre-flight check for `{model_id_or_path}`...", 1.0)
    fit: FitAssessment = assess_compile_fit(model_id_or_path=model_id_or_path, out_dir=target_out, device=device)
    print(fit.format_banner(), flush=True)
    if not fit.ok_to_proceed and not force:
        err_msg = "; ".join(fit.errors) or "Hardware pre-flight check failed."
        raise RuntimeError(f"TensorMMU pre-flight check blocked compilation (pass --force to override): {err_msg}")

    raw_cfg = fetch_or_load_hf_config(model_id_or_path)
    tc = _extract_text_cfg_dict(raw_cfg)
    num_experts = int(tc.get("num_experts") or tc.get("n_routed_experts") or 0)
    tied_emb = bool(tc.get("tie_word_embeddings", False))

    snap_dir = download_hf_checkpoint_if_needed(model_id_or_path, on_progress=on_progress)

    _emit_progress(on_progress, "compiling", f"Preparing contrastive calibration bundle for `{fit.model_label}`...", 10.0)
    tokenizer = AutoTokenizer.from_pretrained(snap_dir, local_files_only=True)
    # Try cached/online datasets if allowed, automatically falling back to deterministic built-in traces offline
    allow_dl = os.environ.get("HF_DATASETS_OFFLINE", "0") != "1"
    try:
        bundle = build_contrastive_datasets(
            tokenizer=tokenizer,
            num_calibration_seqs=16,
            num_eval_seqs=4,
            seq_len=256,
            seed=42,
            allow_hf_download=allow_dl,
        )
    except Exception:
        bundle = build_contrastive_datasets(
            tokenizer=tokenizer,
            num_calibration_seqs=16,
            num_eval_seqs=4,
            seq_len=256,
            seed=42,
            allow_hf_download=False,
        )

    eff_slots = int(tlb_slots) if tlb_slots is not None else int(fit.recommended_tlb_slots or (48 if num_experts > 1 else 24))
    t0 = time.perf_counter()

    if num_experts > 1:
        eff_lm_quant = ("none" if tied_emb else "int8") if lm_head_quant == "auto" else lm_head_quant
        _emit_progress(
            on_progress,
            "compiling",
            f"Stream-compiling Sparse-MoE ({num_experts} experts/layer, UINT4 pages, tlb_slots={eff_slots}) -> `{target_out}`...",
            20.0,
        )
        manifest = compile_streaming_moe_to_nvm(
            model_id_or_path=snap_dir,
            nvm_dir=target_out,
            bundle=bundle,
            tlb_slots=eff_slots,
            expert_group_size=expert_group_size,
            expert_chunk_size=min(64, num_experts),
            attn_quant=attn_quant,
            shared_mlp_quant=shared_mlp_quant,
            lm_head_quant=eff_lm_quant,
            calib_batch_size=4,
            max_calib_tokens=4096,
            device=device,
        )
    else:
        _emit_progress(
            on_progress,
            "compiling",
            f"Stream-compiling Dense LLM (`nvm_v1`, tlb_slots={eff_slots}) -> `{target_out}`...",
            20.0,
        )
        manifest = compile_streaming_to_nvm(
            model_id_or_path=snap_dir,
            nvm_dir=target_out,
            bundle=bundle,
            tlb_slots=eff_slots,
            attn_quant=attn_quant,
            calib_batch_size=4,
            max_calib_tokens=4096,
            device=device,
        )

    ct = manifest.get("compile_telemetry", {})
    _emit_progress(
        on_progress,
        "compiling",
        f"Cartridge compiled in {time.perf_counter() - t0:.1f}s "
        f"(Peak GPU={ct.get('peak_compile_vram_mb', 0.0):.0f} MiB, Total Disk={ct.get('total_disk_mb', 0.0) / 1024.0:.2f} GiB).",
        85.0 if (calibrate_drafter or calibrate_lookahead) else 100.0,
    )

    if calibrate_drafter or calibrate_lookahead:
        calib_seqs: List[torch.Tensor] = []
        for i in range(min(12, bundle.logic_calib.input_ids.shape[0])):
            calib_seqs.append(bundle.logic_calib.input_ids[i].clone())
        for i in range(min(12, bundle.fact_calib.input_ids.shape[0])):
            calib_seqs.append(bundle.fact_calib.input_ids[i].clone())

        if calibrate_lookahead and num_experts > 1:
            from runtime.moe_lookahead_router import calibrate_and_save_lookahead_routers

            _emit_progress(on_progress, "compiling", "Calibrating Cross-Layer Lookahead Router Bank...", 90.0)
            calibrate_and_save_lookahead_routers(
                nvm_dir=target_out,
                calib_token_sequences=calib_seqs,
                tlb_slots_override=eff_slots,
                device=device,
            )

        if calibrate_drafter:
            from runtime.speculative_vm import calibrate_and_save_eagle_draft_head

            _emit_progress(on_progress, "compiling", "Calibrating EagleLiteDraftHead speculative drafter...", 95.0)
            calibrate_and_save_eagle_draft_head(
                nvm_dir=target_out,
                calib_token_sequences=calib_seqs,
                tlb_slots_override=eff_slots,
                device=device,
            )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _emit_progress(on_progress, "compiling", f"Compilation complete: `{target_out}`.", 100.0)
    return target_out
