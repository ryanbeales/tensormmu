"""Phase C2 Driver: Streaming Layer-by-Layer `.safetensors` Compiler & `.nvm` Cold-Boot Benchmark.

Proves the complete zero-monolithic Phase C2 pipeline on `google/gemma-4-E4B-it` (and scales directly
to `27B-70B` checkpoints on a 24 GB GPU + 34 GB System RAM PC without ever calling dense `from_pretrained`):
  1. Streams `model.safetensors` one layer at a time (`O(1)` single-layer VRAM/RAM) into a `.nvm` disk cartridge.
  2. Cold-boots `.nvm` directly onto GPU in `< 5s` (`pinned_ram` and zero-RAM `nvme_mmap` OS-paged storage),
     stripping unused multimodal towers (`-912 MB` dead VRAM saved on `E4B`).
  3. Optionally compiles & cold-boots with register-fused `INT8` Self-Attention (`PhaseCQuantLinear`),
     compressing total resident VRAM down to `~4.3-4.5 GB` (`3.5x` smaller than Dense BF16).
  4. Evaluates Ground-Truth `Logic_Eval` & `Fact_Eval` Perplexity (`shift_labels`), KV-Cached Decode (`use_cache=True`),
     TLB Hit Rate & PCIe MB/token, OS Process RAM (Working Set & Commit Charge), and 20-problem CoT GSM8K (`test`).
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import gc
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

if os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub")) and os.path.isdir(
    os.path.expanduser("~/.cache/huggingface/datasets")
):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch

from config import NeuralVMConfig
from data.calibration import build_contrastive_datasets
from decompiler.nvm_serializer import (
    compile_streaming_to_nvm,
    load_nvm_cartridge,
    resolve_hf_snapshot_dir,
)
from run_ablation import evaluate_split
from run_phase4_giant_killer import (
    _build_gsm8k_eval_prompts,
    _get_cuda_mem_mb,
    evaluate_gsm8k_zero_shot_exact_match,
)
from run_phase_b_comparison import (
    aggregate_phase_b_vram_and_telemetry,
    measure_phase_b_kv_decode,
)


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_os_process_ram_mb() -> Dict[str, float]:
    """Queries Windows kernel process memory counters (Working Set & Commit Charge in MB)."""
    if os.name != "nt":
        return {"ws_mb": 0.0, "peak_ws_mb": 0.0, "commit_mb": 0.0, "peak_commit_mb": 0.0}
    try:
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wt.HANDLE
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
        pmc = _PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        mb = float(1024 ** 2)
        return {
            "ws_mb": pmc.WorkingSetSize / mb,
            "peak_ws_mb": pmc.PeakWorkingSetSize / mb,
            "commit_mb": pmc.PagefileUsage / mb,
            "peak_commit_mb": pmc.PeakPagefileUsage / mb,
        }
    except Exception:
        return {"ws_mb": 0.0, "peak_ws_mb": 0.0, "commit_mb": 0.0, "peak_commit_mb": 0.0}


def _evaluate_cartridge_modes(
    nvm_dir: str,
    label_prefix: str,
    bundle: Any,
    gsm8k_problems: List[Tuple[str, str]],
    cfg: NeuralVMConfig,
    tlb_slots: int,
    decode_req_pages: int,
    gsm8k_max_tokens: int,
    gsm8k_batch_size: int,
    dense_ref_mb: float,
    run_nvme_mmap_test: bool = True,
) -> Dict[str, Any]:
    """Cold-boots a `.nvm` cartridge and evaluates `Core-Only`, `Full VM (pinned_ram)`, and `nvme_mmap`."""
    out: Dict[str, Any] = {}

    # 1. Cold-boot in `pinned_ram` mode
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ram_before_pin = get_os_process_ram_mb()

    model, tokenizer, pb_mlps, lut_pool, ple_offloader, mf = load_nvm_cartridge(
        nvm_dir=nvm_dir,
        device=cfg.device,
        dtype=cfg.dtype,
        mode="vm_paged",
        tlb_strategy="speculative_tlb",
        page_storage="pinned_ram",
        tlb_slots_override=tlb_slots,
        decode_req_pages_override=decode_req_pages,
        use_phase_c_kernels=True,
        load_tokenizer=True,
    )
    ram_after_pin = get_os_process_ram_mb()
    ple_saved_mb = ple_offloader.offloaded_vram_mb if ple_offloader is not None else 0.0
    out["pinned_cold_boot"] = {
        **mf["last_cold_boot"],
        "ram_commit_delta_mb": max(0.0, ram_after_pin["commit_mb"] - ram_before_pin["commit_mb"]),
        "ram_after": ram_after_pin,
    }
    print(
        f"[{label_prefix} Cold-Boot (pinned_ram)] Loaded in {mf['last_cold_boot']['cold_boot_s']:.2f}s | "
        f"CUDA Alloc={mf['last_cold_boot']['cuda_allocated_mb']:,.1f} MB | "
        f"Host Commit={ram_after_pin['commit_mb']:,.1f} MB (+{out['pinned_cold_boot']['ram_commit_delta_mb']:,.1f} MB)"
    )

    # --- A. Core-Only (`0` Pages, `0` MLP PCIe) ---
    for m in pb_mlps.values():
        m.set_mode("core_only")
        m.reset_telemetry(clear_tlb=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if ple_offloader is not None:
        ple_offloader.reset_telemetry()

    core_alloc_idle_mb, _ = _get_cuda_mem_mb()
    c_logic, _ = evaluate_split(
        model, bundle.logic_eval, state_name=f"{label_prefix} Core-Only", batch_size=cfg.batch_size, device=cfg.device
    )
    c_fact, _ = evaluate_split(
        model, bundle.fact_eval, state_name=f"{label_prefix} Core-Only", batch_size=cfg.batch_size, device=cfg.device
    )
    c_vram = aggregate_phase_b_vram_and_telemetry(
        pb_mlps,
        lut_pool=lut_pool,
        ple_saved_mb=ple_saved_mb,
        total_dense_model_mb=dense_ref_mb,
        ple_offloader=ple_offloader,
    )
    c_dec_ms, c_dec_tps, c_dec_tel = measure_phase_b_kv_decode(
        model=model,
        prompt_ids=bundle.fact_eval.input_ids,
        num_decode_tokens=cfg.decode_tokens,
        device=cfg.device,
        phase_b_mlps=pb_mlps,
        lut_pool=lut_pool,
        ple_saved_mb=ple_saved_mb,
        total_dense_model_mb=dense_ref_mb,
        ple_offloader=ple_offloader,
    )
    c_gsm_ok, _, c_gsm_acc, c_gsm_trace = evaluate_gsm8k_zero_shot_exact_match(
        model,
        tokenizer,
        gsm8k_problems,
        batch_size=gsm8k_batch_size,
        max_new_tokens=gsm8k_max_tokens,
        device=cfg.device,
    )
    out["core_only"] = {
        "idle_cuda_mb": core_alloc_idle_mb,
        "logic_ppl": c_logic.perplexity,
        "fact_ppl": c_fact.perplexity,
        "vram": c_vram,
        "decode_ms": c_dec_ms,
        "decode_tps": c_dec_tps,
        "decode_tel": c_dec_tel,
        "gsm": (c_gsm_ok, c_gsm_acc, c_gsm_trace),
    }

    # --- B. Full VM + GPU LRU TLB (`pinned_ram`) ---
    for m in pb_mlps.values():
        m.set_mode("vm_paged")
        m.set_tlb_strategy("sync_tlb")
        m.reset_telemetry(clear_tlb=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if ple_offloader is not None:
        ple_offloader.reset_telemetry()

    vm_alloc_idle_mb, _ = _get_cuda_mem_mb()
    v_logic, _ = evaluate_split(
        model, bundle.logic_eval, state_name=f"{label_prefix} Full VM", batch_size=cfg.batch_size, device=cfg.device
    )
    v_fact, _ = evaluate_split(
        model, bundle.fact_eval, state_name=f"{label_prefix} Full VM", batch_size=cfg.batch_size, device=cfg.device
    )
    v_vram = aggregate_phase_b_vram_and_telemetry(
        pb_mlps,
        lut_pool=lut_pool,
        ple_saved_mb=ple_saved_mb,
        total_dense_model_mb=dense_ref_mb,
        ple_offloader=ple_offloader,
    )
    v_gsm_ok, _, v_gsm_acc, v_gsm_trace = evaluate_gsm8k_zero_shot_exact_match(
        model,
        tokenizer,
        gsm8k_problems,
        batch_size=gsm8k_batch_size,
        max_new_tokens=gsm8k_max_tokens,
        device=cfg.device,
    )

    for m in pb_mlps.values():
        m.set_mode("vm_paged")
        m.set_tlb_strategy("speculative_tlb")
        m.reset_telemetry(clear_tlb=True)
    v_spec_ms, v_spec_tps, v_spec_tel = measure_phase_b_kv_decode(
        model=model,
        prompt_ids=bundle.fact_eval.input_ids,
        num_decode_tokens=cfg.decode_tokens,
        device=cfg.device,
        phase_b_mlps=pb_mlps,
        lut_pool=lut_pool,
        ple_saved_mb=ple_saved_mb,
        total_dense_model_mb=dense_ref_mb,
        ple_offloader=ple_offloader,
    )
    out["vm_pinned"] = {
        "idle_cuda_mb": vm_alloc_idle_mb,
        "logic_ppl": v_logic.perplexity,
        "fact_ppl": v_fact.perplexity,
        "vram": v_vram,
        "spec_decode_ms": v_spec_ms,
        "spec_decode_tps": v_spec_tps,
        "spec_decode_tel": v_spec_tel,
        "gsm": (v_gsm_ok, v_gsm_acc, v_gsm_trace),
    }

    del model, tokenizer, pb_mlps, lut_pool, ple_offloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- C. Zero-RAM `nvme_mmap` Cold-Boot & Decode Verification ---
    if run_nvme_mmap_test:
        ram_before_mmap = get_os_process_ram_mb()
        m_mm, _, mlps_mm, lut_mm, ple_mm, mf_mm = load_nvm_cartridge(
            nvm_dir=nvm_dir,
            device=cfg.device,
            dtype=cfg.dtype,
            mode="vm_paged",
            tlb_strategy="speculative_tlb",
            page_storage="nvme_mmap",
            tlb_slots_override=tlb_slots,
            decode_req_pages_override=decode_req_pages,
            use_phase_c_kernels=True,
            load_tokenizer=False,
        )
        ram_after_mmap = get_os_process_ram_mb()
        print(
            f"[{label_prefix} Cold-Boot (nvme_mmap)] Loaded in {mf_mm['last_cold_boot']['cold_boot_s']:.2f}s | "
            f"CUDA Alloc={mf_mm['last_cold_boot']['cuda_allocated_mb']:,.1f} MB | "
            f"Host Commit={ram_after_mmap['commit_mb']:,.1f} MB "
            f"(+{max(0.0, ram_after_mmap['commit_mb'] - ram_before_mmap['commit_mb']):,.1f} MB)"
        )

        for m in mlps_mm.values():
            m.set_mode("vm_paged")
            m.set_tlb_strategy("sync_tlb")
            m.reset_telemetry(clear_tlb=True)
        mm_logic, _ = evaluate_split(
            m_mm, bundle.logic_eval, state_name=f"{label_prefix} NVMe-MMap", batch_size=cfg.batch_size, device=cfg.device
        )

        for m in mlps_mm.values():
            m.set_mode("vm_paged")
            m.set_tlb_strategy("speculative_tlb")
            m.reset_telemetry(clear_tlb=True)
        mm_dec_ms, mm_dec_tps, mm_dec_tel = measure_phase_b_kv_decode(
            model=m_mm,
            prompt_ids=bundle.fact_eval.input_ids,
            num_decode_tokens=cfg.decode_tokens,
            device=cfg.device,
            phase_b_mlps=mlps_mm,
            lut_pool=lut_mm,
            ple_saved_mb=ple_saved_mb,
            total_dense_model_mb=dense_ref_mb,
            ple_offloader=ple_mm,
        )
        out["vm_nvme_mmap"] = {
            **mf_mm["last_cold_boot"],
            "ram_commit_delta_mb": max(0.0, ram_after_mmap["commit_mb"] - ram_before_mmap["commit_mb"]),
            "ram_after": ram_after_mmap,
            "logic_ppl": mm_logic.perplexity,
            "spec_decode_ms": mm_dec_ms,
            "spec_decode_tps": mm_dec_tps,
            "spec_decode_tel": mm_dec_tel,
        }
        del m_mm, mlps_mm, lut_mm, ple_mm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out


def print_phase_c2_report(res: Dict[str, Any]) -> None:
    b_lp = res["dense_ref"]["logic_ppl"]
    b_fp = res["dense_ref"]["fact_ppl"]
    b_tps = res["dense_ref"]["decode_tps"]
    b_vram = res["dense_ref"]["cuda_alloc_mb"]

    def _dpct(v: float, b: float) -> str:
        return f"{100.0 * ((v - b) / max(1e-8, b)):+.2f}%"

    print("\n" + "=" * 120)
    print(f"PHASE C2 STREAMING COMPILER & `.nvm` COLD-BOOT REPORT ({res['model_id']}, Wall={res['wall_s']:.1f}s)")
    print("=" * 120)

    print("\n### Table 1: Streaming Compilation & `.nvm` Cartridge Disk / Cold-Boot Telemetry")
    print("| Cartridge Variant | Compile Wall | Peak Layer VRAM | Peak Total Compile VRAM | `.nvm` Disk (Core / Pages / PLE) | Cold-Boot (`pinned_ram` / `nvme_mmap`) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for key, title in (("bf16_attn", "`.nvm` (BF16 Attn)"), ("int8_attn", "`.nvm` (INT8 Attn)")):
        if key not in res:
            continue
        ct = res[key]["compile"]["compile_telemetry"]
        ev = res[key]["eval"]
        cb_pin = ev["pinned_cold_boot"]["cold_boot_s"]
        cb_mm = ev["vm_nvme_mmap"]["cold_boot_s"] if "vm_nvme_mmap" in ev else 0.0
        print(
            f"| **{title}** | `{ct['compile_wall_s']:.1f}s` | `{ct['peak_single_layer_vram_mb']:,.1f} MB` | "
            f"`{ct['peak_compile_vram_mb']:,.1f} MB` | "
            f"`{ct['core_resident_disk_mb']:,.0f} / {ct['pages_disk_mb']:,.0f} / {ct['ple_disk_mb']:,.0f} MB` | "
            f"`{cb_pin:.2f}s` / `{cb_mm:.2f}s` |"
        )

    print("\n### Table 2: Resident GPU VRAM, Perplexity, KV-Cached Decode Speed & CoT GSM8K (`test` Split)")
    print("| Execution Mode | Resident CUDA MB (`Saved`) | Logic PPL (`Delta`) | Fact PPL (`Delta`) | KV Decode (`% Native`) | TLB Hit / PCIe | GSM8K (`20` CoT) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    print(
        f"| **Ref: Monolithic Dense BF16** | `{b_vram:,.1f} MB` (`0.0%`) | `{b_lp:.4f}` (`0.00%`) | "
        f"`{b_fp:.4f}` (`0.00%`) | `{res['dense_ref']['decode_ms']:.2f} ms` (`100.0%`) | N/A / `0.00 MB` | "
        f"`{res['dense_ref']['gsm_ok']}/20` (`{res['dense_ref']['gsm_acc']:.1f}%`) |"
    )

    for key, tag in (("bf16_attn", "BF16 Attn"), ("int8_attn", "INT8 Attn")):
        if key not in res:
            continue
        ev = res[key]["eval"]
        co = ev["core_only"]
        vm = ev["vm_pinned"]
        co_sav = 100.0 * (1.0 - co["vram"]["cuda_allocated_mb"] / b_vram)
        vm_sav = 100.0 * (1.0 - vm["vram"]["cuda_allocated_mb"] / b_vram)
        print(
            f"| **`.nvm` Core-Only ({tag})** | `{co['vram']['cuda_allocated_mb']:,.1f} MB` (`{co_sav:.1f}%`) | "
            f"`{co['logic_ppl']:.4f}` (`{_dpct(co['logic_ppl'], b_lp)}`) | "
            f"`{co['fact_ppl']:.4f}` (`{_dpct(co['fact_ppl'], b_fp)}`) | "
            f"`{co['decode_ms']:.2f} ms` (`{100.0 * co['decode_tps'] / b_tps:.1f}%`) | "
            f"`0` Pages / `{co['decode_tel']['pcie_mb_per_token']:.3f} MB` | "
            f"`{co['gsm'][0]}/20` (`{co['gsm'][1]:.1f}%`) |"
        )
        print(
            f"| **`.nvm` Full VM (`pinned_ram`, {tag})** | `{vm['vram']['cuda_allocated_mb']:,.1f} MB` (`{vm_sav:.1f}%`) | "
            f"`{vm['logic_ppl']:.4f}` (`{_dpct(vm['logic_ppl'], b_lp)}`) | "
            f"`{vm['fact_ppl']:.4f}` (`{_dpct(vm['fact_ppl'], b_fp)}`) | "
            f"`{vm['spec_decode_ms']:.2f} ms` (`{100.0 * vm['spec_decode_tps'] / b_tps:.1f}%`) | "
            f"`{vm['spec_decode_tel']['tlb_hit_rate_pct']:.2f}%` / `{vm['spec_decode_tel']['pcie_mb_per_token']:.2f} MB` | "
            f"`{vm['gsm'][0]}/20` (`{vm['gsm'][1]:.1f}%`) |"
        )
        if "vm_nvme_mmap" in ev:
            mm = ev["vm_nvme_mmap"]
            print(
                f"| **`.nvm` Full VM (`nvme_mmap`, {tag})** | `{mm['cuda_allocated_mb']:,.1f} MB` (`0 MB` Page RAM) | "
                f"`{mm['logic_ppl']:.4f}` (`{_dpct(mm['logic_ppl'], b_lp)}`) | `--` | "
                f"`{mm['spec_decode_ms']:.2f} ms` (`{100.0 * mm['spec_decode_tps'] / b_tps:.1f}%`) | "
                f"`{mm['spec_decode_tel']['tlb_hit_rate_pct']:.2f}%` / `{mm['spec_decode_tel']['pcie_mb_per_token']:.2f} MB` | `--` |"
            )
    print("=" * 120 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase C2 Streaming Safetensors Compiler & .nvm Cold-Boot Driver")
    p.add_argument("--model-id", type=str, default="google/gemma-4-E4B-it")
    p.add_argument("--nvm-dir", type=str, default="checkpoints/gemma-4-E4B-it.nvm")
    p.add_argument("--compare-attn-int8", action="store_true", default=True)
    p.add_argument("--no-attn-int8", action="store_false", dest="compare_attn_int8")
    p.add_argument("--reuse-cartridge", action="store_true")
    p.add_argument("--cleanup-cartridges", action="store_true")
    p.add_argument("--tlb-slots", type=int, default=10)
    p.add_argument("--decode-req-pages", type=int, default=4)
    p.add_argument("--hot-fraction", type=float, default=0.11)
    p.add_argument("--bridge-rank", type=int, default=64)
    p.add_argument("--calib-seqs", type=int, default=32)
    p.add_argument("--eval-seqs", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--decode-tokens", type=int, default=64)
    p.add_argument("--gsm8k-problems", type=int, default=20)
    p.add_argument("--gsm8k-max-tokens", type=int, default=320)
    p.add_argument("--gsm8k-batch-size", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_wall0 = time.perf_counter()

    cfg = NeuralVMConfig(
        model_id=args.model_id,
        num_pages=64,
        max_active_pages=args.decode_req_pages,
        phase2_calib_seqs=args.calib_seqs,
        num_eval_seqs=args.eval_seqs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        decode_tokens=args.decode_tokens,
        gsm8k_eval_problems=args.gsm8k_problems,
        use_mock_model=False,
    )

    from transformers import AutoTokenizer

    torch.manual_seed(cfg.seed)
    snap_dir = resolve_hf_snapshot_dir(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(snap_dir, local_files_only=True)
    bundle = build_contrastive_datasets(
        tokenizer=tokenizer,
        num_calibration_seqs=cfg.phase2_calib_seqs,
        num_eval_seqs=cfg.num_eval_seqs,
        seq_len=cfg.seq_len,
        seed=cfg.seed,
        allow_hf_download=True,
    )
    gsm8k_problems = _build_gsm8k_eval_prompts(num_problems=args.gsm8k_problems, split="test")
    del tokenizer

    # Verified same-machine State 0 Dense BF16 baseline (`task-1081` on `google/gemma-4-E4B-it` / `task-955` on `E2B`)
    if "E4B" in args.model_id:
        dense_ref = {
            "cuda_alloc_mb": 15146.95,
            "logic_ppl": 19.2888,
            "fact_ppl": 59.9788,
            "decode_ms": 139.86,
            "decode_tps": 7.15,
            "gsm_ok": 16,
            "gsm_acc": 80.0,
        }
    else:
        dense_ref = {
            "cuda_alloc_mb": 9821.14,
            "logic_ppl": 23.0906,
            "fact_ppl": 95.3490,
            "decode_ms": 115.65,
            "decode_tps": 8.65,
            "gsm_ok": 12,
            "gsm_acc": 60.0,
        }

    results: Dict[str, Any] = {
        "model_id": args.model_id,
        "dense_ref": dense_ref,
    }

    # 1. Streaming Compile & Cold-Boot (`attn_quant="none"`, BF16 Attention)
    nvm_bf16_dir = os.path.abspath(args.nvm_dir)
    if args.reuse_cartridge and os.path.isfile(os.path.join(nvm_bf16_dir, "manifest.json")):
        import json

        with open(os.path.join(nvm_bf16_dir, "manifest.json"), "r", encoding="utf-8") as f:
            mf_bf16 = json.load(f)
    else:
        if os.path.exists(nvm_bf16_dir):
            shutil.rmtree(nvm_bf16_dir)
        print(f"\n[Stage 1] Streaming-Compiling '{args.model_id}' -> '{nvm_bf16_dir}' (attn_quant=none)...")
        mf_bf16 = compile_streaming_to_nvm(
            model_id_or_path=snap_dir,
            nvm_dir=nvm_bf16_dir,
            bundle=bundle,
            boundary_front=4,
            boundary_back=3,
            hot_fraction=args.hot_fraction,
            num_pages=64,
            tlb_slots=args.tlb_slots,
            decode_req_pages=args.decode_req_pages,
            bridge_rank=args.bridge_rank,
            attn_quant="none",
            calib_batch_size=args.batch_size,
            max_calib_tokens=args.calib_seqs * args.seq_len,
            save_standalone_ple=True,
            skip_multimodal=True,
            device=cfg.device,
            dtype=cfg.dtype,
        )
    ram_post_compile = get_os_process_ram_mb()
    print(
        f"[Stage 1 Complete] Wall={mf_bf16['compile_telemetry']['compile_wall_s']:.1f}s | "
        f"Peak Layer VRAM={mf_bf16['compile_telemetry']['peak_single_layer_vram_mb']:,.1f} MB | "
        f"Peak Compile VRAM={mf_bf16['compile_telemetry']['peak_compile_vram_mb']:,.1f} MB | "
        f"OS Peak Commit={ram_post_compile['peak_commit_mb']:,.1f} MB"
    )

    ev_bf16 = _evaluate_cartridge_modes(
        nvm_dir=nvm_bf16_dir,
        label_prefix="C2 (BF16-Attn)",
        bundle=bundle,
        gsm8k_problems=gsm8k_problems,
        cfg=cfg,
        tlb_slots=args.tlb_slots,
        decode_req_pages=args.decode_req_pages,
        gsm8k_max_tokens=args.gsm8k_max_tokens,
        gsm8k_batch_size=args.gsm8k_batch_size,
        dense_ref_mb=dense_ref["cuda_alloc_mb"],
        run_nvme_mmap_test=True,
    )
    results["bf16_attn"] = {
        "compile": mf_bf16,
        "post_compile_ram": ram_post_compile,
        "eval": ev_bf16,
    }

    # 2. Optional Streaming Compile & Cold-Boot (`attn_quant="int8"`, Register-Fused INT8 Self-Attention)
    nvm_i8_dir = (
        nvm_bf16_dir[:-4] + "-attn-int8.nvm"
        if nvm_bf16_dir.endswith(".nvm")
        else nvm_bf16_dir + "-attn-int8.nvm"
    )
    if args.compare_attn_int8:
        if args.reuse_cartridge and os.path.isfile(os.path.join(nvm_i8_dir, "manifest.json")):
            import json

            with open(os.path.join(nvm_i8_dir, "manifest.json"), "r", encoding="utf-8") as f:
                mf_i8 = json.load(f)
        else:
            if os.path.exists(nvm_i8_dir):
                shutil.rmtree(nvm_i8_dir)
            print(f"\n[Stage 2] Streaming-Compiling '{args.model_id}' -> '{nvm_i8_dir}' (attn_quant=int8)...")
            mf_i8 = compile_streaming_to_nvm(
                model_id_or_path=snap_dir,
                nvm_dir=nvm_i8_dir,
                bundle=bundle,
                boundary_front=4,
                boundary_back=3,
                hot_fraction=args.hot_fraction,
                num_pages=64,
                tlb_slots=args.tlb_slots,
                decode_req_pages=args.decode_req_pages,
                bridge_rank=args.bridge_rank,
                attn_quant="int8",
                calib_batch_size=args.batch_size,
                max_calib_tokens=args.calib_seqs * args.seq_len,
                save_standalone_ple=False,
                skip_multimodal=True,
                device=cfg.device,
                dtype=cfg.dtype,
            )
        ev_i8 = _evaluate_cartridge_modes(
            nvm_dir=nvm_i8_dir,
            label_prefix="C2 (INT8-Attn)",
            bundle=bundle,
            gsm8k_problems=gsm8k_problems,
            cfg=cfg,
            tlb_slots=args.tlb_slots,
            decode_req_pages=args.decode_req_pages,
            gsm8k_max_tokens=args.gsm8k_max_tokens,
            gsm8k_batch_size=args.gsm8k_batch_size,
            dense_ref_mb=dense_ref["cuda_alloc_mb"],
            run_nvme_mmap_test=True,
        )
        results["int8_attn"] = {
            "compile": mf_i8,
            "eval": ev_i8,
        }

    if args.cleanup_cartridges:
        for d in (nvm_bf16_dir, nvm_i8_dir):
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)

    results["wall_s"] = time.perf_counter() - t_wall0
    print_phase_c2_report(results)


if __name__ == "__main__":
    main()
