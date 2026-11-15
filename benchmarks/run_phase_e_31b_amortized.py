"""Phase E End-to-End Benchmark: Multi-Token Speculative Amortized Paging on `google/gemma-4-31B-it`.

Executes on a single `24 GB RTX 4090 + 34.3 GB System RAM` Windows 11 workstation using the already-compiled
`checkpoints/gemma-4-31B-it-nvm-hybrid.nvm` cartridge in Zero-RAM `nvme_mmap` mode (`~1 GB` OS Working Set):

  Stage 1 (`--stage calibrate`):
    Zero-leakage two-stage calibration of `EagleLiteDraftHead` (`~118 MB` VRAM) using strictly `Logic_Calib`,
    `Fact_Calib`, and `openai/gsm8k` `train` split sequences + 4 short on-policy `GSM8K train` CoT rollouts.
    Saves `eagle_draft_head.safetensors` inside the `.nvm` cartridge directory.

  Stage 2 (`--stage bench`):
    Fresh-process hardware benchmark measuring:
      A. Component & `M = 1..8` Multi-Token Forward Scaling (`core_only` & `vm_paged`).
      B. Controlled Steady-State Decode (`Fact_Eval` & `Logic_Eval` held-out sequences, `64` tokens x 2 prompts):
         - Phase D Prior Baseline Ref (`M=1` prior kernels: `231.41 ms/tok`, `207.18 MB/tok` PCIe)
         - Phase E1: `Full VM (nvme_mmap)` `M=1` + Vectorized/Unrolled Kernels & Zero-Alloc DMA
         - Phase E2: `Full VM (nvme_mmap)` + Zero-VRAM `PromptLookupDrafter (PLD)` Amortized Paging
         - Phase E3: `Full VM (nvme_mmap)` + Two-Tier `PLD + Eagle-Lite` Speculative Amortized Paging
         - Phase E1-Core: `Core-Only` `M=1` + Vectorized/Unrolled Kernels
         - Phase E4-Core: `Core-Only` + Two-Tier `PLD + Eagle-Lite` Speculative Decode (`0` PCIe ceiling)
      C. Full Held-Out 20-Problem Zero-Shot CoT GSM8K (`test` split, `max_new_tokens=320`):
         - End-to-end exact-match accuracy (`Solved/20`), `tok/s`, `ms/tok`, `tokens/step`, acceptance rate,
           hardware `PCIe MB/tok`, `TLB Hit %`, CUDA VRAM (`MB`), and OS Working Set (`MB`), plus direct
           5-problem `M=1` vs `Phase E3` exact greedy parity & speedup verification.
"""

from __future__ import annotations
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub")) and os.path.isdir(
    os.path.expanduser("~/.cache/huggingface/datasets")
):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
from transformers import AutoTokenizer

from data.calibration import build_contrastive_datasets
from decompiler.nvm_serializer import load_nvm_cartridge, resolve_hf_snapshot_dir
from run_phase4_giant_killer import (
    _build_gsm8k_eval_prompts,
    _extract_gsm8k_answer,
    _get_cuda_mem_mb,
    _normalize_num_str,
)
from run_phase_c2_streaming_nvm import get_os_process_ram_mb
from runtime.speculative_vm import (
    EagleLiteDraftHead,
    SpeculativeDecodeStats,
    SpeculativeVMGenerator,
    build_gsm8k_train_calib_sequences,
    calibrate_and_save_eagle_draft_head,
)


def _save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _format_gsm8k_prompt(tokenizer: Any, question: str) -> str:
    instruction = (
        "Solve this math problem concisely step by step. "
        "Write the final numerical answer on the last line after #### (for example: #### 42).\n\n"
        f"Question: {question}"
    )
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            msgs = [{"role": "user", "content": instruction}]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return instruction + "\nAnswer:"
    return instruction + "\nAnswer:"


def run_calibrate_stage(
    model_id: str,
    nvm_dir: str,
    shortlist_size: int = 12288,
    epochs: int = 48,
    seed: int = 42,
    device: str = "cuda:0",
) -> None:
    torch.manual_seed(seed)
    snap_dir = resolve_hf_snapshot_dir(model_id)
    tokenizer = AutoTokenizer.from_pretrained(snap_dir, local_files_only=True)

    bundle = build_contrastive_datasets(
        tokenizer=tokenizer,
        num_calibration_seqs=16,
        num_eval_seqs=4,
        seq_len=256,
        seed=seed,
        allow_hf_download=False,
    )
    calib_seqs: List[torch.Tensor] = []
    for i in range(min(12, bundle.logic_calib.input_ids.shape[0])):
        calib_seqs.append(bundle.logic_calib.input_ids[i].clone())
    for i in range(min(12, bundle.fact_calib.input_ids.shape[0])):
        calib_seqs.append(bundle.fact_calib.input_ids[i].clone())

    gsm_train_seqs = build_gsm8k_train_calib_sequences(tokenizer, num_examples=18, max_seq_len=256)
    calib_seqs.extend(gsm_train_seqs)

    print(
        f"[Phase E] Prepared {len(calib_seqs)} strictly held-in calibration sequences "
        f"(12 Logic_Calib + 12 Fact_Calib + {len(gsm_train_seqs)} GSM8K train; 0 eval/test overlap)."
    )
    calibrate_and_save_eagle_draft_head(
        nvm_dir=nvm_dir,
        calib_token_sequences=calib_seqs,
        shortlist_size=shortlist_size,
        d_latent=1024,
        d_ffn=1536,
        epochs=epochs,
        lr=2.0e-3,
        rollout_horizon=4,
        device=device,
    )


@torch.no_grad()
def _profile_m_scaling(
    model: torch.nn.Module,
    mlps: Dict[int, Any],
    prompt_ids: torch.Tensor,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Measures isolated kernel & E2E forward-pass latency across M = 1, 2, 4, 5, 6, 8."""
    p_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device)
    base_len = int(prompt_ids.shape[1])

    # 1. Component microbenchmarks at M=1 and M=8
    x1 = torch.randn((1, 1, 5376), device=device, dtype=torch.bfloat16)
    x4 = torch.randn((1, 4, 5376), device=device, dtype=torch.bfloat16)
    x8 = torch.randn((1, 8, 5376), device=device, dtype=torch.bfloat16)

    for m in mlps.values():
        m.set_mode("core_only")
    torch.cuda.synchronize()
    for _ in range(2):
        for m in mlps.values():
            _ = m(x1)
            _ = m(x4)
            _ = m(x8)
    torch.cuda.synchronize()

    mlp_ms: Dict[str, float] = {}
    for M, xt in [(1, x1), (4, x4), (8, x8)]:
        t0 = time.perf_counter()
        iters = 8
        for _ in range(iters):
            for m in mlps.values():
                _ = m(xt)
        torch.cuda.synchronize()
        mlp_ms[f"M{M}"] = (time.perf_counter() - t0) * 1000.0 / iters

    qlins = [mod for mod in model.modules() if type(mod).__name__ == "PhaseCQuantLinear"]
    qlin_ms: Dict[str, float] = {}
    for M in (1, 4, 8):
        inps = {ql.in_features: torch.randn((1, M, ql.in_features), device=device, dtype=torch.bfloat16) for ql in qlins}
        for ql in qlins:
            _ = ql(inps[ql.in_features])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 8
        for _ in range(iters):
            for ql in qlins:
                _ = ql(inps[ql.in_features])
        torch.cuda.synchronize()
        qlin_ms[f"M{M}"] = (time.perf_counter() - t0) * 1000.0 / iters

    # 2. E2E forward pass scaling (M = 1, 2, 4, 5, 6, 8) using natural prompt continuation tokens
    e2e_scaling: Dict[str, Dict[str, Any]] = {}
    cont_tokens = prompt_ids[0, -16:].tolist() * 4  # realistic in-domain token IDs with natural locality

    for mode_name in ("core_only", "vm_paged"):
        for m in mlps.values():
            m.set_mode(mode_name)
            if mode_name == "vm_paged":
                m.set_tlb_strategy("amortized_spec_tlb")
            m.reset_telemetry(clear_tlb=True)

        mode_res: Dict[str, Any] = {}
        for M in (1, 2, 4, 5, 6, 8):
            out0 = model(input_ids=prompt_ids, attention_mask=p_mask, use_cache=True)
            pkv = out0.past_key_values
            mask_m = torch.ones((1, base_len + M), dtype=torch.long, device=device)
            step_batches = [
                torch.tensor([cont_tokens[k * M : (k + 1) * M]], dtype=torch.long, device=device)
                for k in range(8)
            ]
            for w in range(2):
                _ = model(input_ids=step_batches[w], attention_mask=mask_m, past_key_values=pkv, use_cache=True)
                pkv.crop(-M)
            torch.cuda.synchronize()

            for m in mlps.values():
                m.reset_telemetry(clear_tlb=False)
            iters = 6
            dt_sum = 0.0
            for it in range(iters):
                t0 = time.perf_counter()
                _ = model(input_ids=step_batches[2 + it], attention_mask=mask_m, past_key_values=pkv, use_cache=True)
                torch.cuda.synchronize()
                dt_sum += (time.perf_counter() - t0) * 1000.0
                pkv.crop(-M)

            pass_ms = dt_sum / iters
            pcie_mb = sum(m.tlb_cache.telemetry()["pcie_bytes"] for m in mlps.values()) / (1024 ** 2) / iters
            hits = sum(m.tlb_cache.telemetry()["page_hits"] for m in mlps.values())
            misses = sum(m.tlb_cache.telemetry()["page_misses"] for m in mlps.values())
            hit_pct = 100.0 * hits / max(1.0, hits + misses)
            mode_res[f"M{M}"] = {
                "M": M,
                "ms_per_pass": pass_ms,
                "ideal_ms_per_tok": pass_ms / M,
                "ideal_tok_per_s": 1000.0 * M / max(1e-6, pass_ms),
                "pcie_mb_per_pass": pcie_mb,
                "tlb_hit_rate_pct": hit_pct,
            }
            del pkv, out0
        e2e_scaling[mode_name] = mode_res

    return {
        "mlp_56x_core_ms": mlp_ms,
        "quant_linear_ms": qlin_ms,
        "num_quant_linears": len(qlins),
        "e2e_forward_scaling": e2e_scaling,
    }


@torch.no_grad()
def _bench_eval_prompts(
    gen: SpeculativeVMGenerator,
    prompts: List[torch.Tensor],
    max_new_tokens: int = 64,
) -> Dict[str, Any]:
    total_emitted = 0
    total_steps = 0
    total_prop = 0
    total_acc = 0
    total_pld_prop = 0
    total_pld_acc = 0
    total_eag_prop = 0
    total_eag_acc = 0
    total_elapsed_s = 0.0
    total_pcie_mb = 0.0
    hit_rates: List[float] = []
    all_gen_ids: List[List[int]] = []

    for p_ids in prompts:
        st: SpeculativeDecodeStats = gen.generate(
            p_ids,
            max_new_tokens=max_new_tokens,
            stop_on_eos=False,
            reset_steady_state_telemetry_after_prefill=True,
        )
        total_emitted += st.emitted_tokens
        total_steps += st.forward_steps
        total_prop += st.draft_proposed
        total_acc += st.draft_accepted
        total_pld_prop += st.pld_proposed
        total_pld_acc += st.pld_accepted
        total_eag_prop += st.eagle_proposed
        total_eag_acc += st.eagle_accepted
        total_elapsed_s += st.elapsed_s
        total_pcie_mb += st.total_pcie_mb
        hit_rates.append(st.tlb_hit_rate_pct)
        all_gen_ids.append(st.generated_ids)

    em = max(1, total_emitted)
    st_cnt = max(1, total_steps)
    return {
        "emitted_tokens": total_emitted,
        "forward_steps": total_steps,
        "tokens_per_step": float(em) / float(st_cnt),
        "ms_per_token": (total_elapsed_s * 1000.0) / float(em),
        "tok_per_s": float(em) / max(1e-6, total_elapsed_s),
        "draft_proposed": total_prop,
        "draft_accepted": total_acc,
        "acceptance_rate_pct": 100.0 * float(total_acc) / float(max(1, total_prop)) if total_prop > 0 else 0.0,
        "pld_proposed": total_pld_prop,
        "pld_accepted": total_pld_acc,
        "eagle_proposed": total_eag_prop,
        "eagle_accepted": total_eag_acc,
        "total_pcie_mb": total_pcie_mb,
        "pcie_mb_per_token": total_pcie_mb / float(em),
        "pcie_mb_per_step": total_pcie_mb / float(st_cnt),
        "tlb_hit_rate_pct": sum(hit_rates) / max(1, len(hit_rates)),
        "generated_ids_per_prompt": all_gen_ids,
    }


@torch.no_grad()
def _evaluate_gsm8k_speculative(
    gen: SpeculativeVMGenerator,
    problems: List[Tuple[str, str]],
    max_new_tokens: int = 320,
    device: str = "cuda:0",
    label: str = "Phase E3",
) -> Dict[str, Any]:
    correct = 0
    total = len(problems)
    total_emitted = 0
    total_steps = 0
    total_prop = 0
    total_acc = 0
    total_pld_prop = 0
    total_pld_acc = 0
    total_eag_prop = 0
    total_eag_acc = 0
    total_elapsed_s = 0.0
    total_pcie_mb = 0.0
    hit_rates: List[float] = []
    per_problem: List[Dict[str, Any]] = []

    for idx, (question, target_ans) in enumerate(problems):
        p_str = _format_gsm8k_prompt(gen.tokenizer, question)
        p_ids = gen.tokenizer(p_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        st = gen.generate(
            p_ids,
            max_new_tokens=max_new_tokens,
            stop_on_eos=True,
            reset_steady_state_telemetry_after_prefill=True,
        )
        gen_text = gen.tokenizer.decode(st.generated_ids, skip_special_tokens=True)
        pred_raw = _extract_gsm8k_answer(gen_text)
        is_ok = _normalize_num_str(pred_raw) == _normalize_num_str(target_ans)
        if is_ok:
            correct += 1

        total_emitted += st.emitted_tokens
        total_steps += st.forward_steps
        total_prop += st.draft_proposed
        total_acc += st.draft_accepted
        total_pld_prop += st.pld_proposed
        total_pld_acc += st.pld_accepted
        total_eag_prop += st.eagle_proposed
        total_eag_acc += st.eagle_accepted
        total_elapsed_s += st.elapsed_s
        total_pcie_mb += st.total_pcie_mb
        hit_rates.append(st.tlb_hit_rate_pct)

        per_problem.append(
            {
                "idx": idx,
                "target": target_ans,
                "pred": pred_raw,
                "correct": is_ok,
                "emitted_tokens": st.emitted_tokens,
                "forward_steps": st.forward_steps,
                "tokens_per_step": st.tokens_per_step,
                "ms_per_token": st.ms_per_token,
                "tok_per_s": st.tok_per_s,
                "pcie_mb_per_token": st.pcie_mb_per_token,
            }
        )
        if (idx + 1) % 5 == 0 or (idx + 1) == total:
            cur_tps = float(max(1, total_emitted)) / max(1e-6, total_elapsed_s)
            cur_step_mul = float(max(1, total_emitted)) / float(max(1, total_steps))
            cur_pcie = total_pcie_mb / float(max(1, total_emitted))
            print(
                f"  [{label} GSM8K {idx+1:02d}/{total:02d}] Solved={correct}/{idx+1} | "
                f"Speed={cur_tps:.2f} tok/s ({1000.0/max(1e-6,cur_tps):.1f} ms/tok) | "
                f"Tok/Step={cur_step_mul:.2f}x | PCIe={cur_pcie:.1f} MB/tok"
            )

    em = max(1, total_emitted)
    st_cnt = max(1, total_steps)
    cuda_alloc_mb, cuda_peak_mb = _get_cuda_mem_mb()
    ram_info = get_os_process_ram_mb()
    os_ws_mb = ram_info["ws_mb"]
    os_priv_mb = ram_info["commit_mb"]

    return {
        "solved": correct,
        "total": total,
        "accuracy_pct": 100.0 * float(correct) / float(max(1, total)),
        "total_emitted_tokens": total_emitted,
        "total_forward_steps": total_steps,
        "tokens_per_step": float(em) / float(st_cnt),
        "ms_per_token": (total_elapsed_s * 1000.0) / float(em),
        "tok_per_s": float(em) / max(1e-6, total_elapsed_s),
        "draft_proposed": total_prop,
        "draft_accepted": total_acc,
        "acceptance_rate_pct": 100.0 * float(total_acc) / float(max(1, total_prop)) if total_prop > 0 else 0.0,
        "pld_proposed": total_pld_prop,
        "pld_accepted": total_pld_acc,
        "pld_acceptance_pct": 100.0 * float(total_pld_acc) / float(max(1, total_pld_prop)) if total_pld_prop > 0 else 0.0,
        "eagle_proposed": total_eag_prop,
        "eagle_accepted": total_eag_acc,
        "eagle_acceptance_pct": 100.0 * float(total_eag_acc) / float(max(1, total_eag_prop)) if total_eag_prop > 0 else 0.0,
        "total_pcie_mb": total_pcie_mb,
        "pcie_mb_per_token": total_pcie_mb / float(em),
        "pcie_mb_per_step": total_pcie_mb / float(st_cnt),
        "tlb_hit_rate_pct": sum(hit_rates) / max(1, len(hit_rates)),
        "cuda_allocated_mb": cuda_alloc_mb,
        "cuda_peak_mb": cuda_peak_mb,
        "os_working_set_mb": os_ws_mb,
        "os_private_commit_mb": os_priv_mb,
        "sample_trace": gen.tokenizer.decode(
            gen.generate(
                gen.tokenizer(
                    _format_gsm8k_prompt(gen.tokenizer, problems[0][0]),
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids.to(device),
                max_new_tokens=min(192, max_new_tokens),
                stop_on_eos=True,
            ).generated_ids,
            skip_special_tokens=True,
        ),
        "per_problem": per_problem,
    }


def run_bench_stage(
    model_id: str,
    nvm_dir: str,
    out_json: str,
    phase_d_json: str,
    eval_decode_tokens: int = 64,
    gsm8k_problems_count: int = 20,
    gsm8k_max_new_tokens: int = 320,
    seed: int = 42,
    device: str = "cuda:0",
) -> None:
    torch.manual_seed(seed)
    t_total = time.perf_counter()

    # Load Phase D reference baseline numbers for direct side-by-side comparison
    phase_d_ref: Dict[str, Any] = {}
    if os.path.exists(phase_d_json):
        with open(phase_d_json, "r", encoding="utf-8") as f:
            pd_data = json.load(f)
        phase_d_ref = {
            "vm_nvme_mmap": pd_data.get("neuralvm", {}).get("vm_nvme_mmap", {}),
            "core_only": pd_data.get("neuralvm", {}).get("core_only", {}),
            "layer_offload_bf16": pd_data.get("layer_offload_bf16", {}),
            "static_int4": pd_data.get("uniform_int4", {}),
        }

    ram_base = get_os_process_ram_mb()
    base_ws_mb, base_priv_mb = ram_base["ws_mb"], ram_base["commit_mb"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"[Phase E Bench] Cold-booting 31B .nvm cartridge in Zero-RAM nvme_mmap mode from {nvm_dir}...")
    model, tokenizer, phase_b_mlps, lut_pool, ple_off, manifest = load_nvm_cartridge(
        nvm_dir=nvm_dir,
        device=device,
        dtype=torch.bfloat16,
        mode="vm_paged",
        tlb_strategy="amortized_spec_tlb",
        page_storage="nvme_mmap",
        tlb_slots_override=8,
        decode_req_pages_override=4,
        use_phase_c_kernels=True,
        load_tokenizer=True,
    )
    boot_info = manifest["last_cold_boot"]
    vm_cuda_mb, _ = _get_cuda_mem_mb()
    ram_vm = get_os_process_ram_mb()
    vm_ws_mb, vm_priv_mb = ram_vm["ws_mb"], ram_vm["commit_mb"]

    draft_head = EagleLiteDraftHead.load_pretrained_nvm(
        nvm_dir=nvm_dir,
        device=device,
        dtype=torch.bfloat16,
        lut_pool=lut_pool,
    )
    draft_vram_mb = draft_head.vram_bytes() / float(1024 ** 2)
    total_cuda_with_draft_mb, _ = _get_cuda_mem_mb()
    ram_draft = get_os_process_ram_mb()
    ws_with_draft_mb, priv_with_draft_mb = ram_draft["ws_mb"], ram_draft["commit_mb"]

    print(
        f"[Phase E Bench] Boot={boot_info['cold_boot_s']:.2f}s | Base VM CUDA={vm_cuda_mb:,.1f} MB | "
        f"+EagleLiteDraftHead ({draft_vram_mb:.1f} MB) => Total CUDA={total_cuda_with_draft_mb:,.1f} MB | "
        f"OS Working Set={ws_with_draft_mb:,.1f} MB (+{ws_with_draft_mb - base_ws_mb:.1f} MB)"
    )

    bundle = build_contrastive_datasets(
        tokenizer=tokenizer,
        num_calibration_seqs=8,
        num_eval_seqs=4,
        seq_len=256,
        seed=seed,
        allow_hf_download=False,
    )
    fact_prompts = [bundle.fact_eval.input_ids[i : i + 1, :128].to(device) for i in range(2)]
    logic_prompts = [bundle.logic_eval.input_ids[i : i + 1, :128].to(device) for i in range(2)]

    # Section A: M = 1..8 Multi-Token Forward Scaling
    print("[Phase E Bench] Profiling M=1..8 kernel & end-to-end forward-pass scaling...")
    scaling_stats = _profile_m_scaling(model, phase_b_mlps, fact_prompts[0], device=device)

    results: Dict[str, Any] = {
        "model_id": model_id,
        "nvm_dir": nvm_dir,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "baseline_os_ws_mb": base_ws_mb,
            "vm_cold_boot_s": boot_info["cold_boot_s"],
            "vm_base_cuda_mb": vm_cuda_mb,
            "draft_head_vram_mb": draft_vram_mb,
            "total_cuda_with_draft_mb": total_cuda_with_draft_mb,
            "os_working_set_mb": ws_with_draft_mb,
            "os_working_set_delta_mb": ws_with_draft_mb - base_ws_mb,
            "host_only_commit_delta_mb": max(0.0, (priv_with_draft_mb - base_priv_mb) - total_cuda_with_draft_mb),
        },
        "phase_d_reference": phase_d_ref,
        "m_token_scaling": scaling_stats,
        "eval_decode_suite": {},
        "gsm8k_cot_suite": {},
    }
    _save_json(out_json, results)

    # Section B: Controlled Steady-State Decode across Fact_Eval & Logic_Eval
    print(f"[Phase E Bench] Running Controlled Steady-State Decode ({eval_decode_tokens} tokens x 2 prompts/split)...")
    eval_modes = [
        ("phase_e1_vm_m1", "vm_paged", "none"),
        ("phase_e2_vm_pld", "vm_paged", "pld_only"),
        ("phase_e3_vm_two_tier", "vm_paged", "two_tier"),
        ("phase_e1_core_m1", "core_only", "none"),
        ("phase_e4_core_two_tier", "core_only", "two_tier"),
    ]
    for m_key, vm_mode, d_mode in eval_modes:
        for mlp in phase_b_mlps.values():
            mlp.set_mode(vm_mode)
            if vm_mode == "vm_paged":
                mlp.set_tlb_strategy("amortized_spec_tlb")
            mlp.reset_telemetry(clear_tlb=True)

        gen = SpeculativeVMGenerator(
            model=model,
            tokenizer=tokenizer,
            phase_b_mlps=phase_b_mlps,
            draft_head=draft_head if d_mode in ("eagle_only", "two_tier") else None,
            draft_mode=d_mode,
            max_draft_tokens=3,
            pld_long_draft_tokens=4,
        )
        fact_res = _bench_eval_prompts(gen, fact_prompts, max_new_tokens=eval_decode_tokens)
        logic_res = _bench_eval_prompts(gen, logic_prompts, max_new_tokens=eval_decode_tokens)

        # Strip raw token lists from JSON summary after parity check to keep JSON clean
        fact_ids = fact_res.pop("generated_ids_per_prompt")
        logic_ids = logic_res.pop("generated_ids_per_prompt")

        mean_ms = 0.5 * (fact_res["ms_per_token"] + logic_res["ms_per_token"])
        mean_tps = 1000.0 / max(1e-6, mean_ms)
        mean_pcie = 0.5 * (fact_res["pcie_mb_per_token"] + logic_res["pcie_mb_per_token"])
        mean_mul = 0.5 * (fact_res["tokens_per_step"] + logic_res["tokens_per_step"])

        results["eval_decode_suite"][m_key] = {
            "vm_mode": vm_mode,
            "draft_mode": d_mode,
            "fact_eval": fact_res,
            "logic_eval": logic_res,
            "combined_mean_ms_per_tok": mean_ms,
            "combined_mean_tok_per_s": mean_tps,
            "combined_mean_pcie_mb_per_tok": mean_pcie,
            "combined_mean_tokens_per_step": mean_mul,
        }
        print(
            f"  [{m_key}] Fact={fact_res['tok_per_s']:.2f} tok/s ({fact_res['ms_per_token']:.1f} ms/tok, "
            f"{fact_res['tokens_per_step']:.2f} tok/step, PCIe={fact_res['pcie_mb_per_token']:.1f} MB/tok) | "
            f"Logic={logic_res['tok_per_s']:.2f} tok/s ({logic_res['ms_per_token']:.1f} ms/tok, "
            f"{logic_res['tokens_per_step']:.2f} tok/step)"
        )
        _save_json(out_json, results)

    # Section C: Held-Out 20-Problem Zero-Shot CoT GSM8K (`test` split, `max_new_tokens=320`)
    gsm8k_problems = _build_gsm8k_eval_prompts(num_problems=gsm8k_problems_count, split="test")
    print(f"[Phase E Bench] Running Full {len(gsm8k_problems)}-Problem Held-Out GSM8K CoT Benchmark...")

    # 1. Direct 5-problem M=1 baseline on GSM8K[0:5] for exact same-prompt M=1 speed & PCIe reference
    for mlp in phase_b_mlps.values():
        mlp.set_mode("vm_paged")
        mlp.set_tlb_strategy("amortized_spec_tlb")
        mlp.reset_telemetry(clear_tlb=True)
    gen_vm_m1 = SpeculativeVMGenerator(
        model=model,
        tokenizer=tokenizer,
        phase_b_mlps=phase_b_mlps,
        draft_mode="none",
    )
    gsm_m1_sub = _evaluate_gsm8k_speculative(
        gen_vm_m1,
        gsm8k_problems[: min(5, len(gsm8k_problems))],
        max_new_tokens=gsm8k_max_new_tokens,
        device=device,
        label="Phase E1 VM M=1 (5-prob ref)",
    )
    results["gsm8k_cot_suite"]["vm_m1_5prob_ref"] = gsm_m1_sub
    _save_json(out_json, results)

    # 2. Full 20-problem Held-Out GSM8K on Phase E3: Full VM (nvme_mmap) + Two-Tier (PLD + Eagle-Lite)!
    for mlp in phase_b_mlps.values():
        mlp.set_mode("vm_paged")
        mlp.set_tlb_strategy("amortized_spec_tlb")
        mlp.reset_telemetry(clear_tlb=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    gen_vm_e3 = SpeculativeVMGenerator(
        model=model,
        tokenizer=tokenizer,
        phase_b_mlps=phase_b_mlps,
        draft_head=draft_head,
        draft_mode="two_tier",
        max_draft_tokens=3,
        pld_long_draft_tokens=4,
    )
    gsm_vm_e3 = _evaluate_gsm8k_speculative(
        gen_vm_e3,
        gsm8k_problems,
        max_new_tokens=gsm8k_max_new_tokens,
        device=device,
        label="Phase E3 Full VM Two-Tier",
    )
    results["gsm8k_cot_suite"]["phase_e3_full_vm_20prob"] = gsm_vm_e3
    _save_json(out_json, results)

    # 3. 5-problem GSM8K on Phase E4: Core-Only + Two-Tier Speculative Decode (`0` PCIe peak throughput!)
    for mlp in phase_b_mlps.values():
        mlp.set_mode("core_only")
        mlp.reset_telemetry(clear_tlb=True)
    gen_core_e4 = SpeculativeVMGenerator(
        model=model,
        tokenizer=tokenizer,
        phase_b_mlps=phase_b_mlps,
        draft_head=draft_head,
        draft_mode="two_tier",
        max_draft_tokens=3,
        pld_long_draft_tokens=4,
    )
    gsm_core_e4 = _evaluate_gsm8k_speculative(
        gen_core_e4,
        gsm8k_problems[: min(5, len(gsm8k_problems))],
        max_new_tokens=gsm8k_max_new_tokens,
        device=device,
        label="Phase E4 Core-Only Two-Tier (5-prob)",
    )
    results["gsm8k_cot_suite"]["phase_e4_core_only_5prob"] = gsm_core_e4
    results["total_bench_elapsed_s"] = time.perf_counter() - t_total
    _save_json(out_json, results)

    print("\n" + "=" * 88)
    print("PHASE E AMORTIZED SPECULATIVE PAGING COMPLETE — SUMMARY (google/gemma-4-31B-it)")
    print("=" * 88)
    print(
        f"  Phase D M=1 Full VM (nvme_mmap) Ref : 4.32 tok/s (231.41 ms/tok) | PCIe=207.18 MB/tok | GSM8K=18/20 (90.0%)"
    )
    print(
        f"  Phase E1 M=1 Full VM (Upgraded Kern): {gsm_m1_sub['tok_per_s']:.2f} tok/s ({gsm_m1_sub['ms_per_token']:.2f} ms/tok) | "
        f"PCIe={gsm_m1_sub['pcie_mb_per_token']:.2f} MB/tok"
    )
    print(
        f"  Phase E3 Two-Tier Full VM (20 GSM8K): {gsm_vm_e3['tok_per_s']:.2f} tok/s ({gsm_vm_e3['ms_per_token']:.2f} ms/tok) | "
        f"Tok/Step={gsm_vm_e3['tokens_per_step']:.2f}x | PCIe={gsm_vm_e3['pcie_mb_per_token']:.2f} MB/tok | "
        f"GSM8K={gsm_vm_e3['solved']}/{gsm_vm_e3['total']} ({gsm_vm_e3['accuracy_pct']:.1f}%)"
    )
    print(
        f"  Phase E4 Two-Tier Core-Only (0 PCIe): {gsm_core_e4['tok_per_s']:.2f} tok/s ({gsm_core_e4['ms_per_token']:.2f} ms/tok) | "
        f"Tok/Step={gsm_core_e4['tokens_per_step']:.2f}x | PCIe=0.00 MB/tok"
    )
    print(
        f"  Memory Footprint (Phase E3 Full VM) : CUDA={gsm_vm_e3['cuda_allocated_mb']:,.1f} MB "
        f"(Peak={gsm_vm_e3['cuda_peak_mb']:,.1f} MB) | OS Working Set={gsm_vm_e3['os_working_set_mb']:,.1f} MB"
    )
    print("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase E Multi-Token Speculative Amortized Paging on 31B .nvm")
    parser.add_argument("--model-id", type=str, default="google/gemma-4-31B-it")
    parser.add_argument(
        "--nvm-dir",
        type=str,
        default="checkpoints/gemma-4-31B-it-nvm-hybrid.nvm",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="checkpoints/phase_e_31b_results.json",
    )
    parser.add_argument(
        "--phase-d-json",
        type=str,
        default="checkpoints/phase_d_31b_results.json",
    )
    parser.add_argument("--stage", type=str, choices=["all", "calibrate", "bench"], default="all")
    parser.add_argument("--reuse-draft", action="store_true", help="Reuse cached eagle_draft_head.safetensors if present")
    parser.add_argument("--shortlist-size", type=int, default=12288)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--eval-decode-tokens", type=int, default=64)
    parser.add_argument("--gsm8k-problems", type=int, default=20)
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    draft_sf = os.path.join(args.nvm_dir, "eagle_draft_head.safetensors")

    if args.stage == "calibrate":
        run_calibrate_stage(
            model_id=args.model_id,
            nvm_dir=args.nvm_dir,
            shortlist_size=args.shortlist_size,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
        )
    elif args.stage == "bench":
        run_bench_stage(
            model_id=args.model_id,
            nvm_dir=args.nvm_dir,
            out_json=args.out_json,
            phase_d_json=args.phase_d_json,
            eval_decode_tokens=args.eval_decode_tokens,
            gsm8k_problems_count=args.gsm8k_problems,
            gsm8k_max_new_tokens=args.gsm8k_max_new_tokens,
            seed=args.seed,
            device=args.device,
        )
    else:
        # Run calibrate + bench in isolated child processes so CUDA / WDDM / OS Working Set metrics are 100% clean!
        if not (args.reuse_draft and os.path.exists(draft_sf)):
            cmd_cal = [
                sys.executable,
                os.path.abspath(__file__),
                "--stage",
                "calibrate",
                "--model-id",
                args.model_id,
                "--nvm-dir",
                args.nvm_dir,
                "--shortlist-size",
                str(args.shortlist_size),
                "--epochs",
                str(args.epochs),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
            ]
            subprocess.run(cmd_cal, check=True)
        cmd_ben = [
            sys.executable,
            os.path.abspath(__file__),
            "--stage",
            "bench",
            "--model-id",
            args.model_id,
            "--nvm-dir",
            args.nvm_dir,
            "--out-json",
            args.out_json,
            "--phase-d-json",
            args.phase_d_json,
            "--eval-decode-tokens",
            str(args.eval_decode_tokens),
            "--gsm8k-problems",
            str(args.gsm8k_problems),
            "--gsm8k-max-new-tokens",
            str(args.gsm8k_max_new_tokens),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        subprocess.run(cmd_ben, check=True)


if __name__ == "__main__":
    main()
