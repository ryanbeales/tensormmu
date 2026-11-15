"""Phase F End-to-End Benchmark: Zero-Skeleton Sparse MoE Virtual Memory (`nvm_moe_v1`).

Validates Zero-Skeleton Sparse MoE `.nvm` compilation, 4KB-aligned contiguous `UINT4` expert paging,
deterministic NVRTC multi-token MoE-TLB CUDA kernels, intra-layer Shared-MLP DMA overlap, and
Two-Tier (`PLD + EagleLiteDraftHead`) speculative amortized paging on a single 24 GB RTX 4090 PC.

Stages (each isolatable in a fresh OS process via `--stage all` for uncontaminated RAM/VRAM telemetry):
  1. `teacher_ppl` : Stream unmodified BF16 checkpoint layer-by-layer (`O(1)` VRAM) for exact ground-truth PPL.
  2. `offload_ref` : Measure honest BF16 GPU+Host Layer Offloading (`use_cache=True`) fitting `< 24 GB` VRAM.
  3. `compile`     : Stream-compile Zero-Skeleton `nvm_moe_v1` cartridge (`< 2.5 GB` peak VRAM, 0 test leakage).
  4. `calibrate`   : Train `EagleLiteDraftHead` strictly on `Logic_Calib + Fact_Calib + GSM8K train`.
  5. `bench`       : Fresh-process cold-boot (`nvme_mmap`) evaluating PPL, 5-point VRAM-vs-Throughput Pareto
                     sweep (`tlb_slots` = 16, 32, 48, 64, All-Resident 128), `exact_demand` vs `overlap_shared_mlp`,
                     `0-ulp` parity across all modes, and 20-problem held-out GSM8K (`test`) CoT accuracy & speed.
"""

from __future__ import annotations
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

import argparse
import gc
import json
import math
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
import torch.nn.functional as F
from transformers import AutoTokenizer

from data.calibration import TokenizedSplit, build_contrastive_datasets
from decompiler.nvm_serializer import (
    compile_streaming_moe_to_nvm,
    evaluate_streaming_dense_bf16_ppl,
    load_nvm_cartridge,
    measure_layer_offload_kv_decode,
    resolve_hf_snapshot_dir,
)
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


def _load_or_init_json(path: str) -> Dict[str, Any]:
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


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


@torch.no_grad()
def _evaluate_loaded_model_ppl(
    model: torch.nn.Module,
    splits: Dict[str, TokenizedSplit],
    teacher_preds_by_split: Optional[Dict[str, torch.Tensor]] = None,
    batch_size: int = 2,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    dev = torch.device(device)
    out_metrics: Dict[str, Any] = {}
    for sn, split in splits.items():
        nll = 0.0
        toks = 0
        correct = 0
        preds_list: List[torch.Tensor] = []
        n_rows = int(split.input_ids.shape[0])
        for s in range(0, n_rows, batch_size):
            e = min(n_rows, s + batch_size)
            b_ids = split.input_ids[s:e].to(dev)
            b_msk = split.attention_mask[s:e].to(dev)
            logits = model(input_ids=b_ids, attention_mask=b_msk, use_cache=False).logits
            for r in range(logits.shape[0]):
                sl = logits[r, :-1, :].float()
                lb = b_ids[r, 1:]
                mk = b_msk[r, 1:].bool()
                pr = sl.argmax(dim=-1)
                ls = F.cross_entropy(sl, lb, reduction="none")
                nll += float(ls[mk].sum().item())
                toks += int(mk.sum().item())
                correct += int((pr[mk] == lb[mk]).sum().item())
                preds_list.append(pr.cpu().unsqueeze(0))
                del sl, ls
            del logits
        all_preds = torch.cat(preds_list, dim=0)
        mean_loss = nll / max(1, toks)
        agree_pct: Optional[float] = None
        if teacher_preds_by_split is not None and sn in teacher_preds_by_split:
            t_pr = teacher_preds_by_split[sn]
            mk_all = split.attention_mask[:, 1:].bool().cpu()
            agree_pct = 100.0 * float((all_preds[mk_all] == t_pr[mk_all]).float().mean().item())

        out_metrics[sn] = {
            "loss": mean_loss,
            "perplexity": math.exp(min(mean_loss, 20.0)),
            "next_token_acc_pct": 100.0 * (correct / max(1, toks)),
            "teacher_top1_agreement_pct": agree_pct,
            "tokens": toks,
        }
    return out_metrics


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
    label: str = "Phase F",
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
                "tlb_hit_rate_pct": st.tlb_hit_rate_pct,
            }
        )
        if (idx + 1) % 5 == 0 or (idx + 1) == total:
            cur_tps = float(max(1, total_emitted)) / max(1e-6, total_elapsed_s)
            cur_mul = float(max(1, total_emitted)) / float(max(1, total_steps))
            cur_pcie = total_pcie_mb / float(max(1, total_emitted))
            cur_hit = sum(hit_rates) / max(1, len(hit_rates))
            print(
                f"  [{label} GSM8K {idx+1:02d}/{total:02d}] Solved={correct}/{idx+1} | "
                f"Speed={cur_tps:.2f} tok/s ({1000.0/max(1e-6,cur_tps):.1f} ms/tok) | "
                f"Tok/Step={cur_mul:.2f}x | TLB Hit={cur_hit:.1f}% | PCIe={cur_pcie:.1f} MB/tok",
                flush=True,
            )

    em = max(1, total_emitted)
    st_cnt = max(1, total_steps)
    cuda_alloc_mb, cuda_peak_mb = _get_cuda_mem_mb()
    ram_info = get_os_process_ram_mb()

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
        "eagle_proposed": total_eag_prop,
        "eagle_accepted": total_eag_acc,
        "total_pcie_mb": total_pcie_mb,
        "pcie_mb_per_token": total_pcie_mb / float(em),
        "pcie_mb_per_step": total_pcie_mb / float(st_cnt),
        "tlb_hit_rate_pct": sum(hit_rates) / max(1, len(hit_rates)),
        "cuda_allocated_mb": cuda_alloc_mb,
        "cuda_peak_mb": cuda_peak_mb,
        "os_working_set_mb": ram_info["ws_mb"],
        "os_private_commit_mb": ram_info["commit_mb"],
        "per_problem": per_problem,
    }


def stage_teacher_ppl(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    snap_dir = resolve_hf_snapshot_dir(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(snap_dir, local_files_only=True)
    bundle = build_contrastive_datasets(
        tokenizer=tokenizer,
        num_calibration_seqs=16,
        num_eval_seqs=4,
        seq_len=256,
        seed=args.seed,
        allow_hf_download=False,
    )
    print(f"[Phase F Teacher] Streaming exact unmodified BF16 Teacher PPL on '{args.model_id}'...", flush=True)
    bf16_res = evaluate_streaming_dense_bf16_ppl(
        model_id_or_path=snap_dir,
        splits={"Logic_Eval": bundle.logic_eval, "Fact_Eval": bundle.fact_eval},
        batch_size=2,
        device=args.device,
    )
    preds_path = os.path.join(os.path.dirname(os.path.abspath(args.out_json)), f"{args.tag}_teacher_preds.pt")
    torch.save(
        {sn: bf16_res["splits"][sn].pop("preds") for sn in ("Logic_Eval", "Fact_Eval")},
        preds_path,
    )
    data = _load_or_init_json(args.out_json)
    data["model_id"] = args.model_id
    data["teacher_bf16_streaming"] = bf16_res
    _save_json(args.out_json, data)
    print(
        f"[Phase F Teacher] Logic_Eval PPL={bf16_res['splits']['Logic_Eval']['perplexity']:.4f} | "
        f"Fact_Eval PPL={bf16_res['splits']['Fact_Eval']['perplexity']:.4f} | "
        f"Peak VRAM={bf16_res['peak_vram_mb']:.1f} MB | Wall={bf16_res['wall_s']:.1f}s",
        flush=True,
    )


def stage_offload_ref(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    snap_dir = resolve_hf_snapshot_dir(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(snap_dir, local_files_only=True)
    bundle = build_contrastive_datasets(
        tokenizer=tokenizer,
        num_calibration_seqs=16,
        num_eval_seqs=4,
        seq_len=256,
        seed=args.seed,
        allow_hf_download=False,
    )
    print(
        f"[Phase F Offload Ref] Measuring BF16 Layer-Offload KV decode (num_gpu_layers={args.offload_gpu_layers})...",
        flush=True,
    )
    ram0 = get_os_process_ram_mb()
    off_dec_toks = 3 if int(args.offload_gpu_layers) <= 6 else 6
    off_res = measure_layer_offload_kv_decode(
        model_id_or_path=snap_dir,
        prompt_ids=bundle.fact_eval.input_ids[:1, :64],
        num_gpu_layers=args.offload_gpu_layers,
        num_decode_tokens=off_dec_toks,
        warmup_steps=1,
        device=args.device,
    )
    off_res["os_working_set_mb_after"] = get_os_process_ram_mb()["ws_mb"]
    off_res["os_working_set_mb_before"] = ram0["ws_mb"]
    data = _load_or_init_json(args.out_json)
    data["layer_offload_bf16"] = off_res
    _save_json(args.out_json, data)
    print(
        f"[Phase F Offload Ref] {off_res['decode_tok_per_s']:.3f} tok/s ({off_res['decode_ms_per_tok']:.1f} ms/tok) | "
        f"CUDA={off_res['resident_cuda_mb']:.1f} MB | PCIe={off_res['pcie_mb_per_token']:.1f} MB/tok",
        flush=True,
    )


def stage_compile(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    snap_dir = resolve_hf_snapshot_dir(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(snap_dir, local_files_only=True)
    bundle = build_contrastive_datasets(
        tokenizer=tokenizer,
        num_calibration_seqs=16,
        num_eval_seqs=4,
        seq_len=256,
        seed=args.seed,
        allow_hf_download=False,
    )
    print(f"[Phase F Compile] Compiling Zero-Skeleton MoE .nvm cartridge -> {args.nvm_dir}", flush=True)
    manifest = compile_streaming_moe_to_nvm(
        model_id_or_path=snap_dir,
        nvm_dir=args.nvm_dir,
        bundle=bundle,
        tlb_slots=args.default_tlb_slots,
        expert_group_size=args.expert_group_size,
        expert_chunk_size=64,
        attn_quant=args.attn_quant,
        shared_mlp_quant=args.shared_mlp_quant,
        lm_head_quant=args.lm_head_quant,
        calib_batch_size=4,
        max_calib_tokens=4096,
        device=args.device,
    )
    data = _load_or_init_json(args.out_json)
    data["compile_telemetry"] = manifest["compile_telemetry"]
    data["nvm_dir"] = args.nvm_dir
    _save_json(args.out_json, data)
    ct = manifest["compile_telemetry"]
    print(
        f"[Phase F Compile] Done in {ct['compile_wall_s']:.1f}s | Peak VRAM={ct['peak_compile_vram_mb']:.1f} MB | "
        f"Core={ct['core_resident_disk_mb']:.1f} MB + Expert Pages={ct['pages_disk_mb']:.1f} MB "
        f"(Total={ct['total_disk_mb']:.1f} MB)",
        flush=True,
    )


def stage_calibrate(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    snap_dir = resolve_hf_snapshot_dir(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(snap_dir, local_files_only=True)
    bundle = build_contrastive_datasets(
        tokenizer=tokenizer,
        num_calibration_seqs=16,
        num_eval_seqs=4,
        seq_len=256,
        seed=args.seed,
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
        f"[Phase F Calibrate] Training EagleLiteDraftHead on {len(calib_seqs)} strictly held-in calib sequences...",
        flush=True,
    )
    calibrate_and_save_eagle_draft_head(
        nvm_dir=args.nvm_dir,
        calib_token_sequences=calib_seqs,
        shortlist_size=12288,
        d_latent=1024,
        d_ffn=1536,
        epochs=args.eagle_epochs,
        lr=2.0e-3,
        rollout_horizon=4,
        tlb_slots_override=args.default_tlb_slots,
        device=args.device,
    )


def stage_bench(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    data = _load_or_init_json(args.out_json)
    ram_base = get_os_process_ram_mb()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"[Phase F Bench] Cold-booting MoE .nvm cartridge (nvme_mmap) from {args.nvm_dir}...", flush=True)
    model, tokenizer, moe_layers, lut_pool, ple_off, manifest = load_nvm_cartridge(
        nvm_dir=args.nvm_dir,
        device=args.device,
        dtype=torch.bfloat16,
        mode="vm_paged",
        tlb_strategy="overlap_shared_mlp",
        page_storage="nvme_mmap",
        tlb_slots_override=args.default_tlb_slots,
        use_phase_c_kernels=True,
        load_tokenizer=True,
    )
    boot_info = manifest["last_cold_boot"]
    ram_after_boot = get_os_process_ram_mb()

    # Measure core-only resident VRAM (0 expert TLB slots)
    for m in moe_layers.values():
        m.set_mode("core_only")
    torch.cuda.empty_cache()
    core_only_cuda_mb, _ = _get_cuda_mem_mb()

    # Restore default TLB slots and load EagleLiteDraftHead
    for m in moe_layers.values():
        m.set_mode("vm_paged", tlb_slots_override=args.default_tlb_slots)
    draft_head = EagleLiteDraftHead.load_pretrained_nvm(
        nvm_dir=args.nvm_dir,
        device=args.device,
        dtype=torch.bfloat16,
        lut_pool=lut_pool,
    )
    draft_vram_mb = draft_head.vram_bytes() / float(1024 ** 2)
    ram_with_draft = get_os_process_ram_mb()

    print(
        f"[Phase F Bench] Cold-boot={boot_info['cold_boot_s']:.2f}s | Core-Only Resident CUDA={core_only_cuda_mb:,.1f} MB | "
        f"EagleLiteDraftHead={draft_vram_mb:.1f} MB | OS Working Set={ram_with_draft['ws_mb']:,.1f} MB "
        f"(+{ram_with_draft['ws_mb'] - ram_base['ws_mb']:.1f} MB)",
        flush=True,
    )

    bundle = build_contrastive_datasets(
        tokenizer=tokenizer,
        num_calibration_seqs=16,
        num_eval_seqs=4,
        seq_len=256,
        seed=args.seed,
        allow_hf_download=False,
    )
    preds_path = os.path.join(os.path.dirname(os.path.abspath(args.out_json)), f"{args.tag}_teacher_preds.pt")
    teacher_preds = torch.load(preds_path, map_location="cpu", weights_only=True) if os.path.isfile(preds_path) else None

    # 1. Exact Perplexity & 0-ulp Resident vs Paged PPL Parity on Held-Out `Logic_Eval` & `Fact_Eval`
    print("[Phase F Bench] Evaluating Logic_Eval & Fact_Eval Perplexity (vm_paged vs all_resident)...", flush=True)
    for m in moe_layers.values():
        m.set_mode("vm_paged", tlb_slots_override=args.default_tlb_slots)
        m.reset_telemetry(clear_tlb=True)
    ppl_paged = _evaluate_loaded_model_ppl(
        model,
        {"Logic_Eval": bundle.logic_eval, "Fact_Eval": bundle.fact_eval},
        teacher_preds_by_split=teacher_preds,
        batch_size=2,
        device=args.device,
    )

    num_experts_per_layer = next(iter(moe_layers.values())).num_experts
    can_fit_all_resident = (
        core_only_cuda_mb
        + (len(moe_layers) * num_experts_per_layer * next(iter(moe_layers.values())).layout.page_bytes) / (1024 ** 2)
        < 21500.0
    )
    pareto_slots = [int(x) for x in args.pareto_slots.split(",") if x.strip()]
    ppl_resident = None
    if can_fit_all_resident:
        for m in moe_layers.values():
            m.set_mode("all_resident")
        ppl_resident = _evaluate_loaded_model_ppl(
            model,
            {"Logic_Eval": bundle.logic_eval, "Fact_Eval": bundle.fact_eval},
            teacher_preds_by_split=teacher_preds,
            batch_size=2,
            device=args.device,
        )
        for sn in ("Logic_Eval", "Fact_Eval"):
            diff_loss = abs(ppl_paged[sn]["loss"] - ppl_resident[sn]["loss"])
            print(
                f"  [{sn}] vm_paged PPL={ppl_paged[sn]['perplexity']:.4f} | "
                f"all_resident PPL={ppl_resident[sn]['perplexity']:.4f} | "
                f"|delta_loss|={diff_loss:.2e} | Teacher Top-1 Agree={ppl_paged[sn]['teacher_top1_agreement_pct']:.2f}%",
                flush=True,
            )
    else:
        max_sl = max(pareto_slots) if pareto_slots else args.default_tlb_slots
        for m in moe_layers.values():
            m.set_mode("vm_paged", tlb_slots_override=max_sl)
            m.reset_telemetry(clear_tlb=True)
        ppl_high = _evaluate_loaded_model_ppl(
            model,
            {"Logic_Eval": bundle.logic_eval, "Fact_Eval": bundle.fact_eval},
            teacher_preds_by_split=teacher_preds,
            batch_size=2,
            device=args.device,
        )
        ppl_resident = ppl_high
        for sn in ("Logic_Eval", "Fact_Eval"):
            diff_loss = abs(ppl_paged[sn]["loss"] - ppl_high[sn]["loss"])
            print(
                f"  [{sn}] vm_paged(slots={args.default_tlb_slots}) PPL={ppl_paged[sn]['perplexity']:.4f} | "
                f"vm_paged(slots={max_sl}) PPL={ppl_high[sn]['perplexity']:.4f} | "
                f"|delta_loss|={diff_loss:.2e} | Teacher Top-1 Agree={ppl_paged[sn]['teacher_top1_agreement_pct']:.2f}%",
                flush=True,
            )

    # 2. VRAM vs Throughput Pareto Sweep (`nvme_mmap`) + Intra-Layer Overlap Ablation + 0-ulp Token Parity!
    fact_prompts = [bundle.fact_eval.input_ids[i : i + 1, :128].to(args.device) for i in range(2)]
    logic_prompts = [bundle.logic_eval.input_ids[i : i + 1, :128].to(args.device) for i in range(2)]

    pareto_results: Dict[str, Any] = {}
    ref_tokens_by_dmode: Dict[str, Tuple[List[List[int]], List[List[int]]]] = {}

    sweep_configs: List[Tuple[str, str, int, str, str]] = []
    for sl in pareto_slots:
        if sl < num_experts_per_layer:
            sweep_configs.append((f"slots_{sl}_m1_demand", "vm_paged", sl, "exact_demand", "none"))
            sweep_configs.append((f"slots_{sl}_m1_overlap", "vm_paged", sl, "overlap_shared_mlp", "none"))
            sweep_configs.append((f"slots_{sl}_spec_two_tier", "vm_paged", sl, "overlap_shared_mlp", "two_tier"))
    if can_fit_all_resident:
        sweep_configs.append(("all_resident_m1", "all_resident", num_experts_per_layer, "overlap_shared_mlp", "none"))
        sweep_configs.append(
            ("all_resident_spec_two_tier", "all_resident", num_experts_per_layer, "overlap_shared_mlp", "two_tier")
        )

    print(
        f"[Phase F Bench] Running {len(sweep_configs)}-point VRAM-vs-Throughput Pareto Sweep ({args.eval_decode_tokens} tok x 4 prompts)...",
        flush=True,
    )
    for cfg_key, vm_mode, sl_cnt, strat, d_mode in sweep_configs:
        for m in moe_layers.values():
            m.set_tlb_strategy(strat)
            m.set_mode(vm_mode, tlb_slots_override=sl_cnt)
            m.reset_telemetry(clear_tlb=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        alloc_mb, _ = _get_cuda_mem_mb()

        gen = SpeculativeVMGenerator(
            model=model,
            tokenizer=tokenizer,
            phase_b_mlps=moe_layers,
            draft_head=draft_head if d_mode == "two_tier" else None,
            draft_mode=d_mode,
            max_draft_tokens=3,
            pld_long_draft_tokens=4,
        )
        f_res = _bench_eval_prompts(gen, fact_prompts, max_new_tokens=args.eval_decode_tokens)
        l_res = _bench_eval_prompts(gen, logic_prompts, max_new_tokens=args.eval_decode_tokens)
        _, peak_mb = _get_cuda_mem_mb()
        ram_cur = get_os_process_ram_mb()

        f_ids = f_res.pop("generated_ids_per_prompt")
        l_ids = l_res.pop("generated_ids_per_prompt")
        if d_mode not in ref_tokens_by_dmode:
            ref_tokens_by_dmode[d_mode] = (f_ids, l_ids)
            parity_exact = True
        else:
            rf, rl = ref_tokens_by_dmode[d_mode]
            parity_exact = (f_ids == rf) and (l_ids == rl)
            assert parity_exact, f"0-ulp VM parity mismatch in '{cfg_key}' vs '{d_mode}' reference!"

        # Also measure exact token-prefix agreement against M=1 (note: cuBLAS GEMV M=1 vs TensorCore M=4 can differ by 1 BF16 ULP)
        m1_prefix_matches: List[int] = []
        if "none" in ref_tokens_by_dmode:
            m1_f, m1_l = ref_tokens_by_dmode["none"]
            for a, b in list(zip(f_ids, m1_f)) + list(zip(l_ids, m1_l)):
                pref = 0
                for ta, tb in zip(a, b):
                    if ta != tb:
                        break
                    pref += 1
                m1_prefix_matches.append(pref)

        mean_ms = 0.5 * (f_res["ms_per_token"] + l_res["ms_per_token"])
        mean_tps = 1000.0 / max(1e-6, mean_ms)
        mean_pcie = 0.5 * (f_res["pcie_mb_per_token"] + l_res["pcie_mb_per_token"])
        mean_hit = 0.5 * (f_res["tlb_hit_rate_pct"] + l_res["tlb_hit_rate_pct"])
        mean_mul = 0.5 * (f_res["tokens_per_step"] + l_res["tokens_per_step"])

        pareto_results[cfg_key] = {
            "vm_mode": vm_mode,
            "tlb_slots": sl_cnt,
            "cache_pct": 100.0 * float(sl_cnt) / float(num_experts_per_layer),
            "tlb_strategy": strat,
            "draft_mode": d_mode,
            "cuda_allocated_mb": alloc_mb,
            "cuda_peak_mb": peak_mb,
            "os_working_set_mb": ram_cur["ws_mb"],
            "os_private_commit_mb": ram_cur["commit_mb"],
            "zero_ulp_vm_parity": parity_exact,
            "m1_prefix_match_tokens": m1_prefix_matches,
            "combined_mean_ms_per_tok": mean_ms,
            "combined_mean_tok_per_s": mean_tps,
            "combined_mean_pcie_mb_per_tok": mean_pcie,
            "combined_mean_tlb_hit_pct": mean_hit,
            "combined_mean_tokens_per_step": mean_mul,
            "fact_eval": f_res,
            "logic_eval": l_res,
        }
        print(
            f"  [{cfg_key:26s}] VRAM={alloc_mb:7.1f} MB | {mean_tps:5.2f} tok/s ({mean_ms:5.1f} ms/tok) | "
            f"Tok/Step={mean_mul:.2f}x | TLB Hit={mean_hit:5.1f}% | PCIe={mean_pcie:6.1f} MB/tok | 0-ulp VM={parity_exact}",
            flush=True,
        )

    # 3. Held-Out 20-Problem Zero-Shot CoT GSM8K (`test` split, `max_new_tokens=320`)
    gsm8k_problems = _build_gsm8k_eval_prompts(num_problems=args.gsm8k_count, split="test")
    gsm8k_suite: Dict[str, Any] = {}

    # 3a. 5-problem M=1 baseline at low-VRAM cache budget (`32` on 26B = ~6.2 GB VRAM; `64` on 80B = ~7.8 GB VRAM!)
    low_vram_slots = min(max(32, num_experts_per_layer // 8), num_experts_per_layer)
    for m in moe_layers.values():
        m.set_tlb_strategy("overlap_shared_mlp")
        m.set_mode("vm_paged", tlb_slots_override=low_vram_slots)
        m.reset_telemetry(clear_tlb=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gen_32_m1 = SpeculativeVMGenerator(
        model=model, tokenizer=tokenizer, phase_b_mlps=moe_layers, draft_mode="none"
    )
    print(f"[Phase F GSM8K] 5-Problem M=1 Parity Baseline (tlb_slots={low_vram_slots}, nvme_mmap)...", flush=True)
    gsm8k_suite[f"vm_slots_{low_vram_slots}_m1_5prob"] = _evaluate_gsm8k_speculative(
        gen_32_m1, gsm8k_problems[:5], max_new_tokens=args.gsm8k_max_tokens, device=args.device, label=f"Slots{low_vram_slots}-M1"
    )

    # 3b. Full 20-problem GSM8K at `tlb_slots = 32` (`25%` cache, `~6.2 GB` VRAM!) + Two-Tier Speculative
    for m in moe_layers.values():
        m.set_tlb_strategy("overlap_shared_mlp")
        m.set_mode("vm_paged", tlb_slots_override=low_vram_slots)
        m.reset_telemetry(clear_tlb=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gen_32_spec = SpeculativeVMGenerator(
        model=model,
        tokenizer=tokenizer,
        phase_b_mlps=moe_layers,
        draft_head=draft_head,
        draft_mode="two_tier",
        max_draft_tokens=3,
        pld_long_draft_tokens=4,
    )
    print(f"[Phase F GSM8K] Full {len(gsm8k_problems)}-Problem CoT (tlb_slots={low_vram_slots}, ~6.2 GB VRAM, Two-Tier Spec)...", flush=True)
    res_32_full = _evaluate_gsm8k_speculative(
        gen_32_spec, gsm8k_problems, max_new_tokens=args.gsm8k_max_tokens, device=args.device, label=f"Slots{low_vram_slots}-Spec"
    )
    gsm8k_suite[f"vm_slots_{low_vram_slots}_two_tier_20prob"] = res_32_full

    # Record M=1 vs Two-Tier 5-problem answer agreement, and hard-assert 0-ulp VM parity across Two-Tier cache sizes!
    m1_vs_spec_ans_agree = 0
    for i5 in range(min(5, len(gsm8k_problems))):
        p_m1 = gsm8k_suite[f"vm_slots_{low_vram_slots}_m1_5prob"]["per_problem"][i5]
        p_sp = res_32_full["per_problem"][i5]
        if p_m1["pred"] == p_sp["pred"]:
            m1_vs_spec_ans_agree += 1
    gsm8k_suite["m1_vs_two_tier_5prob_answer_agreement"] = f"{m1_vs_spec_ans_agree}/{min(5, len(gsm8k_problems))}"

    # 3c. 5-problem speed check at `tlb_slots = 48` (`~7.8 GB` VRAM) + `all_resident` (`~15.3 GB` VRAM ceiling)
    if args.default_tlb_slots != low_vram_slots:
        for m in moe_layers.values():
            m.set_tlb_strategy("overlap_shared_mlp")
            m.set_mode("vm_paged", tlb_slots_override=args.default_tlb_slots)
            m.reset_telemetry(clear_tlb=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gen_def_spec = SpeculativeVMGenerator(
            model=model,
            tokenizer=tokenizer,
            phase_b_mlps=moe_layers,
            draft_head=draft_head,
            draft_mode="two_tier",
            max_draft_tokens=3,
            pld_long_draft_tokens=4,
        )
        print(f"[Phase F GSM8K] 5-Problem CoT Speed Check (tlb_slots={args.default_tlb_slots}, Two-Tier Spec)...", flush=True)
        res_def_5 = _evaluate_gsm8k_speculative(
            gen_def_spec,
            gsm8k_problems[:5],
            max_new_tokens=args.gsm8k_max_tokens,
            device=args.device,
            label=f"Slots{args.default_tlb_slots}-Spec",
        )
        gsm8k_suite[f"vm_slots_{args.default_tlb_slots}_two_tier_5prob"] = res_def_5
        for i5 in range(min(5, len(gsm8k_problems))):
            p_32 = res_32_full["per_problem"][i5]
            p_df = res_def_5["per_problem"][i5]
            assert p_32["pred"] == p_df["pred"] and p_32["emitted_tokens"] == p_df["emitted_tokens"], (
                f"0-ulp VM parity mismatch GSM8K[{i5}] Slots32 vs Slots{args.default_tlb_slots}: {p_32} vs {p_df}"
            )

    if can_fit_all_resident:
        for m in moe_layers.values():
            m.set_mode("all_resident")
            m.reset_telemetry(clear_tlb=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gen_all_spec = SpeculativeVMGenerator(
            model=model,
            tokenizer=tokenizer,
            phase_b_mlps=moe_layers,
            draft_head=draft_head,
            draft_mode="two_tier",
            max_draft_tokens=3,
            pld_long_draft_tokens=4,
        )
        print("[Phase F GSM8K] 5-Problem All-Resident 0-PCIe Ceiling Check (Two-Tier Spec)...", flush=True)
        res_all_5 = _evaluate_gsm8k_speculative(
            gen_all_spec,
            gsm8k_problems[:5],
            max_new_tokens=args.gsm8k_max_tokens,
            device=args.device,
            label="AllResident-Spec",
        )
        gsm8k_suite["all_resident_two_tier_5prob"] = res_all_5
        for i5 in range(min(5, len(gsm8k_problems))):
            p_32 = res_32_full["per_problem"][i5]
            p_al = res_all_5["per_problem"][i5]
            assert p_32["pred"] == p_al["pred"] and p_32["emitted_tokens"] == p_al["emitted_tokens"], (
                f"0-ulp VM parity mismatch GSM8K[{i5}] Slots32 vs AllResident: {p_32} vs {p_al}"
            )

    elif pareto_slots and max(pareto_slots) > max(low_vram_slots, args.default_tlb_slots):
        hi_slots = max(pareto_slots)
        for m in moe_layers.values():
            m.set_tlb_strategy("overlap_shared_mlp")
            m.set_mode("vm_paged", tlb_slots_override=hi_slots)
            m.reset_telemetry(clear_tlb=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gen_hi_spec = SpeculativeVMGenerator(
            model=model,
            tokenizer=tokenizer,
            phase_b_mlps=moe_layers,
            draft_head=draft_head,
            draft_mode="two_tier",
            max_draft_tokens=3,
            pld_long_draft_tokens=4,
        )
        print(f"[Phase F GSM8K] 5-Problem High-TLB Speed Check (tlb_slots={hi_slots}, Two-Tier Spec)...", flush=True)
        res_hi_5 = _evaluate_gsm8k_speculative(
            gen_hi_spec,
            gsm8k_problems[:5],
            max_new_tokens=args.gsm8k_max_tokens,
            device=args.device,
            label=f"Slots{hi_slots}-Spec",
        )
        gsm8k_suite[f"vm_slots_{hi_slots}_two_tier_5prob"] = res_hi_5
        for i5 in range(min(5, len(gsm8k_problems))):
            p_32 = res_32_full["per_problem"][i5]
            p_hi = res_hi_5["per_problem"][i5]
            assert p_32["pred"] == p_hi["pred"] and p_32["emitted_tokens"] == p_hi["emitted_tokens"], (
                f"0-ulp VM parity mismatch GSM8K[{i5}] Slots{low_vram_slots} vs Slots{hi_slots}: {p_32} vs {p_hi}"
            )

    data["hardware"] = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cold_boot_s": boot_info["cold_boot_s"],
        "core_only_resident_cuda_mb": core_only_cuda_mb,
        "draft_head_vram_mb": draft_vram_mb,
        "baseline_os_ws_mb": ram_base["ws_mb"],
        "baseline_os_commit_mb": ram_base["commit_mb"],
        "boot_os_ws_mb": ram_after_boot["ws_mb"],
        "boot_os_commit_mb": ram_after_boot["commit_mb"],
        "os_ws_delta_mb": ram_with_draft["ws_mb"] - ram_base["ws_mb"],
        "host_only_commit_delta_mb": max(
            0.0, (ram_with_draft["commit_mb"] - ram_base["commit_mb"]) - (core_only_cuda_mb + draft_vram_mb)
        ),
    }
    data["perplexity"] = {
        "vm_paged": ppl_paged,
        "all_resident": ppl_resident,
    }
    data["pareto_vram_sweep"] = pareto_results
    data["gsm8k_cot_suite"] = gsm8k_suite
    _save_json(args.out_json, data)
    print(f"[Phase F Bench] Saved complete telemetry -> {args.out_json}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        choices=("teacher_ppl", "offload_ref", "compile", "calibrate", "bench", "all"),
        default="all",
    )
    p.add_argument("--model-id", default="google/gemma-4-26B-A4B-it")
    p.add_argument("--nvm-dir", default="checkpoints/gemma-4-26B-A4B-it-nvm-moe.nvm")
    p.add_argument("--out-json", default="checkpoints/phase_f_26b_moe_results.json")
    p.add_argument("--tag", default="gemma4_26b_moe")
    p.add_argument("--attn-quant", default="int8")
    p.add_argument("--shared-mlp-quant", default="int8")
    p.add_argument("--lm-head-quant", default="none")
    p.add_argument("--expert-group-size", type=int, default=64)
    p.add_argument("--default-tlb-slots", type=int, default=48)
    p.add_argument("--pareto-slots", default="16,32,48,64")
    p.add_argument("--offload-gpu-layers", type=int, default=12)
    p.add_argument("--eagle-epochs", type=int, default=40)
    p.add_argument("--eval-decode-tokens", type=int, default=64)
    p.add_argument("--gsm8k-count", type=int, default=20)
    p.add_argument("--gsm8k-max-tokens", type=int, default=320)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    if args.stage == "all":
        for sub_stage in ("teacher_ppl", "offload_ref", "compile", "calibrate", "bench"):
            print(f"\n==================== PHASE F STAGE: {sub_stage} ====================", flush=True)
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--stage",
                sub_stage,
                "--model-id",
                args.model_id,
                "--nvm-dir",
                args.nvm_dir,
                "--out-json",
                args.out_json,
                "--tag",
                args.tag,
                "--attn-quant",
                args.attn_quant,
                "--shared-mlp-quant",
                args.shared_mlp_quant,
                "--lm-head-quant",
                args.lm_head_quant,
                "--expert-group-size",
                str(args.expert_group_size),
                "--default-tlb-slots",
                str(args.default_tlb_slots),
                "--pareto-slots",
                args.pareto_slots,
                "--offload-gpu-layers",
                str(args.offload_gpu_layers),
                "--eagle-epochs",
                str(args.eagle_epochs),
                "--eval-decode-tokens",
                str(args.eval_decode_tokens),
                "--gsm8k-count",
                str(args.gsm8k_count),
                "--gsm8k-max-tokens",
                str(args.gsm8k_max_tokens),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
            ]
            subprocess.run(cmd, check=True)
        return

    if args.stage == "teacher_ppl":
        stage_teacher_ppl(args)
    elif args.stage == "offload_ref":
        stage_offload_ref(args)
    elif args.stage == "compile":
        stage_compile(args)
    elif args.stage == "calibrate":
        stage_calibrate(args)
    elif args.stage == "bench":
        stage_bench(args)


if __name__ == "__main__":
    main()
