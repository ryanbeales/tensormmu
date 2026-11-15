"""Phase G End-to-End Benchmark: Fused Sparse-MoE Core + Confidence-Gated Speculative Paging + Cross-Layer Lookahead VM.

Validates the three Phase G architectural upgrades on standalone `nvm_moe_v1` cartridges (`nvme_mmap`)
on a single 24 GB RTX 4090 + 34.3 GB System RAM workstation:
  1. Phase G0 (`qwen3_next_kernels.py`): Fused NVRTC `sm_89` RMSNorm, Gated-RMSNorm, Top-K Softmax Router,
     and GQA-native Conv1d+GDN kernels eliminating ~60-85 ms/pass of Python/CUDA launch overhead.
  2. Phase G1 (`speculative_vm.py` + `moe_expert_tlb.py`): Sparsity-aware confidence + logit-margin draft
     gating and Verified-Protected Probationary Draft-LRU eliminating cold-miss amplification & cache thrashing.
  3. Phase G2 (`moe_lookahead_router.py` + `moe_expert_tlb.py`): Cross-Layer INT8 `LookaheadRouterBank`
     (`h_{l, mid} -> Layer_{l+1}`) + hazard-free background `nvme_mmap` async DMA overlap.

Strict Invariants Enforced:
  - Zero dropped experts & 100% `0-ulp` VM parity across all cache budgets (`slots_min .. slots_max / all_resident`).
  - Honest PCIe & TLB telemetry (every prefetched byte counted in `pcie_bytes`; pre-prefetch `tlb_hit_rate_pct`
    reported separately from `effective_stall_free_hit_pct` and `lookahead_precision_pct`).
  - Zero eval/test benchmark leakage (`LookaheadRouterBank` calibrated strictly on `Logic_Calib + Fact_Calib`).
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
from decompiler.nvm_serializer import load_nvm_cartridge
from run_phase4_giant_killer import (
    _build_gsm8k_eval_prompts,
    _extract_gsm8k_answer,
    _get_cuda_mem_mb,
    _normalize_num_str,
)
from run_phase_c2_streaming_nvm import get_os_process_ram_mb
from runtime.moe_lookahead_router import (
    LookaheadRouterBank,
    calibrate_and_save_lookahead_routers,
)
from runtime.speculative_vm import (
    EagleLiteDraftHead,
    SpeculativeDecodeStats,
    SpeculativeVMGenerator,
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


def _seed_baselines_from_phase_f(data: Dict[str, Any], phase_f_json: str) -> None:
    if not phase_f_json or not os.path.isfile(phase_f_json):
        return
    pf = _load_or_init_json(phase_f_json)
    for k in ("model_id", "nvm_dir", "teacher_bf16_streaming", "layer_offload_bf16", "compile_telemetry"):
        if k in pf and k not in data:
            data[k] = pf[k]
    if "phase_f_baseline_summary" not in data and "pareto_vram_sweep" in pf:
        pf_pareto = {
            cfg: {
                "tok_per_s": v.get("combined_mean_tok_per_s"),
                "ms_per_tok": v.get("combined_mean_ms_per_tok"),
                "pcie_mb_per_tok": v.get("combined_mean_pcie_mb_per_tok"),
                "tlb_hit_pct": v.get("combined_mean_tlb_hit_pct"),
                "tokens_per_step": v.get("combined_mean_tokens_per_step"),
                "cuda_allocated_mb": v.get("cuda_allocated_mb"),
            }
            for cfg, v in pf["pareto_vram_sweep"].items()
        }
        pf_gsm = {}
        for gk, gv in pf.get("gsm8k_cot_suite", {}).items():
            if isinstance(gv, dict) and "tok_per_s" in gv:
                pf_gsm[gk] = {
                    "solved": gv.get("solved"),
                    "total": gv.get("total"),
                    "accuracy_pct": gv.get("accuracy_pct"),
                    "tok_per_s": gv.get("tok_per_s"),
                    "ms_per_token": gv.get("ms_per_token"),
                    "tokens_per_step": gv.get("tokens_per_step"),
                    "acceptance_rate_pct": gv.get("acceptance_rate_pct"),
                    "pcie_mb_per_token": gv.get("pcie_mb_per_token"),
                    "tlb_hit_rate_pct": gv.get("tlb_hit_rate_pct"),
                }
        data["phase_f_baseline_summary"] = {
            "source_json": phase_f_json,
            "pareto_vram_sweep": pf_pareto,
            "gsm8k_cot_suite": pf_gsm,
        }


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
def _measure_core_only_pass_ms(model: torch.nn.Module, moe_layers: Dict[int, Any], device: str = "cuda:0") -> Dict[str, float]:
    """Measures pure `core_only` (`0` expert pages) GPU forward pass time at `M=1` and `M=4`."""
    dev = torch.device(device)
    for m in moe_layers.values():
        m.set_mode("core_only")
    prompt = torch.randint(100, 2000, (1, 128), device=dev)
    mask = torch.ones((1, 256), dtype=torch.long, device=dev)
    results: Dict[str, float] = {}

    for m_tok in (1, 4):
        out = model(input_ids=prompt, attention_mask=mask[:, :128], use_cache=True)
        pkv = out.past_key_values
        cur_len = 128
        for _ in range(4):
            cur_len += m_tok
            out = model(input_ids=prompt[:, :m_tok], attention_mask=mask[:, :cur_len], past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
        torch.cuda.synchronize(dev)

        n_iters = 12
        t0 = time.perf_counter()
        for _ in range(n_iters):
            cur_len += m_tok
            out = model(input_ids=prompt[:, :m_tok], attention_mask=mask[:, :cur_len], past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
        torch.cuda.synchronize(dev)
        results[f"m{m_tok}_ms_per_pass"] = ((time.perf_counter() - t0) * 1000.0) / float(n_iters)
        del pkv, out

    return results


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
    total_demand_misses = 0
    total_la_pref = 0
    total_la_hits = 0
    total_la_wasted = 0
    hit_rates: List[float] = []
    eff_hit_rates: List[float] = []
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
        total_demand_misses += st.demand_misses
        total_la_pref += st.lookahead_prefetched
        total_la_hits += st.lookahead_hits
        total_la_wasted += st.lookahead_wasted
        hit_rates.append(st.tlb_hit_rate_pct)
        eff_hit_rates.append(st.effective_stall_free_hit_pct)
        all_gen_ids.append(st.generated_ids)

    em = max(1, total_emitted)
    st_cnt = max(1, total_steps)
    tot_misses = max(1, total_demand_misses + total_la_hits)
    la_prec = 100.0 * float(total_la_hits) / float(max(1, total_la_pref)) if total_la_pref > 0 else 0.0
    la_cov = 100.0 * float(total_la_hits) / float(tot_misses) if total_la_hits > 0 else 0.0
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
        "effective_stall_free_hit_pct": sum(eff_hit_rates) / max(1, len(eff_hit_rates)),
        "demand_misses": total_demand_misses,
        "lookahead_prefetched": total_la_pref,
        "lookahead_hits": total_la_hits,
        "lookahead_wasted": total_la_wasted,
        "lookahead_precision_pct": la_prec,
        "lookahead_coverage_pct": la_cov,
        "generated_ids_per_prompt": all_gen_ids,
    }


@torch.no_grad()
def _evaluate_gsm8k_speculative(
    gen: SpeculativeVMGenerator,
    problems: List[Tuple[str, str]],
    max_new_tokens: int = 320,
    device: str = "cuda:0",
    label: str = "Phase G",
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
    total_demand_misses = 0
    total_la_pref = 0
    total_la_hits = 0
    total_la_wasted = 0
    hit_rates: List[float] = []
    eff_hit_rates: List[float] = []
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
        total_demand_misses += st.demand_misses
        total_la_pref += st.lookahead_prefetched
        total_la_hits += st.lookahead_hits
        total_la_wasted += st.lookahead_wasted
        hit_rates.append(st.tlb_hit_rate_pct)
        eff_hit_rates.append(st.effective_stall_free_hit_pct)

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
                "acceptance_rate_pct": st.acceptance_rate_pct,
                "pcie_mb_per_token": st.pcie_mb_per_token,
                "tlb_hit_rate_pct": st.tlb_hit_rate_pct,
                "effective_stall_free_hit_pct": st.effective_stall_free_hit_pct,
            }
        )
        if (idx + 1) % 5 == 0 or (idx + 1) == total:
            cur_tps = float(max(1, total_emitted)) / max(1e-6, total_elapsed_s)
            cur_mul = float(max(1, total_emitted)) / float(max(1, total_steps))
            cur_pcie = total_pcie_mb / float(max(1, total_emitted))
            cur_hit = sum(hit_rates) / max(1, len(hit_rates))
            cur_eff = sum(eff_hit_rates) / max(1, len(eff_hit_rates))
            cur_acc = 100.0 * float(total_acc) / float(max(1, total_prop)) if total_prop > 0 else 0.0
            print(
                f"  [{label} GSM8K {idx+1:02d}/{total:02d}] Solved={correct}/{idx+1} | "
                f"Speed={cur_tps:.2f} tok/s ({1000.0/max(1e-6,cur_tps):.1f} ms/tok) | "
                f"Tok/Step={cur_mul:.2f}x (Acc={cur_acc:.1f}%) | TLB={cur_hit:.1f}%->Eff={cur_eff:.1f}% | "
                f"PCIe={cur_pcie:.1f} MB/tok",
                flush=True,
            )

    em = max(1, total_emitted)
    st_cnt = max(1, total_steps)
    cuda_alloc_mb, cuda_peak_mb = _get_cuda_mem_mb()
    ram_info = get_os_process_ram_mb()
    tot_misses = max(1, total_demand_misses + total_la_hits)

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
        "effective_stall_free_hit_pct": sum(eff_hit_rates) / max(1, len(eff_hit_rates)),
        "demand_misses": total_demand_misses,
        "lookahead_prefetched": total_la_pref,
        "lookahead_hits": total_la_hits,
        "lookahead_wasted": total_la_wasted,
        "lookahead_precision_pct": (
            100.0 * float(total_la_hits) / float(max(1, total_la_pref)) if total_la_pref > 0 else 0.0
        ),
        "lookahead_coverage_pct": (
            100.0 * float(total_la_hits) / float(tot_misses) if total_la_hits > 0 else 0.0
        ),
        "cuda_allocated_mb": cuda_alloc_mb,
        "cuda_peak_mb": cuda_peak_mb,
        "os_working_set_mb": ram_info["ws_mb"],
        "os_private_commit_mb": ram_info["commit_mb"],
        "per_problem": per_problem,
    }


def stage_calibrate_lookahead(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    data = _load_or_init_json(args.out_json)
    _seed_baselines_from_phase_f(data, args.phase_f_json)

    tokenizer = AutoTokenizer.from_pretrained(args.nvm_dir, local_files_only=True)
    bundle = build_contrastive_datasets(
        tokenizer=tokenizer,
        num_calibration_seqs=16,
        num_eval_seqs=4,
        seq_len=256,
        seed=args.seed,
        allow_hf_download=False,
    )
    calib_seqs: List[torch.Tensor] = []
    for i in range(int(bundle.logic_calib.input_ids.shape[0])):
        calib_seqs.append(bundle.logic_calib.input_ids[i].clone())
    for i in range(int(bundle.fact_calib.input_ids.shape[0])):
        calib_seqs.append(bundle.fact_calib.input_ids[i].clone())

    print(
        f"[Phase G2 Calibrate] Calibrating Cross-Layer LookaheadRouterBank on {len(calib_seqs)} strictly held-in "
        f"calibration sequences (0 eval/test leakage) -> {args.nvm_dir}",
        flush=True,
    )
    cal_metrics = calibrate_and_save_lookahead_routers(
        nvm_dir=args.nvm_dir,
        calib_token_sequences=calib_seqs,
        k_pre=args.k_pre if args.k_pre > 0 else None,
        ridge_lambda=10.0,
        kl_steps=args.lookahead_kl_steps,
        lr=3.0e-3,
        batch_size=4,
        tlb_slots_override=max(args.default_tlb_slots, 64),
        device=args.device,
    )
    data["lookahead_calibration"] = cal_metrics
    _save_json(args.out_json, data)


def stage_bench(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    data = _load_or_init_json(args.out_json)
    _seed_baselines_from_phase_f(data, args.phase_f_json)

    ram_base = get_os_process_ram_mb()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"[Phase G Bench] Cold-booting MoE .nvm cartridge (nvme_mmap) from {args.nvm_dir}...", flush=True)
    model, tokenizer, moe_layers, lut_pool, ple_off, manifest = load_nvm_cartridge(
        nvm_dir=args.nvm_dir,
        device=args.device,
        dtype=torch.bfloat16,
        mode="vm_paged",
        tlb_strategy="lookahead_overlap",
        page_storage="nvme_mmap",
        tlb_slots_override=args.default_tlb_slots,
        use_phase_c_kernels=True,
        load_tokenizer=True,
    )
    boot_info = manifest["last_cold_boot"]
    ram_after_boot = get_os_process_ram_mb()

    # Measure core-only resident VRAM + Phase G0 fused core forward speed (`M=1` & `M=4`)
    for m in moe_layers.values():
        m.set_mode("core_only")
    torch.cuda.empty_cache()
    core_only_cuda_mb, _ = _get_cuda_mem_mb()
    core_pass_telemetry = _measure_core_only_pass_ms(model, moe_layers, device=args.device)
    print(
        f"[Phase G0 Core] M=1={core_pass_telemetry['m1_ms_per_pass']:.2f} ms/pass | "
        f"M=4={core_pass_telemetry['m4_ms_per_pass']:.2f} ms/pass | Core VRAM={core_only_cuda_mb:,.1f} MB",
        flush=True,
    )

    # Restore default TLB slots and load EagleLiteDraftHead + LookaheadRouterBank
    for m in moe_layers.values():
        m.set_mode("vm_paged", tlb_slots_override=args.default_tlb_slots)
        m.set_tlb_strategy("lookahead_overlap")

    draft_head = EagleLiteDraftHead.load_pretrained_nvm(
        nvm_dir=args.nvm_dir,
        device=args.device,
        dtype=torch.bfloat16,
        lut_pool=lut_pool,
    )
    draft_vram_mb = draft_head.vram_bytes() / float(1024 ** 2)

    assert LookaheadRouterBank.has_saved_nvm(args.nvm_dir), (
        f"Missing `lookahead_routers.safetensors` in {args.nvm_dir}. Run `--stage calibrate_lookahead` first."
    )
    lookahead_bank = LookaheadRouterBank.load_pretrained_nvm(
        nvm_dir=args.nvm_dir,
        device=args.device,
        lut_pool=lut_pool,
        k_pre_override=args.k_pre if args.k_pre > 0 else None,
    )
    n_attached = lookahead_bank.attach_to_moe_layers(moe_layers)
    lookahead_vram_mb = lookahead_bank.vram_bytes() / float(1024 ** 2)
    ram_with_heads = get_os_process_ram_mb()

    print(
        f"[Phase G Bench] Cold-boot={boot_info['cold_boot_s']:.2f}s | Core CUDA={core_only_cuda_mb:,.1f} MB | "
        f"EagleHead={draft_vram_mb:.1f} MB | LookaheadBank({n_attached} pairs)={lookahead_vram_mb:.1f} MB | "
        f"OS Working Set={ram_with_heads['ws_mb']:,.1f} MB (+{ram_with_heads['ws_mb'] - ram_base['ws_mb']:.1f} MB)",
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

    # 1. Exact Perplexity & 0-ulp Resident vs Lookahead-Paged PPL Parity on Held-Out `Logic_Eval` & `Fact_Eval`
    print("[Phase G Bench] Evaluating Logic_Eval & Fact_Eval Perplexity (vm_paged lookahead vs all_resident/high_slots)...", flush=True)
    for m in moe_layers.values():
        m.set_tlb_strategy("lookahead_overlap")
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
        ref_label = "all_resident"
    else:
        max_sl = max(pareto_slots) if pareto_slots else args.default_tlb_slots
        for m in moe_layers.values():
            m.set_mode("vm_paged", tlb_slots_override=max_sl)
            m.reset_telemetry(clear_tlb=True)
        ppl_resident = _evaluate_loaded_model_ppl(
            model,
            {"Logic_Eval": bundle.logic_eval, "Fact_Eval": bundle.fact_eval},
            teacher_preds_by_split=teacher_preds,
            batch_size=2,
            device=args.device,
        )
        ref_label = f"vm_paged(slots={max_sl})"

    for sn in ("Logic_Eval", "Fact_Eval"):
        diff_loss = abs(ppl_paged[sn]["loss"] - ppl_resident[sn]["loss"])
        assert diff_loss < 1e-6, f"PPL parity violation on {sn}: diff_loss={diff_loss}"
        t_ag = ppl_paged[sn]["teacher_top1_agreement_pct"]
        t_ag_str = f"{t_ag:.2f}%" if t_ag is not None else "N/A"
        print(
            f"  [{sn}] vm_paged(lookahead) PPL={ppl_paged[sn]['perplexity']:.4f} | "
            f"{ref_label} PPL={ppl_resident[sn]['perplexity']:.4f} | "
            f"|delta_loss|={diff_loss:.2e} | Teacher Top-1 Agree={t_ag_str}",
            flush=True,
        )

    # 2. VRAM vs Throughput Pareto + Ablation Sweep (`nvme_mmap`) + Hard 0-ulp Cross-Budget Token Parity!
    fact_prompts = [bundle.fact_eval.input_ids[i : i + 1, :128].to(args.device) for i in range(2)]
    logic_prompts = [bundle.logic_eval.input_ids[i : i + 1, :128].to(args.device) for i in range(2)]

    pareto_results: Dict[str, Any] = {}
    ref_tokens_by_family: Dict[str, Tuple[List[List[int]], List[List[int]]]] = {}

    low_vram_slots = min(max(32, num_experts_per_layer // 8), num_experts_per_layer)
    ablation_slots = {pareto_slots[0], low_vram_slots} if pareto_slots else {low_vram_slots}

    # Tuple: `(cfg_key, parity_family, vm_mode, sl_cnt, strat, d_mode, conf_gated)`
    sweep_configs: List[Tuple[str, str, str, int, str, str, bool]] = []
    for sl in pareto_slots:
        if sl < num_experts_per_layer:
            sweep_configs.append((f"slots_{sl}_m1_demand", "m1", "vm_paged", sl, "exact_demand", "none", False))
            sweep_configs.append((f"slots_{sl}_m1_overlap", "m1", "vm_paged", sl, "overlap_shared_mlp", "none", False))
            sweep_configs.append((f"slots_{sl}_m1_lookahead", "m1", "vm_paged", sl, "lookahead_overlap", "none", False))
            if sl in ablation_slots:
                sweep_configs.append(
                    (f"slots_{sl}_spec_ungated_intra", "spec_ungated", "vm_paged", sl, "overlap_shared_mlp", "two_tier", False)
                )
                sweep_configs.append(
                    (f"slots_{sl}_spec_gated_overlap", "spec_phase_g", "vm_paged", sl, "overlap_shared_mlp", "two_tier", True)
                )
            sweep_configs.append(
                (f"slots_{sl}_spec_phase_g", "spec_phase_g", "vm_paged", sl, "lookahead_overlap", "two_tier", True)
            )
    if can_fit_all_resident:
        sweep_configs.append(
            ("all_resident_m1", "m1", "all_resident", num_experts_per_layer, "lookahead_overlap", "none", False)
        )
        sweep_configs.append(
            (
                "all_resident_spec_phase_g",
                "spec_phase_g",
                "all_resident",
                num_experts_per_layer,
                "lookahead_overlap",
                "two_tier",
                True,
            )
        )

    print(
        f"[Phase G Bench] Running {len(sweep_configs)}-point Pareto & Ablation Sweep ({args.eval_decode_tokens} tok x 4 prompts)...",
        flush=True,
    )
    for cfg_key, family, vm_mode, sl_cnt, strat, d_mode, conf_gated in sweep_configs:
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
            confidence_gated_moe=conf_gated,
        )
        f_res = _bench_eval_prompts(gen, fact_prompts, max_new_tokens=args.eval_decode_tokens)
        l_res = _bench_eval_prompts(gen, logic_prompts, max_new_tokens=args.eval_decode_tokens)
        _, peak_mb = _get_cuda_mem_mb()
        ram_cur = get_os_process_ram_mb()

        f_ids = f_res.pop("generated_ids_per_prompt")
        l_ids = l_res.pop("generated_ids_per_prompt")
        if family not in ref_tokens_by_family:
            ref_tokens_by_family[family] = (f_ids, l_ids)
            parity_exact = True
        else:
            rf, rl = ref_tokens_by_family[family]
            parity_exact = (f_ids == rf) and (l_ids == rl)
            assert parity_exact, f"0-ulp VM parity mismatch in '{cfg_key}' vs '{family}' reference!"

        m1_prefix_matches: List[int] = []
        if "m1" in ref_tokens_by_family:
            m1_f, m1_l = ref_tokens_by_family["m1"]
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
        mean_eff_hit = 0.5 * (f_res["effective_stall_free_hit_pct"] + l_res["effective_stall_free_hit_pct"])
        mean_mul = 0.5 * (f_res["tokens_per_step"] + l_res["tokens_per_step"])
        mean_acc = 0.5 * (f_res["acceptance_rate_pct"] + l_res["acceptance_rate_pct"])
        mean_la_prec = 0.5 * (f_res["lookahead_precision_pct"] + l_res["lookahead_precision_pct"])
        mean_la_cov = 0.5 * (f_res["lookahead_coverage_pct"] + l_res["lookahead_coverage_pct"])

        pareto_results[cfg_key] = {
            "parity_family": family,
            "vm_mode": vm_mode,
            "tlb_slots": sl_cnt,
            "cache_pct": 100.0 * float(sl_cnt) / float(num_experts_per_layer),
            "tlb_strategy": strat,
            "draft_mode": d_mode,
            "confidence_gated_moe": conf_gated,
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
            "combined_mean_effective_hit_pct": mean_eff_hit,
            "combined_mean_lookahead_precision_pct": mean_la_prec,
            "combined_mean_lookahead_coverage_pct": mean_la_cov,
            "combined_mean_tokens_per_step": mean_mul,
            "combined_mean_acceptance_pct": mean_acc,
            "fact_eval": f_res,
            "logic_eval": l_res,
        }
        print(
            f"  [{cfg_key:30s}] VRAM={alloc_mb:7.1f} MB | {mean_tps:5.2f} tok/s ({mean_ms:5.1f} ms/tok) | "
            f"Tok/Step={mean_mul:.2f}x (Acc={mean_acc:4.1f}%) | TLB={mean_hit:5.1f}%->Eff={mean_eff_hit:5.1f}% | "
            f"PCIe={mean_pcie:6.1f} MB/tok | 0-ulp={parity_exact}",
            flush=True,
        )

    # 3. Held-Out 20-Problem Zero-Shot CoT GSM8K (`test` split, `max_new_tokens=320`)
    gsm8k_problems = _build_gsm8k_eval_prompts(num_problems=args.gsm8k_count, split="test")
    gsm8k_suite: Dict[str, Any] = {}

    # 3a. 5-problem M=1 (`lookahead_overlap`) baseline at low-VRAM budget (`64` on 80B = ~7.9 GB; `32` on 26B = ~6.3 GB)
    for m in moe_layers.values():
        m.set_tlb_strategy("lookahead_overlap")
        m.set_mode("vm_paged", tlb_slots_override=low_vram_slots)
        m.reset_telemetry(clear_tlb=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gen_low_m1 = SpeculativeVMGenerator(
        model=model,
        tokenizer=tokenizer,
        phase_b_mlps=moe_layers,
        draft_mode="none",
        confidence_gated_moe=False,
    )
    print(f"[Phase G GSM8K] 5-Problem M=1 Lookahead Baseline (tlb_slots={low_vram_slots}, nvme_mmap)...", flush=True)
    gsm8k_suite[f"vm_slots_{low_vram_slots}_m1_lookahead_5prob"] = _evaluate_gsm8k_speculative(
        gen_low_m1,
        gsm8k_problems[:5],
        max_new_tokens=args.gsm8k_max_tokens,
        device=args.device,
        label=f"Slots{low_vram_slots}-M1-LA",
    )

    # 3b. Full 20-problem GSM8K at `low_vram_slots` + Full Phase G (`lookahead_overlap` + Confidence-Gated Two-Tier)
    for m in moe_layers.values():
        m.set_tlb_strategy("lookahead_overlap")
        m.set_mode("vm_paged", tlb_slots_override=low_vram_slots)
        m.reset_telemetry(clear_tlb=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gen_low_spec = SpeculativeVMGenerator(
        model=model,
        tokenizer=tokenizer,
        phase_b_mlps=moe_layers,
        draft_head=draft_head,
        draft_mode="two_tier",
        max_draft_tokens=3,
        pld_long_draft_tokens=4,
        confidence_gated_moe=True,
    )
    print(
        f"[Phase G GSM8K] Full {len(gsm8k_problems)}-Problem CoT (tlb_slots={low_vram_slots}, Phase G Lookahead + Gated Spec)...",
        flush=True,
    )
    res_low_full = _evaluate_gsm8k_speculative(
        gen_low_spec,
        gsm8k_problems,
        max_new_tokens=args.gsm8k_max_tokens,
        device=args.device,
        label=f"Slots{low_vram_slots}-PhaseG",
    )
    gsm8k_suite[f"vm_slots_{low_vram_slots}_phase_g_20prob"] = res_low_full

    m1_vs_spec_ans_agree = 0
    for i5 in range(min(5, len(gsm8k_problems))):
        p_m1 = gsm8k_suite[f"vm_slots_{low_vram_slots}_m1_lookahead_5prob"]["per_problem"][i5]
        p_sp = res_low_full["per_problem"][i5]
        if p_m1["pred"] == p_sp["pred"]:
            m1_vs_spec_ans_agree += 1
    gsm8k_suite["m1_vs_phase_g_5prob_answer_agreement"] = f"{m1_vs_spec_ans_agree}/{min(5, len(gsm8k_problems))}"

    # 3c. Cross-slot 5-problem speed & hard `0-ulp` parity checks across extreme low, default, and ceiling VRAM!
    check_slots_list: List[Tuple[str, str, int]] = []
    if pareto_slots and min(pareto_slots) < low_vram_slots:
        mn_sl = min(pareto_slots)
        check_slots_list.append((f"vm_slots_{mn_sl}_phase_g_5prob", "vm_paged", mn_sl))
    if args.default_tlb_slots != low_vram_slots:
        check_slots_list.append(
            (f"vm_slots_{args.default_tlb_slots}_phase_g_5prob", "vm_paged", args.default_tlb_slots)
        )
    if can_fit_all_resident:
        check_slots_list.append(("all_resident_phase_g_5prob", "all_resident", num_experts_per_layer))
    elif pareto_slots and max(pareto_slots) > max(low_vram_slots, args.default_tlb_slots):
        hi_sl = max(pareto_slots)
        check_slots_list.append((f"vm_slots_{hi_sl}_phase_g_5prob", "vm_paged", hi_sl))

    for suite_key, v_mode, sl_val in check_slots_list:
        for m in moe_layers.values():
            m.set_tlb_strategy("lookahead_overlap")
            m.set_mode(v_mode, tlb_slots_override=sl_val)
            m.reset_telemetry(clear_tlb=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gen_chk = SpeculativeVMGenerator(
            model=model,
            tokenizer=tokenizer,
            phase_b_mlps=moe_layers,
            draft_head=draft_head,
            draft_mode="two_tier",
            max_draft_tokens=3,
            pld_long_draft_tokens=4,
            confidence_gated_moe=True,
        )
        print(f"[Phase G GSM8K] 5-Problem Cross-Slot Speed & 0-ulp Check ({suite_key})...", flush=True)
        res_chk = _evaluate_gsm8k_speculative(
            gen_chk,
            gsm8k_problems[:5],
            max_new_tokens=args.gsm8k_max_tokens,
            device=args.device,
            label=suite_key,
        )
        gsm8k_suite[suite_key] = res_chk
        for i5 in range(min(5, len(gsm8k_problems))):
            p_ref = res_low_full["per_problem"][i5]
            p_cur = res_chk["per_problem"][i5]
            assert p_ref["pred"] == p_cur["pred"] and p_ref["emitted_tokens"] == p_cur["emitted_tokens"], (
                f"0-ulp VM parity mismatch GSM8K[{i5}] Slots{low_vram_slots} vs {suite_key}: {p_ref} vs {p_cur}"
            )

    data["hardware"] = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cold_boot_s": boot_info["cold_boot_s"],
        "core_only_resident_cuda_mb": core_only_cuda_mb,
        "core_only_pass_ms": core_pass_telemetry,
        "draft_head_vram_mb": draft_vram_mb,
        "lookahead_bank_vram_mb": lookahead_vram_mb,
        "lookahead_pairs_attached": n_attached,
        "baseline_os_ws_mb": ram_base["ws_mb"],
        "baseline_os_commit_mb": ram_base["commit_mb"],
        "boot_os_ws_mb": ram_after_boot["ws_mb"],
        "boot_os_commit_mb": ram_after_boot["commit_mb"],
        "os_ws_delta_mb": ram_with_heads["ws_mb"] - ram_base["ws_mb"],
        "host_only_commit_delta_mb": max(
            0.0,
            (ram_with_heads["commit_mb"] - ram_base["commit_mb"])
            - (core_only_cuda_mb + draft_vram_mb + lookahead_vram_mb),
        ),
    }
    data["perplexity"] = {
        "vm_paged": ppl_paged,
        "all_resident": ppl_resident,
    }
    data["pareto_vram_sweep"] = pareto_results
    data["gsm8k_cot_suite"] = gsm8k_suite
    _save_json(args.out_json, data)
    print(f"[Phase G Bench] Saved complete telemetry -> {args.out_json}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        choices=("calibrate_lookahead", "bench", "all"),
        default="all",
    )
    p.add_argument("--nvm-dir", default="checkpoints/Qwen3-Next-80B-A3B-Instruct-nvm-moe.nvm")
    p.add_argument("--out-json", default="checkpoints/phase_g_80b_moe_results.json")
    p.add_argument("--phase-f-json", default="checkpoints/phase_f_80b_moe_results.json")
    p.add_argument("--tag", default="qwen3_next_80b_moe")
    p.add_argument("--default-tlb-slots", type=int, default=96)
    p.add_argument("--pareto-slots", default="32,64,96,128,192")
    p.add_argument("--k-pre", type=int, default=0, help="0 = auto-select from model top_k")
    p.add_argument("--lookahead-kl-steps", type=int, default=60)
    p.add_argument("--eval-decode-tokens", type=int, default=64)
    p.add_argument("--gsm8k-count", type=int, default=20)
    p.add_argument("--gsm8k-max-tokens", type=int, default=320)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    if args.stage == "all":
        for sub_stage in ("calibrate_lookahead", "bench"):
            print(f"\n==================== PHASE G STAGE: {sub_stage} ====================", flush=True)
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--stage",
                sub_stage,
                "--nvm-dir",
                args.nvm_dir,
                "--out-json",
                args.out_json,
                "--phase-f-json",
                args.phase_f_json,
                "--tag",
                args.tag,
                "--default-tlb-slots",
                str(args.default_tlb_slots),
                "--pareto-slots",
                args.pareto_slots,
                "--k-pre",
                str(args.k_pre),
                "--lookahead-kl-steps",
                str(args.lookahead_kl_steps),
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

    if args.stage == "calibrate_lookahead":
        stage_calibrate_lookahead(args)
    elif args.stage == "bench":
        stage_bench(args)


if __name__ == "__main__":
    main()
