import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
"""Phase B Driver: Bit-Tiered Resident Core + Persistent GPU LRU Page Cache (TLB) Benchmark.

Performs a direct, hardware-honest apple-to-apple comparison on `google/gemma-4-E2B-it`
(and scales directly to `google/gemma-4-E4B-it`) against the Phase A baseline across:
  1. Ground-Truth `Logic_Eval` & `Fact_Eval` Perplexity (`shift_labels`, zero argmax leakage).
  2. Real PyTorch CUDA Allocator Footprint (`torch.cuda.memory_allocated()` & Peak MB) with
     genuinely bit-packed `int16` (5x 3-bit) and `uint8` (2x 4-bit) weights.
  3. True KV-Cached (`use_cache=True`) Autoregressive Decode Speed, Isolated 28-Layer CUDA-Event
     Latency, Measured GPU TLB Cache Hit Rate (%), and Measured PCIe MB/token.
  4. True Zero-Shot Chain-of-Thought GSM8K (`test` split, question-only prompt, `max_new_tokens=320`).
"""

import argparse
import gc
import os
import time
from typing import Any, Dict, List, Optional, Tuple

if os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub")) and os.path.isdir(
    os.path.expanduser("~/.cache/huggingface/datasets")
):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.nn as nn

from config import NeuralVMConfig
from data.calibration import ContrastiveCalibrationBundle, build_contrastive_datasets
from decompiler.full_mlp_svd import capture_teacher_full_mlp_traces
from decompiler.full_model_vm import (
    DeterministicPLEOffloader,
    decompile_full_28_layer_quantized_stack,
    offload_per_layer_embeddings_to_host_ram,
)
from decompiler.hooks import resolve_decoder_layers
from decompiler.phase_b_compiler import (
    PhaseBLayerStats,
    build_uniform_awq_4bit_stack,
    decompile_phase_b_stack,
)
from decompiler.sequential_cascade import snapshot_original_mlp_weights
from run_ablation import MockGemma4ForCausalLM, evaluate_split, load_model_and_tokenizer
from run_phase4_giant_killer import (
    _aggregate_phase4_vram_and_telemetry,
    _build_gsm8k_eval_prompts,
    _get_cuda_mem_mb,
    _restore_dense_full_mlps,
    evaluate_gsm8k_zero_shot_exact_match,
)
from runtime.gpu_page_tlb import BitTieredTLBMLP, SharedQuantLUTPool, UniformAWQ4BitMLP


def aggregate_phase_b_vram_and_telemetry(
    phase_b_mlps: Dict[int, BitTieredTLBMLP],
    lut_pool: SharedQuantLUTPool,
    ple_saved_mb: float = 0.0,
    total_dense_model_mb: float = 0.0,
    ple_offloader: Optional[DeterministicPLEOffloader] = None,
) -> Dict[str, float]:
    """Aggregates 28-layer Phase B VRAM, real CUDA allocated MB, GPU TLB hit rates, and PCIe MB/token."""
    dense_mlp_mb = 0.0
    hot_core_mb = 0.0
    cold_base_mb = 0.0
    bridge_mb = 0.0
    core_total_mb = 0.0
    tlb_cache_mb = 0.0
    host_pages_mb = 0.0

    total_pcie_mb = 0.0
    pcie_mb_per_token = 0.0
    total_hits = 0.0
    total_misses = 0.0
    distinct_list: List[float] = []

    for _, mod in sorted(phase_b_mlps.items()):
        vs = mod.vram_footprint_stats()
        bt = mod.bandwidth_telemetry()
        dense_mlp_mb += vs["dense_vram_mb"]
        hot_core_mb += vs["hot_core_mb"]
        cold_base_mb += vs["cold_base_mb"]
        bridge_mb += vs["bridge_mb"]
        core_total_mb += vs["core_total_mb"]
        tlb_cache_mb += vs["tlb_cache_mb"]
        host_pages_mb += vs["host_ram_pages_mb"]

        total_pcie_mb += bt["total_pcie_mb"]
        pcie_mb_per_token += bt["pcie_mb_per_token"]
        total_hits += bt["tlb_hits"]
        total_misses += bt["tlb_misses"]
        distinct_list.append(bt["distinct_pages"])

    lut_mb = lut_pool.total_vram_bytes / float(1024 ** 2)
    sample_mod = next(iter(phase_b_mlps.values()))
    ws_mb = 0.0
    if sample_mod._phase_c_launcher is not None:
        ws_mb = sample_mod._phase_c_launcher.ws.total_vram_bytes / float(1024 ** 2)
    overhead_mb = lut_mb + ws_mb
    if sample_mod.mode == "core_only":
        active_mlp_mb = core_total_mb + overhead_mb
        tlb_cache_mb = 0.0
    else:
        active_mlp_mb = core_total_mb + tlb_cache_mb + overhead_mb

    ple_kb_tok = 0.0
    if ple_offloader is not None:
        pt = ple_offloader.bandwidth_telemetry()
        total_pcie_mb += pt["ple_total_pcie_mb"]
        ple_kb_tok = pt["ple_pcie_kb_per_token"]
        pcie_mb_per_token += ple_kb_tok / 1024.0

    mlp_saved_mb = dense_mlp_mb - active_mlp_mb
    mlp_red_pct = 100.0 * (1.0 - (active_mlp_mb / max(1e-6, dense_mlp_mb)))
    total_saved_mb = mlp_saved_mb + ple_saved_mb
    active_model_mb = total_dense_model_mb - total_saved_mb
    total_red_pct = 100.0 * (total_saved_mb / max(1e-6, total_dense_model_mb))
    hit_rate_pct = 100.0 * (total_hits / max(1.0, total_hits + total_misses))
    avg_distinct = sum(distinct_list) / max(1, len(distinct_list))
    cuda_alloc_mb, cuda_peak_mb = _get_cuda_mem_mb()

    return {
        "dense_mlp_mb": dense_mlp_mb,
        "hot_core_mb": hot_core_mb,
        "cold_base_mb": cold_base_mb,
        "bridge_mb": bridge_mb,
        "core_total_mb": core_total_mb + lut_mb,
        "tlb_cache_mb": tlb_cache_mb,
        "active_mlp_mb": active_mlp_mb,
        "host_pages_mb": host_pages_mb,
        "mlp_reduction_pct": mlp_red_pct,
        "ple_saved_mb": ple_saved_mb,
        "active_total_model_mb": active_model_mb,
        "total_model_reduction_pct": total_red_pct,
        "cuda_allocated_mb": cuda_alloc_mb,
        "cuda_peak_mb": cuda_peak_mb,
        "pcie_mb_per_token": pcie_mb_per_token,
        "ple_kb_per_token": ple_kb_tok,
        "tlb_hit_rate_pct": hit_rate_pct,
        "tlb_hits": total_hits,
        "tlb_misses": total_misses,
        "avg_distinct_pages": avg_distinct,
    }


@torch.no_grad()
def measure_phase_b_kv_decode(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    num_decode_tokens: int = 64,
    device: str = "cuda:0",
    phase_b_mlps: Optional[Dict[int, BitTieredTLBMLP]] = None,
    lut_pool: Optional[SharedQuantLUTPool] = None,
    ple_saved_mb: float = 0.0,
    total_dense_model_mb: float = 0.0,
    ple_offloader: Optional[DeterministicPLEOffloader] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """Measures KV-cached (`use_cache=True`) single-token decode speed after prompt prefill TLB warmup."""
    model.eval()
    is_mock = isinstance(model, MockGemma4ForCausalLM)
    p_ids = prompt_ids[:1, : min(64, prompt_ids.shape[1])].to(device)
    p_mask = torch.ones_like(p_ids, device=device)

    if is_mock:
        if phase_b_mlps is not None:
            for m in phase_b_mlps.values():
                m.reset_telemetry(clear_tlb=True)
            _ = model(input_ids=p_ids, attention_mask=p_mask, use_cache=False)
            for m in phase_b_mlps.values():
                m.reset_telemetry(clear_tlb=False)
        tok_in = p_ids[:, -1:]
        t_mask = torch.ones_like(tok_in, device=device)
        t0 = time.perf_counter()
        for _ in range(num_decode_tokens):
            out = model(input_ids=tok_in, attention_mask=t_mask, use_cache=False)
            tok_in = out.logits[:, -1:, :].argmax(dim=-1)
        elapsed = time.perf_counter() - t0
        ms = (elapsed * 1000.0) / max(1, num_decode_tokens)
        tps = num_decode_tokens / max(1e-6, elapsed)
        tel = (
            aggregate_phase_b_vram_and_telemetry(
                phase_b_mlps,
                lut_pool=lut_pool,
                ple_saved_mb=ple_saved_mb,
                total_dense_model_mb=total_dense_model_mb,
                ple_offloader=ple_offloader,
            )
            if (phase_b_mlps is not None and lut_pool is not None)
            else {}
        )
        return ms, tps, tel

    # Kernel warmup
    w_out = model(input_ids=p_ids, attention_mask=p_mask, use_cache=True)
    w_pkv = w_out.past_key_values
    w_tok = w_out.logits[:, -1:, :].argmax(dim=-1)
    w_mask = p_mask.clone()
    for _ in range(4):
        w_mask = torch.cat([w_mask, torch.ones((1, 1), dtype=p_mask.dtype, device=device)], dim=1)
        w_out = model(input_ids=w_tok, attention_mask=w_mask, past_key_values=w_pkv, use_cache=True)
        w_pkv = w_out.past_key_values
        w_tok = w_out.logits[:, -1:, :].argmax(dim=-1)
    del w_pkv, w_out
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Clear TLB before prefill so prefill genuinely warms the M-slot GPU TLB from cold start
    if phase_b_mlps is not None:
        for mod in phase_b_mlps.values():
            mod.reset_telemetry(clear_tlb=True)

    out = model(input_ids=p_ids, attention_mask=p_mask, use_cache=True)
    pkv = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    cur_mask = p_mask.clone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Reset counters (keep warm TLB contents from prompt prefill!) to measure steady-state decode
    if phase_b_mlps is not None:
        for mod in phase_b_mlps.values():
            mod.reset_telemetry(clear_tlb=False)
    if ple_offloader is not None:
        ple_offloader.reset_telemetry()

    t0 = time.perf_counter()
    for _ in range(num_decode_tokens):
        cur_mask = torch.cat([cur_mask, torch.ones((1, 1), dtype=p_mask.dtype, device=device)], dim=1)
        out = model(
            input_ids=next_tok,
            attention_mask=cur_mask,
            past_key_values=pkv,
            use_cache=True,
        )
        pkv = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0
    del pkv, out

    ms_per_tok = (elapsed_s * 1000.0) / max(1, num_decode_tokens)
    tok_per_s = num_decode_tokens / max(1e-6, elapsed_s)
    tel = (
        aggregate_phase_b_vram_and_telemetry(
            phase_b_mlps,
            lut_pool=lut_pool,
            ple_saved_mb=ple_saved_mb,
            total_dense_model_mb=total_dense_model_mb,
            ple_offloader=ple_offloader,
        )
        if (phase_b_mlps is not None and lut_pool is not None)
        else {}
    )
    return ms_per_tok, tok_per_s, tel


@torch.no_grad()
def benchmark_isolated_phase_b_28layer_hw(
    phase_b_mlps: Dict[int, BitTieredTLBMLP],
    attn_compute_iters: int = 3,
    num_steps: int = 64,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    """Isolated 28-Layer CUDA-Event microbenchmark comparing Dense BF16 vs Phase B TLB strategies."""
    if not torch.cuda.is_available():
        return {
            "hw_dense_ms": 0.0,
            "hw_core_only_ms": 0.0,
            "hw_stateless_dma_ms": 0.0,
            "hw_sync_tlb_ms": 0.0,
            "hw_spec_tlb_ms": 0.0,
        }

    sorted_mlps = [phase_b_mlps[idx] for idx in sorted(phase_b_mlps.keys())]
    d_model = sorted_mlps[0].d_model
    act_fn = sorted_mlps[0].act_fn

    x_prefill = torch.randn((1, 32, d_model), device=device, dtype=dtype)
    x_tok = x_prefill[:, -1:, :].contiguous()  # Coherent context token for realistic temporal locality!
    w_attn_qkv = torch.randn((2 * d_model, d_model), device=device, dtype=dtype) * 0.02
    w_attn_out = torch.randn((d_model, 2 * d_model), device=device, dtype=dtype) * 0.02

    def _run_attn(h_in: torch.Tensor) -> torch.Tensor:
        cur = h_in.view(-1, d_model)
        for _ in range(attn_compute_iters):
            qkv = cur @ w_attn_qkv.T
            cur = cur + (qkv @ w_attn_out.T)
        return cur.view_as(h_in)

    # 1. Dense BF16 Reference
    min_ffn = min(m.d_ffn for m in sorted_mlps)
    wg_d = torch.randn((min_ffn, d_model), device=device, dtype=dtype) * 0.02
    wu_d = torch.randn((min_ffn, d_model), device=device, dtype=dtype) * 0.02
    wd_d = torch.randn((d_model, min_ffn), device=device, dtype=dtype) * 0.02

    for _ in range(5):
        h = x_tok
        for m in sorted_mlps:
            h = _run_attn(h)
            fh = h.view(1, d_model)
            for _ in range(max(1, m.d_ffn // min_ffn)):
                fh = fh + ((act_fn(fh @ wg_d.T) * (fh @ wu_d.T)) @ wd_d.T)
            h = fh.view(1, 1, d_model)
    torch.cuda.synchronize()

    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(num_steps):
        h = x_tok
        for m in sorted_mlps:
            h = _run_attn(h)
            fh = h.view(1, d_model)
            for _ in range(max(1, m.d_ffn // min_ffn)):
                fh = fh + ((act_fn(fh @ wg_d.T) * (fh @ wu_d.T)) @ wd_d.T)
            h = fh.view(1, 1, d_model)
    ev1.record()
    torch.cuda.synchronize()
    hw_dense_ms = ev0.elapsed_time(ev1) / num_steps
    del wg_d, wu_d, wd_d

    def _time_mode(mode: str, strat: str, use_phase_c: bool = True) -> float:
        for m in sorted_mlps:
            m.set_phase_c_enabled(use_phase_c)
            m.set_mode(mode)
            m.set_tlb_strategy(strat)
            m.reset_telemetry(clear_tlb=True)
        # Prefill warm-up
        hp = x_prefill
        for m in sorted_mlps:
            hp = _run_attn(hp)
            hp = hp + m(hp)
        for _ in range(4):
            ht = x_tok
            for m in sorted_mlps:
                ht = _run_attn(ht)
                ht = ht + m(ht)
        torch.cuda.synchronize()

        e_s, e_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e_s.record()
        for step in range(num_steps):
            ht = x_tok
            for m in sorted_mlps:
                ht = _run_attn(ht)
                ht = ht + m(ht)
        e_e.record()
        torch.cuda.synchronize()
        return e_s.elapsed_time(e_e) / num_steps

    hw_py_core_ms = _time_mode("core_only", "sync_tlb", use_phase_c=False)
    hw_py_spec_ms = _time_mode("vm_paged", "speculative_tlb", use_phase_c=False)

    hw_core_only_ms = _time_mode("core_only", "sync_tlb", use_phase_c=True)
    hw_stateless_ms = _time_mode("vm_paged", "stateless_dma", use_phase_c=True)
    hw_sync_tlb_ms = _time_mode("vm_paged", "sync_tlb", use_phase_c=True)
    hw_spec_tlb_ms = _time_mode("vm_paged", "speculative_tlb", use_phase_c=True)

    for m in sorted_mlps:
        m.set_phase_c_enabled(True)
        m.set_mode("vm_paged")
        m.set_tlb_strategy("sync_tlb")
        m.reset_telemetry(clear_tlb=True)

    del w_attn_qkv, w_attn_out, x_prefill, x_tok
    torch.cuda.empty_cache()
    return {
        "hw_dense_ms": hw_dense_ms,
        "hw_py_core_only_ms": hw_py_core_ms,
        "hw_py_spec_tlb_ms": hw_py_spec_ms,
        "hw_core_only_ms": hw_core_only_ms,
        "hw_stateless_dma_ms": hw_stateless_ms,
        "hw_sync_tlb_ms": hw_sync_tlb_ms,
        "hw_spec_tlb_ms": hw_spec_tlb_ms,
    }


def run_phase_b_pipeline(
    cfg: NeuralVMConfig,
    tlb_slots: int = 12,
    decode_req_pages: int = 4,
    hot_fraction: float = 1.0 / 12.0,
    bridge_rank: int = 64,
    gsm8k_max_new_tokens: int = 320,
    gsm8k_batch_size: int = 5,
    include_phase_a_rerun: bool = True,
    include_uniform_4bit_ref: bool = True,
) -> Dict[str, Any]:
    """Executes the full Phase B vs Phase A benchmark on `cfg.model_id`."""
    torch.manual_seed(cfg.seed)
    t_total_start = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_model_and_tokenizer(cfg)
    dense_cuda_alloc_mb, _ = _get_cuda_mem_mb()
    total_dense_model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    decoder_layers = resolve_decoder_layers(model)
    layer_indices = [idx for idx in range(cfg.phase4_start_layer, cfg.phase4_end_layer + 1) if idx < len(decoder_layers)]
    num_layers = len(layer_indices)

    print(
        f"[Data] Building Contrastive Calibration/Eval Splits "
        f"(calib={cfg.phase2_calib_seqs}x{cfg.seq_len}, eval={cfg.num_eval_seqs}x{cfg.seq_len}) "
        f"& {cfg.gsm8k_eval_problems} Held-Out GSM8K Test Problems..."
    )
    bundle: ContrastiveCalibrationBundle = build_contrastive_datasets(
        tokenizer=tokenizer,
        num_calibration_seqs=cfg.phase2_calib_seqs,
        num_eval_seqs=cfg.num_eval_seqs,
        seq_len=cfg.seq_len,
        seed=cfg.seed,
        allow_hf_download=not cfg.use_mock_model,
    )
    gsm8k_problems = _build_gsm8k_eval_prompts(num_problems=cfg.gsm8k_eval_problems, split="test")

    # =========================================================================
    # 1. State 0: Unmodified Dense BF16 Baseline
    # =========================================================================
    print(
        f"\n[State 0: Dense BF16 Baseline] Evaluating unmodified {num_layers}-layer model "
        f"(Layers {layer_indices[0]}..{layer_indices[-1]}) + CoT GSM8K (max_new_tokens={gsm8k_max_new_tokens})..."
    )
    base_logic, _ = evaluate_split(
        model, bundle.logic_eval, state_name="State 0: Dense Baseline", batch_size=cfg.batch_size, device=cfg.device
    )
    base_fact, _ = evaluate_split(
        model, bundle.fact_eval, state_name="State 0: Dense Baseline", batch_size=cfg.batch_size, device=cfg.device
    )
    gsm_base_ok, gsm_total, gsm_base_acc, gsm_base_trace = evaluate_gsm8k_zero_shot_exact_match(
        model,
        tokenizer,
        gsm8k_problems,
        batch_size=gsm8k_batch_size,
        max_new_tokens=gsm8k_max_new_tokens,
        device=cfg.device,
    )
    base_decode_ms, base_decode_tps, _ = measure_phase_b_kv_decode(
        model=model,
        prompt_ids=bundle.fact_eval.input_ids,
        num_decode_tokens=cfg.decode_tokens,
        device=cfg.device,
    )
    _, dense_cuda_peak_mb = _get_cuda_mem_mb()

    # =========================================================================
    # 2. Snapshot Weights, Capture Teacher Traces & Offload PLE Table
    # =========================================================================
    max_calib_tokens = cfg.phase2_calib_seqs * cfg.seq_len
    print(f"[Teacher] Snapshotting {num_layers} MLPs & capturing calibration X/H traces...")
    snapshots = snapshot_original_mlp_weights(model, layer_indices)
    orig_mlp_modules = {idx: decoder_layers[idx].mlp for idx in layer_indices}

    teacher_h_logic: Dict[int, torch.Tensor] = {}
    teacher_h_fact: Dict[int, torch.Tensor] = {}
    if include_phase_a_rerun:
        teacher_x_logic, teacher_h_logic = capture_teacher_full_mlp_traces(
            model, bundle.logic_calib, layer_indices, batch_size=cfg.batch_size, device=cfg.device, max_tokens=max_calib_tokens
        )
        teacher_x_fact, teacher_h_fact = capture_teacher_full_mlp_traces(
            model, bundle.fact_calib, layer_indices, batch_size=cfg.batch_size, device=cfg.device, max_tokens=max_calib_tokens
        )
    else:
        from decompiler.full_mlp_svd import MultiLayerMLPInputAndActCollector
        from decompiler.hooks import _EarlyExitAfterHook

        def _cap_x_only(split_data: Any) -> Dict[int, torch.Tensor]:
            with MultiLayerMLPInputAndActCollector(
                model, layer_indices, max_tokens=max_calib_tokens, capture_x=True, capture_h=False
            ) as col:
                for st in range(0, split_data.num_sequences, cfg.batch_size):
                    en = min(st + cfg.batch_size, split_data.num_sequences)
                    ids = split_data.input_ids[st:en].to(cfg.device)
                    msk = split_data.attention_mask[st:en].to(cfg.device)
                    col.set_batch_mask(msk)
                    try:
                        model(input_ids=ids, attention_mask=msk, use_cache=False)
                    except _EarlyExitAfterHook:
                        pass
                    if all(c >= max_calib_tokens for c in col.x_counts.values()):
                        break
                return col.get_x_activations()

        teacher_x_logic = _cap_x_only(bundle.logic_calib)
        teacher_x_fact = _cap_x_only(bundle.fact_calib)

    print("[PLE Offload] Offloading static Per-Layer Embedding table to Host RAM...")
    ple_offloader, ple_saved_mb = offload_per_layer_embeddings_to_host_ram(model)

    results: Dict[str, Any] = {
        "model_id": cfg.model_id,
        "num_layers": num_layers,
        "layer_range": (layer_indices[0], layer_indices[-1]),
        "dense_cuda_alloc_mb": dense_cuda_alloc_mb,
        "dense_cuda_peak_mb": dense_cuda_peak_mb,
        "total_dense_model_mb": total_dense_model_mb,
        "ple_saved_mb": ple_saved_mb,
        "base_logic_ppl": base_logic.perplexity,
        "base_fact_ppl": base_fact.perplexity,
        "base_decode_ms": base_decode_ms,
        "base_decode_tps": base_decode_tps,
        "gsm_total": gsm_total,
        "gsm_base": (gsm_base_ok, gsm_base_acc, gsm_base_trace),
    }

    # =========================================================================
    # 3. Optional Direct Phase A Rerun (Same Session / Same GSM8K Token Budget)
    # =========================================================================
    if include_phase_a_rerun:
        print(f"\n[Phase A Reference] Compiling 28-Layer Tri-Matrix SVD + 4/64 INT8 Paged Stack...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        pa_mlps, _ = decompile_full_28_layer_quantized_stack(
            model=model,
            snapshots=snapshots,
            teacher_x_logic=teacher_x_logic,
            teacher_x_fact=teacher_x_fact,
            teacher_h_logic=teacher_h_logic,
            teacher_h_fact=teacher_h_fact,
            logic_calib_split=bundle.logic_calib,
            fact_calib_split=bundle.fact_calib,
            layer_indices=layer_indices,
            rank_ratio=cfg.rank_ratio,
            num_pages=cfg.num_pages,
            max_pages=cfg.max_active_pages,
            displace_fact_cols=0,
            lambda_reg=cfg.lambda_reg,
            closed_loop_ridge=cfg.closed_loop_ridge,
            batch_size=cfg.batch_size,
            max_calib_tokens=max_calib_tokens,
            device=cfg.device,
            dtype=cfg.dtype,
        )
        for m in pa_mlps.values():
            m.set_mode("vm_paged")
            m.set_paging_strategy("async_int8")
            m.reset_telemetry()
        if ple_offloader is not None:
            ple_offloader.reset_telemetry()

        pa_logic, _ = evaluate_split(
            model, bundle.logic_eval, state_name="Phase A: INT8 Paged", batch_size=cfg.batch_size, device=cfg.device
        )
        pa_fact, _ = evaluate_split(
            model, bundle.fact_eval, state_name="Phase A: INT8 Paged", batch_size=cfg.batch_size, device=cfg.device
        )
        pa_vram = _aggregate_phase4_vram_and_telemetry(
            pa_mlps, ple_saved_mb=ple_saved_mb, total_dense_model_mb=total_dense_model_mb, ple_offloader=ple_offloader
        )

        # Decode benchmark for Phase A
        from run_phase4_giant_killer import measure_kv_cached_decode_latency

        pa_dec_ms, pa_dec_tps, pa_dec_tel = measure_kv_cached_decode_latency(
            model=model,
            prompt_ids=bundle.fact_eval.input_ids,
            num_decode_tokens=cfg.decode_tokens,
            device=cfg.device,
            quant_mlps=pa_mlps,
            ple_offloader=ple_offloader,
        )
        pa_gsm_ok, _, pa_gsm_acc, pa_gsm_trace = evaluate_gsm8k_zero_shot_exact_match(
            model,
            tokenizer,
            gsm8k_problems,
            batch_size=gsm8k_batch_size,
            max_new_tokens=gsm8k_max_new_tokens,
            device=cfg.device,
        )
        results["phase_a"] = {
            "logic_ppl": pa_logic.perplexity,
            "fact_ppl": pa_fact.perplexity,
            "vram": pa_vram,
            "decode_ms": pa_dec_ms,
            "decode_tps": pa_dec_tps,
            "decode_pcie_mb": pa_dec_tel["pcie_mb_per_token"],
            "gsm": (pa_gsm_ok, pa_gsm_acc, pa_gsm_trace),
        }

        _restore_dense_full_mlps(model, snapshots, orig_mlp_modules, device="cpu", dtype=cfg.dtype)
        del pa_mlps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # 4. Phase B Decompilation (`Hot INT8 Core + 3-Bit Base8 + GPU LRU TLB`)
    # =========================================================================
    print(
        f"\n[Phase B Compile] Compiling {num_layers}-Layer Bit-Tiered Stack "
        f"(Hot={100.0*hot_fraction:.1f}% INT8, Cold=3-Bit Base8, TLB={tlb_slots}/{cfg.num_pages} INT4 Slots, BridgeRank={bridge_rank})..."
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t_b_comp = time.perf_counter()
    pb_mlps, pb_lut_pool, pb_stats = decompile_phase_b_stack(
        model=model,
        snapshots=snapshots,
        teacher_x_logic=teacher_x_logic,
        teacher_x_fact=teacher_x_fact,
        teacher_h_logic=teacher_h_logic,
        teacher_h_fact=teacher_h_fact,
        logic_calib_split=bundle.logic_calib,
        fact_calib_split=bundle.fact_calib,
        layer_indices=layer_indices,
        hot_fraction=hot_fraction,
        num_pages=cfg.num_pages,
        tlb_slots=tlb_slots,
        decode_req_pages=decode_req_pages,
        bridge_rank=bridge_rank,
        closed_loop_ridge=0.08,
        batch_size=cfg.batch_size,
        max_calib_tokens=max_calib_tokens,
        refresh_every=2,
        device=cfg.device,
        dtype=cfg.dtype,
    )
    pb_compile_sec = time.perf_counter() - t_b_comp
    avg_core_err = sum(s.core_only_rel_err for s in pb_stats) / max(1, len(pb_stats))
    avg_tlb_err = sum(s.tlb_paged_rel_err for s in pb_stats) / max(1, len(pb_stats))
    print(
        f"[Phase B Compile] Completed {num_layers} layers in {pb_compile_sec:.1f}s "
        f"(Avg Layer RelErr: CoreOnly={avg_core_err:.4f} -> TLBPaged={avg_tlb_err:.4f})."
    )

    # --- State B1: Resident Core Only (`0` TLB Pages, `0` MLP PCIe) ---
    print("\n[Phase B1: Core Only] Evaluating Resident Core Only (0 Pages / 0 MLP PCIe)...")
    for m in pb_mlps.values():
        m.set_mode("core_only")
        m.reset_telemetry(clear_tlb=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if ple_offloader is not None:
        ple_offloader.reset_telemetry()

    b1_logic, _ = evaluate_split(
        model, bundle.logic_eval, state_name="Phase B1: Core Only", batch_size=cfg.batch_size, device=cfg.device
    )
    b1_fact, _ = evaluate_split(
        model, bundle.fact_eval, state_name="Phase B1: Core Only", batch_size=cfg.batch_size, device=cfg.device
    )
    b1_vram = aggregate_phase_b_vram_and_telemetry(
        pb_mlps,
        lut_pool=pb_lut_pool,
        ple_saved_mb=ple_saved_mb,
        total_dense_model_mb=total_dense_model_mb,
        ple_offloader=ple_offloader,
    )
    for m in pb_mlps.values():
        m.set_phase_c_enabled(False)
    b1_py_dec_ms, b1_py_dec_tps, _ = measure_phase_b_kv_decode(
        model=model,
        prompt_ids=bundle.fact_eval.input_ids,
        num_decode_tokens=cfg.decode_tokens,
        device=cfg.device,
        phase_b_mlps=pb_mlps,
        lut_pool=pb_lut_pool,
        ple_saved_mb=ple_saved_mb,
        total_dense_model_mb=total_dense_model_mb,
        ple_offloader=ple_offloader,
    )
    for m in pb_mlps.values():
        m.set_phase_c_enabled(True)
    b1_dec_ms, b1_dec_tps, b1_dec_tel = measure_phase_b_kv_decode(
        model=model,
        prompt_ids=bundle.fact_eval.input_ids,
        num_decode_tokens=cfg.decode_tokens,
        device=cfg.device,
        phase_b_mlps=pb_mlps,
        lut_pool=pb_lut_pool,
        ple_saved_mb=ple_saved_mb,
        total_dense_model_mb=total_dense_model_mb,
        ple_offloader=ple_offloader,
    )
    b1_gsm_ok, _, b1_gsm_acc, b1_gsm_trace = evaluate_gsm8k_zero_shot_exact_match(
        model,
        tokenizer,
        gsm8k_problems,
        batch_size=gsm8k_batch_size,
        max_new_tokens=gsm8k_max_new_tokens,
        device=cfg.device,
    )

    # --- State B2: Full Phase B (`vm_paged` + Exact Sync GPU LRU TLB) ---
    print(f"\n[Phase B2: Exact GPU LRU TLB] Evaluating {tlb_slots}-Slot GPU TLB ({decode_req_pages} req/tok)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for m in pb_mlps.values():
        m.set_mode("vm_paged")
        m.set_tlb_strategy("sync_tlb")
        m.reset_telemetry(clear_tlb=True)
    if ple_offloader is not None:
        ple_offloader.reset_telemetry()

    b2_logic, _ = evaluate_split(
        model, bundle.logic_eval, state_name="Phase B2: Exact GPU TLB", batch_size=cfg.batch_size, device=cfg.device
    )
    b2_fact, _ = evaluate_split(
        model, bundle.fact_eval, state_name="Phase B2: Exact GPU TLB", batch_size=cfg.batch_size, device=cfg.device
    )
    b2_vram = aggregate_phase_b_vram_and_telemetry(
        pb_mlps,
        lut_pool=pb_lut_pool,
        ple_saved_mb=ple_saved_mb,
        total_dense_model_mb=total_dense_model_mb,
        ple_offloader=ple_offloader,
    )
    b2_dec_ms, b2_dec_tps, b2_dec_tel = measure_phase_b_kv_decode(
        model=model,
        prompt_ids=bundle.fact_eval.input_ids,
        num_decode_tokens=cfg.decode_tokens,
        device=cfg.device,
        phase_b_mlps=pb_mlps,
        lut_pool=pb_lut_pool,
        ple_saved_mb=ple_saved_mb,
        total_dense_model_mb=total_dense_model_mb,
        ple_offloader=ple_offloader,
    )
    b2_gsm_ok, _, b2_gsm_acc, b2_gsm_trace = evaluate_gsm8k_zero_shot_exact_match(
        model,
        tokenizer,
        gsm8k_problems,
        batch_size=gsm8k_batch_size,
        max_new_tokens=gsm8k_max_new_tokens,
        device=cfg.device,
    )

    # --- State B2-Spec: Zero-Stall Speculative GPU TLB Decode ---
    print("\n[Phase B2-Spec: Speculative Zero-Stall TLB] Measuring KV-Cached Decode & Stateless DMA Control...")
    for m in pb_mlps.values():
        m.set_phase_c_enabled(False)
        m.set_mode("vm_paged")
        m.set_tlb_strategy("speculative_tlb")
        m.reset_telemetry(clear_tlb=True)
    b2_py_spec_ms, b2_py_spec_tps, _ = measure_phase_b_kv_decode(
        model=model,
        prompt_ids=bundle.fact_eval.input_ids,
        num_decode_tokens=cfg.decode_tokens,
        device=cfg.device,
        phase_b_mlps=pb_mlps,
        lut_pool=pb_lut_pool,
        ple_saved_mb=ple_saved_mb,
        total_dense_model_mb=total_dense_model_mb,
        ple_offloader=ple_offloader,
    )

    for m in pb_mlps.values():
        m.set_phase_c_enabled(True)
        m.set_mode("vm_paged")
        m.set_tlb_strategy("speculative_tlb")
        m.reset_telemetry(clear_tlb=True)
    b2_spec_ms, b2_spec_tps, b2_spec_tel = measure_phase_b_kv_decode(
        model=model,
        prompt_ids=bundle.fact_eval.input_ids,
        num_decode_tokens=cfg.decode_tokens,
        device=cfg.device,
        phase_b_mlps=pb_mlps,
        lut_pool=pb_lut_pool,
        ple_saved_mb=ple_saved_mb,
        total_dense_model_mb=total_dense_model_mb,
        ple_offloader=ple_offloader,
    )

    for m in pb_mlps.values():
        m.set_mode("vm_paged")
        m.set_tlb_strategy("stateless_dma")
        m.reset_telemetry(clear_tlb=True)
    b2_stateless_ms, b2_stateless_tps, b2_stateless_tel = measure_phase_b_kv_decode(
        model=model,
        prompt_ids=bundle.fact_eval.input_ids,
        num_decode_tokens=cfg.decode_tokens,
        device=cfg.device,
        phase_b_mlps=pb_mlps,
        lut_pool=pb_lut_pool,
        ple_saved_mb=ple_saved_mb,
        total_dense_model_mb=total_dense_model_mb,
        ple_offloader=ple_offloader,
    )

    hw_bench = benchmark_isolated_phase_b_28layer_hw(
        pb_mlps,
        num_steps=min(64, max(16, cfg.decode_tokens)),
        device=cfg.device,
        dtype=cfg.dtype,
    )

    results["phase_b_stats"] = pb_stats
    results["phase_b_compile_sec"] = pb_compile_sec
    results["phase_b1"] = {
        "logic_ppl": b1_logic.perplexity,
        "fact_ppl": b1_fact.perplexity,
        "vram": b1_vram,
        "py_decode_ms": b1_py_dec_ms,
        "py_decode_tps": b1_py_dec_tps,
        "decode_ms": b1_dec_ms,
        "decode_tps": b1_dec_tps,
        "decode_tel": b1_dec_tel,
        "gsm": (b1_gsm_ok, b1_gsm_acc, b1_gsm_trace),
    }
    results["phase_b2"] = {
        "logic_ppl": b2_logic.perplexity,
        "fact_ppl": b2_fact.perplexity,
        "vram": b2_vram,
        "decode_ms": b2_dec_ms,
        "decode_tps": b2_dec_tps,
        "decode_tel": b2_dec_tel,
        "py_spec_decode_ms": b2_py_spec_ms,
        "py_spec_decode_tps": b2_py_spec_tps,
        "spec_decode_ms": b2_spec_ms,
        "spec_decode_tps": b2_spec_tps,
        "spec_decode_tel": b2_spec_tel,
        "stateless_decode_ms": b2_stateless_ms,
        "stateless_decode_tps": b2_stateless_tps,
        "stateless_decode_tel": b2_stateless_tel,
        "hw_bench": hw_bench,
        "gsm": (b2_gsm_ok, b2_gsm_acc, b2_gsm_trace),
    }

    for idx, orig_mod in orig_mlp_modules.items():
        decoder_layers[idx].mlp = orig_mod
    del pb_mlps, pb_lut_pool
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # 5. Optional Reference: Static Uniform 4-Bit AWQ (`g=64`)
    # =========================================================================
    if include_uniform_4bit_ref:
        print("\n[Uniform 4-Bit AWQ Reference] Building packed 4-bit AWQ (g=64) reference stack...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        q4_mlps, q4_lut = build_uniform_awq_4bit_stack(
            model=model,
            snapshots=snapshots,
            teacher_x_logic=teacher_x_logic,
            teacher_x_fact=teacher_x_fact,
            layer_indices=layer_indices,
            group_size=64,
            device=cfg.device,
            dtype=cfg.dtype,
        )
        del snapshots, teacher_x_logic, teacher_x_fact
        gc.collect()
        q4_logic, _ = evaluate_split(
            model, bundle.logic_eval, state_name="Uniform 4-Bit AWQ", batch_size=cfg.batch_size, device=cfg.device
        )
        q4_fact, _ = evaluate_split(
            model, bundle.fact_eval, state_name="Uniform 4-Bit AWQ", batch_size=cfg.batch_size, device=cfg.device
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        q4_mlp_mb = sum(m.vram_footprint_mb() for m in q4_mlps.values()) + (q4_lut.total_vram_bytes / float(1024 ** 2))
        q4_cuda_alloc, q4_cuda_peak = _get_cuda_mem_mb()
        q4_dec_ms, q4_dec_tps, _ = measure_phase_b_kv_decode(
            model=model,
            prompt_ids=bundle.fact_eval.input_ids,
            num_decode_tokens=cfg.decode_tokens,
            device=cfg.device,
        )
        q4_gsm_ok, _, q4_gsm_acc, q4_gsm_trace = evaluate_gsm8k_zero_shot_exact_match(
            model,
            tokenizer,
            gsm8k_problems,
            batch_size=gsm8k_batch_size,
            max_new_tokens=gsm8k_max_new_tokens,
            device=cfg.device,
        )
        results["uniform_4bit"] = {
            "logic_ppl": q4_logic.perplexity,
            "fact_ppl": q4_fact.perplexity,
            "active_mlp_mb": q4_mlp_mb,
            "cuda_allocated_mb": q4_cuda_alloc,
            "cuda_peak_mb": q4_cuda_peak,
            "decode_ms": q4_dec_ms,
            "decode_tps": q4_dec_tps,
            "gsm": (q4_gsm_ok, q4_gsm_acc, q4_gsm_trace),
        }
        del q4_mlps, q4_lut
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results["wall_time_sec"] = time.perf_counter() - t_total_start
    return results


def print_phase_b_report(res: Dict[str, Any]) -> None:
    """Prints clean GitHub-Markdown tables comparing Dense, Phase A, Phase B, and Uniform 4-Bit."""
    b_lp = res["base_logic_ppl"]
    b_fp = res["base_fact_ppl"]
    dense_mlp_mb = res["phase_b2"]["vram"]["dense_mlp_mb"]

    def _dpct(val: float, base: float) -> str:
        return f"{100.0 * ((val - base) / max(1e-8, base)):+.2f}%"

    print("\n" + "=" * 116)
    print(
        f"PHASE B HARDWARE-HONEST BENCHMARK REPORT ({res['model_id']}, "
        f"Layers {res['layer_range'][0]}..{res['layer_range'][1]}, Wall={res['wall_time_sec']:.1f}s)"
    )
    print("=" * 116)

    nl_str = f"{res['num_layers']}L"
    print("\n### Table 1: Ground-Truth Perplexity & Real Packed GPU VRAM Footprint")
    print(f"| Architecture / State | Logic PPL (Delta) | Fact PPL (Delta) | {nl_str} MLP+TLB MB | Real CUDA Alloc MB | Peak CUDA MB |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    print(
        f"| **State 0: Dense BF16 Baseline** | `{b_lp:.4f}` (`0.00%`) | `{b_fp:.4f}` (`0.00%`) | "
        f"`{dense_mlp_mb:,.2f} MB` | `{res['dense_cuda_alloc_mb']:,.2f} MB` | `{res['dense_cuda_peak_mb']:,.2f} MB` |"
    )
    if "phase_a" in res:
        pa = res["phase_a"]
        pav = pa["vram"]
        print(
            f"| **Phase A (State 2b): SVD + 4/64 INT8 Paged** | `{pa['logic_ppl']:.4f}` (`{_dpct(pa['logic_ppl'], b_lp)}`) | "
            f"`{pa['fact_ppl']:.4f}` (`{_dpct(pa['fact_ppl'], b_fp)}`) | `{pav['active_mlp_mb']:,.2f} MB` | "
            f"`{pav['cuda_allocated_mb']:,.2f} MB` | `{pav['cuda_peak_mb']:,.2f} MB` |"
        )

    b1 = res["phase_b1"]
    b1v = b1["vram"]
    print(
        f"| **Phase B1: Bit-Tiered Core Only (0 Pages)** | `{b1['logic_ppl']:.4f}` (`{_dpct(b1['logic_ppl'], b_lp)}`) | "
        f"`{b1['fact_ppl']:.4f}` (`{_dpct(b1['fact_ppl'], b_fp)}`) | `{b1v['active_mlp_mb']:,.2f} MB` | "
        f"`{b1v['cuda_allocated_mb']:,.2f} MB` | `{b1v['cuda_peak_mb']:,.2f} MB` |"
    )

    b2 = res["phase_b2"]
    b2v = b2["vram"]
    print(
        f"| **Phase B2: Full VM + GPU LRU TLB** | `{b2['logic_ppl']:.4f}` (`{_dpct(b2['logic_ppl'], b_lp)}`) | "
        f"`{b2['fact_ppl']:.4f}` (`{_dpct(b2['fact_ppl'], b_fp)}`) | `{b2v['active_mlp_mb']:,.2f} MB` | "
        f"`{b2v['cuda_allocated_mb']:,.2f} MB` | `{b2v['cuda_peak_mb']:,.2f} MB` |"
    )

    if "uniform_4bit" in res:
        u4 = res["uniform_4bit"]
        print(
            f"| **Ref: Static Uniform 4-Bit AWQ (g=64)** | `{u4['logic_ppl']:.4f}` (`{_dpct(u4['logic_ppl'], b_lp)}`) | "
            f"`{u4['fact_ppl']:.4f}` (`{_dpct(u4['fact_ppl'], b_fp)}`) | `{u4['active_mlp_mb']:,.2f} MB` | "
            f"`{u4['cuda_allocated_mb']:,.2f} MB` | `{u4['cuda_peak_mb']:,.2f} MB` |"
        )

    print("\n### Table 2: KV-Cached (`use_cache=True`) Decode Speed, GPU TLB Hit Rate & PCIe Bandwidth")
    base_tps = res["base_decode_tps"]
    hw = b2["hw_bench"]
    print(f"| Runtime Strategy | E2E KV Decode | Throughput (`% Native`) | Isolated {nl_str} GPU | GPU TLB Hit Rate | Measured PCIe/Tok |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    print(
        f"| **State 0: Native Dense BF16** | `{res['base_decode_ms']:.2f} ms/tok` | `{base_tps:.2f} tok/s` (`100.0%`) | "
        f"`{hw['hw_dense_ms']:.2f} ms/tok` | N/A | `0.00 MB/tok` |"
    )
    if "phase_a" in res:
        pa = res["phase_a"]
        print(
            f"| **Phase A: Stateless INT8 Paged (4/64)** | `{pa['decode_ms']:.2f} ms/tok` | "
            f"`{pa['decode_tps']:.2f} tok/s` (`{100.0*pa['decode_tps']/max(1e-6, base_tps):.1f}%`) | "
            f"`--` | `0.0%` (No TLB) | `{pa['decode_pcie_mb']:.2f} MB/tok` |"
        )
    print(
        f"| **Phase B1 (PyTorch): Core Only (0 Pages)** | `{b1['py_decode_ms']:.2f} ms/tok` | "
        f"`{b1['py_decode_tps']:.2f} tok/s` (`{100.0*b1['py_decode_tps']/max(1e-6, base_tps):.1f}%`) | "
        f"`{hw.get('hw_py_core_only_ms', 0.0):.2f} ms/tok` | N/A (`0` Pages) | `{b1['decode_tel']['pcie_mb_per_token']:.3f} MB/tok` |"
    )
    print(
        f"| **Phase C1 (Fused CUDA): Core Only (0 Pages)** | `{b1['decode_ms']:.2f} ms/tok` | "
        f"`{b1['decode_tps']:.2f} tok/s` (`{100.0*b1['decode_tps']/max(1e-6, base_tps):.1f}%`) | "
        f"`{hw['hw_core_only_ms']:.2f} ms/tok` | N/A (`0` Pages) | `{b1['decode_tel']['pcie_mb_per_token']:.3f} MB/tok` |"
    )
    print(
        f"| **Phase C Control: Stateless INT4 DMA (No TLB)** | `{b2['stateless_decode_ms']:.2f} ms/tok` | "
        f"`{b2['stateless_decode_tps']:.2f} tok/s` (`{100.0*b2['stateless_decode_tps']/max(1e-6, base_tps):.1f}%`) | "
        f"`{hw['hw_stateless_dma_ms']:.2f} ms/tok` | `0.0%` (Forced Miss) | `{b2['stateless_decode_tel']['pcie_mb_per_token']:.2f} MB/tok` |"
    )
    print(
        f"| **Phase B2-Spec (PyTorch): Speculative TLB** | `{b2['py_spec_decode_ms']:.2f} ms/tok` | "
        f"`{b2['py_spec_decode_tps']:.2f} tok/s` (`{100.0*b2['py_spec_decode_tps']/max(1e-6, base_tps):.1f}%`) | "
        f"`{hw.get('hw_py_spec_tlb_ms', 0.0):.2f} ms/tok` | `{b2['spec_decode_tel']['tlb_hit_rate_pct']:.2f}%` | `{b2['spec_decode_tel']['pcie_mb_per_token']:.3f} MB/tok` |"
    )
    print(
        f"| **Phase C2: Exact Sync GPU LRU TLB (CUDA)** | `{b2['decode_ms']:.2f} ms/tok` | "
        f"`{b2['decode_tps']:.2f} tok/s` (`{100.0*b2['decode_tps']/max(1e-6, base_tps):.1f}%`) | "
        f"`{hw['hw_sync_tlb_ms']:.2f} ms/tok` | `{b2['decode_tel']['tlb_hit_rate_pct']:.2f}%` | `{b2['decode_tel']['pcie_mb_per_token']:.3f} MB/tok` |"
    )
    print(
        f"| **Phase C2-Spec (Fused CUDA): Speculative TLB** | `{b2['spec_decode_ms']:.2f} ms/tok` | "
        f"`{b2['spec_decode_tps']:.2f} tok/s` (`{100.0*b2['spec_decode_tps']/max(1e-6, base_tps):.1f}%`) | "
        f"`{hw['hw_spec_tlb_ms']:.2f} ms/tok` | `{b2['spec_decode_tel']['tlb_hit_rate_pct']:.2f}%` | `{b2['spec_decode_tel']['pcie_mb_per_token']:.3f} MB/tok` |"
    )
    if "uniform_4bit" in res:
        u4 = res["uniform_4bit"]
        print(
            f"| **Ref: Static Uniform 4-Bit AWQ (g=64)** | `{u4['decode_ms']:.2f} ms/tok` | "
            f"`{u4['decode_tps']:.2f} tok/s` (`{100.0*u4['decode_tps']/max(1e-6, base_tps):.1f}%`) | "
            f"`--` | N/A (Static) | `0.00 MB/tok` |"
        )

    print("\n### Table 3: Zero-Shot Chain-of-Thought GSM8K Exact-Match (`test` Split, Question-Only)")
    tot = res["gsm_total"]
    print("| Architecture / State | Solved / Total | Exact-Match (`%`) | Sample Generated Trace |")
    print("| :--- | :---: | :---: | :--- |")
    g0_ok, g0_acc, g0_tr = res["gsm_base"]
    print(f"| **State 0: Dense BF16 Baseline** | `{g0_ok} / {tot}` | `{g0_acc:.1f}%` | {g0_tr} |")
    if "phase_a" in res:
        ga_ok, ga_acc, ga_tr = res["phase_a"]["gsm"]
        print(f"| **Phase A: SVD + 4/64 INT8 Paged** | `{ga_ok} / {tot}` | `{ga_acc:.1f}%` | {ga_tr} |")
    gb1_ok, gb1_acc, gb1_tr = b1["gsm"]
    print(f"| **Phase B1: Bit-Tiered Core Only** | `{gb1_ok} / {tot}` | `{gb1_acc:.1f}%` | {gb1_tr} |")
    gb2_ok, gb2_acc, gb2_tr = b2["gsm"]
    print(f"| **Phase B2: Full VM + GPU LRU TLB** | `{gb2_ok} / {tot}` | `{gb2_acc:.1f}%` | {gb2_tr} |")
    if "uniform_4bit" in res:
        gu4_ok, gu4_acc, gu4_tr = res["uniform_4bit"]["gsm"]
        print(f"| **Ref: Static Uniform 4-Bit AWQ** | `{gu4_ok} / {tot}` | `{gu4_acc:.1f}%` | {gu4_tr} |")
    print("=" * 116 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase B Neural VM Hardware-Honest Comparison Driver")
    parser.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--start-layer", type=int, default=4)
    parser.add_argument("--end-layer", type=int, default=-1, help="Defaults to num_layers - 4 (e.g. 31 on E2B, 38 on E4B)")
    parser.add_argument("--num-pages", type=int, default=64)
    parser.add_argument("--tlb-slots", type=int, default=10)
    parser.add_argument("--decode-req-pages", type=int, default=4)
    parser.add_argument("--hot-fraction", type=float, default=0.11)
    parser.add_argument("--bridge-rank", type=int, default=64)
    parser.add_argument("--calib-seqs", type=int, default=32)
    parser.add_argument("--eval-seqs", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--gsm8k-problems", type=int, default=20)
    parser.add_argument("--gsm8k-max-tokens", type=int, default=320)
    parser.add_argument("--gsm8k-batch-size", type=int, default=5)
    parser.add_argument("--skip-phase-a", action="store_true")
    parser.add_argument("--skip-uniform-4bit", action="store_true")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Resolve default end_layer dynamically if -1 (`31` for 35-layer E2B, `21` for 26-layer mock)
    resolved_end = args.end_layer
    if resolved_end < 0:
        if args.mock:
            resolved_end = 21
        elif "E4B" in args.model_id:
            resolved_end = 38
        else:
            resolved_end = 31

    cfg = NeuralVMConfig(
        model_id=args.model_id,
        phase4_start_layer=args.start_layer,
        phase4_end_layer=resolved_end,
        num_pages=args.num_pages,
        max_active_pages=args.decode_req_pages,
        phase2_calib_seqs=args.calib_seqs,
        num_eval_seqs=args.eval_seqs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        decode_tokens=args.decode_tokens,
        gsm8k_eval_problems=args.gsm8k_problems,
        use_mock_model=args.mock,
    )
    res = run_phase_b_pipeline(
        cfg=cfg,
        tlb_slots=args.tlb_slots,
        decode_req_pages=args.decode_req_pages,
        hot_fraction=args.hot_fraction,
        bridge_rank=args.bridge_rank,
        gsm8k_max_new_tokens=args.gsm8k_max_tokens,
        gsm8k_batch_size=args.gsm8k_batch_size,
        include_phase_a_rerun=not args.skip_phase_a,
        include_uniform_4bit_ref=not args.skip_uniform_4bit,
    )
    print_phase_b_report(res)
