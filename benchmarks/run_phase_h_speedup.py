"""Phase H End-to-End Verification & Speedup Benchmark on RTX 4090 (24 GB VRAM + 34.3 GB DDR5).

Benchmarks all three Phase H runtime upgrades on existing `nvm_moe_v1` cartridges (ZERO recompilation):
  1. Phase H1 (`cuda_graph_decode.py` + `qwen3_next_kernels.py`):
     Static-KV NVRTC Split-K Flash-Decoding + In-Place GDN States + Full-Model & Piecewise Layer-Fused
     CUDA Graphs eliminating the ~70-87 ms/step Windows WDDM Python/ATen launch wall.
  2. Phase H2 (`moe_tlb_kernels.py` + `moe_expert_tlb.py`):
     3-Tier GPU Virtual Address Hierarchy (L1 VRAM + L2 Zero-Copy Pinned Host DDR5 via
     `cuMemHostGetDevicePointer_v2` + L3 `nvme_mmap` 32-slot Pinned DMA Staging Ring at `22.9 GB/s`).
  3. Phase H3 (`moe_expert_tlb.py` + `qwen3_next_kernels.py`):
     Bit-Exact (`0.0-ulp`) Multi-Wave Grouped NVRTC MoE Prefill + NVRTC Streaming 4-tap Conv1d Prefill.
"""

from __future__ import annotations
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

import gc
import json
import os
import time
from typing import Any, Dict, List

if os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub")) and os.path.isdir(
    os.path.expanduser("~/.cache/huggingface/datasets")
):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch

from decompiler.nvm_serializer import load_nvm_cartridge
from run_phase4_giant_killer import (
    _build_gsm8k_eval_prompts,
    _extract_gsm8k_answer,
    _normalize_num_str,
)
from run_phase_c2_streaming_nvm import get_os_process_ram_mb
from run_phase_g_lookahead_moe import _format_gsm8k_prompt
from runtime.moe_expert_tlb import auto_configure_moe_l2_zero_copy
from runtime.moe_lookahead_router import LookaheadRouterBank
from runtime.speculative_vm import EagleLiteDraftHead, SpeculativeVMGenerator


def _save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@torch.no_grad()
def measure_core_only_graph_pass_ms(model: torch.nn.Module, moe_layers: Dict[int, Any], device: str = "cuda:0") -> Dict[str, float]:
    """Measures `core_only` pass latency (`ms/pass`) at `M=1` and `M=4` with Phase H1 Full CUDA Graphs."""
    dev = torch.device(device)
    prev_mode = next(iter(moe_layers.values())).mode
    prev_slots = next(iter(moe_layers.values())).tlb_cache.tlb_slots
    for m in moe_layers.values():
        m.set_mode("core_only")
    runner = getattr(model, "_nvm_graph_runner", None)
    assert runner is not None and runner.is_supported
    runner.invalidate_graphs()

    prompt = torch.randint(100, 2000, (1, 128), device=dev)
    mask = torch.ones((1, 128), dtype=torch.long, device=dev)
    res: Dict[str, float] = {}

    for m_tok in (1, 4):
        out = model(input_ids=prompt, attention_mask=mask, use_cache=True)
        assert runner.bind_prefill_cache(out.past_key_values, seq_len=128, max_total_len=512)
        del out
        runner.ensure_captured([m_tok])
        step_toks = [101 + j for j in range(m_tok)]
        for _ in range(4):
            runner.step(step_toks, return_logits=False)
            runner.finalize_step(m_tok, m_tok)
        torch.cuda.synchronize(dev)

        n_iters = 25
        t0 = time.perf_counter()
        for _ in range(n_iters):
            runner.step(step_toks, return_logits=False)
            runner.finalize_step(m_tok, m_tok)
        torch.cuda.synchronize(dev)
        ms_pass = ((time.perf_counter() - t0) * 1000.0) / float(n_iters)
        res[f"m{m_tok}_ms_per_pass"] = ms_pass
        res[f"m{m_tok}_equiv_tok_per_s"] = (1000.0 * float(m_tok)) / ms_pass

    # Restore `vm_paged` + L2 Zero-Copy
    for m in moe_layers.values():
        m.set_mode(prev_mode, tlb_slots_override=prev_slots)
    auto_configure_moe_l2_zero_copy(moe_layers)
    runner.invalidate_graphs()
    return res


@torch.no_grad()
def measure_prefill_ttft_256(model: torch.nn.Module, moe_layers: Dict[int, Any], tokenizer: Any, device: str = "cuda:0") -> Dict[str, Any]:
    """Compares ~260-token multi-turn prompt TTFT: Phase H3 Grouped NVRTC vs legacy per-expert loop."""
    dev = torch.device(device)
    base_text = (
        "You are an expert systems architect designing virtual-memory sparse MoE inference engines on consumer GPUs. "
        "Explain in clear technical detail how PCIe 4.0 x16 DMA staging, L1 HBM page caching, L2 device-mapped host DDR5, "
        "and CUDA Graph stream capture interact with Windows WDDM kernel launch scheduling across multi-turn chat sessions. "
        * 3
    )
    msgs = [{"role": "user", "content": base_text}]
    prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids[:, :264].to(dev)
    p_mask = torch.ones_like(p_ids, dtype=torch.long, device=dev)
    seq_len = int(p_ids.shape[1])

    # Warmup fast prefill once
    out_w = model(input_ids=p_ids, attention_mask=p_mask, use_cache=True)
    del out_w
    torch.cuda.synchronize(dev)

    # 1. Phase H3 Grouped NVRTC Prefill (`use_fused_kernels = True`)
    for m in moe_layers.values():
        m.tlb_cache.clear_cache()
        m.tlb_cache.seed_default_experts()
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    out_fast = model(input_ids=p_ids, attention_mask=p_mask, use_cache=True)
    tok_fast = int(out_fast.logits[0, -1].argmax(dim=-1).item())
    torch.cuda.synchronize(dev)
    fast_ttft_ms = (time.perf_counter() - t0) * 1000.0
    del out_fast

    # 2. Legacy MoE Python Reference Prefill (`use_fused_kernels = False` on MoE layers)
    for m in moe_layers.values():
        m.use_fused_kernels = False
        m.tlb_cache.clear_cache()
        m.tlb_cache.seed_default_experts()
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    out_leg = model(input_ids=p_ids, attention_mask=p_mask, use_cache=True)
    tok_leg = int(out_leg.logits[0, -1].argmax(dim=-1).item())
    torch.cuda.synchronize(dev)
    legacy_ttft_ms = (time.perf_counter() - t0) * 1000.0
    del out_leg

    for m in moe_layers.values():
        m.use_fused_kernels = True
        m.tlb_cache.clear_cache()
        m.tlb_cache.seed_default_experts()

    return {
        "prompt_tokens": seq_len,
        "legacy_ttft_ms": round(legacy_ttft_ms, 2),
        "phase_h3_ttft_ms": round(fast_ttft_ms, 2),
        "speedup_x": round(legacy_ttft_ms / max(0.1, fast_ttft_ms), 2),
        "first_token_match": bool(tok_fast == tok_leg),
    }


@torch.no_grad()
def run_gsm8k_5prob_suite(
    model: torch.nn.Module,
    tokenizer: Any,
    moe_layers: Dict[int, Any],
    draft_head: Any,
    draft_mode: str,
    device: str = "cuda:0",
    max_new_tokens: int = 384,
) -> Dict[str, Any]:
    problems = _build_gsm8k_eval_prompts()[:5]
    gen = SpeculativeVMGenerator(
        model=model,
        tokenizer=tokenizer,
        phase_b_mlps=moe_layers,
        draft_head=draft_head,
        draft_mode=draft_mode,
        confidence_gated_moe=True,
    )

    solved = 0
    tot_emitted = 0
    tot_steps = 0
    tot_elapsed = 0.0
    tot_prop = 0
    tot_acc = 0
    tot_pcie_mb = 0.0
    per_prob: List[Dict[str, Any]] = []

    for idx, (question, target_ans) in enumerate(problems):
        p_str = _format_gsm8k_prompt(tokenizer, question)
        p_ids = tokenizer(p_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        st = gen.generate(p_ids, max_new_tokens=max_new_tokens, stop_on_eos=True)
        txt = tokenizer.decode(st.generated_ids, skip_special_tokens=True)
        pred = _extract_gsm8k_answer(txt)
        ok = _normalize_num_str(pred) == _normalize_num_str(target_ans)
        if ok:
            solved += 1
        dec_toks = max(1, st.emitted_tokens - 1) if st.forward_steps > 0 else st.emitted_tokens
        tot_emitted += dec_toks
        tot_steps += max(1, st.forward_steps)
        tot_elapsed += st.elapsed_s
        tot_prop += st.draft_proposed
        tot_acc += st.draft_accepted
        tot_pcie_mb += st.total_pcie_mb
        per_prob.append(
            {
                "idx": idx,
                "target": target_ans,
                "pred": pred,
                "correct": ok,
                "emitted_tokens": st.emitted_tokens,
                "forward_steps": st.forward_steps,
                "tokens_per_step": round(st.tokens_per_step, 3),
                "tok_per_s": round(st.tok_per_s, 2),
                "ms_per_token": round(st.ms_per_token, 2),
                "tlb_hit_rate_pct": round(st.tlb_hit_rate_pct, 2),
                "pcie_mb_per_token": round(st.pcie_mb_per_token, 2),
            }
        )

    agg_tok_s = float(tot_emitted) / max(1e-6, tot_elapsed)
    ram_info = get_os_process_ram_mb()
    return {
        "solved": solved,
        "total": len(problems),
        "accuracy_pct": 100.0 * solved / len(problems),
        "tok_per_s": round(agg_tok_s, 2),
        "ms_per_token": round(1000.0 / agg_tok_s, 2),
        "tokens_per_step": round(float(tot_emitted) / float(max(1, tot_steps)), 3),
        "acceptance_rate_pct": round(100.0 * float(tot_acc) / float(max(1, tot_prop)), 2) if tot_prop > 0 else 0.0,
        "pcie_mb_per_token": round(tot_pcie_mb / float(max(1, tot_emitted)), 2),
        "cuda_allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 1),
        "os_working_set_mb": round(float(ram_info["ws_mb"]), 1),
        "per_problem": per_prob,
    }


@torch.no_grad()
def benchmark_cartridge(
    label: str,
    nvm_dir: str,
    tlb_slots: int,
    tlb_strategy: str,
    phase_g_ref: Dict[str, Any],
) -> Dict[str, Any]:
    print(f"\n{'=' * 78}")
    print(f"  PHASE H BENCHMARK: {label} ({nvm_dir}, slots={tlb_slots}, strategy={tlb_strategy})")
    print(f"{'=' * 78}")

    device = "cuda:0"
    model, tokenizer, moe_layers, lut_pool, ple_off, manifest = load_nvm_cartridge(
        nvm_dir=nvm_dir,
        device=device,
        dtype=torch.bfloat16,
        mode="vm_paged",
        tlb_strategy=tlb_strategy,
        page_storage="nvme_mmap",
        tlb_slots_override=tlb_slots,
        use_phase_c_kernels=True,
        load_tokenizer=True,
    )

    draft_head = None
    if os.path.isfile(os.path.join(nvm_dir, "eagle_draft_head.safetensors")):
        draft_head = EagleLiteDraftHead.load_pretrained_nvm(
            nvm_dir=nvm_dir, device=device, dtype=torch.bfloat16, lut_pool=lut_pool
        )

    lookahead_bank = None
    if LookaheadRouterBank.has_saved_nvm(nvm_dir):
        lookahead_bank = LookaheadRouterBank.load_pretrained_nvm(
            nvm_dir=nvm_dir, device=device, lut_pool=lut_pool
        )
        lookahead_bank.attach_to_moe_layers(moe_layers)
        if getattr(model, "_nvm_graph_runner", None) is not None:
            model._nvm_graph_runner.invalidate_graphs()

    boot_info = manifest.get("last_cold_boot", {})
    runner = getattr(model, "_nvm_graph_runner", None)
    first_m = next(iter(moe_layers.values()))
    l2_slots = int(first_m.tlb_cache.l2_slots)
    l2_gb = sum(m.tlb_cache.total_l2_pinned_ram_bytes for m in moe_layers.values()) / (1024 ** 3)
    all_mapped = all(m.tlb_cache.all_experts_mapped for m in moe_layers.values())
    graph_mode = runner.active_graph_mode if runner is not None else "disabled"

    print(
        f"  Loaded in {boot_info.get('cold_boot_s', 0):.2f}s | L1={tlb_slots} + L2={l2_slots}/{first_m.num_experts} "
        f"({l2_gb:.2f} GB pinned DDR5) | all_mapped={all_mapped} | graph_mode={graph_mode}"
    )

    # 1. Multi-turn ~264-token Prefill TTFT
    ttft_res = measure_prefill_ttft_256(model, moe_layers, tokenizer, device=device)
    print(
        f"  [Prefill TTFT ({ttft_res['prompt_tokens']} toks)] Legacy={ttft_res['legacy_ttft_ms']:.1f} ms -> "
        f"Phase H3={ttft_res['phase_h3_ttft_ms']:.1f} ms ({ttft_res['speedup_x']:.2f}x faster, match={ttft_res['first_token_match']})"
    )

    # 2. Greedy Parity Check (32 tokens: Eager vs Phase H CUDA Graph) + Open-Ended 96-token M=1 Speed
    chat_prompt = _format_gsm8k_prompt(
        tokenizer,
        "Explain in three concise bullet points how virtual memory paging works in operating systems and GPUs.",
    )
    p_ids = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    for m in moe_layers.values():
        m.tlb_cache.clear_cache()
        m.tlb_cache.seed_default_experts()
    model._disable_cuda_graphs = True
    gen_eager = SpeculativeVMGenerator(
        model=model, tokenizer=tokenizer, phase_b_mlps=moe_layers, draft_head=None, draft_mode="none"
    )
    st_parity_eager = gen_eager.generate(p_ids, max_new_tokens=32, stop_on_eos=True)
    model._disable_cuda_graphs = False

    for m in moe_layers.values():
        m.tlb_cache.clear_cache()
        m.tlb_cache.seed_default_experts()
    if runner is not None:
        runner.invalidate_graphs()
    gen_m1 = SpeculativeVMGenerator(
        model=model, tokenizer=tokenizer, phase_b_mlps=moe_layers, draft_head=None, draft_mode="none"
    )
    st_parity_graph = gen_m1.generate(p_ids, max_new_tokens=32, stop_on_eos=True)

    exact_32 = st_parity_eager.generated_ids == st_parity_graph.generated_ids
    prefix_match = 0
    for a, b in zip(st_parity_eager.generated_ids, st_parity_graph.generated_ids):
        if a == b:
            prefix_match += 1
        else:
            break
    sample_text = tokenizer.decode(st_parity_graph.generated_ids, skip_special_tokens=True)
    print(f"  [Greedy Parity (32 toks)] exact_match={exact_32} (prefix={prefix_match}/32) | sample={sample_text[:90]!r}")

    # Steady-state Open-Ended M=1 (96 tokens)
    st_open_m1 = gen_m1.generate(p_ids, max_new_tokens=96, stop_on_eos=True)
    print(
        f"  [Open-Ended M=1 (96 toks)] {st_open_m1.tok_per_s:.2f} tok/s ({st_open_m1.ms_per_token:.2f} ms/tok) | "
        f"L1+L2 Hit={st_open_m1.tlb_hit_rate_pct:.2f}% | PCIe={st_open_m1.pcie_mb_per_token:.1f} MB/tok"
    )

    # 3. GSM8K 5-Problem Suite: M=1 AND Two-Tier Speculative (in untouched production boot state!)
    gsm_m1 = run_gsm8k_5prob_suite(
        model=model,
        tokenizer=tokenizer,
        moe_layers=moe_layers,
        draft_head=None,
        draft_mode="none",
        device=device,
    )
    print(
        f"  [GSM8K 5-Prob M=1]      Acc={gsm_m1['solved']}/{gsm_m1['total']} ({gsm_m1['accuracy_pct']:.0f}%) | "
        f"{gsm_m1['tok_per_s']:.2f} tok/s ({gsm_m1['ms_per_token']:.2f} ms/tok) | VRAM={gsm_m1['cuda_allocated_mb']:.0f} MB"
    )

    gsm_spec = run_gsm8k_5prob_suite(
        model=model,
        tokenizer=tokenizer,
        moe_layers=moe_layers,
        draft_head=draft_head,
        draft_mode="two_tier",
        device=device,
    )
    print(
        f"  [GSM8K 5-Prob Two-Tier] Acc={gsm_spec['solved']}/{gsm_spec['total']} ({gsm_spec['accuracy_pct']:.0f}%) | "
        f"{gsm_spec['tok_per_s']:.2f} tok/s ({gsm_spec['ms_per_token']:.2f} ms/tok, {gsm_spec['tokens_per_step']:.2f} tok/step) | "
        f"VRAM={gsm_spec['cuda_allocated_mb']:.0f} MB"
    )

    # 4. Core-only CUDA Graph ceiling (`M=1` & `M=4`) — run last so production L2 Zero-Copy mapping isn't disturbed!
    core_pass = measure_core_only_graph_pass_ms(model, moe_layers, device=device)
    print(
        f"  [Core-Only CUDA Graph] M=1: {core_pass['m1_ms_per_pass']:.2f} ms/pass ({core_pass['m1_equiv_tok_per_s']:.1f} tok/s) "
        f"| M=4: {core_pass['m4_ms_per_pass']:.2f} ms/pass ({core_pass['m4_equiv_tok_per_s']:.1f} tok/s)"
    )

    result = {
        "label": label,
        "nvm_dir": nvm_dir,
        "tlb_slots_l1": tlb_slots,
        "l2_zero_copy_slots": l2_slots,
        "l2_pinned_ram_gb": round(l2_gb, 3),
        "all_experts_mapped": all_mapped,
        "cuda_graph_mode": graph_mode,
        "tlb_strategy": tlb_strategy,
        "phase_g_reference": phase_g_ref,
        "core_only_cuda_graph": core_pass,
        "prefill_ttft_256": ttft_res,
        "greedy_parity_32": {
            "exact_match": exact_32,
            "prefix_match_tokens": prefix_match,
            "sample_output": sample_text,
        },
        "open_ended_m1_96": {
            "tok_per_s": round(st_open_m1.tok_per_s, 2),
            "ms_per_token": round(st_open_m1.ms_per_token, 2),
            "tlb_hit_rate_pct": round(st_open_m1.tlb_hit_rate_pct, 2),
            "pcie_mb_per_token": round(st_open_m1.pcie_mb_per_token, 2),
            "cuda_allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 1),
        },
        "gsm8k_5prob_m1": gsm_m1,
        "gsm8k_5prob_two_tier": gsm_spec,
    }

    # Cleanly release VRAM & pinned L2 RAM before loading next model!
    if runner is not None:
        runner.release()
        model._nvm_graph_runner = None
    for m in moe_layers.values():
        m.release_tlb()
    del gen_eager, gen_m1, model, tokenizer, moe_layers, lut_pool, ple_off, draft_head, lookahead_bank
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache()
    return result


def main() -> None:
    out_path = os.path.join("checkpoints", "phase_h_results.json")
    summary: Dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "models": {},
    }

    # 1. Gemma-4-26B-A4B-it (`slots=48` L1 + `80` L2 Zero-Copy = 128/128 mapped -> `full_graph`!)
    res_26b = benchmark_cartridge(
        label="Gemma-4-26B-A4B-it",
        nvm_dir=os.path.join("checkpoints", "gemma-4-26B-A4B-it-nvm-moe.nvm"),
        tlb_slots=48,
        tlb_strategy="overlap_shared_mlp",
        phase_g_ref={
            "core_only_m1_ms": 86.93,
            "slots_48_m1_tok_s": 8.42,
            "slots_48_spec_tok_s": 13.04,
            "all_resident_gsm8k_5prob_tok_s": 16.66,
            "slots_48_vram_mb": 7752.3,
            "all_resident_vram_mb": 15396.3,
        },
    )
    summary["models"]["gemma_4_26b_a4b"] = res_26b
    _save_json(out_path, summary)

    # 2. Qwen3-Next-80B-A3B-Instruct (`slots=192` L1 + `~88` L2 Zero-Copy + Pinned L3 Ring -> `piecewise_graph`!)
    res_80b = benchmark_cartridge(
        label="Qwen3-Next-80B-A3B-Instruct",
        nvm_dir=os.path.join("checkpoints", "Qwen3-Next-80B-A3B-Instruct-nvm-moe.nvm"),
        tlb_slots=192,
        tlb_strategy="overlap_shared_mlp",
        phase_g_ref={
            "core_only_m1_ms": 77.82,
            "slots_192_m1_tok_s": 5.44,
            "slots_192_gsm8k_5prob_tok_s": 7.96,
            "slots_192_vram_mb": 18282.7,
        },
    )
    summary["models"]["qwen3_next_80b_a3b"] = res_80b
    _save_json(out_path, summary)
    print(f"\nSaved complete Phase H benchmark results to {out_path}!")


if __name__ == "__main__":
    main()
