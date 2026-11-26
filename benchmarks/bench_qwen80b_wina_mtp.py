"""End-to-End RTX 4090 Benchmark on Real `Qwen3-Next-80B-A3B-Instruct.tmmu`:
  1. Baseline (`UINT4` L2 `wina=1.0`, `draft_mode=none`)
  2. `UINT4-WINA` (`wina=0.50`, `draft_mode=none`)
  3. `UINT4-WINA` + CUDA-Graphed Native `mtp.*` (`wina=0.50`, `draft_mode=mtp_only`)

Measures real wall-clock `tok/s`, TTFT, MTP acceptance rate, and 5-problem GSM8K CoT accuracy.
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Dict, List, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from runtime.openai_server import TensorMMUEngine

CART_DIR = os.path.join(_REPO_ROOT, "checkpoints", "Qwen3-Next-80B-A3B-Instruct.tmmu")

# 5 canonical GSM8K-style multi-step math problems with exact ground-truth numeric answers
EVAL_PROBLEMS: List[Tuple[str, str]] = [
    (
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "18",
    ),
    (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "3",
    ),
    (
        "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "70000",
    ),
    (
        "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "540",
    ),
    (
        "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
        "20",
    ),
]


def _extract_last_number(text: str) -> str:
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not nums:
        return ""
    return nums[-1].replace(",", "").rstrip(".")


def run_config(label: str, l2_wina_ratio: float, draft_mode: str, tlb_slots: int = 170) -> Dict[str, float]:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os.environ["TMMU_L2_WINA_RATIO"] = str(l2_wina_ratio)
    print(f"\n{'=' * 88}", flush=True)
    print(f"  CONFIG: {label} (l2_wina_ratio={l2_wina_ratio}, draft_mode={draft_mode}, tlb_slots={tlb_slots})", flush=True)
    print(f"{'=' * 88}", flush=True)

    engine = TensorMMUEngine(
        nvm_dir=CART_DIR,
        device="cuda:0",
        vm_mode="vm_paged",
        tlb_slots=tlb_slots,
        draft_mode=draft_mode,
        defer_load=False,
    )
    st = engine.status_dict()
    from runtime.moe_expert_tlb import query_available_system_ram_bytes
    avail_b, tot_b = query_available_system_ram_bytes()
    print(
        f"  Loaded: VRAM={st['cuda_allocated_mb']:.0f} MiB | L1={st['tlb_slots']} | "
        f"L2={st['l2_slots']} ({st['l2_ram_gb']:.2f} GiB, wina={st['l2_wina_active']}@{st['l2_wina_ratio']}) | "
        f"RAM_avail={avail_b/(1024**3):.2f}/{tot_b/(1024**3):.2f} GiB | "
        f"NativeMTP={st['has_native_mtp']} | Graph={st['cuda_graph_mode']}",
        flush=True,
    )

    total_toks = 0
    total_decode_s = 0.0
    total_ttft_ms = 0.0
    total_prop = 0
    total_acc = 0
    total_steps = 0
    correct = 0
    peak_tps = 0.0

    for idx, (q, ans) in enumerate(EVAL_PROBLEMS):
        msgs = [
            {"role": "system", "content": "Solve step by step concisely and end with the final numeric answer."},
            {"role": "user", "content": q},
        ]
        prompt_ids, _ = engine.format_chat_prompt_ids(msgs, enable_thinking=False)
        t_call = time.perf_counter()
        first_tok_ts = [0.0]

        def _on_toks(_ids: List[int]) -> None:
            if first_tok_ts[0] == 0.0:
                first_tok_ts[0] = time.perf_counter()

        stats, _ = engine.generate_streaming_sync(
            prompt_ids=prompt_ids,
            max_new_tokens=384,
            temperature=0.0,
            on_step_tokens=_on_toks,
        )
        ttft_ms = (first_tok_ts[0] - t_call) * 1000.0 if first_tok_ts[0] > 0.0 else 0.0
        out_txt = engine.tokenizer.decode(stats.generated_ids, skip_special_tokens=True)
        got = _extract_last_number(out_txt)
        ok = got == ans
        if ok:
            correct += 1

        tps = stats.emitted_tokens / max(1e-6, stats.elapsed_s)
        peak_tps = max(peak_tps, tps)
        total_toks += stats.emitted_tokens
        total_decode_s += stats.elapsed_s
        total_ttft_ms += ttft_ms
        total_prop += stats.draft_proposed
        total_acc += stats.draft_accepted
        total_steps += stats.forward_steps
        acc_pct = (100.0 * stats.draft_accepted / stats.draft_proposed) if stats.draft_proposed > 0 else 0.0
        print(
            f"  [P{idx+1}] {'PASS' if ok else 'FAIL'} (got={got}, exp={ans}) | "
            f"{stats.emitted_tokens:3d} toks @ {tps:5.2f} tok/s | TTFT={ttft_ms:6.0f}ms | "
            f"TLB={stats.tlb_hit_rate_pct:4.1f}% | draft={stats.draft_accepted}/{stats.draft_proposed} ({acc_pct:4.1f}%)",
            flush=True,
        )

    mean_tps = total_toks / max(1e-6, total_decode_s)
    mean_ttft = total_ttft_ms / len(EVAL_PROBLEMS)
    overall_acc = (100.0 * total_acc / total_prop) if total_prop > 0 else 0.0
    toks_per_step = total_toks / max(1, total_steps)
    print(
        f"  --> SUMMARY [{label}]: {mean_tps:.2f} tok/s (peak {peak_tps:.2f}) | "
        f"TTFT={mean_ttft:.0f}ms | GSM8K={correct}/{len(EVAL_PROBLEMS)} | "
        f"Draft Acc={total_acc}/{total_prop} ({overall_acc:.1f}%) | {toks_per_step:.2f} tok/step",
        flush=True,
    )

    engine._unload_current()
    del engine
    torch.cuda.empty_cache()
    return {
        "mean_tps": mean_tps,
        "peak_tps": peak_tps,
        "mean_ttft_ms": mean_ttft,
        "gsm8k_correct": correct,
        "draft_acc_pct": overall_acc,
        "toks_per_step": toks_per_step,
    }


def main() -> None:
    if not os.path.isfile(os.path.join(CART_DIR, "manifest.json")):
        raise FileNotFoundError(f"Cartridge not found: {CART_DIR}")

    r_base = run_config("Baseline UINT4 (no WINA, no MTP)", l2_wina_ratio=1.0, draft_mode="none")
    r_wina = run_config("UINT4-WINA 50% (no MTP)", l2_wina_ratio=0.50, draft_mode="none")
    r_mtp = run_config("UINT4-WINA 50% + Native MTP", l2_wina_ratio=0.50, draft_mode="mtp_only")

    print("\n" + "=" * 88)
    print("  FINAL RTX 4090 COMPARISON — Qwen3-Next-80B-A3B-Instruct")
    print("=" * 88)
    for lbl, r in [
        ("1. Baseline UINT4 (no WINA, no MTP) ", r_base),
        ("2. UINT4-WINA 50% (no MTP)          ", r_wina),
        ("3. UINT4-WINA 50% + Native MTP      ", r_mtp),
    ]:
        sp = r["mean_tps"] / max(1e-6, r_base["mean_tps"])
        print(
            f"  {lbl} | {r['mean_tps']:6.2f} tok/s (peak {r['peak_tps']:5.2f}, {sp:.2f}x) | "
            f"GSM8K={r['gsm8k_correct']}/5 | {r['toks_per_step']:.2f} tok/step"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()
