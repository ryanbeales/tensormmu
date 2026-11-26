"""TensorMMU OpenAI & Anthropic Compatible HTTP/SSE Server (`Qwen3-Next` & `Gemma 4`).

Features:
  - OpenAI Endpoints:
      * `GET  /v1/models` (`/models`)
      * `POST /v1/chat/completions` (`/chat/completions`) — JSON + SSE, native `tools` / `tool_choice`,
        `max_tokens=32768` agent default + `thinking_token_budget=6144` (`doug.sh` tuning)
      * `POST /v1/completions` (`/completions`) — JSON + SSE
  - Anthropic Endpoints (for Claude Code, Anthropic SDK, Oh My Pi, etc.):
      * `POST /v1/messages` (`/messages`) — JSON + SSE (`message_start`, `thinking_delta`, `text_delta`,
        `tool_use` / `input_json_delta`, `message_delta`, `message_stop`)
      * `POST /v1/messages/count_tokens` (`/messages/count_tokens`)
  - Kubernetes & Container Probes + Async Download/Compile Readiness Gating:
      * `GET  /livez` & `GET /healthz` — `200 OK` liveness probe
      * `GET  /readyz` — `200 OK` when model is loaded, `503 Service Unavailable` while downloading/compiling/loading
      * Message endpoints return `503` (`Retry-After: 10`) until ready
  - Hardware Safety & Telemetry:
      * `GET  /health`, `GET /v1/tmmu/status` (`/v1/nvm/status`)
      * `POST /v1/tmmu/config` (`/v1/nvm/config`)
      * `GET  /` (Interactive Playground) & `GET /about` (Architecture Explanation Site)
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from aiohttp import web

from decompiler.nvm_serializer import load_nvm_cartridge
from runtime.moe_expert_tlb import SparseMoETLBExperts, auto_configure_moe_l2_zero_copy
from runtime.moe_lookahead_router import LookaheadRouterBank
from runtime.native_mtp import NativeMTPModule
from runtime.speculative_vm import (
    EagleLiteDraftHead,
    PromptLookupDrafter,
    SpeculativeDecodeStats,
    SpeculativeVMGenerator,
    _LMHeadInputCaptureHook,
    resolve_eos_token_ids,
)
from tensormmu.hardware_safety import FitAssessment, assess_cartridge_fit
from tensormmu.tool_parser import (
    ParsedToolCall,
    ToolAwareStreamFilter,
    anthropic_messages_to_openai,
    normalize_chat_messages_with_tools,
    normalize_openai_tools,
    parse_tool_calls_from_output,
)

# Per `doug.sh/posts/tuning-a-local-coding-agent-oh-my-pi/`:
# 8,192 truncates file writes mid-tool-call once thinking uses ~6,000 tokens.
# Defaulting `max_tokens` to 32,768 (backed by growable static KV buffers) + bounding thinking
# to `6,144` tokens prevents mid-file truncation without wasting upfront VRAM.
DEFAULT_AGENT_MAX_TOKENS = int(os.environ.get("TMMU_DEFAULT_MAX_TOKENS", "32768"))
DEFAULT_THINKING_BUDGET = int(os.environ.get("TMMU_THINKING_BUDGET", "6144"))

_EFFORT_TO_BUDGET = {
    "minimal": 1024,
    "low": 2048,
    "medium": 6144,
    "high": 12288,
    "xhigh": 20480,
}


def _has_eagle_draft_head(nvm_dir: str) -> bool:
    return os.path.isfile(os.path.join(nvm_dir, "eagle_draft_head.safetensors")) and os.path.isfile(
        os.path.join(nvm_dir, "eagle_draft_meta.json")
    )


def _strip_cart_suffix(name: str) -> str:
    for suf in (".tmmu", ".nvm"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _discover_nvm_cartridges(checkpoints_dir: str, active_nvm_dir: str) -> Dict[str, str]:
    """Maps model IDs (with and without `.nvm`/`.tmmu`) to absolute cartridge paths."""
    mapping: Dict[str, str] = {}
    if active_nvm_dir:
        abs_active = os.path.abspath(active_nvm_dir)
        if os.path.isdir(abs_active) and os.path.isfile(os.path.join(abs_active, "manifest.json")):
            base = os.path.basename(abs_active)
            mapping[base] = abs_active
            mapping[_strip_cart_suffix(base)] = abs_active

    if checkpoints_dir:
        abs_ckpt = os.path.abspath(checkpoints_dir)
        if os.path.isdir(abs_ckpt):
            for entry in sorted(os.listdir(abs_ckpt)):
                cand = os.path.join(abs_ckpt, entry)
                if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "manifest.json")):
                    mapping[entry] = cand
                    mapping[_strip_cart_suffix(entry)] = cand
    return mapping


def _sample_next_token(
    logits_1d: torch.Tensor,
    seen_ids: Sequence[int],
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    gen_rng: Optional[torch.Generator] = None,
) -> int:
    """Samples a single token from `logits_1d` (`[V]`) on GPU."""
    scores = logits_1d.float().clone()
    if repetition_penalty != 1.0 and repetition_penalty > 0.0 and seen_ids:
        uniq_ids = torch.tensor(list(set(seen_ids[-512:])), dtype=torch.long, device=scores.device)
        vals = scores.index_select(0, uniq_ids)
        vals = torch.where(vals < 0, vals * repetition_penalty, vals / repetition_penalty)
        scores.index_copy_(0, uniq_ids, vals)

    if temperature <= 1e-5:
        return int(scores.argmax(dim=-1).item())

    scores.div_(max(1e-5, float(temperature)))

    if top_k is not None and int(top_k) > 0:
        k_val = min(int(top_k), int(scores.shape[-1]))
        kth = torch.topk(scores, k_val, dim=-1).values[..., -1]
        scores = scores.masked_fill(scores < kth, float("-inf"))

    probs = F.softmax(scores, dim=-1)
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cum_probs - sorted_probs > float(top_p)
        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12))
        sampled_pos = torch.multinomial(sorted_probs, num_samples=1, generator=gen_rng)
        return int(sorted_idx.gather(-1, sampled_pos).item())

    sampled = torch.multinomial(probs, num_samples=1, generator=gen_rng)
    return int(sampled.item())


class _IncrementalTokenTextDecoder:
    """Decodes streamed token IDs into valid UTF-8 deltas without emitting broken multi-byte `\\ufffd` chars."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.all_ids: List[int] = []
        self.emitted_text_len: int = 0

    def push(self, new_ids: Sequence[int], final: bool = False) -> str:
        if new_ids:
            self.all_ids.extend(int(t) for t in new_ids)
        full_text = self.tokenizer.decode(self.all_ids, skip_special_tokens=True)
        if not final and full_text.endswith("\ufffd"):
            safe_len = len(full_text.rstrip("\ufffd"))
            if safe_len <= self.emitted_text_len:
                return ""
            delta = full_text[self.emitted_text_len : safe_len]
            self.emitted_text_len = safe_len
            return delta
        if len(full_text) <= self.emitted_text_len:
            return ""
        delta = full_text[self.emitted_text_len :]
        self.emitted_text_len = len(full_text)
        return delta

    @property
    def full_text(self) -> str:
        return self.tokenizer.decode(self.all_ids, skip_special_tokens=True)


def _strip_known_special(text: str) -> str:
    for sp in ("<bos>", "<eos>", "<end_of_turn>", "<turn|>", "<|eot_id|>", "<|im_end|>", "<pad>"):
        text = text.replace(sp, "")
    return text


def _split_thinking_and_answer(raw_ids: Sequence[int], tokenizer: Any) -> Tuple[Optional[str], str, str]:
    """Extracts `(reasoning_or_None, answer_clean, raw_answer_with_special)` for Gemma 4 & Qwen3-Next."""
    id_list = list(raw_ids)
    raw_with_spec = tokenizer.decode(id_list, skip_special_tokens=False)
    clean_full = tokenizer.decode(id_list, skip_special_tokens=True)

    # Gemma 4 thinking block (`<|channel>thought...<channel|>`)
    if "<channel|>" in raw_with_spec:
        if "<|channel>thought" in raw_with_spec:
            _, after_open = raw_with_spec.split("<|channel>thought", 1)
        else:
            after_open = raw_with_spec
        thought_part, rest_raw = after_open.split("<channel|>", 1)
        thought_clean = _strip_known_special(thought_part).lstrip("\n").strip()
        # Decode answer tokens cleanly if `<channel|>` is a single token ID
        chan_id = tokenizer.convert_tokens_to_ids("<channel|>") if hasattr(tokenizer, "convert_tokens_to_ids") else -1
        if isinstance(chan_id, int) and chan_id in id_list:
            pos = id_list.index(chan_id)
            rest_clean = tokenizer.decode(id_list[pos + 1 :], skip_special_tokens=True).lstrip("\n")
        else:
            rest_clean = _strip_known_special(rest_raw).lstrip("\n")
        return (thought_clean if thought_clean else None), rest_clean, rest_raw

    # Qwen3 / DeepSeek thinking block (`<think>...</think>`)
    if "</think>" in clean_full:
        if "<think>" in clean_full:
            _, after_open = clean_full.split("<think>", 1)
        else:
            after_open = clean_full
        thought_part, rest_part = after_open.split("</think>", 1)
        _, rest_raw = raw_with_spec.split("</think>", 1) if "</think>" in raw_with_spec else ("", rest_part)
        thought_clean = thought_part.strip()
        return (thought_clean if thought_clean else None), rest_part.lstrip("\n"), rest_raw

    return None, clean_full, raw_with_spec


class TensorMMUEngine:
    """Thread-safe GPU Virtual Memory engine exposing `.tmmu`/`.nvm` cartridges to OpenAI & Anthropic agents."""

    def __init__(
        self,
        nvm_dir: Optional[str] = None,
        checkpoints_dir: str = "checkpoints",
        device: str = "cuda:0",
        vm_mode: str = "vm_paged",
        tlb_slots: Optional[int] = None,
        tlb_strategy: Optional[str] = None,
        page_storage: Optional[str] = None,
        draft_mode: str = "two_tier",
        confidence_gated_moe: bool = True,
        auto_swap_model: bool = False,
        default_max_tokens: int = DEFAULT_AGENT_MAX_TOKENS,
        default_thinking_budget: int = DEFAULT_THINKING_BUDGET,
        force_unsafe_load: bool = False,
        defer_load: bool = False,
        target_model_hint: str = "",
    ) -> None:
        self.checkpoints_dir = os.path.abspath(checkpoints_dir)
        self.device = device
        self.auto_swap_model = bool(auto_swap_model)
        self.confidence_gated_moe = bool(confidence_gated_moe)
        self.default_max_tokens = max(64, int(default_max_tokens))
        self.default_thinking_budget = max(0, int(default_thinking_budget))
        self.force_unsafe_load = bool(force_unsafe_load)

        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Any = None
        self.paged_layers: Dict[int, Any] = {}
        self.lut_pool: Any = None
        self.ple_offloader: Any = None
        self.manifest: Dict[str, Any] = {}
        self.draft_head: Optional[EagleLiteDraftHead] = None
        self.native_mtp: Optional[NativeMTPModule] = None
        self.lookahead_bank: Optional[LookaheadRouterBank] = None
        self.pld_drafter = PromptLookupDrafter()
        self.fit_assessment: Optional[FitAssessment] = None

        self.active_nvm_dir: str = ""
        self.active_model_id: str = target_model_hint or (_strip_cart_suffix(os.path.basename(nvm_dir)) if nvm_dir else "")
        self.vm_mode: str = vm_mode
        self.tlb_slots: Optional[int] = tlb_slots
        self.tlb_strategy: str = tlb_strategy or ""
        self.page_storage: str = page_storage or ""
        self.draft_mode: str = draft_mode
        self.requests_served: int = 0
        self.total_emitted_tokens: int = 0
        self.last_request_telemetry: Optional[Dict[str, Any]] = None

        # Async lifecycle state for Docker / Kubernetes startup (`initializing` -> `downloading` -> `compiling` -> `loading` -> `ready` | `failed`)
        self._state_lock = threading.Lock()
        self.lifecycle_state: str = "initializing"
        self.lifecycle_detail: str = "Starting TensorMMU server..."
        self.lifecycle_progress_pct: Optional[float] = None
        self.lifecycle_error: Optional[str] = None
        self.started_at_ts: float = time.time()

        if nvm_dir and not defer_load:
            self.load_cartridge(
                nvm_dir=nvm_dir,
                vm_mode=vm_mode,
                tlb_slots=tlb_slots,
                tlb_strategy=tlb_strategy,
                page_storage=page_storage,
                draft_mode=draft_mode,
            )

    @property
    def is_ready(self) -> bool:
        return self.lifecycle_state == "ready" and self.model is not None and self.tokenizer is not None

    def set_bootstrap_state(
        self,
        state: str,
        detail: str = "",
        progress_pct: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._state_lock:
            self.lifecycle_state = state
            if detail:
                self.lifecycle_detail = detail
            self.lifecycle_progress_pct = progress_pct
            self.lifecycle_error = error

    def _unload_current(self) -> None:
        if self.model is not None:
            runner = getattr(self.model, "_nvm_graph_runner", None)
            if runner is not None:
                runner.release()
                self.model._nvm_graph_runner = None
            for m in self.paged_layers.values():
                if hasattr(m, "release_tlb"):
                    m.release_tlb()
            self.paged_layers.clear()
            self.draft_head = None
            if self.native_mtp is not None:
                self.native_mtp.release()
                self.native_mtp = None
            self.lookahead_bank = None
            self.ple_offloader = None
            self.lut_pool = None
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()

    def load_cartridge(
        self,
        nvm_dir: str,
        vm_mode: str = "vm_paged",
        tlb_slots: Optional[int] = None,
        tlb_strategy: Optional[str] = None,
        page_storage: Optional[str] = None,
        draft_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        cart_dir = os.path.abspath(nvm_dir)
        if not os.path.isfile(os.path.join(cart_dir, "manifest.json")):
            raise FileNotFoundError(f"No `manifest.json` found in '{cart_dir}'.")

        self.set_bootstrap_state("loading", f"Checking GPU fit and loading cartridge `{os.path.basename(cart_dir)}`...")
        self._unload_current()

        # Pre-flight GPU & RAM safety check + dynamic `tlb_slots` auto-scaling (RTX 30/40/50 -> AWS L40S/A100/H100)
        fit = assess_cartridge_fit(
            cartridge_dir=cart_dir,
            device=self.device,
            requested_vm_mode=vm_mode,
            requested_tlb_slots=tlb_slots,
        )
        self.fit_assessment = fit
        print(fit.format_banner(), flush=True)
        if not fit.ok_to_proceed and not self.force_unsafe_load:
            err_msg = "; ".join(fit.errors) or "Hardware pre-flight check failed."
            self.set_bootstrap_state("failed", detail=err_msg, error=err_msg)
            raise RuntimeError(f"TensorMMU GPU Safety Check blocked unsafe load (pass --force to override): {err_msg}")

        eff_vm_mode = fit.recommended_vm_mode if not self.force_unsafe_load else vm_mode
        eff_tlb_slots = fit.recommended_tlb_slots if (tlb_slots is None or not self.force_unsafe_load) else tlb_slots

        with open(os.path.join(cart_dir, "manifest.json"), "r", encoding="utf-8") as f:
            raw_manifest = json.load(f)

        is_moe = raw_manifest.get("format_version") == "nvm_moe_v1"
        eff_storage = page_storage or ("nvme_mmap" if is_moe else "pinned_ram")
        if tlb_strategy:
            eff_strategy = tlb_strategy
        elif is_moe:
            eff_strategy = "overlap_shared_mlp"
        else:
            eff_strategy = "amortized_spec_tlb"

        try:
            model, tokenizer, paged_layers, lut_pool, ple_off, manifest = load_nvm_cartridge(
                nvm_dir=cart_dir,
                device=self.device,
                dtype=torch.bfloat16,
                mode=eff_vm_mode,
                tlb_strategy=eff_strategy,
                page_storage=eff_storage,
                tlb_slots_override=eff_tlb_slots,
                use_phase_c_kernels=True,
                load_tokenizer=True,
            )

            draft_head: Optional[EagleLiteDraftHead] = None
            if _has_eagle_draft_head(cart_dir):
                draft_head = EagleLiteDraftHead.load_pretrained_nvm(
                    nvm_dir=cart_dir,
                    device=self.device,
                    dtype=torch.bfloat16,
                    lut_pool=lut_pool,
                )

            req_draft = draft_mode if draft_mode is not None else self.draft_mode
            native_mtp: Optional[NativeMTPModule] = None
            if req_draft != "none" and NativeMTPModule.has_saved_mtp(cart_dir):
                native_mtp = NativeMTPModule.load_from_cartridge(
                    cartridge_dir=cart_dir,
                    main_model=model,
                    lut_pool=lut_pool,
                    device=self.device,
                    dtype=torch.bfloat16,
                )

            lookahead_bank: Optional[LookaheadRouterBank] = None
            if is_moe and LookaheadRouterBank.has_saved_nvm(cart_dir):
                lookahead_bank = LookaheadRouterBank.load_pretrained_nvm(
                    nvm_dir=cart_dir,
                    device=self.device,
                    lut_pool=lut_pool,
                )
                lookahead_bank.attach_to_moe_layers(paged_layers)
                runner_init = getattr(model, "_nvm_graph_runner", None)
                if runner_init is not None:
                    runner_init.invalidate_graphs()

            if req_draft == "mtp_only" and native_mtp is None:
                eff_draft = "none"
            elif req_draft in ("two_tier", "eagle_only") and draft_head is None and native_mtp is None:
                eff_draft = "pld_only" if req_draft == "two_tier" else "none"
            else:
                eff_draft = req_draft

            base_name = os.path.basename(cart_dir)
            model_id = _strip_cart_suffix(base_name)

            self.model = model
            self.tokenizer = tokenizer
            self.paged_layers = paged_layers
            self.lut_pool = lut_pool
            self.ple_offloader = ple_off
            self.manifest = manifest
            self.draft_head = draft_head
            self.native_mtp = native_mtp
            self.lookahead_bank = lookahead_bank
            self.active_nvm_dir = cart_dir
            self.active_model_id = model_id
            self.vm_mode = eff_vm_mode
            first_layer = next(iter(paged_layers.values()), None)
            self.tlb_slots = int(first_layer.tlb_cache.tlb_slots) if first_layer is not None else 0
            self.tlb_strategy = eff_strategy
            self.page_storage = eff_storage
            self.draft_mode = eff_draft

            self.set_bootstrap_state("loading", f"Capturing initial M=1..4 CUDA Graphs for `{model_id}`...")
            with torch.no_grad():
                w_ids = torch.tensor([[1, 2]], dtype=torch.long, device=self.device)
                w_out = self.model(input_ids=w_ids, attention_mask=torch.ones_like(w_ids), use_cache=True)
                runner_warm = getattr(self.model, "_nvm_graph_runner", None)
                if runner_warm is not None and runner_warm.is_supported:
                    if runner_warm.bind_prefill_cache(w_out.past_key_values, seq_len=2, max_total_len=4096):
                        runner_warm.ensure_captured([1, 2, 3, 4] if self.draft_mode != "none" else [1])
                del w_out
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            for mlp in self.paged_layers.values():
                mlp.reset_telemetry(clear_tlb=False)

            self.set_bootstrap_state("ready", f"Serving `{model_id}`", progress_pct=100.0)
            return self.status_dict()
        except Exception as exc:
            self.set_bootstrap_state("failed", detail=str(exc), error=str(exc))
            raise

    def update_runtime_config(
        self,
        vm_mode: Optional[str] = None,
        tlb_slots: Optional[int] = None,
        tlb_strategy: Optional[str] = None,
        draft_mode: Optional[str] = None,
        confidence_gated_moe: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Dynamically resizes GPU TLB slots or switches strategies in `< 50 ms` without reloading weights."""
        runner = getattr(self.model, "_nvm_graph_runner", None) if self.model is not None else None
        if tlb_strategy is not None:
            for m in self.paged_layers.values():
                m.set_tlb_strategy(tlb_strategy)
            self.tlb_strategy = tlb_strategy
            if runner is not None:
                runner.invalidate_graphs()

        if vm_mode is not None or tlb_slots is not None:
            target_mode = vm_mode or self.vm_mode
            for m in self.paged_layers.values():
                m.set_mode(target_mode, tlb_slots_override=tlb_slots)
            moe_sub = {k: v for k, v in self.paged_layers.items() if isinstance(v, SparseMoETLBExperts)}
            if moe_sub:
                auto_configure_moe_l2_zero_copy(moe_sub)
            self.vm_mode = target_mode
            first_layer = next(iter(self.paged_layers.values()), None)
            self.tlb_slots = int(first_layer.tlb_cache.tlb_slots) if first_layer is not None else 0
            if runner is not None:
                runner.invalidate_graphs()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if draft_mode is not None:
            if draft_mode in ("two_tier", "eagle_only") and self.draft_head is None:
                self.draft_mode = "pld_only" if draft_mode == "two_tier" else "none"
            else:
                self.draft_mode = draft_mode

        if confidence_gated_moe is not None:
            self.confidence_gated_moe = bool(confidence_gated_moe)

        return self.status_dict()

    def status_dict(self) -> Dict[str, Any]:
        alloc_mb = float(torch.cuda.memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
        peak_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
        first_layer = next(iter(self.paged_layers.values()), None)
        max_pages = int(getattr(first_layer, "num_experts", None) or getattr(first_layer, "num_pages", 0))
        l2_slots = int(getattr(first_layer.tlb_cache, "l2_slots", 0)) if first_layer is not None else 0
        l2_wina_active = bool(first_layer is not None and getattr(first_layer.tlb_cache, "l2_wina_dev_ptr", 0) != 0)
        l2_wina_ratio = (
            float(getattr(first_layer.tlb_cache, "wina_keep_ratio", 1.0)) if l2_wina_active else 1.0
        )
        l2_ram_gb = round(
            sum(getattr(m.tlb_cache, "total_l2_pinned_ram_bytes", 0) for m in self.paged_layers.values()) / (1024 ** 3),
            3,
        )
        all_mapped = bool(
            self.paged_layers
            and all(getattr(m.tlb_cache, "all_experts_mapped", False) for m in self.paged_layers.values())
        )
        runner = getattr(self.model, "_nvm_graph_runner", None) if self.model is not None else None
        graph_mode = runner.active_graph_mode if runner is not None else "disabled"
        return {
            "engine": "tensormmu",
            "status": self.lifecycle_state,
            "ready": self.is_ready,
            "lifecycle_detail": self.lifecycle_detail,
            "lifecycle_progress_pct": self.lifecycle_progress_pct,
            "lifecycle_error": self.lifecycle_error,
            "uptime_s": round(time.time() - self.started_at_ts, 1),
            "active_model": self.active_model_id,
            "nvm_dir": self.active_nvm_dir,
            "format_version": self.manifest.get("format_version", "unknown"),
            "source_model": self.manifest.get("source_model", ""),
            "device": self.device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cuda_allocated_mb": round(alloc_mb, 2),
            "cuda_peak_mb": round(peak_mb, 2),
            "vm_mode": self.vm_mode,
            "tlb_slots": self.tlb_slots,
            "l2_slots": l2_slots,
            "l2_wina_active": l2_wina_active,
            "l2_wina_ratio": l2_wina_ratio,
            "l2_ram_gb": l2_ram_gb,
            "all_experts_mapped": all_mapped,
            "cuda_graph_mode": graph_mode,
            "max_pages_per_layer": max_pages,
            "tlb_strategy": self.tlb_strategy,
            "page_storage": self.page_storage,
            "draft_mode": self.draft_mode,
            "default_max_tokens": self.default_max_tokens,
            "default_thinking_budget": self.default_thinking_budget,
            "has_eagle_draft_head": self.draft_head is not None,
            "has_native_mtp": self.native_mtp is not None,
            "has_lookahead_bank": self.lookahead_bank is not None,
            "confidence_gated_moe": self.confidence_gated_moe,
            "hardware_fit": self.fit_assessment.to_dict() if self.fit_assessment is not None else None,
            "requests_served": self.requests_served,
            "total_emitted_tokens": self.total_emitted_tokens,
            "last_request_telemetry": self.last_request_telemetry,
        }

    def format_chat_prompt_ids(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        enable_thinking: bool = False,
    ) -> Tuple[torch.Tensor, bool]:
        """Formats `messages` + optional `tools` into token IDs; returns `(prompt_ids_gpu, tools_active)`."""
        norm_tools, forced_instr = normalize_openai_tools(tools, tool_choice=tool_choice)
        norm_msgs = normalize_chat_messages_with_tools(messages, forced_tool_instruction=forced_instr)

        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            tpl_kwargs: Dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": bool(enable_thinking),
            }
            if norm_tools:
                tpl_kwargs["tools"] = norm_tools

            try:
                prompt_str = self.tokenizer.apply_chat_template(norm_msgs, **tpl_kwargs)
            except TypeError:
                tpl_kwargs.pop("enable_thinking", None)
                try:
                    prompt_str = self.tokenizer.apply_chat_template(norm_msgs, **tpl_kwargs)
                except TypeError:
                    tpl_kwargs.pop("tools", None)
                    prompt_str = self.tokenizer.apply_chat_template(norm_msgs, **tpl_kwargs)

            return (
                self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device),
                bool(norm_tools),
            )

        lines = [f"{m['role'].capitalize()}: {m.get('content', '')}" for m in norm_msgs]
        lines.append("Assistant:")
        return self.tokenizer("\n".join(lines), return_tensors="pt").input_ids.to(self.device), False

    def _resolve_thought_close_spec(self, enable_thinking: bool) -> Tuple[Optional[int], List[int]]:
        """Returns `(thought_end_token_id, force_close_token_ids)` for Gemma 4 or Qwen3-Next thinking budgets."""
        if not enable_thinking or self.tokenizer is None:
            return None, []
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            unk = getattr(self.tokenizer, "unk_token_id", -1)
            cid = self.tokenizer.convert_tokens_to_ids("<channel|>")
            if isinstance(cid, int) and cid >= 0 and cid != unk:
                close_ids = self.tokenizer.encode("\n<channel|>", add_special_tokens=False)
                return cid, (close_ids if close_ids else [cid])
            tid = self.tokenizer.convert_tokens_to_ids("</think>")
            if isinstance(tid, int) and tid >= 0 and tid != unk:
                close_ids = self.tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
                return tid, (close_ids if close_ids else [tid])
        return None, []

    def _resolve_tool_stop_token_ids(self, tools_active: bool) -> Set[int]:
        """Adds Gemma-4 `<|tool_response>` token ID to EOS when tools are active so it stops right after `<tool_call|>`."""
        extra: Set[int] = set()
        if not tools_active or self.tokenizer is None:
            return extra
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            unk = getattr(self.tokenizer, "unk_token_id", -1)
            tr_id = self.tokenizer.convert_tokens_to_ids("<|tool_response>")
            if isinstance(tr_id, int) and tr_id >= 0 and tr_id != unk:
                extra.add(int(tr_id))
        return extra

    @torch.no_grad()
    def generate_streaming_sync(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = DEFAULT_AGENT_MAX_TOKENS,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        stop_strings: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        enable_thinking: bool = False,
        thinking_token_budget: Optional[int] = None,
        tools_active: bool = False,
        on_step_tokens: Optional[Callable[[List[int]], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Tuple[SpeculativeDecodeStats, str]:
        """Executes greedy speculative (`temperature == 0`) or sampled (`temperature > 0`) VM generation."""
        assert self.model is not None and self.tokenizer is not None
        self.model.eval()
        dev = prompt_ids.device
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        eos_ids, _ = resolve_eos_token_ids(self.tokenizer)
        eos_ids = set(eos_ids) | self._resolve_tool_stop_token_ids(tools_active)

        thought_end_id, force_close_ids = self._resolve_thought_close_spec(enable_thinking)
        eff_think_budget = (
            int(thinking_token_budget)
            if thinking_token_budget is not None
            else (self.default_thinking_budget if enable_thinking else 0)
        )
        still_thinking = bool(enable_thinking and thought_end_id is not None and eff_think_budget > 0)

        use_greedy_spec = (float(temperature) <= 1e-5) and (float(repetition_penalty) == 1.0)
        eff_draft_mode = self.draft_mode if use_greedy_spec else "none"

        gen_helper = SpeculativeVMGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            phase_b_mlps=self.paged_layers,
            draft_head=self.draft_head if eff_draft_mode in ("two_tier", "eagle_only") else None,
            pld_drafter=self.pld_drafter,
            draft_mode=eff_draft_mode,
            max_draft_tokens=3,
            pld_long_draft_tokens=4,
            confidence_gated_moe=self.confidence_gated_moe,
        )
        for mlp in self.paged_layers.values():
            if mlp.mode == "vm_paged" and mlp.tlb_strategy == "speculative_tlb":
                mlp.set_tlb_strategy("amortized_spec_tlb")

        eff_s0_conf, eff_lt_conf, eff_s0_marg, eff_lt_marg, eff_min_cum, allow_2gram_pld = (
            gen_helper._resolve_moe_draft_gates()
        )

        gen_rng: Optional[torch.Generator] = None
        if seed is not None:
            gen_rng = torch.Generator(device=dev)
            gen_rng.manual_seed(int(seed))

        stop_list = [s for s in (stop_strings or []) if s]
        max_new_tokens = max(1, int(max_new_tokens))

        embed_fn = self.model.get_input_embeddings()
        hook = _LMHeadInputCaptureHook(self.model.lm_head)
        finish_reason = "length"

        try:
            t_start = time.perf_counter()
            p_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=dev)
            out = self.model(input_ids=prompt_ids, attention_mask=p_mask, use_cache=True)
            pkv = out.past_key_values

            prompt_list = prompt_ids[0].tolist()
            if use_greedy_spec:
                first_tok = int(out.logits[0, -1].argmax(dim=-1).item())
            else:
                first_tok = _sample_next_token(
                    out.logits[0, -1],
                    seen_ids=prompt_list,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    gen_rng=gen_rng,
                )

            assert hook.last_hidden is not None
            h_prefill = hook.last_hidden
            h_verified = h_prefill[:, -1:, :].clone()
            ctx_tail = h_prefill[:, -min(8, int(h_prefill.shape[1])) :, :]
            c_ema = ctx_tail.mean(dim=1, keepdim=True).clone()
            if self.native_mtp is not None and use_greedy_spec and eff_draft_mode in ("two_tier", "mtp_only"):
                self.native_mtp.prefill_from_hidden(prompt_ids, h_prefill)

            if torch.cuda.is_available() and dev.type == "cuda":
                torch.cuda.synchronize(dev)
            ttft_ms = (time.perf_counter() - t_start) * 1000.0
            t_decode_start = time.perf_counter()

            for mlp in self.paged_layers.values():
                mlp.reset_telemetry(clear_tlb=False)

            seq_ids: List[int] = prompt_list + [first_tok]
            gen_ids: List[int] = []
            last_tok = first_tok

            forward_steps = 0
            draft_proposed = 0
            draft_accepted = 0
            pld_proposed = 0
            pld_accepted = 0
            eagle_proposed = 0
            eagle_accepted = 0

            if first_tok in eos_ids:
                finish_reason = "stop"
            else:
                if still_thinking and first_tok == thought_end_id:
                    still_thinking = False
                gen_ids.append(first_tok)
                if on_step_tokens is not None:
                    on_step_tokens([first_tok])

            if stop_list and gen_ids:
                cur_txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                if any(st in cur_txt for st in stop_list):
                    finish_reason = "stop"

            max_total_len = int(prompt_ids.shape[1]) + max_new_tokens + 16
            runner = (
                None
                if getattr(self.model, "_disable_cuda_graphs", False)
                else getattr(self.model, "_nvm_graph_runner", None)
            )
            runner_bound = False
            if finish_reason != "stop" and runner is not None and runner.is_supported:
                runner_bound = runner.bind_prefill_cache(
                    pkv, seq_len=int(prompt_ids.shape[1]), max_total_len=max_total_len
                )
                if runner_bound:
                    pkv = None
            ones_mask_buf = None if runner_bound else torch.ones((1, max_total_len), dtype=torch.long, device=dev)

            while finish_reason != "stop" and len(gen_ids) < max_new_tokens:
                if cancel_event is not None and cancel_event.is_set():
                    finish_reason = "cancelled"
                    break

                # Enforce separate `thinking_token_budget` (`doug.sh`) so reasoning never starves file writes!
                if still_thinking and len(gen_ids) >= eff_think_budget and force_close_ids:
                    still_thinking = False
                    if len(gen_ids) + len(force_close_ids) < max_new_tokens:
                        feed_seq = [last_tok, *force_close_ids[:-1]]
                        advanced_ok = True
                        for ft in feed_seq:
                            if runner_bound and runner is not None and runner.can_run_step(1):
                                _, h_ft, _ = runner.step([int(ft)], return_logits=False)
                                runner.finalize_step(1, 1)
                                forward_steps += 1
                                h_verified = h_ft[:, -1:, :].clone()
                            elif not runner_bound and ones_mask_buf is not None:
                                clen = len(seq_ids) - 1
                                s_t = torch.tensor([[int(ft)]], dtype=torch.long, device=dev)
                                out_ft = self.model(
                                    input_ids=s_t,
                                    attention_mask=ones_mask_buf[:, : clen + 1],
                                    past_key_values=pkv,
                                    use_cache=True,
                                )
                                pkv = out_ft.past_key_values
                                forward_steps += 1
                                if hook.last_hidden is not None:
                                    h_verified = hook.last_hidden[:, -1:, :].clone()
                            else:
                                advanced_ok = False
                                break
                        if advanced_ok:
                            gen_ids.extend(force_close_ids)
                            seq_ids.extend(force_close_ids)
                            last_tok = int(force_close_ids[-1])
                            if on_step_tokens is not None:
                                on_step_tokens(list(force_close_ids))
                            continue

                rem = max_new_tokens - len(gen_ids)
                k_budget = max(0, rem - 1) if use_greedy_spec else 0
                if runner_bound:
                    k_budget = min(k_budget, 7)

                draft_toks: List[int] = []
                draft_source = "none"
                if k_budget > 0:
                    if eff_draft_mode == "pld_only":
                        k_lim = min(gen_helper.pld_long_draft_tokens, k_budget)
                        min_ord = 2 if allow_2gram_pld else 3
                        draft_toks, _ = self.pld_drafter.draft(
                            seq_ids, k_lim, eos_ids=eos_ids, min_order_override=min_ord
                        )
                        if draft_toks:
                            draft_source = "pld"
                    elif eff_draft_mode == "eagle_only":
                        if self.draft_head is not None:
                            k_lim = min(gen_helper.max_draft_tokens, k_budget)
                            draft_toks = self.draft_head.draft(
                                h_verified=h_verified,
                                last_tok_id=last_tok,
                                c_ema=c_ema,
                                embed_fn=embed_fn,
                                max_draft_tokens=k_lim,
                                step0_conf_threshold=eff_s0_conf,
                                later_conf_threshold=eff_lt_conf,
                                step0_margin_threshold=eff_s0_marg,
                                later_margin_threshold=eff_lt_marg,
                                min_cum_prob=eff_min_cum,
                                eos_ids=eos_ids,
                            )
                            if draft_toks:
                                draft_source = "eagle"
                    elif eff_draft_mode == "mtp_only":
                        if self.native_mtp is not None:
                            all_mapped_now = bool(
                                self.paged_layers
                                and all(getattr(m.tlb_cache, "all_experts_mapped", False) for m in self.paged_layers.values())
                            )
                            k_mtp = min(3 if all_mapped_now else 1, k_budget)
                            min_c = 0.25 if all_mapped_now else 0.55
                            draft_toks, _ = self.native_mtp.draft_k_tokens(
                                last_tok_id=last_tok,
                                last_hidden=h_verified,
                                max_draft_tokens=k_mtp,
                                min_conf=min_c,
                                eos_ids=eos_ids,
                            )
                            if draft_toks:
                                draft_source = "mtp"
                    elif eff_draft_mode == "two_tier":
                        is_moe_graph = runner_bound and not allow_2gram_pld
                        pld_min_ord = 4 if is_moe_graph else 3
                        k_pld = min(3 if is_moe_graph else gen_helper.pld_long_draft_tokens, k_budget)
                        cand_pld, matched_n = self.pld_drafter.draft(
                            seq_ids, k_pld, eos_ids=eos_ids, min_order_override=pld_min_ord
                        )
                        if cand_pld and matched_n >= pld_min_ord:
                            draft_toks = cand_pld
                            draft_source = "pld"
                        elif self.native_mtp is not None:
                            all_mapped_now = bool(
                                self.paged_layers
                                and all(getattr(m.tlb_cache, "all_experts_mapped", False) for m in self.paged_layers.values())
                            )
                            k_mtp = min(3 if all_mapped_now else 1, k_budget)
                            min_c = 0.25 if all_mapped_now else 0.55
                            cand_mtp, _ = self.native_mtp.draft_k_tokens(
                                last_tok_id=last_tok,
                                last_hidden=h_verified,
                                max_draft_tokens=k_mtp,
                                min_conf=min_c,
                                eos_ids=eos_ids,
                            )
                            if cand_mtp:
                                draft_toks = cand_mtp
                                draft_source = "mtp"
                        elif not is_moe_graph:
                            if self.draft_head is not None:
                                k_eag = min(gen_helper.max_draft_tokens, k_budget)
                                draft_toks = self.draft_head.draft(
                                    h_verified=h_verified,
                                    last_tok_id=last_tok,
                                    c_ema=c_ema,
                                    embed_fn=embed_fn,
                                    max_draft_tokens=k_eag,
                                    step0_conf_threshold=eff_s0_conf,
                                    later_conf_threshold=eff_lt_conf,
                                    step0_margin_threshold=eff_s0_marg,
                                    later_margin_threshold=eff_lt_marg,
                                    min_cum_prob=eff_min_cum,
                                    eos_ids=eos_ids,
                                )
                                if draft_toks:
                                    draft_source = "eagle"
                            if not draft_toks and allow_2gram_pld:
                                cand_bi, _ = self.pld_drafter.draft(
                                    seq_ids, min(2, k_budget), eos_ids=eos_ids, min_order_override=2
                                )
                                if cand_bi:
                                    draft_toks = cand_bi
                                    draft_source = "pld"

                K = len(draft_toks)
                M = K + 1
                if runner_bound and runner is not None and not runner.can_run_step(M):
                    # Try falling back from speculative `M>1` to `M=1` first; if even `M=1` hits VRAM ceiling, stop cleanly!
                    if K > 0 and runner.can_run_step(1):
                        draft_toks = []
                        draft_source = "none"
                        K = 0
                        M = 1
                    else:
                        finish_reason = "length"
                        break

                draft_proposed += K
                if draft_source == "pld":
                    pld_proposed += K
                elif draft_source == "eagle":
                    eagle_proposed += K

                use_graph_step = runner_bound and runner is not None and runner.can_run_step(M)
                step_logits_gpu: Optional[torch.Tensor] = None
                if use_graph_step:
                    pred_ids_graph, h_step_all, step_logits_gpu = runner.step(
                        [last_tok, *draft_toks], return_logits=not use_greedy_spec
                    )
                    forward_steps += 1
                else:
                    assert ones_mask_buf is not None
                    cache_len = len(seq_ids) - 1
                    step_ids = torch.tensor([[last_tok, *draft_toks]], dtype=torch.long, device=dev)
                    cur_mask = ones_mask_buf[:, : cache_len + M]

                    if K > 0:
                        from runtime.qwen3_next_kernels import prepare_hybrid_cache_for_speculative_step

                        prepare_hybrid_cache_for_speculative_step(pkv, K)

                    out = self.model(
                        input_ids=step_ids,
                        attention_mask=cur_mask,
                        past_key_values=pkv,
                        use_cache=True,
                    )
                    pkv = out.past_key_values
                    forward_steps += 1
                    assert hook.last_hidden is not None
                    h_step_all = hook.last_hidden
                    step_logits_gpu = out.logits
                    pred_ids_graph = []

                emitted_visible: List[int] = []
                n_acc = 0
                hit_eos = False

                if use_greedy_spec:
                    pred_ids = pred_ids_graph if use_graph_step else step_logits_gpu[0].argmax(dim=-1).tolist()  # type: ignore[index]
                    valid_kv_added = 0
                    for j in range(K):
                        if pred_ids[j] == draft_toks[j]:
                            n_acc += 1
                            tid = int(draft_toks[j])
                            valid_kv_added += 1
                            if tid in eos_ids:
                                hit_eos = True
                                finish_reason = "stop"
                                break
                            emitted_visible.append(tid)
                        else:
                            break

                    draft_accepted += n_acc
                    if draft_source == "pld":
                        pld_accepted += n_acc
                    elif draft_source == "eagle":
                        eagle_accepted += n_acc

                    if not hit_eos and (len(gen_ids) + len(emitted_visible) < max_new_tokens):
                        bonus_tok = int(pred_ids[n_acc])
                        valid_kv_added += 1
                        if bonus_tok in eos_ids:
                            hit_eos = True
                            finish_reason = "stop"
                        else:
                            emitted_visible.append(bonus_tok)

                    if use_graph_step:
                        assert runner is not None
                        runner.finalize_step(M, max(1, valid_kv_added))
                        if self.native_mtp is not None:
                            self.native_mtp.commit_verified_tokens(n_acc)
                        if K > 0:
                            for mlp in self.paged_layers.values():
                                if hasattr(mlp, "commit_speculative_step"):
                                    mlp.commit_speculative_step(n_acc)
                    elif K > 0:
                        from runtime.qwen3_next_kernels import finalize_or_rollback_hybrid_cache

                        finalize_or_rollback_hybrid_cache(pkv, M, max(1, valid_kv_added))
                        if self.native_mtp is not None:
                            self.native_mtp.commit_verified_tokens(n_acc)
                        for mlp in self.paged_layers.values():
                            if hasattr(mlp, "commit_speculative_step"):
                                mlp.commit_speculative_step(n_acc)
                    elif self.native_mtp is not None:
                        self.native_mtp.commit_verified_tokens(0)

                    last_valid_pos = max(0, valid_kv_added - 1)
                    h_verified = h_step_all[:, last_valid_pos : last_valid_pos + 1, :].clone()
                    for pos_i in range(max(1, valid_kv_added)):
                        c_ema.mul_(0.75).add_(h_step_all[:, pos_i : pos_i + 1, :], alpha=0.25)
                else:
                    assert step_logits_gpu is not None
                    sampled_tok = _sample_next_token(
                        step_logits_gpu[0, 0],
                        seen_ids=seq_ids,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        gen_rng=gen_rng,
                    )
                    if use_graph_step:
                        assert runner is not None
                        runner.finalize_step(1, 1)
                    if sampled_tok in eos_ids:
                        hit_eos = True
                        finish_reason = "stop"
                    else:
                        emitted_visible.append(sampled_tok)

                if emitted_visible:
                    if still_thinking and thought_end_id in emitted_visible:
                        still_thinking = False
                    gen_ids.extend(emitted_visible)
                    seq_ids.extend(emitted_visible)
                    last_tok = emitted_visible[-1]
                    if on_step_tokens is not None:
                        on_step_tokens(emitted_visible)

                    if stop_list:
                        cur_txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                        if any(st in cur_txt for st in stop_list):
                            finish_reason = "stop"
                            break

                if hit_eos:
                    break

            if torch.cuda.is_available() and dev.type == "cuda":
                torch.cuda.synchronize(dev)
            decode_elapsed_s = time.perf_counter() - t_decode_start
            elapsed_s = time.perf_counter() - t_start
            del pkv, out
        finally:
            hook.close()

        tot_pcie_bytes = 0.0
        tot_hits = 0.0
        tot_misses = 0.0
        tot_demand_misses = 0.0
        tot_la_pref = 0.0
        tot_la_hits = 0.0
        tot_la_wasted = 0.0
        for mlp in self.paged_layers.values():
            t = mlp.tlb_cache.telemetry()
            tot_pcie_bytes += t["pcie_bytes"]
            tot_hits += t["page_hits"]
            tot_misses += t["page_misses"]
            tot_demand_misses += t.get("demand_misses", t["page_misses"])
            tot_la_pref += t.get("lookahead_prefetched", 0.0)
            tot_la_hits += t.get("lookahead_hits", 0.0)
            tot_la_wasted += t.get("lookahead_wasted", 0.0)

        decode_emitted = max(1, len(gen_ids) - 1) if forward_steps > 0 else max(1, len(gen_ids))
        eff_decode_s = decode_elapsed_s if forward_steps > 0 else elapsed_s
        f_steps = max(1, forward_steps)
        tot_pcie_mb = tot_pcie_bytes / float(1024 ** 2)
        tot_req = max(1.0, tot_hits + tot_misses)
        hit_rate = 100.0 * tot_hits / tot_req
        eff_hit_rate = 100.0 * (tot_hits + tot_la_hits) / tot_req
        la_prec = 100.0 * tot_la_hits / max(1.0, tot_la_pref) if tot_la_pref > 0 else 0.0
        la_cov = 100.0 * tot_la_hits / max(1.0, tot_misses) if tot_misses > 0 else 0.0
        acc_rate = 100.0 * float(draft_accepted) / float(max(1, draft_proposed)) if draft_proposed > 0 else 0.0

        stats = SpeculativeDecodeStats(
            generated_ids=gen_ids,
            emitted_tokens=len(gen_ids),
            forward_steps=forward_steps,
            draft_proposed=draft_proposed,
            draft_accepted=draft_accepted,
            pld_proposed=pld_proposed,
            pld_accepted=pld_accepted,
            eagle_proposed=eagle_proposed,
            eagle_accepted=eagle_accepted,
            elapsed_s=elapsed_s,
            ms_per_token=(eff_decode_s * 1000.0) / float(decode_emitted),
            tok_per_s=float(decode_emitted) / max(1e-6, eff_decode_s),
            tokens_per_step=float(decode_emitted) / float(f_steps) if forward_steps > 0 else 1.0,
            acceptance_rate_pct=acc_rate,
            total_pcie_mb=tot_pcie_mb,
            pcie_mb_per_token=tot_pcie_mb / float(decode_emitted),
            pcie_mb_per_step=tot_pcie_mb / float(f_steps),
            tlb_hit_rate_pct=hit_rate,
            demand_misses=int(tot_demand_misses),
            lookahead_prefetched=int(tot_la_pref),
            lookahead_hits=int(tot_la_hits),
            lookahead_wasted=int(tot_la_wasted),
            lookahead_precision_pct=la_prec,
            lookahead_coverage_pct=la_cov,
            effective_stall_free_hit_pct=eff_hit_rate,
        )

        self.requests_served += 1
        self.total_emitted_tokens += len(gen_ids)
        self.last_request_telemetry = self._format_telemetry(stats, ttft_ms=ttft_ms)
        return stats, finish_reason

    def _format_telemetry(self, stats: SpeculativeDecodeStats, ttft_ms: Optional[float] = None) -> Dict[str, Any]:
        alloc_mb = float(torch.cuda.memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
        out_tel: Dict[str, Any] = {
            "tok_per_s": round(stats.tok_per_s, 2),
            "ms_per_token": round(stats.ms_per_token, 2),
            "elapsed_s": round(stats.elapsed_s, 3),
            "emitted_tokens": stats.emitted_tokens,
            "forward_steps": stats.forward_steps,
            "tokens_per_step": round(stats.tokens_per_step, 2),
            "acceptance_rate_pct": round(stats.acceptance_rate_pct, 1),
            "tlb_hit_rate_pct": round(stats.tlb_hit_rate_pct, 2),
            "effective_stall_free_hit_pct": round(stats.effective_stall_free_hit_pct, 2),
            "pcie_mb_per_token": round(stats.pcie_mb_per_token, 2),
            "cuda_allocated_mb": round(alloc_mb, 1),
            "tlb_slots": self.tlb_slots,
            "vm_mode": self.vm_mode,
            "draft_mode": self.draft_mode,
        }
        if ttft_ms is not None:
            out_tel["ttft_ms"] = round(ttft_ms, 1)
        return out_tel


# Backward-compatible alias
NeuralVMOpenAIEngine = TensorMMUEngine


_PLAYGROUND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>TensorMMU Playground</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; background: #0d1117; color: #e6edf3; max-width: 980px; margin: 24px auto; padding: 0 16px; }
  .hdr { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
  h2 { margin: 0; font-size: 1.25rem; }
  a { color: #58a6ff; text-decoration: none; font-size: 0.9rem; }
  a:hover { text-decoration: underline; }
  .bar { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px 14px; font-size: 0.85rem; margin-bottom: 12px; }
  .pill { color: #58a6ff; font-weight: 600; }
  .notready { color: #f0883e; font-weight: 600; }
  .warn { color: #d29922; font-size: 0.82rem; margin-top: 6px; font-style: italic; }
  #chat { background: #161b22; border: 1px solid #30363d; border-radius: 8px; height: 480px; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 12px; }
  .msg { padding: 10px 14px; border-radius: 8px; white-space: pre-wrap; line-height: 1.45; }
  .user { background: #1f6feb; align-self: flex-end; max-width: 80%; }
  .assistant { background: #21262d; border: 1px solid #30363d; align-self: flex-start; max-width: 88%; }
  .row { display: flex; gap: 8px; margin-top: 12px; }
  .ctrl { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; font-size: 0.85rem; margin-top: 8px; color: #8b949e; }
  .ctrl input { width: 88px; background: #0d1117; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; padding: 4px 8px; }
  textarea { flex: 1; background: #0d1117; color: #e6edf3; border: 1px solid #30363d; border-radius: 8px; padding: 10px; font-family: inherit; resize: vertical; min-height: 52px; }
  button { background: #238636; color: #fff; border: none; border-radius: 8px; padding: 0 18px; font-weight: 600; cursor: pointer; }
  button.sec { background: #30363d; padding: 6px 12px; font-weight: 500; }
  button.stop { background: #da3633; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
</head>
<body>
  <div class="hdr">
    <h2>TensorMMU Agent Server (<code>/v1/chat/completions</code> &amp; <code>/v1/messages</code>)</h2>
    <a href="/about">How TensorMMU Works &rarr;</a>
  </div>
  <div class="bar" id="status">Loading TensorMMU status...</div>
  <div id="chat"></div>
  <div class="row">
    <textarea id="inp" placeholder="Ask anything... (Shift+Enter for newline, Enter to send)"></textarea>
    <button id="send">Send</button>
    <button id="stopBtn" class="stop" style="display:none;">Stop</button>
  </div>
  <div class="ctrl">
    <label>Max tokens: <input id="maxTok" type="number" value="32768" min="16" max="65536" step="1024" /></label>
    <label>Thinking budget: <input id="thinkBud" type="number" value="6144" min="512" max="32768" step="512" /></label>
    <label>Temperature: <input id="temp" type="number" value="0.0" min="0.0" max="2.0" step="0.1" /></label>
    <button id="clearBtn" class="sec">Clear Chat</button>
  </div>
<script>
const msgs = [];
let abortCtrl = null;
async function refreshStatus() {
  const r = await fetch('/v1/tmmu/status');
  const d = await r.json();
  if (!d.ready) {
    const pct = d.lifecycle_progress_pct != null ? ` (${d.lifecycle_progress_pct.toFixed(1)}%)` : '';
    document.getElementById('status').innerHTML =
      `State: <span class="notready">${d.status.toUpperCase()}${pct}</span> &mdash; ${d.lifecycle_detail || 'Preparing model...'}`;
    setTimeout(refreshStatus, 2500);
    return;
  }
  const tel = d.last_request_telemetry;
  const perf = tel ? ` | Last: <span class="pill">${tel.tok_per_s} tok/s</span> (${tel.emitted_tokens} tok, TTFT ${tel.ttft_ms ?? 0} ms, TLB ${tel.tlb_hit_rate_pct}%, PCIe ${tel.pcie_mb_per_token} MB/tok)` : '';
  document.getElementById('status').innerHTML =
    `Model: <span class="pill">${d.active_model}</span> | VRAM: <span class="pill">${d.cuda_allocated_mb} MB</span> | ` +
    `L1: <span class="pill">${d.tlb_slots}</span> + L2: <span class="pill">${d.l2_slots}</span>/${d.max_pages_per_layer} (${d.cuda_graph_mode})` + perf;
}
function addBubble(role, text) {
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  el.textContent = text;
  const c = document.getElementById('chat');
  c.appendChild(el);
  c.scrollTop = c.scrollHeight;
  return el;
}
async function sendMsg() {
  const inp = document.getElementById('inp');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  const maxTokens = parseInt(document.getElementById('maxTok').value || '32768', 10);
  const thinkBudget = parseInt(document.getElementById('thinkBud').value || '6144', 10);
  const temperature = parseFloat(document.getElementById('temp').value || '0.0');
  document.getElementById('send').disabled = true;
  document.getElementById('stopBtn').style.display = 'inline-block';
  msgs.push({role: 'user', content: text});
  addBubble('user', text);
  const botEl = addBubble('assistant', 'Prefilling prompt...');
  let acc = '';
  let finishReason = null;
  abortCtrl = new AbortController();
  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      signal: abortCtrl.signal,
      body: JSON.stringify({
        model: 'default',
        messages: msgs,
        stream: true,
        temperature,
        max_tokens: maxTokens,
        thinking_token_budget: thinkBudget,
        stream_options: {include_usage: true}
      })
    });
    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      botEl.textContent = `[HTTP ${res.status}] ` + (errJson.error?.message || JSON.stringify(errJson));
      msgs.pop();
    } else {
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buf += dec.decode(value, {stream: true});
        const lines = buf.split('\\n');
        buf = lines.pop();
        for (const ln of lines) {
          if (!ln.startsWith('data: ')) continue;
          const payload = ln.slice(6).trim();
          if (payload === '[DONE]') break;
          const j = JSON.parse(payload);
          const ch = j.choices?.[0];
          const delta = ch?.delta?.content || ch?.delta?.reasoning_content || '';
          if (ch?.finish_reason) finishReason = ch.finish_reason;
          if (delta) {
            acc += delta;
            botEl.textContent = acc;
            document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
          }
        }
      }
      if (finishReason === 'length') {
        const note = document.createElement('div');
        note.className = 'warn';
        note.textContent = `[Stopped at max_tokens=${maxTokens} limit]`;
        botEl.appendChild(note);
      }
      msgs.push({role: 'assistant', content: acc});
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      if (acc) msgs.push({role: 'assistant', content: acc});
      else botEl.textContent = '[Generation cancelled]';
    } else {
      botEl.textContent = (acc ? acc + '\\n\\n' : '') + '[Error: ' + e + ']';
    }
  }
  abortCtrl = null;
  document.getElementById('send').disabled = false;
  document.getElementById('stopBtn').style.display = 'none';
  refreshStatus();
}
document.getElementById('send').onclick = sendMsg;
document.getElementById('stopBtn').onclick = () => { if (abortCtrl) abortCtrl.abort(); };
document.getElementById('clearBtn').onclick = () => { msgs.length = 0; document.getElementById('chat').innerHTML = ''; };
document.getElementById('inp').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
});
refreshStatus();
</script>
</body>
</html>
"""


def _trim_at_stop_strings(text: str, stop_strings: Optional[Sequence[str]]) -> Tuple[str, Optional[str]]:
    """Returns `(trimmed_text, matched_stop_sequence_or_None)`."""
    if not stop_strings:
        return text, None
    best_idx: Optional[int] = None
    matched: Optional[str] = None
    for s in stop_strings:
        if s:
            pos = text.find(s)
            if pos != -1 and (best_idx is None or pos < best_idx):
                best_idx = pos
                matched = s
    if best_idx is not None:
        return text[:best_idx], matched
    return text, None


def _resolve_thinking_options(body: Dict[str, Any], default_budget: int) -> Tuple[bool, int]:
    """Parses OpenAI (`enable_thinking`, `reasoning_effort`, `thinking_token_budget`) & Anthropic (`thinking`)."""
    ct_kwargs = body.get("chat_template_kwargs") or {}
    anth_think = body.get("thinking")
    enable = bool(body.get("enable_thinking", ct_kwargs.get("enable_thinking", False)))
    budget = int(body.get("thinking_token_budget") or ct_kwargs.get("thinking_token_budget") or default_budget)

    effort = str(body.get("reasoning_effort", "") or "").strip().lower()
    if effort in _EFFORT_TO_BUDGET:
        if effort != "none":
            enable = True
        budget = _EFFORT_TO_BUDGET[effort]
    elif effort == "none":
        enable = False

    if isinstance(anth_think, dict):
        t_mode = str(anth_think.get("type", "disabled")).lower()
        if t_mode == "enabled":
            enable = True
            if anth_think.get("budget_tokens") is not None:
                budget = int(anth_think["budget_tokens"])
        elif t_mode == "disabled":
            enable = False

    return enable, max(256, budget)


def _format_sse_event(event_name: str, payload: Dict[str, Any]) -> bytes:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def create_openai_web_app(
    engine: TensorMMUEngine,
    api_key: Optional[str] = None,
    fail_livez_on_error: bool = False,
) -> web.Application:
    gpu_lock = asyncio.Lock()
    auth_protected_prefixes = ("/v1/", "/chat/", "/completions", "/messages", "/models")

    @web.middleware
    async def cors_and_auth_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
        if request.method == "OPTIONS":
            resp: web.StreamResponse = web.Response(status=204)
        else:
            if api_key and request.path.startswith(auth_protected_prefixes):
                auth_hdr = request.headers.get("Authorization", "")
                x_api = request.headers.get("x-api-key", "")
                valid = (auth_hdr == f"Bearer {api_key}") or (x_api == api_key)
                if not valid:
                    if "messages" in request.path:
                        return web.json_response(
                            {"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key."}},
                            status=401,
                        )
                    return web.json_response(
                        {"error": {"message": "Invalid API key.", "type": "invalid_request_error", "code": "invalid_api_key"}},
                        status=401,
                    )
            try:
                resp = await handler(request)
            except web.HTTPException:
                raise
            except Exception as exc:
                import traceback

                traceback.print_exc()
                if "messages" in request.path:
                    resp = web.json_response(
                        {"type": "error", "error": {"type": "api_error", "message": str(exc)}},
                        status=500,
                    )
                else:
                    resp = web.json_response(
                        {"error": {"message": str(exc), "type": "server_error", "code": "internal_error"}},
                        status=500,
                    )

        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = (
            "Authorization, x-api-key, anthropic-version, anthropic-beta, Content-Type, X-Requested-With"
        )
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

    app = web.Application(middlewares=[cors_and_auth_middleware], client_max_size=32 * 1024 * 1024)

    def _not_ready_response(anthropic_format: bool = False) -> web.Response:
        pct = f" ({engine.lifecycle_progress_pct:.1f}%)" if engine.lifecycle_progress_pct is not None else ""
        msg = (
            f"TensorMMU server is not ready yet [state={engine.lifecycle_state}{pct}]: "
            f"{engine.lifecycle_detail}"
        )
        headers = {"Retry-After": "10"}
        if anthropic_format:
            return web.json_response(
                {
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": msg,
                    },
                    "state": engine.lifecycle_state,
                    "progress_pct": engine.lifecycle_progress_pct,
                },
                status=503,
                headers=headers,
            )
        return web.json_response(
            {
                "error": {
                    "message": msg,
                    "type": "server_error",
                    "code": "model_not_ready",
                    "state": engine.lifecycle_state,
                    "progress_pct": engine.lifecycle_progress_pct,
                }
            },
            status=503,
            headers=headers,
        )

    async def handle_playground(_request: web.Request) -> web.Response:
        return web.Response(text=_PLAYGROUND_HTML, content_type="text/html")

    async def handle_about(_request: web.Request) -> web.Response:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        about_path = os.path.join(repo_root, "docs", "index.html")
        if os.path.isfile(about_path):
            with open(about_path, "r", encoding="utf-8") as f:
                return web.Response(text=f.read(), content_type="text/html")
        return web.Response(text="TensorMMU docs/index.html not found.", status=404)

    async def handle_livez(_request: web.Request) -> web.Response:
        failed = engine.lifecycle_state == "failed"
        http_code = 500 if (failed and fail_livez_on_error) else 200
        return web.json_response(
            {
                "alive": not (failed and fail_livez_on_error),
                "ready": engine.is_ready,
                "state": engine.lifecycle_state,
                "detail": engine.lifecycle_detail,
                "progress_pct": engine.lifecycle_progress_pct,
                "error": engine.lifecycle_error,
            },
            status=http_code,
        )

    async def handle_readyz(_request: web.Request) -> web.Response:
        if engine.is_ready:
            return web.json_response(
                {
                    "ready": True,
                    "state": "ready",
                    "active_model": engine.active_model_id,
                    "cuda_graph_mode": engine.status_dict().get("cuda_graph_mode"),
                },
                status=200,
            )
        return web.json_response(
            {
                "ready": False,
                "state": engine.lifecycle_state,
                "detail": engine.lifecycle_detail,
                "progress_pct": engine.lifecycle_progress_pct,
                "error": engine.lifecycle_error,
            },
            status=503,
            headers={"Retry-After": "10"},
        )

    async def handle_health(_request: web.Request) -> web.Response:
        return web.json_response(engine.status_dict())

    async def handle_models(_request: web.Request) -> web.Response:
        discovered = _discover_nvm_cartridges(engine.checkpoints_dir, engine.active_nvm_dir)
        seen_paths: Set[str] = set()
        models_list: List[Dict[str, Any]] = []
        created_ts = int(time.time())

        if engine.active_model_id:
            models_list.append(
                {
                    "id": engine.active_model_id,
                    "object": "model",
                    "created": created_ts,
                    "owned_by": "tensormmu",
                    "active": engine.is_ready,
                    "ready": engine.is_ready,
                    "state": engine.lifecycle_state,
                    "nvm_dir": engine.active_nvm_dir,
                }
            )
            if engine.active_nvm_dir:
                seen_paths.add(engine.active_nvm_dir)

        for mid, path in discovered.items():
            if path in seen_paths or mid.endswith(".nvm") or mid.endswith(".tmmu"):
                continue
            seen_paths.add(path)
            models_list.append(
                {
                    "id": mid,
                    "object": "model",
                    "created": created_ts,
                    "owned_by": "tensormmu",
                    "active": False,
                    "ready": True,
                    "state": "available",
                    "nvm_dir": path,
                }
            )

        return web.json_response({"object": "list", "data": models_list})

    async def handle_nvm_config(request: web.Request) -> web.Response:
        body = await request.json()
        async with gpu_lock:
            loop = asyncio.get_running_loop()
            if "nvm_dir" in body and body["nvm_dir"]:
                st = await loop.run_in_executor(
                    None,
                    lambda: engine.load_cartridge(
                        nvm_dir=str(body["nvm_dir"]),
                        vm_mode=str(body.get("vm_mode", engine.vm_mode)),
                        tlb_slots=int(body["tlb_slots"]) if body.get("tlb_slots") is not None else None,
                        tlb_strategy=body.get("tlb_strategy"),
                        page_storage=body.get("page_storage"),
                        draft_mode=body.get("draft_mode"),
                    ),
                )
            else:
                if not engine.is_ready:
                    return _not_ready_response(anthropic_format=False)
                st = await loop.run_in_executor(
                    None,
                    lambda: engine.update_runtime_config(
                        vm_mode=body.get("vm_mode"),
                        tlb_slots=int(body["tlb_slots"]) if body.get("tlb_slots") is not None else None,
                        tlb_strategy=body.get("tlb_strategy"),
                        draft_mode=body.get("draft_mode"),
                        confidence_gated_moe=body.get("confidence_gated_moe"),
                    ),
                )
        return web.json_response(st)

    async def _maybe_swap_model_locked(requested_model: Optional[str]) -> str:
        if not requested_model or requested_model in (
            "default",
            engine.active_model_id,
            f"{engine.active_model_id}.nvm",
            f"{engine.active_model_id}.tmmu",
        ):
            return engine.active_model_id
        if not engine.auto_swap_model:
            return engine.active_model_id
        discovered = _discover_nvm_cartridges(engine.checkpoints_dir, engine.active_nvm_dir)
        target_dir = discovered.get(requested_model)
        if target_dir and os.path.abspath(target_dir) != engine.active_nvm_dir:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: engine.load_cartridge(nvm_dir=target_dir))
        return engine.active_model_id

    # -----------------------------------------------------------------------
    # OpenAI `/v1/chat/completions` (JSON + SSE + Native Tool Calling)
    # -----------------------------------------------------------------------
    async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
        if not engine.is_ready:
            return _not_ready_response(anthropic_format=False)

        body = await request.json()
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return web.json_response(
                {"error": {"message": "`messages` must be a non-empty list.", "type": "invalid_request_error"}},
                status=400,
            )

        req_model = body.get("model")
        stream = bool(body.get("stream", False))
        temperature = float(body.get("temperature", 0.0) or 0.0)
        top_p = float(body.get("top_p", 1.0) if body.get("top_p") is not None else 1.0)
        top_k = int(body.get("top_k", 0) or 0)
        repetition_penalty = float(body.get("repetition_penalty", 1.0) or 1.0)
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or engine.default_max_tokens)
        seed = body.get("seed")
        raw_stop = body.get("stop")
        stop_strings = [raw_stop] if isinstance(raw_stop, str) else (list(raw_stop) if isinstance(raw_stop, list) else [])
        raw_tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        enable_thinking, thinking_budget = _resolve_thinking_options(body, engine.default_thinking_budget)
        include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))

        req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not stream:
            async with gpu_lock:
                if not engine.is_ready:
                    return _not_ready_response(anthropic_format=False)
                model_id = await _maybe_swap_model_locked(req_model)
                prompt_ids, tools_active = engine.format_chat_prompt_ids(
                    messages, tools=raw_tools, tool_choice=tool_choice, enable_thinking=enable_thinking
                )
                prompt_tok_cnt = int(prompt_ids.shape[-1])
                loop = asyncio.get_running_loop()
                stats, finish_reason = await loop.run_in_executor(
                    None,
                    lambda: engine.generate_streaming_sync(
                        prompt_ids=prompt_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        stop_strings=stop_strings,
                        seed=seed,
                        enable_thinking=enable_thinking,
                        thinking_token_budget=thinking_budget,
                        tools_active=tools_active,
                    ),
                )

            reasoning_txt, answer_txt, raw_ans_spec = _split_thinking_and_answer(stats.generated_ids, engine.tokenizer)
            answer_txt, _ = _trim_at_stop_strings(answer_txt, stop_strings)
            parsed_tools: List[ParsedToolCall] = []
            if tools_active:
                answer_txt, parsed_tools = parse_tool_calls_from_output(answer_txt, raw_with_special=raw_ans_spec)

            msg_obj: Dict[str, Any] = {
                "role": "assistant",
                "content": (answer_txt if (answer_txt or not parsed_tools) else None),
            }
            if reasoning_txt:
                msg_obj["reasoning_content"] = reasoning_txt
            if parsed_tools:
                msg_obj["tool_calls"] = [tc.to_openai_dict() for tc in parsed_tools]
                eff_finish = "tool_calls"
            else:
                eff_finish = finish_reason if finish_reason in ("stop", "length") else "stop"

            comp_toks = stats.emitted_tokens
            tel_obj = engine.last_request_telemetry or engine._format_telemetry(stats)
            return web.json_response(
                {
                    "id": req_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": msg_obj,
                            "finish_reason": eff_finish,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tok_cnt,
                        "completion_tokens": comp_toks,
                        "total_tokens": prompt_tok_cnt + comp_toks,
                    },
                    "tensormmu_telemetry": tel_obj,
                    "neuralvm_telemetry": tel_obj,
                }
            )

        # Streaming SSE (`text/event-stream`)
        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await resp.prepare(request)

        async with gpu_lock:
            model_id = await _maybe_swap_model_locked(req_model)
            prompt_ids, tools_active = engine.format_chat_prompt_ids(
                messages, tools=raw_tools, tool_choice=tool_choice, enable_thinking=enable_thinking
            )
            prompt_tok_cnt = int(prompt_ids.shape[-1])

            init_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
            await resp.write(f"data: {json.dumps(init_chunk)}\n\n".encode("utf-8"))

            loop = asyncio.get_running_loop()
            tok_queue: asyncio.Queue[Optional[List[int]]] = asyncio.Queue()
            cancel_ev = threading.Event()
            inc_dec = _IncrementalTokenTextDecoder(engine.tokenizer)
            reason_dec = _IncrementalTokenTextDecoder(engine.tokenizer)
            tool_filter = ToolAwareStreamFilter(active=tools_active)

            thought_end_id, _ = engine._resolve_thought_close_spec(enable_thinking)
            in_thought = thought_end_id is not None and enable_thinking

            def _push_cb(step_toks: List[int]) -> None:
                loop.call_soon_threadsafe(tok_queue.put_nowait, list(step_toks))

            def _worker() -> Tuple[SpeculativeDecodeStats, str]:
                try:
                    return engine.generate_streaming_sync(
                        prompt_ids=prompt_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        stop_strings=stop_strings,
                        seed=seed,
                        enable_thinking=enable_thinking,
                        thinking_token_budget=thinking_budget,
                        tools_active=tools_active,
                        on_step_tokens=_push_cb,
                        cancel_event=cancel_ev,
                    )
                finally:
                    loop.call_soon_threadsafe(tok_queue.put_nowait, None)

            gen_fut = loop.run_in_executor(None, _worker)

            try:
                while True:
                    item = await tok_queue.get()
                    if item is None:
                        break
                    ans_toks: List[int] = []
                    if in_thought:
                        if thought_end_id in item:
                            split_pos = item.index(thought_end_id)  # type: ignore[arg-type]
                            r_delta = reason_dec.push(item[:split_pos], final=True).replace("<think>", "")
                            in_thought = False
                            ans_toks = item[split_pos + 1 :]
                        else:
                            r_delta = reason_dec.push(item, final=False).replace("<think>", "")
                        if r_delta:
                            r_chunk = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [{"index": 0, "delta": {"reasoning_content": r_delta}, "finish_reason": None}],
                            }
                            await resp.write(f"data: {json.dumps(r_chunk)}\n\n".encode("utf-8"))
                    else:
                        ans_toks = item

                    if not ans_toks:
                        continue
                    raw_delta = inc_dec.push(ans_toks, final=False)
                    delta_txt = tool_filter.push_delta(raw_delta)
                    if stop_strings and delta_txt:
                        full_now = inc_dec.full_text
                        trimmed, matched = _trim_at_stop_strings(full_now, stop_strings)
                        if matched is not None:
                            excess = len(full_now) - len(trimmed)
                            delta_txt = delta_txt[:-excess] if excess < len(delta_txt) else ""
                            cancel_ev.set()
                    if delta_txt:
                        chunk_payload = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {"content": delta_txt}, "finish_reason": None}],
                        }
                        await resp.write(f"data: {json.dumps(chunk_payload)}\n\n".encode("utf-8"))

                tail_txt = tool_filter.push_delta(inc_dec.push([], final=True)) + tool_filter.flush()
                if tail_txt and not stop_strings:
                    chunk_payload = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": tail_txt}, "finish_reason": None}],
                    }
                    await resp.write(f"data: {json.dumps(chunk_payload)}\n\n".encode("utf-8"))

                stats, finish_reason = await gen_fut
                parsed_tools = []
                if tools_active:
                    _, full_ans, raw_ans_spec = _split_thinking_and_answer(stats.generated_ids, engine.tokenizer)
                    _, parsed_tools = parse_tool_calls_from_output(full_ans, raw_with_special=raw_ans_spec)

                if parsed_tools:
                    tc_deltas = [
                        {
                            "index": idx,
                            "id": tc.call_id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments_json},
                        }
                        for idx, tc in enumerate(parsed_tools)
                    ]
                    tc_chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"tool_calls": tc_deltas}, "finish_reason": None}],
                    }
                    await resp.write(f"data: {json.dumps(tc_chunk)}\n\n".encode("utf-8"))
                    eff_finish = "tool_calls"
                else:
                    eff_finish = finish_reason if finish_reason in ("stop", "length") else "stop"

                fin_chunk: Dict[str, Any] = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": eff_finish}],
                }
                if include_usage:
                    tel_obj = engine.last_request_telemetry or engine._format_telemetry(stats)
                    fin_chunk["usage"] = {
                        "prompt_tokens": prompt_tok_cnt,
                        "completion_tokens": stats.emitted_tokens,
                        "total_tokens": prompt_tok_cnt + stats.emitted_tokens,
                    }
                    fin_chunk["tensormmu_telemetry"] = tel_obj
                    fin_chunk["neuralvm_telemetry"] = tel_obj

                await resp.write(f"data: {json.dumps(fin_chunk)}\n\n".encode("utf-8"))
                await resp.write(b"data: [DONE]\n\n")
            except (ConnectionResetError, asyncio.CancelledError):
                cancel_ev.set()
                await gen_fut

        await resp.write_eof()
        return resp

    # -----------------------------------------------------------------------
    # Anthropic `/v1/messages` & `/v1/messages/count_tokens` (Claude Code / SDK)
    # -----------------------------------------------------------------------
    async def handle_anthropic_count_tokens(request: web.Request) -> web.Response:
        if not engine.is_ready:
            return _not_ready_response(anthropic_format=True)
        body = await request.json()
        anth_msgs = body.get("messages") or []
        canon_msgs = anthropic_messages_to_openai(body.get("system"), anth_msgs)
        enable_thinking, _ = _resolve_thinking_options(body, engine.default_thinking_budget)
        async with gpu_lock:
            if not engine.is_ready:
                return _not_ready_response(anthropic_format=True)
            prompt_ids, _ = engine.format_chat_prompt_ids(
                canon_msgs,
                tools=body.get("tools"),
                tool_choice=body.get("tool_choice"),
                enable_thinking=enable_thinking,
            )
        return web.json_response({"input_tokens": int(prompt_ids.shape[-1])})

    async def handle_anthropic_messages(request: web.Request) -> web.StreamResponse:
        if not engine.is_ready:
            return _not_ready_response(anthropic_format=True)

        body = await request.json()
        anth_msgs = body.get("messages")
        if not anth_msgs or not isinstance(anth_msgs, list):
            return web.json_response(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "`messages` is required."}},
                status=400,
            )

        req_model = body.get("model")
        stream = bool(body.get("stream", False))
        temperature = float(body.get("temperature", 0.0) or 0.0)
        top_p = float(body.get("top_p", 1.0) if body.get("top_p") is not None else 1.0)
        top_k = int(body.get("top_k", 0) or 0)
        max_tokens = int(body.get("max_tokens") or engine.default_max_tokens)
        stop_strings = list(body.get("stop_sequences") or [])
        raw_tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        enable_thinking, thinking_budget = _resolve_thinking_options(body, engine.default_thinking_budget)

        canon_msgs = anthropic_messages_to_openai(body.get("system"), anth_msgs)
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        if not stream:
            async with gpu_lock:
                if not engine.is_ready:
                    return _not_ready_response(anthropic_format=True)
                model_id = await _maybe_swap_model_locked(req_model)
                prompt_ids, tools_active = engine.format_chat_prompt_ids(
                    canon_msgs, tools=raw_tools, tool_choice=tool_choice, enable_thinking=enable_thinking
                )
                in_toks = int(prompt_ids.shape[-1])
                loop = asyncio.get_running_loop()
                stats, finish_reason = await loop.run_in_executor(
                    None,
                    lambda: engine.generate_streaming_sync(
                        prompt_ids=prompt_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        stop_strings=stop_strings,
                        enable_thinking=enable_thinking,
                        thinking_token_budget=thinking_budget,
                        tools_active=tools_active,
                    ),
                )

            reasoning_txt, answer_txt, raw_ans_spec = _split_thinking_and_answer(stats.generated_ids, engine.tokenizer)
            answer_txt, matched_stop = _trim_at_stop_strings(answer_txt, stop_strings)
            parsed_tools = []
            if tools_active:
                answer_txt, parsed_tools = parse_tool_calls_from_output(answer_txt, raw_with_special=raw_ans_spec)

            content_blocks: List[Dict[str, Any]] = []
            if reasoning_txt:
                content_blocks.append({"type": "thinking", "thinking": reasoning_txt, "signature": "tensormmu"})
            if answer_txt or not parsed_tools:
                content_blocks.append({"type": "text", "text": answer_txt})
            for tc in parsed_tools:
                content_blocks.append(tc.to_anthropic_block())

            if parsed_tools:
                stop_reason = "tool_use"
            elif matched_stop is not None:
                stop_reason = "stop_sequence"
            elif finish_reason == "length":
                stop_reason = "max_tokens"
            else:
                stop_reason = "end_turn"

            return web.json_response(
                {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": req_model or model_id,
                    "content": content_blocks,
                    "stop_reason": stop_reason,
                    "stop_sequence": matched_stop,
                    "usage": {
                        "input_tokens": in_toks,
                        "output_tokens": stats.emitted_tokens,
                    },
                }
            )

        # Anthropic Streaming SSE (`event: ...`)
        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await resp.prepare(request)

        async with gpu_lock:
            model_id = await _maybe_swap_model_locked(req_model)
            prompt_ids, tools_active = engine.format_chat_prompt_ids(
                canon_msgs, tools=raw_tools, tool_choice=tool_choice, enable_thinking=enable_thinking
            )
            in_toks = int(prompt_ids.shape[-1])
            eff_model_name = req_model or model_id

            await resp.write(
                _format_sse_event(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "model": eff_model_name,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": in_toks, "output_tokens": 0},
                        },
                    },
                )
            )

            loop = asyncio.get_running_loop()
            tok_queue: asyncio.Queue[Optional[List[int]]] = asyncio.Queue()
            cancel_ev = threading.Event()
            inc_dec = _IncrementalTokenTextDecoder(engine.tokenizer)
            reason_dec = _IncrementalTokenTextDecoder(engine.tokenizer)
            tool_filter = ToolAwareStreamFilter(active=tools_active)

            thought_end_id, _ = engine._resolve_thought_close_spec(enable_thinking)
            in_thought = thought_end_id is not None and enable_thinking
            cur_block_idx = 0
            thought_block_open = False
            text_block_open = False
            matched_stop: Optional[str] = None

            def _push_cb(step_toks: List[int]) -> None:
                loop.call_soon_threadsafe(tok_queue.put_nowait, list(step_toks))

            def _worker() -> Tuple[SpeculativeDecodeStats, str]:
                try:
                    return engine.generate_streaming_sync(
                        prompt_ids=prompt_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        stop_strings=stop_strings,
                        enable_thinking=enable_thinking,
                        thinking_token_budget=thinking_budget,
                        tools_active=tools_active,
                        on_step_tokens=_push_cb,
                        cancel_event=cancel_ev,
                    )
                finally:
                    loop.call_soon_threadsafe(tok_queue.put_nowait, None)

            gen_fut = loop.run_in_executor(None, _worker)

            try:
                while True:
                    item = await tok_queue.get()
                    if item is None:
                        break
                    ans_toks: List[int] = []
                    if in_thought:
                        if thought_end_id in item:
                            split_pos = item.index(thought_end_id)  # type: ignore[arg-type]
                            r_delta = reason_dec.push(item[:split_pos], final=True).replace("<think>", "")
                            in_thought = False
                            ans_toks = item[split_pos + 1 :]
                            if r_delta:
                                if not thought_block_open:
                                    await resp.write(
                                        _format_sse_event(
                                            "content_block_start",
                                            {
                                                "type": "content_block_start",
                                                "index": cur_block_idx,
                                                "content_block": {"type": "thinking", "thinking": ""},
                                            },
                                        )
                                    )
                                    thought_block_open = True
                                await resp.write(
                                    _format_sse_event(
                                        "content_block_delta",
                                        {
                                            "type": "content_block_delta",
                                            "index": cur_block_idx,
                                            "delta": {"type": "thinking_delta", "thinking": r_delta},
                                        },
                                    )
                                )
                            if thought_block_open:
                                await resp.write(
                                    _format_sse_event(
                                        "content_block_delta",
                                        {
                                            "type": "content_block_delta",
                                            "index": cur_block_idx,
                                            "delta": {"type": "signature_delta", "signature": "tensormmu"},
                                        },
                                    )
                                )
                                await resp.write(
                                    _format_sse_event(
                                        "content_block_stop",
                                        {"type": "content_block_stop", "index": cur_block_idx},
                                    )
                                )
                                thought_block_open = False
                                cur_block_idx += 1
                        else:
                            r_delta = reason_dec.push(item, final=False).replace("<think>", "")
                            if r_delta:
                                if not thought_block_open:
                                    await resp.write(
                                        _format_sse_event(
                                            "content_block_start",
                                            {
                                                "type": "content_block_start",
                                                "index": cur_block_idx,
                                                "content_block": {"type": "thinking", "thinking": ""},
                                            },
                                        )
                                    )
                                    thought_block_open = True
                                await resp.write(
                                    _format_sse_event(
                                        "content_block_delta",
                                        {
                                            "type": "content_block_delta",
                                            "index": cur_block_idx,
                                            "delta": {"type": "thinking_delta", "thinking": r_delta},
                                        },
                                    )
                                )
                    else:
                        ans_toks = item

                    if not ans_toks:
                        continue
                    raw_delta = inc_dec.push(ans_toks, final=False)
                    delta_txt = tool_filter.push_delta(raw_delta)
                    if stop_strings and delta_txt:
                        full_now = inc_dec.full_text
                        trimmed, m_stop = _trim_at_stop_strings(full_now, stop_strings)
                        if m_stop is not None:
                            matched_stop = m_stop
                            excess = len(full_now) - len(trimmed)
                            delta_txt = delta_txt[:-excess] if excess < len(delta_txt) else ""
                            cancel_ev.set()

                    if delta_txt:
                        if not text_block_open:
                            await resp.write(
                                _format_sse_event(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": cur_block_idx,
                                        "content_block": {"type": "text", "text": ""},
                                    },
                                )
                            )
                            text_block_open = True
                        await resp.write(
                            _format_sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": cur_block_idx,
                                    "delta": {"type": "text_delta", "text": delta_txt},
                                },
                            )
                        )

                if thought_block_open:
                    await resp.write(
                        _format_sse_event("content_block_stop", {"type": "content_block_stop", "index": cur_block_idx})
                    )
                    thought_block_open = False
                    cur_block_idx += 1

                tail_txt = tool_filter.push_delta(inc_dec.push([], final=True)) + tool_filter.flush()
                if tail_txt and matched_stop is None:
                    if not text_block_open:
                        await resp.write(
                            _format_sse_event(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": cur_block_idx,
                                    "content_block": {"type": "text", "text": ""},
                                },
                            )
                        )
                        text_block_open = True
                    await resp.write(
                        _format_sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": cur_block_idx,
                                "delta": {"type": "text_delta", "text": tail_txt},
                            },
                        )
                    )

                if text_block_open:
                    await resp.write(
                        _format_sse_event("content_block_stop", {"type": "content_block_stop", "index": cur_block_idx})
                    )
                    text_block_open = False
                    cur_block_idx += 1

                stats, finish_reason = await gen_fut
                parsed_tools = []
                if tools_active:
                    _, full_ans, raw_ans_spec = _split_thinking_and_answer(stats.generated_ids, engine.tokenizer)
                    _, parsed_tools = parse_tool_calls_from_output(full_ans, raw_with_special=raw_ans_spec)

                if parsed_tools:
                    for tc in parsed_tools:
                        await resp.write(
                            _format_sse_event(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": cur_block_idx,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tc.anthropic_id,
                                        "name": tc.name,
                                        "input": {},
                                    },
                                },
                            )
                        )
                        await resp.write(
                            _format_sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": cur_block_idx,
                                    "delta": {"type": "input_json_delta", "partial_json": tc.arguments_json},
                                },
                            )
                        )
                        await resp.write(
                            _format_sse_event(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": cur_block_idx},
                            )
                        )
                        cur_block_idx += 1
                    stop_reason = "tool_use"
                elif matched_stop is not None:
                    stop_reason = "stop_sequence"
                elif finish_reason == "length":
                    stop_reason = "max_tokens"
                else:
                    stop_reason = "end_turn"

                # Ensure at least one content block was emitted per Anthropic spec
                if cur_block_idx == 0:
                    await resp.write(
                        _format_sse_event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                    )
                    await resp.write(
                        _format_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
                    )

                await resp.write(
                    _format_sse_event(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": stop_reason, "stop_sequence": matched_stop},
                            "usage": {"output_tokens": stats.emitted_tokens},
                        },
                    )
                )
                await resp.write(_format_sse_event("message_stop", {"type": "message_stop"}))
            except (ConnectionResetError, asyncio.CancelledError):
                cancel_ev.set()
                await gen_fut

        await resp.write_eof()
        return resp

    # -----------------------------------------------------------------------
    # OpenAI `/v1/completions` (Legacy Text Completions)
    # -----------------------------------------------------------------------
    async def handle_completions(request: web.Request) -> web.StreamResponse:
        if not engine.is_ready:
            return _not_ready_response(anthropic_format=False)

        body = await request.json()
        prompt_raw = body.get("prompt", "")
        prompt_str = prompt_raw[0] if isinstance(prompt_raw, list) and prompt_raw else str(prompt_raw)
        req_model = body.get("model")
        stream = bool(body.get("stream", False))
        temperature = float(body.get("temperature", 0.0) or 0.0)
        top_p = float(body.get("top_p", 1.0) if body.get("top_p") is not None else 1.0)
        top_k = int(body.get("top_k", 0) or 0)
        repetition_penalty = float(body.get("repetition_penalty", 1.0) or 1.0)
        max_tokens = int(body.get("max_tokens") or 1024)
        seed = body.get("seed")
        raw_stop = body.get("stop")
        stop_strings = [raw_stop] if isinstance(raw_stop, str) else (list(raw_stop) if isinstance(raw_stop, list) else [])

        req_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not stream:
            async with gpu_lock:
                if not engine.is_ready:
                    return _not_ready_response(anthropic_format=False)
                model_id = await _maybe_swap_model_locked(req_model)
                prompt_ids = engine.tokenizer(prompt_str, return_tensors="pt").input_ids.to(engine.device)
                prompt_tok_cnt = int(prompt_ids.shape[-1])
                loop = asyncio.get_running_loop()
                stats, finish_reason = await loop.run_in_executor(
                    None,
                    lambda: engine.generate_streaming_sync(
                        prompt_ids=prompt_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        stop_strings=stop_strings,
                        seed=seed,
                    ),
                )
            text_out, _ = _trim_at_stop_strings(
                engine.tokenizer.decode(stats.generated_ids, skip_special_tokens=True),
                stop_strings,
            )
            tel_obj = engine.last_request_telemetry or engine._format_telemetry(stats)
            return web.json_response(
                {
                    "id": req_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "text": text_out,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": finish_reason if finish_reason in ("stop", "length") else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tok_cnt,
                        "completion_tokens": stats.emitted_tokens,
                        "total_tokens": prompt_tok_cnt + stats.emitted_tokens,
                    },
                    "tensormmu_telemetry": tel_obj,
                    "neuralvm_telemetry": tel_obj,
                }
            )

        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)

        async with gpu_lock:
            model_id = await _maybe_swap_model_locked(req_model)
            prompt_ids = engine.tokenizer(prompt_str, return_tensors="pt").input_ids.to(engine.device)
            loop = asyncio.get_running_loop()
            tok_queue: asyncio.Queue[Optional[List[int]]] = asyncio.Queue()
            cancel_ev = threading.Event()
            inc_dec = _IncrementalTokenTextDecoder(engine.tokenizer)

            def _push_cb(step_toks: List[int]) -> None:
                loop.call_soon_threadsafe(tok_queue.put_nowait, list(step_toks))

            def _worker() -> Tuple[SpeculativeDecodeStats, str]:
                try:
                    return engine.generate_streaming_sync(
                        prompt_ids=prompt_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        stop_strings=stop_strings,
                        seed=seed,
                        on_step_tokens=_push_cb,
                        cancel_event=cancel_ev,
                    )
                finally:
                    loop.call_soon_threadsafe(tok_queue.put_nowait, None)

            gen_fut = loop.run_in_executor(None, _worker)
            try:
                while True:
                    item = await tok_queue.get()
                    if item is None:
                        break
                    delta_txt = inc_dec.push(item, final=False)
                    if delta_txt:
                        chunk_payload = {
                            "id": req_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_id,
                            "choices": [{"text": delta_txt, "index": 0, "logprobs": None, "finish_reason": None}],
                        }
                        await resp.write(f"data: {json.dumps(chunk_payload)}\n\n".encode("utf-8"))
                _, finish_reason = await gen_fut
                fin_chunk = {
                    "id": req_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": finish_reason}],
                }
                await resp.write(f"data: {json.dumps(fin_chunk)}\n\n".encode("utf-8"))
                await resp.write(b"data: [DONE]\n\n")
            except (ConnectionResetError, asyncio.CancelledError):
                cancel_ev.set()
                await gen_fut

        await resp.write_eof()
        return resp

    # UI & Docs
    app.router.add_get("/", handle_playground)
    app.router.add_get("/about", handle_about)

    # Kubernetes Liveness / Readiness & Telemetry
    app.router.add_get("/livez", handle_livez)
    app.router.add_get("/healthz", handle_livez)
    app.router.add_get("/health/live", handle_livez)
    app.router.add_get("/readyz", handle_readyz)
    app.router.add_get("/health/ready", handle_readyz)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/tmmu/status", handle_health)
    app.router.add_post("/v1/tmmu/config", handle_nvm_config)
    app.router.add_get("/v1/nvm/status", handle_health)
    app.router.add_post("/v1/nvm/config", handle_nvm_config)

    # OpenAI Compatible Endpoints
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_post("/chat/completions", handle_chat_completions)
    app.router.add_post("/v1/completions", handle_completions)
    app.router.add_post("/completions", handle_completions)

    # Anthropic Compatible Endpoints (Claude Code, Anthropic SDK, Oh My Pi)
    app.router.add_post("/v1/messages", handle_anthropic_messages)
    app.router.add_post("/messages", handle_anthropic_messages)
    app.router.add_post("/v1/messages/count_tokens", handle_anthropic_count_tokens)
    app.router.add_post("/messages/count_tokens", handle_anthropic_count_tokens)
    return app


create_tensormmu_web_app = create_openai_web_app
