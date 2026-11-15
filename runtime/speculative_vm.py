"""Phase E Runtime: Lossless Two-Tier Speculative Amortized Paging for NeuralVM `.nvm` Cartridges.

Combines:
  1. Tier 1 (`0.0 MB` VRAM, `~0.02 ms` CPU): `PromptLookupDrafter` — recency-prioritized multi-order
     (`n = 5..2`) suffix n-gram drafter over `[prompt_ids + generated_ids]`, capable of drafting arbitrary
     out-of-shortlist 262k-vocab prompt entities, numbers, equations, and repeated CoT phrases.
  2. Tier 2 (`~118 MB` VRAM, `< 0.25 ms/tok` GPU): `EagleLiteDraftHead` — self-speculative gated SwiGLU
     feature + token + EMA-context transition head conditioned on `31B`'s exact final RMSNorm hidden state
     and resident scaled token embeddings, projecting through a calibrated `INT8` `PhaseCQuantLinear`
     shortlist classifier (`12,288` tokens) initialized directly from `31B`'s `lm_head.weight`.
  3. `SpeculativeVMGenerator`: Single-pass `M = K + 1` (`M <= 5` Ada SM occupancy sweet spot) exact greedy
     verification on `BitTieredTLBMLP(tlb_strategy="amortized_spec_tlb")`, amortizing both the 60-layer
     resident forward pass AND PCIe specialist page-miss DMA across all accepted + bonus tokens per step,
     with exact `HybridCache.crop(-n_reject)` KV rollback.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from runtime.gpu_page_tlb import BitTieredTLBMLP, SharedQuantLUTPool
from runtime.quant_linear_phase_c import PhaseCQuantLinear


class DraftRMSNorm(nn.Module):
    """Lightweight FP32-accumulated RMSNorm for `EagleLiteDraftHead`."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dt = x.dtype
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(orig_dt)


class PromptLookupDrafter:
    """Zero-VRAM recency-prioritized multi-order suffix lookup drafter (`n = 5..2`)."""

    def __init__(
        self,
        max_order: int = 5,
        min_order: int = 2,
        max_window: int = 2048,
    ) -> None:
        self.max_order = int(max_order)
        self.min_order = int(min_order)
        self.max_window = int(max_window)

    def draft(
        self,
        seq_ids: Sequence[int],
        max_draft_tokens: int,
        eos_ids: Optional[Set[int]] = None,
        min_order_override: Optional[int] = None,
    ) -> Tuple[List[int], int]:
        """Returns `(draft_token_ids, matched_ngram_order)` (`([], 0)` if no suffix match)."""
        if max_draft_tokens <= 0:
            return [], 0
        seq_len = len(seq_ids)
        lo_order = int(min_order_override) if min_order_override is not None else self.min_order
        hi_order = min(self.max_order, seq_len - 1)
        if hi_order < lo_order:
            return [], 0

        start_bound = max(0, seq_len - self.max_window)
        # Convert window slice to plain list once for fast CPython slice equality
        win = list(seq_ids[start_bound:])
        wlen = len(win)

        for n in range(hi_order, lo_order - 1, -1):
            if wlen <= n:
                continue
            target = win[wlen - n :]
            last_tok = target[-1]
            first_tok = target[0]
            # Cap 2-gram drafts to 2 tokens to avoid speculative over-reach on common bigrams
            k_cap = max_draft_tokens if n >= 3 else min(2, max_draft_tokens)

            # Scan backwards from most recent occurrence for highest local context relevance
            for pos in range(wlen - n - 1, -1, -1):
                if win[pos] == first_tok and win[pos + n - 1] == last_tok:
                    if win[pos : pos + n] == target:
                        end_pos = min(wlen, pos + n + k_cap)
                        cand: List[int] = []
                        for idx in range(pos + n, end_pos):
                            tid = int(win[idx])
                            cand.append(tid)
                            if eos_ids and tid in eos_ids:
                                break
                        if cand:
                            return cand, n
        return [], 0


class EagleLiteDraftHead(nn.Module):
    """Self-Speculative Feature + Token + EMA-Context Gated SwiGLU Draft Head (`~118 MB` VRAM on `31B`).

    Predicts `h_{t+1}, ..., h_{t+K}` autoregressively in latent + residual `d_model` space starting from
    `31B`'s exact verified final RMSNorm hidden state `h_t` and resident token embeddings `embed(y_{t+1})`,
    scoring candidates via an `INT8` `PhaseCQuantLinear` Top-`K_vocab` shortlist head derived from `lm_head`.
    """

    def __init__(
        self,
        d_model: int = 5376,
        d_latent: int = 1024,
        d_ffn: int = 1536,
        shortlist_size: int = 12288,
        max_steps: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_latent = int(d_latent)
        self.d_ffn = int(d_ffn)
        self.shortlist_size = int(shortlist_size)
        self.max_steps = int(max_steps)

        self.rms_h = DraftRMSNorm(self.d_model)
        self.rms_e = DraftRMSNorm(self.d_model)
        self.rms_c = DraftRMSNorm(self.d_model)
        self.rms_lat = DraftRMSNorm(2 * self.d_latent)
        self.rms_out = DraftRMSNorm(self.d_model)

        self.proj_h = nn.Linear(self.d_model, self.d_latent, bias=False)
        self.proj_e = nn.Linear(self.d_model, self.d_latent, bias=False)
        self.proj_c = nn.Linear(self.d_model, self.d_latent, bias=False)
        self.step_emb = nn.Embedding(self.max_steps, self.d_latent)

        self.w_gu = nn.Linear(2 * self.d_latent, 2 * self.d_ffn, bias=False)
        self.w_down = nn.Linear(self.d_ffn, self.d_latent, bias=False)
        self.w_gate = nn.Linear(2 * self.d_latent, self.d_latent, bias=True)

        self.proj_out = nn.Linear(self.d_latent, self.d_model, bias=False)
        self.res_scale = nn.Parameter(torch.full((self.d_model,), 0.5))
        self.logit_bias = nn.Parameter(torch.zeros(self.shortlist_size))

        self.register_buffer(
            "h_chan_scale",
            torch.ones(self.d_model, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "shortlist_ids",
            torch.arange(self.shortlist_size, dtype=torch.long),
            persistent=True,
        )
        # Populated via `attach_int8_shortlist_head(...)` for compact VRAM inference
        self.shortlist_head: Optional[PhaseCQuantLinear] = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.step_emb.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.w_gate.bias)
        with torch.no_grad():
            self.w_gate.bias.fill_(-0.5)
            # Small initial residual projection so step-0 starts close to smooth continuation
            self.proj_out.weight.mul_(0.25)

    @torch.no_grad()
    def attach_int8_shortlist_head(
        self,
        shortlist_weight_bf16: torch.Tensor,
        shortlist_ids: torch.Tensor,
        lut_pool: Optional[SharedQuantLUTPool] = None,
    ) -> None:
        """Packs `[shortlist_size, d_model]` `lm_head` rows + `logit_bias` into `INT8` `PhaseCQuantLinear`."""
        dev = shortlist_weight_bf16.device
        self.shortlist_ids = shortlist_ids.to(device=dev, dtype=torch.long).contiguous()
        tmp_lin = nn.Linear(self.d_model, self.shortlist_size, bias=True, device=dev, dtype=torch.bfloat16)
        tmp_lin.weight.copy_(shortlist_weight_bf16.to(device=dev, dtype=torch.bfloat16))
        tmp_lin.bias.copy_(self.logit_bias.detach().to(device=dev, dtype=torch.bfloat16))
        self.shortlist_head = PhaseCQuantLinear.from_linear(
            tmp_lin,
            quant_mode="int8",
            target_device=str(dev),
            scale_dtype=torch.bfloat16,
            lut_pool=lut_pool,
        )
        del tmp_lin

    def vram_bytes(self) -> int:
        total = 0
        for p in self.parameters():
            total += int(p.numel() * p.element_size())
        total += int(self.shortlist_ids.numel() * self.shortlist_ids.element_size())
        total += int(self.h_chan_scale.numel() * self.h_chan_scale.element_size())
        if self.shortlist_head is not None:
            # Avoid double-counting if registered as submodule parameter/buffer: PhaseCQuantLinear uses buffers
            for b in self.shortlist_head.buffers():
                total += int(b.numel() * b.element_size())
        return total

    def step_transition(
        self,
        h_cur: torch.Tensor,
        u_state: torch.Tensor,
        tok_emb: torch.Tensor,
        c_lat: torch.Tensor,
        step_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advances `(h_cur, u_state)` by 1 draft step in channel-whitened space (`[..., d_model]`)."""
        s_idx = min(int(step_idx), self.max_steps - 1)
        s_vec = self.step_emb.weight[s_idx]
        e_lat = self.proj_e(self.rms_e(tok_emb)) + c_lat + s_vec

        cat_in = self.rms_lat(torch.cat([u_state, e_lat], dim=-1))
        g, up = self.w_gu(cat_in).chunk(2, dim=-1)
        cand_u = self.w_down(F.silu(g) * up)
        gate = torch.sigmoid(self.w_gate(cat_in))
        u_next = (1.0 - gate) * u_state + gate * cand_u

        # Operate residual & RMSNorm in unit-variance whitened space, then restore exact Gemma norm scale!
        ch_sc = self.h_chan_scale.to(h_cur.dtype).clamp_min(1e-4)
        h_white = h_cur / ch_sc
        delta_white = self.proj_out(u_next) * self.res_scale.to(h_cur.dtype)
        h_next = self.rms_out(h_white + delta_white) * ch_sc
        return h_next, u_next

    def init_state(self, h_verified: torch.Tensor, c_ema: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes initial `(u_0, c_lat)` from channel-whitened `h_verified` and `c_ema` (`[B, d_model]`)."""
        ch_sc = self.h_chan_scale.to(h_verified.dtype).clamp_min(1e-4)
        u0 = self.proj_h(self.rms_h(h_verified / ch_sc))
        c_lat = self.proj_c(self.rms_c(c_ema / ch_sc))
        return u0, c_lat

    @torch.no_grad()
    def draft(
        self,
        h_verified: torch.Tensor,
        last_tok_id: int,
        c_ema: torch.Tensor,
        embed_fn: Callable[[torch.Tensor], torch.Tensor],
        max_draft_tokens: int = 3,
        step0_conf_threshold: float = 0.05,
        later_conf_threshold: float = 0.10,
        step0_margin_threshold: float = 0.0,
        later_margin_threshold: float = 0.0,
        min_cum_prob: float = 0.01,
        eos_ids: Optional[Set[int]] = None,
    ) -> List[int]:
        """Autoregressively drafts up to `max_draft_tokens` on GPU with confidence + logit-margin early-stopping."""
        if max_draft_tokens <= 0 or self.shortlist_head is None:
            return []

        dev = h_verified.device
        dt = self.proj_h.weight.dtype
        h_cur = h_verified.reshape(1, self.d_model).to(dt)
        c_cur = c_ema.reshape(1, self.d_model).to(dt)
        u_cur, c_lat = self.init_state(h_cur, c_cur)

        # Keep token IDs on GPU throughout the multi-step draft loop -> single D2H copy at the end!
        cur_tok_gpu = torch.tensor([[int(last_tok_id)]], dtype=torch.long, device=dev)
        drafted_gpu: List[torch.Tensor] = []
        stats_gpu: List[torch.Tensor] = []

        n_steps = min(int(max_draft_tokens), self.max_steps)
        for s in range(n_steps):
            tok_emb = embed_fn(cur_tok_gpu).reshape(1, self.d_model).to(dt)
            h_cur, u_cur = self.step_transition(h_cur, u_cur, tok_emb, c_lat, step_idx=s)
            # NVRTC INT8 shortlist projection: [1, 1, d_model] -> [1, 1, shortlist_size]
            logits = self.shortlist_head(h_cur.unsqueeze(0)).reshape(1, self.shortlist_size).float()
            # Top-2 over shortlist gives both exact Top-1 ID, Top-1 prob (via logsumexp), and Top-1 - Top-2 margin
            top2_vals, top2_idx = torch.topk(logits, k=2, dim=-1)
            max_val = top2_vals[:, 0]
            margin = top2_vals[:, 0] - top2_vals[:, 1]
            best_short_idx = top2_idx[:, 0]
            lse = torch.logsumexp(logits, dim=-1)
            top_prob = torch.exp(max_val - lse)

            next_vocab_id = self.shortlist_ids[best_short_idx]  # [1]
            drafted_gpu.append(next_vocab_id)
            stats_gpu.append(torch.stack((top_prob.squeeze(0), margin.squeeze(0))))
            cur_tok_gpu = next_vocab_id.view(1, 1)

        # Single D2H sync for all drafted IDs + (confidence, margin) pairs!
        all_ids = torch.cat(drafted_gpu, dim=0).tolist()
        all_stats = torch.stack(stats_gpu, dim=0).tolist()

        accepted_draft: List[int] = []
        cum_prob = 1.0
        for s, (tid, (p_val, m_val)) in enumerate(zip(all_ids, all_stats)):
            # Beyond M=4 (s >= 3 -> K=4, M=5), require strong confidence since SM occupancy steps from 4 -> 3 blocks
            thresh = step0_conf_threshold if s == 0 else (max(0.45, later_conf_threshold) if s >= 3 else later_conf_threshold)
            m_thresh = step0_margin_threshold if s == 0 else later_margin_threshold
            cum_prob *= float(p_val)
            if float(p_val) < thresh or float(m_val) < m_thresh or (s >= 1 and cum_prob < min_cum_prob):
                break
            accepted_draft.append(int(tid))
            if eos_ids and int(tid) in eos_ids:
                break
        return accepted_draft

    def save_pretrained_nvm(self, nvm_dir: str) -> str:
        """Saves `EagleLiteDraftHead` weights + INT8 shortlist head into `nvm_dir`."""
        os.makedirs(nvm_dir, exist_ok=True)
        sd: Dict[str, torch.Tensor] = {}
        for k, v in self.state_dict().items():
            sd[k] = v.detach().cpu().contiguous()
        out_path = os.path.join(nvm_dir, "eagle_draft_head.safetensors")
        meta_path = os.path.join(nvm_dir, "eagle_draft_meta.json")
        save_file(sd, out_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "d_model": self.d_model,
                    "d_latent": self.d_latent,
                    "d_ffn": self.d_ffn,
                    "shortlist_size": self.shortlist_size,
                    "max_steps": self.max_steps,
                    "has_int8_shortlist": self.shortlist_head is not None,
                },
                f,
                indent=2,
            )
        return out_path

    @classmethod
    def load_pretrained_nvm(
        cls,
        nvm_dir: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        lut_pool: Optional[SharedQuantLUTPool] = None,
    ) -> "EagleLiteDraftHead":
        meta_path = os.path.join(nvm_dir, "eagle_draft_meta.json")
        sf_path = os.path.join(nvm_dir, "eagle_draft_head.safetensors")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        head = cls(
            d_model=int(meta["d_model"]),
            d_latent=int(meta["d_latent"]),
            d_ffn=int(meta["d_ffn"]),
            shortlist_size=int(meta["shortlist_size"]),
            max_steps=int(meta["max_steps"]),
        )
        raw_sd = load_file(sf_path, device="cpu")
        dev = torch.device(device)

        if meta.get("has_int8_shortlist", False) and "shortlist_head.q_weight" in raw_sd:
            q_w = raw_sd.pop("shortlist_head.q_weight").to(device=dev)
            sc = raw_sd.pop("shortlist_head.scale").to(device=dev)
            bias = raw_sd.pop("shortlist_head.bias").to(device=dev) if "shortlist_head.bias" in raw_sd else None
            raw_sd.pop("shortlist_head.wmin", None)
            head.load_state_dict(raw_sd, strict=True)
            head.to(device=dev, dtype=dtype)
            head.shortlist_ids = head.shortlist_ids.to(device=dev, dtype=torch.long)
            head.shortlist_head = PhaseCQuantLinear(
                in_features=head.d_model,
                out_features=head.shortlist_size,
                quant_mode="int8",
                q_weight=q_w,
                scale=sc,
                wmin=None,
                bias=bias,
                lut_pool=lut_pool,
            )
        else:
            head.load_state_dict(raw_sd, strict=True)
            head.to(device=dev, dtype=dtype)
            head.shortlist_ids = head.shortlist_ids.to(device=dev, dtype=torch.long)

        head.eval()
        return head


class _LMHeadInputCaptureHook:
    """Zero-copy pre-hook on `model.lm_head` capturing the final normalized hidden state `[B, S, D]`."""

    def __init__(self, lm_head: nn.Module) -> None:
        self.last_hidden: Optional[torch.Tensor] = None
        self._handle = lm_head.register_forward_pre_hook(self._hook)

    def _hook(self, module: nn.Module, args: Tuple[Any, ...]) -> None:
        if args and isinstance(args[0], torch.Tensor):
            self.last_hidden = args[0].detach()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.last_hidden = None


def resolve_eos_token_ids(tokenizer: Any) -> Tuple[Set[int], int]:
    """Collects standard EOS + chat turn-end token IDs (`<end_of_turn>`, `<turn|>`, etc.)."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, list):
        eos_ids = {int(x) for x in eos_id}
        primary_eos = int(eos_id[0])
    elif eos_id is not None:
        eos_ids = {int(eos_id)}
        primary_eos = int(eos_id)
    else:
        eos_ids = set()
        primary_eos = 0

    if hasattr(tokenizer, "convert_tokens_to_ids"):
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for tok_str in ("<end_of_turn>", "<turn|>", "<|eot_id|>", "<|im_end|>"):
            try:
                tid = tokenizer.convert_tokens_to_ids(tok_str)
                if isinstance(tid, int) and tid > 0 and tid != unk_id:
                    eos_ids.add(tid)
            except Exception:
                pass
    return eos_ids, primary_eos


@dataclass
class SpeculativeDecodeStats:
    generated_ids: List[int]
    emitted_tokens: int
    forward_steps: int
    draft_proposed: int
    draft_accepted: int
    pld_proposed: int
    pld_accepted: int
    eagle_proposed: int
    eagle_accepted: int
    elapsed_s: float
    ms_per_token: float
    tok_per_s: float
    tokens_per_step: float
    acceptance_rate_pct: float
    total_pcie_mb: float
    pcie_mb_per_token: float
    pcie_mb_per_step: float
    tlb_hit_rate_pct: float
    demand_misses: int = 0
    lookahead_prefetched: int = 0
    lookahead_hits: int = 0
    lookahead_wasted: int = 0
    lookahead_precision_pct: float = 0.0
    lookahead_coverage_pct: float = 0.0
    effective_stall_free_hit_pct: float = 0.0


class SpeculativeVMGenerator:
    """Lossless Exact-Greedy Multi-Token Speculative Decoder over a NeuralVM `.nvm` Model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        phase_b_mlps: Optional[Dict[int, Any]] = None,
        draft_head: Optional[EagleLiteDraftHead] = None,
        pld_drafter: Optional[PromptLookupDrafter] = None,
        draft_mode: str = "two_tier",  # "none", "pld_only", "eagle_only", "two_tier"
        max_draft_tokens: int = 3,     # K=3 -> M=4 tokens/pass (Ada 96KB SM occupancy sweet spot!)
        pld_long_draft_tokens: int = 4,  # K=4 -> M=5 on exact >=3-gram suffix matches!
        step0_conf_threshold: float = 0.05,
        later_conf_threshold: float = 0.10,
        step0_margin_threshold: float = 0.0,
        later_margin_threshold: float = 0.0,
        min_cum_prob: float = 0.01,
        confidence_gated_moe: bool = True,
    ) -> None:
        if draft_mode not in ("none", "pld_only", "eagle_only", "two_tier"):
            raise ValueError(f"Unsupported draft_mode '{draft_mode}'.")
        self.model = model
        self.tokenizer = tokenizer
        self.phase_b_mlps = phase_b_mlps or {}
        self.draft_head = draft_head
        self.pld_drafter = pld_drafter or PromptLookupDrafter()
        self.draft_mode = draft_mode
        self.max_draft_tokens = int(max_draft_tokens)
        self.pld_long_draft_tokens = int(pld_long_draft_tokens)
        self.step0_conf_threshold = float(step0_conf_threshold)
        self.later_conf_threshold = float(later_conf_threshold)
        self.step0_margin_threshold = float(step0_margin_threshold)
        self.later_margin_threshold = float(later_margin_threshold)
        self.min_cum_prob = float(min_cum_prob)
        self.confidence_gated_moe = bool(confidence_gated_moe)
        self.eos_ids, self.primary_eos = resolve_eos_token_ids(tokenizer)

    def _resolve_moe_draft_gates(self) -> Tuple[float, float, float, float, float, bool]:
        """Returns `(step0_conf, later_conf, step0_margin, later_margin, min_cum_prob, allow_2gram_pld)`.

        Crucially, thresholds depend ONLY on static model sparsity (`num_experts`), NEVER on `tlb_slots`,
        preserving 100% bit-for-bit (`0-ulp`) token parity across all GPU TLB cache budgets.
        """
        s0_c = self.step0_conf_threshold
        lt_c = self.later_conf_threshold
        s0_m = self.step0_margin_threshold
        lt_m = self.later_margin_threshold
        min_cp = self.min_cum_prob
        allow_2g = True

        if self.confidence_gated_moe and self.phase_b_mlps:
            max_exp = max((int(getattr(m, "num_experts", 0)) for m in self.phase_b_mlps.values()), default=0)
            if max_exp >= 256:
                # Ultra-sparse MoE (e.g. Qwen3-Next-80B: E=512, K=10): require high confidence + margin
                # so uncertain open-ended tokens run stall-free M=1 while structured/CoT spans draft at high precision!
                s0_c = max(s0_c, 0.36)
                lt_c = max(lt_c, 0.46)
                s0_m = max(s0_m, 1.45)
                lt_m = max(lt_m, 1.80)
                min_cp = max(min_cp, 0.18)
                allow_2g = False
            elif max_exp >= 64:
                # Moderate-sparse MoE (e.g. Gemma-4-26B: E=128, K=8)
                s0_c = max(s0_c, 0.22)
                lt_c = max(lt_c, 0.32)
                s0_m = max(s0_m, 0.90)
                lt_m = max(lt_m, 1.20)
                min_cp = max(min_cp, 0.08)
                allow_2g = False

        return s0_c, lt_c, s0_m, lt_m, min_cp, allow_2g

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        stop_on_eos: bool = True,
        reset_steady_state_telemetry_after_prefill: bool = True,
    ) -> SpeculativeDecodeStats:
        """Runs exact lossless greedy generation (`M=1` or speculative `M=K+1`) and collects VM telemetry."""
        self.model.eval()
        dev = prompt_ids.device
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        assert prompt_ids.shape[0] == 1, "SpeculativeVMGenerator supports single-sequence (batch=1) low-latency decode."

        # Ensure compiled MLPs use zero-stall amortized speculative TLB when VM paging is active
        for mlp in self.phase_b_mlps.values():
            if mlp.mode == "vm_paged" and mlp.tlb_strategy == "speculative_tlb":
                mlp.set_tlb_strategy("amortized_spec_tlb")

        eff_s0_conf, eff_lt_conf, eff_s0_marg, eff_lt_marg, eff_min_cum, allow_2gram_pld = (
            self._resolve_moe_draft_gates()
        )

        embed_fn = self.model.get_input_embeddings()
        hook = _LMHeadInputCaptureHook(self.model.lm_head)

        try:
            p_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=dev)
            out = self.model(input_ids=prompt_ids, attention_mask=p_mask, use_cache=True)
            pkv = out.past_key_values
            first_tok = int(out.logits[0, -1].argmax(dim=-1).item())
            assert hook.last_hidden is not None
            h_prefill = hook.last_hidden  # [1, P, D]
            h_verified = h_prefill[:, -1:, :].clone()
            ctx_tail = h_prefill[:, -min(8, int(h_prefill.shape[1])) :, :]
            c_ema = ctx_tail.mean(dim=1, keepdim=True).clone()

            max_total_len = int(prompt_ids.shape[1]) + max_new_tokens + 16
            runner = (
                None
                if getattr(self.model, "_disable_cuda_graphs", False)
                else getattr(self.model, "_nvm_graph_runner", None)
            )
            runner_bound = False
            if runner is not None and runner.is_supported:
                runner_bound = runner.bind_prefill_cache(
                    pkv, seq_len=int(prompt_ids.shape[1]), max_total_len=max_total_len
                )
                if runner_bound:
                    # Free DynamicCache `pkv` immediately when static CUDA-Graph slabs own the session state!
                    pkv = None
                    warm_ms = [1] if self.draft_mode == "none" else [1, 2, 3, 4]
                    runner.ensure_captured(warm_ms)

            if torch.cuda.is_available() and dev.type == "cuda":
                torch.cuda.synchronize(dev)

            if reset_steady_state_telemetry_after_prefill:
                for mlp in self.phase_b_mlps.values():
                    mlp.reset_telemetry(clear_tlb=False)

            seq_ids: List[int] = prompt_ids[0].tolist() + [first_tok]
            gen_ids: List[int] = [first_tok]
            last_tok = first_tok

            forward_steps = 0
            draft_proposed = 0
            draft_accepted = 0
            pld_proposed = 0
            pld_accepted = 0
            eagle_proposed = 0
            eagle_accepted = 0

            # Pre-allocate reusable 1D attention_mask buffer up to max sequence length (avoids per-step torch.cat!)
            ones_mask_buf = None if runner_bound else torch.ones((1, max_total_len), dtype=torch.long, device=dev)

            t0 = time.perf_counter()
            if not (stop_on_eos and (first_tok in self.eos_ids)):
                while len(gen_ids) < max_new_tokens:
                    rem = max_new_tokens - len(gen_ids)
                    k_budget = max(0, rem - 1)
                    if runner_bound:
                        k_budget = min(k_budget, 7)

                    draft_toks: List[int] = []
                    draft_source = "none"

                    if k_budget > 0 and self.draft_mode != "none":
                        eos_filter = self.eos_ids if stop_on_eos else None
                        if self.draft_mode == "pld_only":
                            k_lim = min(self.pld_long_draft_tokens, k_budget)
                            min_ord = None if allow_2gram_pld else 3
                            draft_toks, _ = self.pld_drafter.draft(
                                seq_ids, k_lim, eos_ids=eos_filter, min_order_override=min_ord
                            )
                            if draft_toks:
                                draft_source = "pld"
                        elif self.draft_mode == "eagle_only":
                            if self.draft_head is not None:
                                k_lim = min(self.max_draft_tokens, k_budget)
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
                                    eos_ids=eos_filter,
                                )
                                if draft_toks:
                                    draft_source = "eagle"
                        elif self.draft_mode == "two_tier":
                            # Under Phase H CUDA Graphs on Sparse-MoE (`runner_bound and not allow_2gram_pld`),
                            # 1 step is 7-20 ms so ungraphed Eagle (~6 ms WDDM launches + 2.6x MoE cold misses)
                            # is net-negative; run 0.02 ms CPU 4+-gram PLD (`M <= 4` graph) + stall-free M=1!
                            is_moe_graph = runner_bound and not allow_2gram_pld
                            pld_min_ord = 4 if is_moe_graph else 3
                            k_pld = min(3 if is_moe_graph else self.pld_long_draft_tokens, k_budget)
                            cand_pld, matched_n = self.pld_drafter.draft(
                                seq_ids, k_pld, eos_ids=eos_filter, min_order_override=pld_min_ord
                            )
                            if cand_pld and matched_n >= pld_min_ord:
                                draft_toks = cand_pld
                                draft_source = "pld"
                            elif not is_moe_graph:
                                # 2. Self-speculative Eagle-Lite feature + token + context transition head
                                if self.draft_head is not None:
                                    k_eag = min(self.max_draft_tokens, k_budget)
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
                                        eos_ids=eos_filter,
                                    )
                                    if draft_toks:
                                        draft_source = "eagle"
                                # 3. Fallback to 2-gram prompt lookup ONLY on dense / low-sparsity models
                                if not draft_toks and allow_2gram_pld:
                                    cand_bi, _ = self.pld_drafter.draft(
                                        seq_ids, min(2, k_budget), eos_ids=eos_filter, min_order_override=2
                                    )
                                    if cand_bi:
                                        draft_toks = cand_bi
                                        draft_source = "pld"

                    K = len(draft_toks)
                    M = K + 1
                    draft_proposed += K
                    if draft_source == "pld":
                        pld_proposed += K
                    elif draft_source == "eagle":
                        eagle_proposed += K

                    # Current KV cache has `len(seq_ids) - 1` entries (everything strictly before `last_tok`)
                    use_graph_step = runner_bound and runner is not None and runner.can_run_step(M)
                    if use_graph_step:
                        pred_ids, h_step_all, _ = runner.step([last_tok, *draft_toks], return_logits=False)
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

                        pred_ids = out.logits[0].argmax(dim=-1).tolist()
                        assert hook.last_hidden is not None
                        h_step_all = hook.last_hidden  # [1, M, D]

                    n_acc = 0
                    hit_eos = False
                    emitted_step: List[int] = []
                    for j in range(K):
                        if pred_ids[j] == draft_toks[j]:
                            n_acc += 1
                            tid = int(draft_toks[j])
                            emitted_step.append(tid)
                            if stop_on_eos and (tid in self.eos_ids):
                                hit_eos = True
                                break
                        else:
                            break

                    draft_accepted += n_acc
                    if draft_source == "pld":
                        pld_accepted += n_acc
                    elif draft_source == "eagle":
                        eagle_accepted += n_acc

                    if not hit_eos and (len(gen_ids) + len(emitted_step) < max_new_tokens):
                        bonus_tok = int(pred_ids[n_acc])
                        emitted_step.append(bonus_tok)
                        if stop_on_eos and (bonus_tok in self.eos_ids):
                            hit_eos = True

                    # Crop unaccepted draft KV entries (and linear-attention conv/recurrent states if hybrid)
                    # and commit/demote probationary speculative TLB slots based on exact `n_acc`!
                    valid_kv_added = len(emitted_step)
                    if use_graph_step:
                        assert runner is not None
                        runner.finalize_step(M, valid_kv_added)
                        if K > 0:
                            for mlp in self.phase_b_mlps.values():
                                if hasattr(mlp, "commit_speculative_step"):
                                    mlp.commit_speculative_step(n_acc)
                    elif K > 0:
                        from runtime.qwen3_next_kernels import finalize_or_rollback_hybrid_cache

                        finalize_or_rollback_hybrid_cache(pkv, M, valid_kv_added)
                        for mlp in self.phase_b_mlps.values():
                            if hasattr(mlp, "commit_speculative_step"):
                                mlp.commit_speculative_step(n_acc)

                    # Update verified hidden state & EMA context using ONLY verified positions `0 .. valid_kv_added - 1`
                    last_valid_pos = valid_kv_added - 1
                    h_verified = h_step_all[:, last_valid_pos : last_valid_pos + 1, :].clone()
                    for pos_i in range(valid_kv_added):
                        c_ema.mul_(0.75).add_(h_step_all[:, pos_i : pos_i + 1, :], alpha=0.25)

                    gen_ids.extend(emitted_step)
                    seq_ids.extend(emitted_step)
                    last_tok = emitted_step[-1]

                    if hit_eos:
                        break

            if torch.cuda.is_available() and dev.type == "cuda":
                torch.cuda.synchronize(dev)
            elapsed_s = time.perf_counter() - t0
            del pkv, out
        finally:
            hook.close()

        # Steady-state decode emitted `len(gen_ids) - 1` tokens after the prefill token `first_tok`
        decode_emitted = max(1, len(gen_ids) - 1) if forward_steps > 0 else len(gen_ids)
        f_steps = max(1, forward_steps)

        tot_pcie_bytes = 0.0
        tot_hits = 0.0
        tot_misses = 0.0
        tot_demand_misses = 0.0
        tot_la_pref = 0.0
        tot_la_hits = 0.0
        tot_la_wasted = 0.0
        for mlp in self.phase_b_mlps.values():
            t = mlp.tlb_cache.telemetry()
            tot_pcie_bytes += t["pcie_bytes"]
            tot_hits += t["page_hits"]
            tot_misses += t["page_misses"]
            tot_demand_misses += t.get("demand_misses", t["page_misses"])
            tot_la_pref += t.get("lookahead_prefetched", 0.0)
            tot_la_hits += t.get("lookahead_hits", 0.0)
            tot_la_wasted += t.get("lookahead_wasted", 0.0)

        tot_pcie_mb = tot_pcie_bytes / float(1024 ** 2)
        tot_req = max(1.0, tot_hits + tot_misses)
        hit_rate = 100.0 * tot_hits / tot_req
        eff_hit_rate = 100.0 * (tot_hits + tot_la_hits) / tot_req
        la_prec = 100.0 * tot_la_hits / max(1.0, tot_la_pref) if tot_la_pref > 0 else 0.0
        la_cov = 100.0 * tot_la_hits / max(1.0, tot_misses) if tot_misses > 0 else 0.0
        acc_rate = 100.0 * float(draft_accepted) / float(max(1, draft_proposed)) if draft_proposed > 0 else 0.0

        return SpeculativeDecodeStats(
            generated_ids=gen_ids,
            emitted_tokens=decode_emitted,
            forward_steps=forward_steps,
            draft_proposed=draft_proposed,
            draft_accepted=draft_accepted,
            pld_proposed=pld_proposed,
            pld_accepted=pld_accepted,
            eagle_proposed=eagle_proposed,
            eagle_accepted=eagle_accepted,
            elapsed_s=elapsed_s,
            ms_per_token=(elapsed_s * 1000.0) / float(decode_emitted),
            tok_per_s=float(decode_emitted) / max(1e-6, elapsed_s),
            tokens_per_step=float(decode_emitted) / float(f_steps),
            acceptance_rate_pct=acc_rate,
            total_pcie_mb=tot_pcie_mb,
            pcie_mb_per_token=tot_pcie_mb / float(decode_emitted),
            pcie_mb_per_step=tot_pcie_mb / float(f_steps),
            tlb_hit_rate_pct=hit_rate,
            demand_misses=int(round(tot_demand_misses)),
            lookahead_prefetched=int(round(tot_la_pref)),
            lookahead_hits=int(round(tot_la_hits)),
            lookahead_wasted=int(round(tot_la_wasted)),
            lookahead_precision_pct=la_prec,
            lookahead_coverage_pct=la_cov,
            effective_stall_free_hit_pct=eff_hit_rate,
        )


def build_gsm8k_train_calib_sequences(
    tokenizer: Any,
    num_examples: int = 24,
    max_seq_len: int = 256,
) -> List[torch.Tensor]:
    """Builds zero-leakage CoT training sequences strictly from `openai/gsm8k` `train` split."""
    seqs: List[torch.Tensor] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="train")
        for row in ds:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if not q or "####" not in a:
                continue
            instruction = (
                "Solve this math problem concisely step by step. "
                "Write the final numerical answer on the last line after #### (for example: #### 42).\n\n"
                f"Question: {q}"
            )
            # Clean GSM8K `<<...>>` calculator annotations so tokens match natural Gemma CoT output
            import re
            clean_a = re.sub(r"<<[^>]*>>", "", a)
            if hasattr(tokenizer, "apply_chat_template"):
                msgs = [
                    {"role": "user", "content": instruction},
                    {"role": "model", "content": clean_a},
                ]
                try:
                    full_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                except Exception:
                    msgs[1]["role"] = "assistant"
                    full_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            else:
                full_str = f"{instruction}\nAnswer: {clean_a}"

            ids = tokenizer(full_str, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if ids.numel() > max_seq_len:
                ids = ids[:max_seq_len]
            if ids.numel() >= 32:
                seqs.append(ids.long())
            if len(seqs) >= num_examples:
                break
    except Exception:
        pass
    return seqs


def calibrate_and_save_eagle_draft_head(
    nvm_dir: str,
    calib_token_sequences: Sequence[torch.Tensor],
    shortlist_size: int = 12288,
    d_latent: int = 1024,
    d_ffn: int = 1536,
    epochs: int = 14,
    lr: float = 1.5e-3,
    rollout_horizon: int = 4,
    tlb_slots_override: Optional[int] = None,
    device: str = "cuda:0",
) -> str:
    """Two-stage VRAM-safe calibration of `EagleLiteDraftHead` (`zero` eval/test split leakage):

      Stage 1 (Teacher Feature & Top-32 Logit Extraction):
        Loads the `.nvm` cartridge (`vm_paged`), captures exact pre-`lm_head` hidden states `H`,
        token embeddings, and exact Top-32 `lm_head` teacher logits across `calib_token_sequences`
        (`Logic_Calib` + `Fact_Calib` + `GSM8K train` + short on-policy rollouts on `GSM8K train`),
        extracts `embed_tokens` + `lm_head` shortlist rows to CPU, and **unloads the model** completely.
      Stage 2 (Multi-Step Self-Feeding Distillation on Free GPU):
        With 22+ GB VRAM free, trains `EagleLiteDraftHead` via 4-step scheduled-sampling rollout
        (Top-32 Teacher KL Divergence + Cosine/Smooth-L1 Hidden Matching), packs the `INT8` shortlist
        `PhaseCQuantLinear` classifier, and writes `eagle_draft_head.safetensors` inside `nvm_dir`.
    """
    from decompiler.nvm_serializer import load_nvm_cartridge

    dev = torch.device(device)
    t_start = time.perf_counter()
    with open(os.path.join(nvm_dir, "manifest.json"), "r", encoding="utf-8") as mf:
        fmt_ver = json.load(mf).get("format_version", "nvm_v1")
    is_moe_cart = fmt_ver == "nvm_moe_v1"
    eff_tlb_strat = "overlap_shared_mlp" if is_moe_cart else "amortized_spec_tlb"
    eff_slots_ovr = tlb_slots_override if is_moe_cart else (8 if tlb_slots_override is None else tlb_slots_override)

    print(f"[Phase E Calib] Stage 1: Loading .nvm cartridge ({fmt_ver}) from {nvm_dir} to capture teacher states...")
    model, tokenizer, phase_b_mlps, lut_pool, ple_off, _ = load_nvm_cartridge(
        nvm_dir=nvm_dir,
        device=device,
        dtype=torch.bfloat16,
        mode="vm_paged",
        tlb_strategy=eff_tlb_strat,
        page_storage="nvme_mmap",
        tlb_slots_override=eff_slots_ovr,
        decode_req_pages_override=4,
        use_phase_c_kernels=True,
        load_tokenizer=True,
    )

    # Also generate 6 short (96-token) on-policy greedy CoT traces on GSM8K TRAIN questions so the draft head
    # learns 31B's exact own step-by-step phrasing ("Step 1:", "**", "####", etc.) with ZERO test-set touch!
    on_policy_seqs: List[torch.Tensor] = []
    try:
        from datasets import load_dataset

        ds_tr = load_dataset("openai/gsm8k", "main", split="train")
        gen_helper = SpeculativeVMGenerator(
            model=model,
            tokenizer=tokenizer,
            phase_b_mlps=phase_b_mlps,
            draft_mode="pld_only",
            max_draft_tokens=3,
            pld_long_draft_tokens=4,
        )
        count_op = 0
        for row in ds_tr:
            q = (row.get("question") or "").strip()
            if not q:
                continue
            instr = (
                "Solve this math problem concisely step by step. "
                "Write the final numerical answer on the last line after #### (for example: #### 42).\n\n"
                f"Question: {q}"
            )
            msgs = [{"role": "user", "content": instr}]
            p_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            p_ids = tokenizer(p_str, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            st = gen_helper.generate(p_ids, max_new_tokens=80, stop_on_eos=True)
            full_ids = torch.cat(
                [p_ids[0].cpu(), torch.tensor(st.generated_ids, dtype=torch.long)], dim=0
            )
            on_policy_seqs.append(full_ids)
            count_op += 1
            if count_op >= 4:
                break
        del gen_helper
        print(f"[Phase E Calib] Collected {len(on_policy_seqs)} on-policy 31B rollouts on GSM8K train split.")
    except Exception as exc:
        print(f"[Phase E Calib] Note: skipped on-policy train rollouts ({exc}).")

    all_seqs: List[torch.Tensor] = [s.detach().cpu().long() for s in calib_token_sequences] + on_policy_seqs

    hook = _LMHeadInputCaptureHook(model.lm_head)
    captured_h: List[torch.Tensor] = []
    captured_ids: List[torch.Tensor] = []
    captured_topv: List[torch.Tensor] = []
    captured_topi: List[torch.Tensor] = []

    if isinstance(model.lm_head, PhaseCQuantLinear):
        vocab_size, d_model = int(model.lm_head.out_features), int(model.lm_head.in_features)
    else:
        vocab_size, d_model = int(model.lm_head.weight.shape[0]), int(model.lm_head.weight.shape[1])
    tok_counts = torch.zeros((vocab_size,), dtype=torch.float32)

    with torch.no_grad():
        for seq in all_seqs:
            if seq.numel() < 16:
                continue
            inp = seq.unsqueeze(0).to(dev)
            msk = torch.ones_like(inp, dtype=torch.long, device=dev)
            out = model(input_ids=inp, attention_mask=msk, use_cache=False)
            assert hook.last_hidden is not None
            h_cpu = hook.last_hidden[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            # Top-32 exact teacher logits per position
            top_v, top_i = torch.topk(out.logits[0].float(), k=32, dim=-1)
            captured_h.append(h_cpu)
            captured_ids.append(seq.clone())
            captured_topv.append(top_v.to(torch.float16).cpu())
            captured_topi.append(top_i.to(torch.int32).cpu())

            # Count both ground-truth tokens and top-8 teacher-predicted tokens for shortlist selection
            uniq_gt, cnt_gt = torch.unique(seq, return_counts=True)
            tok_counts[uniq_gt] += cnt_gt.float() * 4.0
            uniq_tp, cnt_tp = torch.unique(top_i[:, :8].cpu().long(), return_counts=True)
            tok_counts[uniq_tp] += cnt_tp.float()
            del out, inp, msk

    hook.close()

    # Build Shortlist Vocabulary (`shortlist_size = 12,288`):
    #   1. All observed calibration + teacher top-8 tokens weighted by frequency
    #   2. Essential ASCII / digits / math / EOS / chat formatting tokens
    #   3. Lowest token-ID BPE core pieces (in SentencePiece/Gemma vocab, low IDs after byte tokens are highest-prior BPE pieces!)
    eos_ids, _ = resolve_eos_token_ids(tokenizer)
    for eid in eos_ids:
        if 0 <= eid < vocab_size:
            tok_counts[eid] += 1e6
    for probe_str in (
        "0 1 2 3 4 5 6 7 8 9 10 12 15 16 18 20 24 25 30 36 40 45 48 50 60 72 75 80 90 100 120 150 200 500 1000\n"
        "+ - * / = $ % . , : ; ( ) [ ] ** #### Step Answer Therefore Total First Next Then So Let's\n"
        "\n\n  1. 2. 3. 4. 5."
    ):
        try:
            pids = tokenizer(probe_str, add_special_tokens=False).input_ids
            for pid in pids:
                if 0 <= int(pid) < vocab_size:
                    tok_counts[int(pid)] += 1e4
        except Exception:
            pass

    # Smooth BPE rank prior so unseen common English words enter shortlist ahead of rare CJK/multilingual tails
    rank_prior = 1.0 / (torch.arange(vocab_size, dtype=torch.float32) + 512.0)
    shortlist_score = tok_counts + rank_prior
    eff_shortlist = min(int(shortlist_size), vocab_size)
    _, shortlist_ids_cpu = torch.topk(shortlist_score, k=eff_shortlist, largest=True, sorted=True)
    shortlist_ids_cpu = shortlist_ids_cpu.long().contiguous()

    # Extract exact shortlist `lm_head` rows + scaled input embedding parameters BEFORE unloading!
    with torch.no_grad():
        sl_gpu = shortlist_ids_cpu.to(dev)
        if isinstance(model.lm_head, PhaseCQuantLinear):
            assert model.lm_head.quant_mode == "int8", "Only INT8 PhaseCQuantLinear lm_head supported for shortlist extraction."
            qw_rows = model.lm_head.q_weight[sl_gpu, :d_model].float()
            sc_rows = model.lm_head.scale[sl_gpu].float().reshape(-1, 1)
            short_w_cpu = (qw_rows * sc_rows).to("cpu", torch.bfloat16).contiguous()
        else:
            short_w_cpu = model.lm_head.weight[sl_gpu].detach().to("cpu", torch.bfloat16).contiguous()
        emb_mod = model.get_input_embeddings()
        emb_w_cpu = emb_mod.weight.detach().to("cpu", torch.bfloat16).contiguous()
        emb_scale = float(getattr(emb_mod, "embed_scale", 1.0))
        if emb_scale == 1.0 and hasattr(model, "config"):
            # Probe actual scaling factor directly from `emb_mod` on token 10!
            probe_t = torch.tensor([[10]], dtype=torch.long, device=dev)
            probe_out = emb_mod(probe_t)[0, 0].float()
            raw_row = emb_mod.weight[10].float()
            ratio = (probe_out.norm() / raw_row.norm().clamp_min(1e-8)).item()
            emb_scale = ratio

    # Completely unload 31B so Stage 2 distillation has ~23 GB free GPU VRAM!
    mlp_tmp = None
    for mlp_tmp in phase_b_mlps.values():
        mlp_tmp.release_tlb() if hasattr(mlp_tmp, "release_tlb") else mlp_tmp.set_mode("core_only")
    mlp_tmp = None
    del model, phase_b_mlps, lut_pool, ple_off
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"[Phase E Calib] Stage 2: Training EagleLiteDraftHead on {len(captured_h)} sequences "
        f"(shortlist={eff_shortlist}, embed_scale={emb_scale:.3f}, CUDA_used={torch.cuda.memory_allocated(dev)/(1024**2):.1f} MB)..."
    )

    # Build reverse lookup `vocab_to_short[vocab_id] -> [0..eff_shortlist-1] or -1`
    vocab_to_short = torch.full((vocab_size,), -1, dtype=torch.long, device=dev)
    shortlist_ids_gpu = shortlist_ids_cpu.to(dev)
    vocab_to_short[shortlist_ids_gpu] = torch.arange(eff_shortlist, dtype=torch.long, device=dev)

    emb_w_gpu = emb_w_cpu.to(dev)
    short_w_gpu = short_w_cpu.to(dev)  # [K_short, D] BF16

    def _fast_embed(t_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(t_ids, emb_w_gpu) * emb_scale

    # Assemble fixed-window training chunks of length `W = 64` across all captured sequences
    win_len = 64
    stride = 24
    h_windows: List[torch.Tensor] = []
    id_windows: List[torch.Tensor] = []
    tv_windows: List[torch.Tensor] = []
    ti_windows: List[torch.Tensor] = []
    for h_s, id_s, tv_s, ti_s in zip(captured_h, captured_ids, captured_topv, captured_topi):
        L = int(h_s.shape[0])
        if L < rollout_horizon + 4:
            continue
        if L <= win_len:
            pad_n = win_len - L
            if pad_n == 0:
                h_windows.append(h_s)
                id_windows.append(id_s)
                tv_windows.append(tv_s)
                ti_windows.append(ti_s)
            continue
        for st in range(0, L - win_len + 1, stride):
            h_windows.append(h_s[st : st + win_len])
            id_windows.append(id_s[st : st + win_len])
            tv_windows.append(tv_s[st : st + win_len])
            ti_windows.append(ti_s[st : st + win_len])
        if (L - win_len) % stride != 0:
            h_windows.append(h_s[-win_len:])
            id_windows.append(id_s[-win_len:])
            tv_windows.append(tv_s[-win_len:])
            ti_windows.append(ti_s[-win_len:])

    H_all = torch.stack(h_windows, dim=0).to(dev)        # [N, W, D] bf16
    IDs_all = torch.stack(id_windows, dim=0).to(dev)     # [N, W] int64
    TopV_all = torch.stack(tv_windows, dim=0).to(dev)    # [N, W, 32] fp16
    TopI_all = torch.stack(ti_windows, dim=0).to(dev)    # [N, W, 32] int32

    # Precompute Causal EMA context `C_all` (`c_t = 0.75 * c_{t-1} + 0.25 * h_t`) matching inference!
    C_all = torch.empty_like(H_all)
    C_all[:, 0, :] = H_all[:, 0, :]
    for t_i in range(1, win_len):
        C_all[:, t_i, :] = 0.75 * C_all[:, t_i - 1, :] + 0.25 * H_all[:, t_i, :]

    # Coverage stats of shortlist over calibration ground-truth tokens
    short_targets_all = vocab_to_short[IDs_all]
    cov_pct = 100.0 * (short_targets_all >= 0).float().mean().item()
    print(f"[Phase E Calib] Assembled {H_all.shape[0]} windows (x{win_len} tokens) | Shortlist GT coverage: {cov_pct:.2f}%")

    # Empirical per-channel RMS of Gemma 4's final normed hidden state (preserves heterogeneous RMSNorm scales!)
    h_chan_rms = H_all.float().pow(2).mean(dim=(0, 1)).sqrt().clamp_min(1e-3)
    print(
        f"[Phase E Calib] Hidden channel RMS range: min={h_chan_rms.min().item():.3f}, "
        f"mean={h_chan_rms.mean().item():.3f}, max={h_chan_rms.max().item():.3f}"
    )

    # Train in FP32 master weights with BF16 autocast for numerical precision + TensorCore speed!
    draft_head = EagleLiteDraftHead(
        d_model=d_model,
        d_latent=d_latent,
        d_ffn=d_ffn,
        shortlist_size=eff_shortlist,
        max_steps=8,
    ).to(device=dev, dtype=torch.float32)
    draft_head.shortlist_ids = shortlist_ids_gpu
    draft_head.h_chan_scale.copy_(h_chan_rms)
    draft_head.train()

    optimizer = torch.optim.AdamW(draft_head.parameters(), lr=lr, weight_decay=1e-2)
    num_win = int(H_all.shape[0])
    batch_size = min(16, max(4, num_win))
    steps_per_epoch = max(1, (num_win + batch_size - 1) // batch_size)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.05)

    horizon_weights = [1.0, 0.80, 0.60, 0.45]
    K_roll = min(rollout_horizon, len(horizon_weights))
    T_usable = win_len - K_roll - 1

    for ep in range(epochs):
        perm = torch.randperm(num_win, device=dev)
        # Scheduled sampling probability: smoothly ramp self-fed token/state usage from 0.0 -> 0.55
        ss_prob = min(0.55, 0.65 * (float(ep) / max(1.0, float(epochs - 1))))
        ep_loss = 0.0
        ep_top1_acc = 0.0
        n_batches = 0

        for b_start in range(0, num_win, batch_size):
            idx = perm[b_start : b_start + batch_size]
            # Anchor positions `t in 0 .. T_usable - 1`:
            # At anchor `t`, verified hidden state is `H[:, t]`, verified bonus token is `IDs[:, t + 1]`,
            # and we predict hidden states `H[:, t + 1 + s]` & tokens `IDs[:, t + 2 + s]` for `s = 0 .. K_roll - 1`!
            h_anchor = H_all[idx, :T_usable, :]    # [B, T, D]
            c_anchor = C_all[idx, :T_usable, :]    # [B, T, D]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                u_cur, c_lat = draft_head.init_state(h_anchor.float(), c_anchor.float())
                h_cur = h_anchor.float()
                cur_tok_ids = IDs_all[idx, 1 : 1 + T_usable]  # verified bonus token y_{t+1}

                total_loss = torch.zeros((), device=dev, dtype=torch.float32)
                step0_acc_val = 0.0

                for s in range(K_roll):
                    tok_emb = _fast_embed(cur_tok_ids).float()
                    h_pred, u_cur = draft_head.step_transition(h_cur, u_cur, tok_emb, c_lat, step_idx=s)

                    # Target hidden state at `t + 1 + s` (whose lm_head produces target token at `t + 2 + s`!)
                    h_tgt = H_all[idx, 1 + s : 1 + s + T_usable, :].float()
                    inv_ch = 1.0 / h_chan_rms
                    cos_loss = 1.0 - F.cosine_similarity(h_pred * inv_ch, h_tgt * inv_ch, dim=-1).mean()
                    l1_loss = F.smooth_l1_loss(h_pred * inv_ch, h_tgt * inv_ch, beta=0.5)

                    # Shortlist logits: `[B, T, K_short]`
                    logits_short = F.linear(
                        h_pred.to(torch.bfloat16),
                        short_w_gpu,
                        draft_head.logit_bias.to(torch.bfloat16),
                    ).float()

                    # Exact Teacher Top-32 KL Divergence restricted to shortlist + GT Hard CE!
                    t_vals = TopV_all[idx, 1 + s : 1 + s + T_usable, :].float()          # [B, T, 32]
                    t_inds = TopI_all[idx, 1 + s : 1 + s + T_usable, :].long()           # [B, T, 32]
                    t_short = vocab_to_short[t_inds]                                      # [B, T, 32], -1 if OOV
                    valid_m = t_short >= 0
                    safe_short = t_short.clamp_min(0)

                    # Gather student logits at teacher's top-32 shortlist indices
                    log_p_short = F.log_softmax(logits_short, dim=-1)
                    stu_log_p = torch.gather(log_p_short, dim=-1, index=safe_short)
                    masked_t_vals = t_vals.masked_fill(~valid_m, -1e4)
                    tea_probs = F.softmax(masked_t_vals, dim=-1)
                    has_any = valid_m.any(dim=-1)
                    kl_per_pos = -(tea_probs * stu_log_p.masked_fill(~valid_m, 0.0)).sum(dim=-1)
                    kl_loss = kl_per_pos[has_any].mean() if has_any.any() else torch.zeros((), device=dev)

                    # Hard token CE on ground-truth token `IDs[:, 2 + s]` when inside shortlist
                    gt_vocab = IDs_all[idx, 2 + s : 2 + s + T_usable]
                    gt_short = vocab_to_short[gt_vocab]
                    in_sl = gt_short >= 0
                    ce_loss = (
                        F.nll_loss(log_p_short[in_sl], gt_short[in_sl])
                        if in_sl.any()
                        else torch.zeros((), device=dev)
                    )

                    if s == 0 and has_any.any():
                        pred_top = logits_short.argmax(dim=-1)
                        tea_top = safe_short[..., 0]
                        valid_top1 = valid_m[..., 0]
                        if valid_top1.any():
                            step0_acc_val = (pred_top[valid_top1] == tea_top[valid_top1]).float().mean().item()

                    step_loss = 0.55 * kl_loss + 0.25 * ce_loss + 2.5 * cos_loss + 1.0 * l1_loss
                    total_loss = total_loss + horizon_weights[s] * step_loss

                    # Scheduled sampling for next rollout step `s + 1`:
                    if s + 1 < K_roll:
                        with torch.no_grad():
                            pred_vocab = shortlist_ids_gpu[logits_short.argmax(dim=-1)]
                            gt_next = IDs_all[idx, 2 + s : 2 + s + T_usable]
                            if ss_prob > 0.0:
                                use_pred = torch.rand_like(gt_next.float()) < ss_prob
                                cur_tok_ids = torch.where(use_pred, pred_vocab, gt_next)
                            else:
                                cur_tok_ids = gt_next
                    h_cur = h_pred

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(draft_head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ep_loss += float(total_loss.item())
            ep_top1_acc += step0_acc_val
            n_batches += 1

        if (ep + 1) in (1, epochs // 2, epochs):
            print(
                f"  [Epoch {ep+1:02d}/{epochs:02d}] loss={ep_loss/max(1,n_batches):.4f} | "
                f"step0_teacher_top1_agree={100.0*ep_top1_acc/max(1,n_batches):.1f}%"
            )

    draft_head.eval()
    draft_head.to(dtype=torch.bfloat16)
    lut_tmp = SharedQuantLUTPool(dev)
    draft_head.attach_int8_shortlist_head(short_w_gpu, shortlist_ids_gpu, lut_pool=lut_tmp)
    saved_path = draft_head.save_pretrained_nvm(nvm_dir)
    vram_mb = draft_head.vram_bytes() / float(1024 ** 2)

    del H_all, C_all, IDs_all, TopV_all, TopI_all, emb_w_gpu, short_w_gpu, draft_head, lut_tmp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"[Phase E Calib] Saved EagleLiteDraftHead ({vram_mb:.1f} MB VRAM footprint) -> {saved_path} "
        f"in {time.perf_counter() - t_start:.1f}s"
    )
    return saved_path
