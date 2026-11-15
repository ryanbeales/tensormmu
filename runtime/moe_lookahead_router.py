"""Phase G2 Runtime & Zero-Leakage Calibration: Cross-Layer Lookahead Router Bank (`LookaheadRouterBank`).

Predicts Layer `l + 1`'s high-confidence MoE expert routes directly from Layer `l`'s pre-MLP hidden state
`h_{l, mid}`, allowing cold expert pages in `Layer_{l+1}.tlb_cache` to be prefetched asynchronously across
PCIe on a background DMA stream while the GPU executes:
  `MoE_l -> PostNorm_l -> InputNorm_{l+1} -> GDN/SelfAttn_{l+1} -> PostAttnNorm_{l+1} -> Router_{l+1} -> SharedMLP_{l+1}`.

Strict Invariants:
  1. **Zero mathematical approximation (`0.0-ulp` VM parity):** `LookaheadRouterBank` ONLY pre-warms
     `Layer_{l+1}.tlb_cache`. Layer `l + 1` always executes its exact `Router_{l+1}` Top-`K` indices and weights.
  2. **Zero eval/test benchmark leakage:** Calibration (`calibrate_and_save_lookahead_routers`) strictly uses
     held-in `Logic_Calib + Fact_Calib` sequences (`0` overlap with `Logic_Eval`, `Fact_Eval`, or `GSM8K test`).
  3. **Compact INT8 VRAM & `< 10 us/layer` GPU latency:** Quantized via `PhaseCQuantLinear(quant_mode="int8")`
     (`~10.0 MB` total across 29 pairs on `Gemma-4-26B-A4B`, `~47.1 MB` total across 47 pairs on `Qwen3-Next-80B-A3B`).
"""

from __future__ import annotations

import gc
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from runtime.gpu_page_tlb import SharedQuantLUTPool
from runtime.quant_linear_phase_c import PhaseCQuantLinear
from runtime.qwen3_next_kernels import moe_router_topk_softmax_cuda, nvm_rmsnorm_bf16_cuda


class LookaheadLayerHead(nn.Module):
    """Single-layer `h_{l, mid} -> top_k_pre(Layer_{l+1})` INT8 lookahead predictor (`~9.5 us` on RTX 4090)."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        proj: PhaseCQuantLinear,
        default_k_pre: int = 8,
        min_prob_m1: float = 0.035,
        min_prob_spec: float = 0.065,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        self.proj = proj
        self.default_k_pre = int(max(1, min(default_k_pre, 16, num_experts)))
        self.min_prob_m1 = float(min_prob_m1)
        self.min_prob_spec = float(min_prob_spec)

    def vram_bytes(self) -> int:
        total = 0
        for b in self.proj.buffers():
            total += int(b.numel() * b.element_size())
        return total

    @torch.no_grad()
    def predict_topk_indices(self, h_mid_flat: torch.Tensor) -> Optional[torch.Tensor]:
        """Returns `[T_eff, k_pre]` `int64` predicted expert IDs (`-1` where below `min_prob`)."""
        if not h_mid_flat.is_cuda:
            return None
        T = int(h_mid_flat.shape[0])
        if T <= 0:
            return None

        # On speculative steps (`T > 1`), predict lookahead for verified `tok_0` + top step-0 `draft_1` (`T_eff <= 2`)
        # so deeper speculative drafts never pollute Layer `l+1` PCIe bandwidth!
        t_eff = min(T, 2)
        x_slice = h_mid_flat[:t_eff]
        if x_slice.dtype != torch.bfloat16:
            x_slice = x_slice.to(torch.bfloat16)

        x_norm = nvm_rmsnorm_bf16_cuda(x_slice, None, eps=1e-6, w_offset=0.0, use_rsqrt=True)
        logits_bf16 = self.proj(x_norm.unsqueeze(0)).squeeze(0)  # [t_eff, E]
        min_p = self.min_prob_m1 if T == 1 else self.min_prob_spec
        k_eff = self.default_k_pre if T == 1 else max(2, (self.default_k_pre * 3) // 4)
        _, pred_idx = moe_router_topk_softmax_cuda(
            logits_bf16,
            top_k=k_eff,
            norm_topk_prob=False,
            round_bf16_before_scale=False,
            per_expert_scale_bf16=None,
            min_prob_thresh=min_p,
        )
        return pred_idx


class LookaheadRouterBank(nn.Module):
    """Collection of Cross-Layer `LookaheadLayerHead` predictors (`src_layer -> dst_layer`)."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        default_k_pre: int = 8,
        min_prob_m1: float = 0.035,
        min_prob_spec: float = 0.065,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        self.default_k_pre = int(default_k_pre)
        self.min_prob_m1 = float(min_prob_m1)
        self.min_prob_spec = float(min_prob_spec)
        self.heads = nn.ModuleDict()
        self.pair_map: Dict[int, int] = {}
        self.calibration_metrics: Dict[str, Any] = {}

    def add_head(self, src_layer: int, dst_layer: int, proj: PhaseCQuantLinear) -> None:
        self.pair_map[int(src_layer)] = int(dst_layer)
        self.heads[str(int(src_layer))] = LookaheadLayerHead(
            d_model=self.d_model,
            num_experts=self.num_experts,
            proj=proj,
            default_k_pre=self.default_k_pre,
            min_prob_m1=self.min_prob_m1,
            min_prob_spec=self.min_prob_spec,
        )

    def vram_bytes(self) -> int:
        return sum(h.vram_bytes() for h in self.heads.values())  # type: ignore[attr-defined]

    def attach_to_moe_layers(self, moe_layers: Dict[int, Any]) -> int:
        """Wires each `moe_layers[src]` to prefetch into `moe_layers[dst]` during `"lookahead_overlap"`."""
        attached = 0
        for m in moe_layers.values():
            m.clear_lookahead_target()
        for src_idx, dst_idx in sorted(self.pair_map.items()):
            key = str(src_idx)
            if src_idx in moe_layers and dst_idx in moe_layers and key in self.heads:
                moe_layers[src_idx].set_lookahead_target(self.heads[key], moe_layers[dst_idx])
                attached += 1
        return attached

    def save_pretrained_nvm(self, nvm_dir: str) -> str:
        os.makedirs(nvm_dir, exist_ok=True)
        sd: Dict[str, torch.Tensor] = {}
        pairs_meta: Dict[str, Dict[str, Any]] = {}
        for src_str, head in self.heads.items():
            src_i = int(src_str)
            dst_i = int(self.pair_map[src_i])
            qlin: PhaseCQuantLinear = head.proj  # type: ignore[assignment]
            sd[f"head_{src_i}.q_weight"] = qlin.q_weight.detach().cpu().contiguous()
            sd[f"head_{src_i}.scale"] = qlin.scale.detach().cpu().contiguous()
            if qlin.bias is not None:
                sd[f"head_{src_i}.bias"] = qlin.bias.detach().cpu().contiguous()
            pairs_meta[src_str] = {
                "dst_layer": dst_i,
                "has_bias": qlin.bias is not None,
            }

        sf_path = os.path.join(nvm_dir, "lookahead_routers.safetensors")
        meta_path = os.path.join(nvm_dir, "lookahead_routers_meta.json")
        save_file(sd, sf_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "d_model": self.d_model,
                    "num_experts": self.num_experts,
                    "default_k_pre": self.default_k_pre,
                    "min_prob_m1": self.min_prob_m1,
                    "min_prob_spec": self.min_prob_spec,
                    "pairs": pairs_meta,
                    "calibration_metrics": self.calibration_metrics,
                },
                f,
                indent=2,
            )
        return sf_path

    @classmethod
    def has_saved_nvm(cls, nvm_dir: str) -> bool:
        return os.path.isfile(os.path.join(nvm_dir, "lookahead_routers.safetensors")) and os.path.isfile(
            os.path.join(nvm_dir, "lookahead_routers_meta.json")
        )

    @classmethod
    def load_pretrained_nvm(
        cls,
        nvm_dir: str,
        device: str = "cuda:0",
        lut_pool: Optional[SharedQuantLUTPool] = None,
        k_pre_override: Optional[int] = None,
        min_prob_m1_override: Optional[float] = None,
    ) -> "LookaheadRouterBank":
        meta_path = os.path.join(nvm_dir, "lookahead_routers_meta.json")
        sf_path = os.path.join(nvm_dir, "lookahead_routers.safetensors")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        dev = torch.device(device)
        raw_sd = load_file(sf_path, device="cpu")
        k_pre = int(k_pre_override) if k_pre_override is not None else int(meta["default_k_pre"])
        min_p1 = float(min_prob_m1_override) if min_prob_m1_override is not None else float(meta["min_prob_m1"])

        bank = cls(
            d_model=int(meta["d_model"]),
            num_experts=int(meta["num_experts"]),
            default_k_pre=k_pre,
            min_prob_m1=min_p1,
            min_prob_spec=float(meta.get("min_prob_spec", max(0.055, min_p1 * 1.6))),
        )
        bank.calibration_metrics = meta.get("calibration_metrics", {})

        for src_str, p_info in meta["pairs"].items():
            src_i = int(src_str)
            dst_i = int(p_info["dst_layer"])
            qlin = PhaseCQuantLinear(
                in_features=bank.d_model,
                out_features=bank.num_experts,
                quant_mode="int8",
                q_weight=raw_sd[f"head_{src_i}.q_weight"].to(device=dev, non_blocking=True),
                scale=raw_sd[f"head_{src_i}.scale"].to(device=dev, non_blocking=True),
                wmin=None,
                bias=(
                    raw_sd[f"head_{src_i}.bias"].to(device=dev, non_blocking=True)
                    if p_info.get("has_bias", False) and f"head_{src_i}.bias" in raw_sd
                    else None
                ),
                lut_pool=lut_pool,
            )
            bank.add_head(src_i, dst_i, qlin)

        bank.eval()
        return bank


def _resolve_layer_router_module(decoder_layer: nn.Module) -> Tuple[Optional[nn.Module], str]:
    """Returns `(router_module, router_kind)` where `router_kind` is `'gemma4'` or `'qwen3'`."""
    if getattr(decoder_layer, "enable_moe_block", False) and hasattr(decoder_layer, "router"):
        return decoder_layer.router, "gemma4"
    moe_block = getattr(decoder_layer, "mlp", None)
    if moe_block is not None and hasattr(moe_block, "gate"):
        return moe_block.gate, "qwen3"
    return None, "none"


def calibrate_and_save_lookahead_routers(
    nvm_dir: str,
    calib_token_sequences: Sequence[torch.Tensor],
    k_pre: Optional[int] = None,
    ridge_lambda: float = 10.0,
    kl_steps: int = 60,
    lr: float = 3.0e-3,
    batch_size: int = 4,
    tlb_slots_override: int = 96,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Calibrates Cross-Layer `LookaheadRouterBank` strictly on held-in calibration sequences (`0` eval/test leakage).

    1. Runs 1 forward sweep over `calib_token_sequences` (`Logic_Calib + Fact_Calib`), capturing exact router
       inputs `X_l` and next-layer router probability targets `P_{l+1}` on CPU (`~20-35s`).
    2. Unloads the `.nvm` model to free VRAM, then fits each `(l -> l+1)` head via closed-form Ridge + 60-step
       KL distillation (`~0.12s/layer` on RTX 4090), evaluates held-out calibration-split Precision & Recall,
       quantizes to `INT8` `PhaseCQuantLinear`, and writes `lookahead_routers.safetensors` into `nvm_dir`.
    """
    from decompiler.nvm_serializer import load_nvm_cartridge, resolve_decoder_layers

    t0 = time.perf_counter()
    dev = torch.device(device)
    seqs = [s.long().cpu() for s in calib_token_sequences if s.numel() >= 32]
    assert len(seqs) >= 8, "Expected at least 8 held-in calibration sequences for LookaheadRouter calibration."
    min_len = min(int(s.numel()) for s in seqs)
    calib_ids = torch.stack([s[:min_len] for s in seqs], dim=0)  # [N_seq, S]

    model, _, moe_layers, lut_pool, _, _ = load_nvm_cartridge(
        nvm_dir=nvm_dir,
        device=device,
        dtype=torch.bfloat16,
        mode="vm_paged",
        tlb_strategy="exact_demand",
        page_storage="nvme_mmap",
        tlb_slots_override=tlb_slots_override,
        use_phase_c_kernels=True,
        load_tokenizer=False,
    )
    decoder_layers = resolve_decoder_layers(model)
    moe_indices = sorted(moe_layers.keys())
    assert len(moe_indices) >= 2, "Need at least 2 Sparse MoE layers for cross-layer lookahead."

    sample_moe = moe_layers[moe_indices[0]]
    d_model = int(sample_moe.d_model)
    num_experts = int(sample_moe.num_experts)

    # Determine model's exact `top_k` per token from router
    r0, r_kind0 = _resolve_layer_router_module(decoder_layers[moe_indices[0]])
    assert r0 is not None
    exact_top_k = int(getattr(r0, "top_k", None) or getattr(getattr(r0, "config", None), "top_k_experts", 8))
    eff_k_pre = int(k_pre) if k_pre is not None else max(3, min(4, exact_top_k // 2))
    min_prob_m1 = 0.080 if num_experts >= 256 else 0.070
    min_prob_spec = 0.115 if num_experts >= 256 else 0.100

    pairs: List[Tuple[int, int]] = [
        (moe_indices[i], moe_indices[i + 1]) for i in range(len(moe_indices) - 1)
    ]
    src_set = {p[0] for p in pairs}
    dst_set = {p[1] for p in pairs}

    cap_x_cpu: Dict[int, List[torch.Tensor]] = {l: [] for l in src_set}
    cap_p_cpu: Dict[int, List[torch.Tensor]] = {l1: [] for l1 in dst_set}
    hooks = []

    for l_idx in sorted(src_set | dst_set):
        router_mod, r_kind = _resolve_layer_router_module(decoder_layers[l_idx])
        assert router_mod is not None

        if l_idx in src_set:
            def _make_pre_hook(li: int):
                def _pre(mod: nn.Module, args: Tuple[Any, ...]) -> None:
                    x_in = args[0].detach().reshape(-1, d_model).to(dtype=torch.bfloat16, device="cpu")
                    cap_x_cpu[li].append(x_in)
                return _pre

            hooks.append(router_mod.register_forward_pre_hook(_make_pre_hook(l_idx)))

        if l_idx in dst_set:
            def _make_post_hook(li1: int, kind: str):
                def _post(mod: nn.Module, args: Tuple[Any, ...], out: Any) -> None:
                    first_out = out[0].detach().reshape(-1, num_experts).float()
                    if kind == "gemma4":
                        probs = first_out.clamp_min(1e-12)
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                    else:
                        probs = F.softmax(first_out, dim=-1)
                    cap_p_cpu[li1].append(probs.to(dtype=torch.bfloat16, device="cpu"))
                return _post

            hooks.append(router_mod.register_forward_hook(_make_post_hook(l_idx, r_kind)))

    print(
        f"[Phase G2 Calibrate] Capturing cross-layer router states across {calib_ids.shape[0]}x{calib_ids.shape[1]} "
        f"held-in calib tokens ({len(pairs)} MoE layer pairs, E={num_experts}, K={exact_top_k}, k_pre={eff_k_pre})...",
        flush=True,
    )
    with torch.no_grad():
        for s in range(0, int(calib_ids.shape[0]), batch_size):
            e = min(int(calib_ids.shape[0]), s + batch_size)
            inp = calib_ids[s:e].to(dev)
            msk = torch.ones_like(inp)
            _ = model(input_ids=inp, attention_mask=msk, use_cache=False)

    for h in hooks:
        h.remove()
    del model, moe_layers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lut_calib = SharedQuantLUTPool(device=dev, dtype=torch.bfloat16)
    bank = LookaheadRouterBank(
        d_model=d_model,
        num_experts=num_experts,
        default_k_pre=eff_k_pre,
        min_prob_m1=min_prob_m1,
        min_prob_spec=min_prob_spec,
    )

    total_toks = int(calib_ids.shape[0] * calib_ids.shape[1])
    val_toks = max(512, total_toks // 5)
    tr_toks = total_toks - val_toks

    precs_raw: List[float] = []
    precs_gated: List[float] = []
    recs_gated: List[float] = []

    reg_eye = float(ridge_lambda) * torch.eye(d_model, device=dev, dtype=torch.float32)

    for p_idx, (src_l, dst_l) in enumerate(pairs):
        X_all = torch.cat(cap_x_cpu[src_l], dim=0).to(device=dev, dtype=torch.float32)
        P_all = torch.cat(cap_p_cpu[dst_l], dim=0).to(device=dev, dtype=torch.float32).clamp_min(1e-7)
        P_all = P_all / P_all.sum(dim=-1, keepdim=True)
        # Free CPU list for `src_l` immediately
        cap_x_cpu[src_l].clear()

        Xn_all = X_all * torch.rsqrt(X_all.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        Xn_tr, Xn_va = Xn_all[:tr_toks], Xn_all[tr_toks:]
        P_tr, P_va = P_all[:tr_toks], P_all[tr_toks:]
        Y_log_tr = torch.log(P_tr)

        with torch.no_grad():
            XtX = Xn_tr.T @ Xn_tr
            W_init = torch.linalg.solve(XtX + reg_eye, Xn_tr.T @ Y_log_tr).T.contiguous()
            b_init = Y_log_tr.mean(dim=0) - (Xn_tr.mean(dim=0) @ W_init.T)

        lin = nn.Linear(d_model, num_experts, bias=True, device=dev, dtype=torch.float32)
        with torch.no_grad():
            lin.weight.copy_(W_init)
            lin.bias.copy_(b_init)

        if kl_steps > 0:
            opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=1e-2)
            for _ in range(kl_steps):
                opt.zero_grad(set_to_none=True)
                pred_logits = lin(Xn_tr)
                loss = -(P_tr * F.log_softmax(pred_logits, dim=-1)).sum(dim=-1).mean()
                loss.backward()
                opt.step()

        with torch.no_grad():
            tmp_bf = nn.Linear(d_model, num_experts, bias=True, device=dev, dtype=torch.bfloat16)
            tmp_bf.weight.copy_(lin.weight.detach().to(torch.bfloat16))
            tmp_bf.bias.copy_(lin.bias.detach().to(torch.bfloat16))
            qlin = PhaseCQuantLinear.from_linear(
                tmp_bf,
                quant_mode="int8",
                target_device=str(dev),
                scale_dtype=torch.bfloat16,
                lut_pool=lut_calib,
            )
            bank.add_head(src_l, dst_l, qlin)

            # Evaluate INT8 quantized head on held-out calibration validation slice
            _, true_topk = torch.topk(P_va, k=exact_top_k, dim=-1)
            va_logits = qlin(Xn_va.to(torch.bfloat16).unsqueeze(0)).squeeze(0)
            va_probs = F.softmax(va_logits.float(), dim=-1)
            top_p, top_i = torch.topk(va_probs, k=eff_k_pre, dim=-1)
            in_true = torch.zeros_like(top_i, dtype=torch.bool)
            for j in range(exact_top_k):
                in_true |= (top_i == true_topk[:, j : j + 1])

            raw_prec = 100.0 * float(in_true.float().mean().item())
            mask_g = top_p >= min_prob_m1
            n_g = int(mask_g.sum().item())
            gated_prec = 100.0 * float(in_true[mask_g].float().mean().item()) if n_g > 0 else raw_prec
            gated_rec = 100.0 * float((in_true & mask_g).float().sum().item()) / float(P_va.shape[0] * exact_top_k)

            precs_raw.append(raw_prec)
            precs_gated.append(gated_prec)
            recs_gated.append(gated_rec)

        del lin, tmp_bf, X_all, P_all, Xn_all
        if (p_idx + 1) % 12 == 0 or (p_idx + 1) == len(pairs):
            print(
                f"  [Lookahead Fit {p_idx+1:02d}/{len(pairs):02d} (L{src_l:02d}->L{dst_l:02d})] "
                f"INT8 Val Precision={gated_prec:.1f}% (raw@{eff_k_pre}={raw_prec:.1f}%) | "
                f"Recall@{exact_top_k}={gated_rec:.1f}%",
                flush=True,
            )

    wall_s = time.perf_counter() - t0
    vram_mb = bank.vram_bytes() / float(1024 ** 2)
    bank.calibration_metrics = {
        "num_pairs": len(pairs),
        "exact_top_k": exact_top_k,
        "k_pre": eff_k_pre,
        "min_prob_m1": min_prob_m1,
        "min_prob_spec": min_prob_spec,
        "calib_train_tokens": tr_toks,
        "calib_val_tokens": val_toks,
        "mean_val_raw_precision_pct": sum(precs_raw) / max(1, len(precs_raw)),
        "mean_val_gated_precision_pct": sum(precs_gated) / max(1, len(precs_gated)),
        "mean_val_gated_recall_pct": sum(recs_gated) / max(1, len(recs_gated)),
        "bank_vram_mb": vram_mb,
        "calibration_wall_s": wall_s,
    }
    out_path = bank.save_pretrained_nvm(nvm_dir)
    print(
        f"[Phase G2 Calibrate] Saved LookaheadRouterBank ({vram_mb:.1f} MB INT8) -> {out_path} in {wall_s:.1f}s | "
        f"Mean Val Gated Precision={bank.calibration_metrics['mean_val_gated_precision_pct']:.2f}% | "
        f"Recall@{exact_top_k}={bank.calibration_metrics['mean_val_gated_recall_pct']:.2f}%",
        flush=True,
    )
    return bank.calibration_metrics
