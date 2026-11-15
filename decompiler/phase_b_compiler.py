"""Phase B Decompiler: AWQ Input+Mid Bit-Tiered Core + Co-Activated INT4 Pages + Validated Closed-Loop Bridge."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.calibration import TokenizedSplit
from decompiler.full_mlp_svd import capture_single_layer_mlp_input
from decompiler.hooks import get_target_mlp, resolve_decoder_layers
from runtime.gpu_page_tlb import (
    BitTieredTLBMLP,
    GPUPageTLBCache,
    SharedBatchTLBController,
    SharedQuantLUTPool,
    UniformAWQ4BitMLP,
    choose_base8_group_size,
    quantize_and_pack_3bit_base8,
    quantize_and_pack_int4_pages_pinned,
    quantize_and_pack_uint4_group,
    quantize_int8_rowwise,
)


@dataclass
class PhaseBLayerStats:
    """Per-layer compilation and error diagnostics for Phase B."""

    layer_idx: int
    d_model: int
    d_ffn: int
    n_hot: int
    n_cold: int
    page_size: int
    tlb_slots: int
    core_only_rel_err: float
    tlb_paged_rel_err: float
    core_vram_mb: float
    tlb_vram_mb: float
    active_vram_mb: float


@torch.no_grad()
def compile_phase_b_layer(
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    act_fn: nn.Module,
    X_orig_logic: torch.Tensor,
    X_orig_fact: torch.Tensor,
    H_orig_logic: Optional[torch.Tensor],
    H_orig_fact: Optional[torch.Tensor],
    lut_pool: SharedQuantLUTPool,
    X_actual_logic: Optional[torch.Tensor] = None,
    X_actual_fact: Optional[torch.Tensor] = None,
    hot_fraction: float = 0.11,
    num_pages: int = 64,
    tlb_slots: int = 10,
    decode_req_pages: int = 4,
    bridge_rank: int = 64,
    closed_loop_ridge: float = 0.08,
    compute_device: str = "cuda:0",
    target_device: str = "cuda:0",
    target_dtype: torch.dtype = torch.bfloat16,
    layer_idx: int = -1,
    shared_dma_stream: Optional[torch.cuda.Stream] = None,
    batch_controller: Optional[SharedBatchTLBController] = None,
    controller_slot_idx: int = 0,
) -> Tuple[BitTieredTLBMLP, PhaseBLayerStats]:
    """Compiles a single dense GeGLU MLP into a Phase B `BitTieredTLBMLP` with exact token-aligned calibration."""
    dev = torch.device(compute_device)
    t_dev = torch.device(target_device)

    Wg = W_gate.to(device=dev, dtype=torch.float32)
    Wu = W_up.to(device=dev, dtype=torch.float32)
    Wd = W_down.to(device=dev, dtype=torch.float32)
    d_ffn, d_model = Wg.shape

    # Align token counts PER SPLIT BEFORE concatenating so X_orig[i] and X_act[i] are 100% token-identical!
    if X_actual_logic is not None and X_actual_fact is not None:
        n_l = min(int(X_orig_logic.shape[0]), int(X_actual_logic.shape[0]))
        n_f = min(int(X_orig_fact.shape[0]), int(X_actual_fact.shape[0]))
        X_orig = torch.cat([X_orig_logic[:n_l], X_orig_fact[:n_f]], dim=0).to(device=dev, dtype=torch.float32)
        X_act = torch.cat([X_actual_logic[:n_l], X_actual_fact[:n_f]], dim=0).to(device=dev, dtype=torch.float32)
    else:
        X_orig = torch.cat([X_orig_logic, X_orig_fact], dim=0).to(device=dev, dtype=torch.float32)
        X_act = X_orig

    # Compute exact teacher GeGLU states H_orig and outputs Y_teacher directly from X_orig across ALL tokens!
    H_orig_chunks: List[torch.Tensor] = []
    Y_teacher_chunks: List[torch.Tensor] = []
    for s in range(0, X_orig.shape[0], 2048):
        e = min(X_orig.shape[0], s + 2048)
        xo = X_orig[s:e]
        ho = act_fn(xo @ Wg.T) * (xo @ Wu.T)
        H_orig_chunks.append(ho)
        Y_teacher_chunks.append(ho @ Wd.T)
    H_orig = torch.cat(H_orig_chunks, dim=0)
    Y_teacher = torch.cat(Y_teacher_chunks, dim=0)
    del H_orig_chunks, Y_teacher_chunks

    # 1. Activation-Aware (AWQ) Input & Intermediate GeGLU Channel Scales (`s_in` & `s_mid`)
    s_in = X_orig.abs().mean(dim=0).clamp_min(1e-5).pow(0.5)   # [d_model]
    s_mid = H_orig.abs().mean(dim=0).clamp_min(1e-5).pow(0.5)  # [d_ffn]

    # 2. Post-GeGLU Output Energy Neuron Partitioning (`Hot Core` vs `Cold Specialist Pages`)
    col_w_norm = torch.linalg.norm(Wd, dim=0)
    imp = (
        H_orig.square().mean(dim=0) + 0.15 * H_orig.abs().amax(dim=0).square()
    ) * col_w_norm.square()
    _, order = torch.sort(imp, descending=True)

    eff_pages = min(int(num_pages), max(1, d_ffn // 10))
    target_cold = int(round(d_ffn * (1.0 - hot_fraction)))
    # Make page_size a multiple of 5 so `n_cold = eff_pages * page_size` packs cleanly into 5-tuples!
    page_size = max(5, (target_cold // (eff_pages * 5)) * 5)
    if page_size * eff_pages >= d_ffn:
        page_size = max(5, ((d_ffn - max(16, d_ffn // 16)) // (eff_pages * 5)) * 5)
    if page_size * eff_pages >= d_ffn:
        page_size = max(1, (d_ffn - max(4, d_ffn // 16)) // eff_pages)
    n_cold = eff_pages * page_size
    n_hot = d_ffn - n_cold

    hot_cols = order[:n_hot]
    cold_raw = order[n_hot:]

    # 3. Unbiased Spectral Co-Activation Clustering of Cold Specialist Neurons
    sample_n = min(1536, H_orig.shape[0])
    hc_abs = H_orig[:sample_n, cold_raw].abs()
    hc_centered = hc_abs - hc_abs.mean(dim=0, keepdim=True)
    hc_norm = F.normalize(hc_centered, p=2.0, dim=0, eps=1e-8)
    q_proj = min(4, sample_n, n_cold)
    if q_proj >= 2:
        _, _, v_spec = torch.svd_lowrank(hc_norm, q=q_proj, niter=2)
        angles = torch.atan2(v_spec[:, 1], v_spec[:, 0])
        _, cluster_perm = torch.sort(angles, descending=False)
        cold_cols = cold_raw[cluster_perm]
    else:
        cold_cols = cold_raw

    s_mid_hot = s_mid[hot_cols]
    s_mid_cold = s_mid[cold_cols]
    del H_orig, hc_abs, hc_centered, hc_norm

    # 4. Tier 1: Fused Hot Generalist Neurons in AWQ Row-Wise INT8
    W_gu_hot = torch.cat([Wg[hot_cols], Wu[hot_cols]], dim=0).contiguous()
    q_guh_i8, sc_guh = quantize_int8_rowwise(
        W_gu_hot, s_in=s_in, target_device=target_device, scale_dtype=target_dtype
    )
    del W_gu_hot
    q_dh_i8, sc_dh = quantize_int8_rowwise(
        Wd[:, hot_cols].contiguous(),
        s_in=s_mid_hot,
        target_device=target_device,
        scale_dtype=target_dtype,
    )

    # 5. Tier 0: Fused Cold Specialist Neurons in Optimal-Clipped 3-Bit Base8 (`s_in` on gu, `s_mid_cold` on down!)
    gu_gsize = choose_base8_group_size(d_model)
    down_gsize = choose_base8_group_size(n_cold)

    W_gu_cold = torch.cat([Wg[cold_cols], Wu[cold_cols]], dim=0).contiguous()
    p_guc, mn_guc, sc_guc, deq_guc_fp32, _ = quantize_and_pack_3bit_base8(
        W_gu_cold,
        s_in=s_in,
        group_size=gu_gsize,
        target_device=target_device,
        scale_dtype=target_dtype,
    )
    del W_gu_cold
    deq_gc_fp32 = deq_guc_fp32[:n_cold]
    deq_uc_fp32 = deq_guc_fp32[n_cold:]

    p_dc, mn_dc, sc_dc, deq_dc_fp32, _ = quantize_and_pack_3bit_base8(
        Wd[:, cold_cols].contiguous(),
        s_in=s_mid_cold,
        group_size=down_gsize,
        target_device=target_device,
        scale_dtype=target_dtype,
    )
    # `deq_dc_fp32` has shape `[d_model, n_cold]` in unscaled space; transpose to `[n_cold, d_model]` for page residuals
    deq_dc_T_fp32 = deq_dc_fp32.T.contiguous()
    del deq_dc_fp32

    # 6. Tier 2: Pack Cold Precision-Upgrade Residuals `dW = W_exact - W_tier0` into Pinned Host INT4 Pages
    dw_gate = (Wg[cold_cols] - deq_gc_fp32).view(eff_pages, page_size, d_model)
    dw_up = (Wu[cold_cols] - deq_uc_fp32).view(eff_pages, page_size, d_model)
    dw_down_T = (Wd[:, cold_cols].T - deq_dc_T_fp32).view(eff_pages, page_size, d_model)
    del deq_guc_fp32, deq_gc_fp32, deq_uc_fp32

    col_err_weight = (
        torch.linalg.norm(Wd[:, cold_cols].T - deq_dc_T_fp32, dim=-1)
        + 0.5 * col_w_norm[cold_cols]
    ).to(device=t_dev, dtype=target_dtype)
    del deq_dc_T_fp32, Wg, Wu, Wd

    dw_pages = torch.stack([dw_gate, dw_up, dw_down_T], dim=1).contiguous()
    del dw_gate, dw_up, dw_down_T

    q4_pages_pinned, page_scales_gpu = quantize_and_pack_int4_pages_pinned(
        dw_pages,
        s_in_gu=s_in,
        target_device=target_device,
        scale_dtype=target_dtype,
    )
    del dw_pages

    tlb_cache = GPUPageTLBCache(
        num_pages=eff_pages,
        tlb_slots=min(tlb_slots, eff_pages),
        page_size=page_size,
        d_model=d_model,
        q4_pages_cpu_pinned=q4_pages_pinned,
        page_scales_gpu=page_scales_gpu,
        device=t_dev,
        scale_dtype=target_dtype,
        dma_stream=shared_dma_stream,
    )
    if batch_controller is not None:
        batch_controller.register_layer_cache(controller_slot_idx, tlb_cache)

    eff_bridge_rank = min(int(bridge_rank), d_model)
    zero_U = torch.zeros((d_model, eff_bridge_rank), device=t_dev, dtype=target_dtype)
    zero_V = torch.zeros((eff_bridge_rank, d_model), device=t_dev, dtype=target_dtype)

    mlp = BitTieredTLBMLP(
        q_gu_hot_i8=q_guh_i8,
        sc_gu_hot=sc_guh,
        q_down_hot_i8=q_dh_i8,
        sc_down_hot=sc_dh,
        s_mid_hot=s_mid_hot.to(device=t_dev, dtype=target_dtype),
        p_gu_cold_i16=p_guc,
        mn_gu_cold=mn_guc,
        sc_gu_cold=sc_guc,
        p_down_cold_i16=p_dc,
        mn_down_cold=mn_dc,
        sc_down_cold=sc_dc,
        s_mid_cold=s_mid_cold.to(device=t_dev, dtype=target_dtype),
        gu_group_size=gu_gsize,
        down_group_size=down_gsize,
        s_in=s_in.to(device=t_dev, dtype=target_dtype),
        col_err_weight=col_err_weight,
        U_bridge=zero_U,
        V_bridge=zero_V,
        tlb_cache=tlb_cache,
        lut_pool=lut_pool,
        act_fn=act_fn,
        num_pages=eff_pages,
        decode_req_pages=decode_req_pages,
        mode="vm_paged",
        tlb_strategy="sync_tlb",
        batch_controller=batch_controller,
        controller_slot_idx=controller_slot_idx,
    )

    # 7. Closed-Loop Drift & Residual Bridge Calibration with Held-Out 80/20 Validation Split
    x_act_bf = X_act.to(device=t_dev, dtype=target_dtype)
    y_paged_chunks: List[torch.Tensor] = []
    for s in range(0, x_act_bf.shape[0], 256):
        e = min(x_act_bf.shape[0], s + 256)
        mlp.tlb_cache.clear_cache()
        y_paged_chunks.append(mlp(x_act_bf[s:e]).float())
    Y_paged_pre = torch.cat(y_paged_chunks, dim=0)

    R_cal = Y_teacher - Y_paged_pre
    y_norm = torch.linalg.norm(Y_teacher).clamp_min(1e-8)
    pre_err = float((torch.linalg.norm(R_cal) / y_norm).item())

    n_tot = X_act.shape[0]
    if eff_bridge_rank > 0 and n_tot >= max(256, 2 * eff_bridge_rank):
        idx_all = torch.arange(n_tot, device=dev)
        val_mask = (idx_all % 5) == 4
        tr_mask = ~val_mask

        X_tr = X_act[tr_mask]
        R_tr = R_cal[tr_mask]
        X_val_bf = x_act_bf[val_mask]
        Y_val_target = Y_teacher[val_mask]
        Y_val_pre = Y_paged_pre[val_mask]
        val_norm = torch.linalg.norm(Y_val_target).clamp_min(1e-8)
        best_val_err = float((torch.linalg.norm(Y_val_target - Y_val_pre) / val_norm).item())

        XtX = X_tr.T @ X_tr
        ridge_val = max(1e-4, float(closed_loop_ridge) * float(torch.trace(XtX).item() / max(1, d_model)))
        XtX_reg = XtX + ridge_val * torch.eye(d_model, device=dev, dtype=torch.float32)
        W_corr = torch.linalg.solve(XtX_reg, X_tr.T @ R_tr)
        U_svd, S_svd, Vh_svd = torch.linalg.svd(W_corr, full_matrices=False)
        U_base = (U_svd[:, :eff_bridge_rank] * S_svd[:eff_bridge_rank].unsqueeze(0)).to(
            device=t_dev, dtype=target_dtype
        )
        V_base = Vh_svd[:eff_bridge_rank, :].to(device=t_dev, dtype=target_dtype)
        delta_val_full = ((X_val_bf @ U_base) @ V_base).float()

        best_gamma = 0.0
        for gamma in (0.25, 0.50, 0.75, 0.90, 1.00):
            v_err = float((torch.linalg.norm(Y_val_target - (Y_val_pre + gamma * delta_val_full)) / val_norm).item())
            if v_err < best_val_err - 1e-4:
                best_val_err = v_err
                best_gamma = gamma

        if best_gamma > 0.0:
            mlp.U_bridge.copy_((U_base.float() * best_gamma).to(target_dtype))
            mlp.V_bridge.copy_(V_base)
            Y_post_all = Y_paged_pre + best_gamma * ((x_act_bf @ U_base) @ V_base).float()
            tlb_rel_err = float((torch.linalg.norm(Y_teacher - Y_post_all) / y_norm).item())
        else:
            tlb_rel_err = pre_err
    else:
        tlb_rel_err = pre_err

    # Measure core_only diagnostic relative error
    mlp.set_mode("core_only")
    y_core_chunks: List[torch.Tensor] = []
    for s in range(0, x_act_bf.shape[0], 512):
        e = min(x_act_bf.shape[0], s + 512)
        y_core_chunks.append(mlp(x_act_bf[s:e]).float())
    Y_core = torch.cat(y_core_chunks, dim=0)
    core_rel_err = float((torch.linalg.norm(Y_teacher - Y_core) / y_norm).item())

    mlp.set_mode("vm_paged")
    mlp.reset_telemetry(clear_tlb=True)
    v_stats = mlp.vram_footprint_stats()

    stats = PhaseBLayerStats(
        layer_idx=layer_idx,
        d_model=d_model,
        d_ffn=d_ffn,
        n_hot=n_hot,
        n_cold=n_cold,
        page_size=page_size,
        tlb_slots=tlb_cache.tlb_slots,
        core_only_rel_err=core_rel_err,
        tlb_paged_rel_err=tlb_rel_err,
        core_vram_mb=v_stats["core_total_mb"],
        tlb_vram_mb=v_stats["tlb_cache_mb"],
        active_vram_mb=v_stats["active_total_mb"],
    )

    del X_orig, X_act, Y_teacher
    return mlp, stats


@torch.no_grad()
def decompile_phase_b_stack(
    model: nn.Module,
    snapshots: Dict[int, Dict[str, torch.Tensor]],
    teacher_x_logic: Dict[int, torch.Tensor],
    teacher_x_fact: Dict[int, torch.Tensor],
    teacher_h_logic: Dict[int, torch.Tensor],
    teacher_h_fact: Dict[int, torch.Tensor],
    logic_calib_split: TokenizedSplit,
    fact_calib_split: TokenizedSplit,
    layer_indices: List[int],
    hot_fraction: float = 0.11,
    num_pages: int = 64,
    tlb_slots: int = 10,
    decode_req_pages: int = 4,
    bridge_rank: int = 64,
    closed_loop_ridge: float = 0.08,
    batch_size: int = 4,
    max_calib_tokens: int = 8192,
    refresh_every: int = 2,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Dict[int, BitTieredTLBMLP], SharedQuantLUTPool, List[PhaseBLayerStats]]:
    """Sequentially compiles all interior MLP layers into Fused Phase B `BitTieredTLBMLP` modules."""
    decoder_layers = resolve_decoder_layers(model)
    t_dev = torch.device(device)
    lut_pool = SharedQuantLUTPool(device=t_dev, dtype=dtype)
    shared_dma_stream = (
        torch.cuda.Stream(device=t_dev) if (torch.cuda.is_available() and t_dev.type == "cuda") else None
    )

    sorted_layers = sorted(layer_indices)
    batch_controller = SharedBatchTLBController(
        num_layers=len(sorted_layers),
        decode_req_pages=decode_req_pages,
        device=t_dev,
        dma_stream=shared_dma_stream,
    )

    phase_b_mlps: Dict[int, BitTieredTLBMLP] = {}
    stats_list: List[PhaseBLayerStats] = []

    for step_i, idx in enumerate(sorted_layers):
        w_dict = snapshots[idx]
        orig_mlp = get_target_mlp(model, idx)
        act_fn = getattr(orig_mlp, "act_fn", nn.GELU(approximate="tanh"))

        if step_i == 0 or refresh_every <= 0 or (step_i % refresh_every != 0):
            x_act_l = teacher_x_logic[idx]
            x_act_f = teacher_x_fact[idx]
        else:
            x_act_l = capture_single_layer_mlp_input(
                model=model,
                dataset_split=logic_calib_split,
                layer_idx=idx,
                batch_size=batch_size,
                device=device,
                max_tokens=max_calib_tokens,
            )
            x_act_f = capture_single_layer_mlp_input(
                model=model,
                dataset_split=fact_calib_split,
                layer_idx=idx,
                batch_size=batch_size,
                device=device,
                max_tokens=max_calib_tokens,
            )

        b_mlp, st = compile_phase_b_layer(
            W_gate=w_dict["gate_proj"],
            W_up=w_dict["up_proj"],
            W_down=w_dict["down_proj"],
            act_fn=act_fn,
            X_orig_logic=teacher_x_logic[idx],
            X_orig_fact=teacher_x_fact[idx],
            H_orig_logic=teacher_h_logic.get(idx),
            H_orig_fact=teacher_h_fact.get(idx),
            lut_pool=lut_pool,
            X_actual_logic=x_act_l,
            X_actual_fact=x_act_f,
            hot_fraction=hot_fraction,
            num_pages=num_pages,
            tlb_slots=tlb_slots,
            decode_req_pages=decode_req_pages,
            bridge_rank=bridge_rank,
            closed_loop_ridge=closed_loop_ridge,
            compute_device=device,
            target_device=device,
            target_dtype=dtype,
            layer_idx=idx,
            shared_dma_stream=shared_dma_stream,
            batch_controller=batch_controller,
            controller_slot_idx=step_i,
        )

        # Free GPU storage on orig_mlp without duplicating the CPU snapshot already in `snapshots[idx]`
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(orig_mlp, proj_name, None)
            if proj is not None and hasattr(proj, "weight"):
                proj.weight = nn.Parameter(torch.empty(0, dtype=dtype), requires_grad=False)
        decoder_layers[idx].mlp = b_mlp
        phase_b_mlps[idx] = b_mlp
        stats_list.append(st)

        del x_act_l, x_act_f, orig_mlp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Pre-size shared Phase C workspace to the widest layer and bind NVRTC launchers once upfront
    if phase_b_mlps and torch.cuda.is_available() and t_dev.type == "cuda" and dtype == torch.bfloat16:
        from runtime.phase_c_kernels import get_shared_phase_c_workspace

        ws = get_shared_phase_c_workspace(t_dev)
        ws.ensure_capacity(
            d_model=max(m.d_model for m in phase_b_mlps.values()),
            n_hot=max(m.n_hot for m in phase_b_mlps.values()),
            n_cold=max(m.n_cold for m in phase_b_mlps.values()),
            c_act=max(m.tlb_cache.tlb_slots * m.page_size for m in phase_b_mlps.values()),
            bridge_rank=max((int(m.U_bridge.shape[1]) if m.U_bridge.ndim == 2 else 0) for m in phase_b_mlps.values()),
            num_pages=max(m.num_pages for m in phase_b_mlps.values()),
        )
        for m in phase_b_mlps.values():
            m.set_phase_c_enabled(True)

    return phase_b_mlps, lut_pool, stats_list


@torch.no_grad()
def build_uniform_awq_4bit_stack(
    model: nn.Module,
    snapshots: Dict[int, Dict[str, torch.Tensor]],
    teacher_x_logic: Dict[int, torch.Tensor],
    teacher_x_fact: Dict[int, torch.Tensor],
    layer_indices: List[int],
    group_size: int = 64,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Dict[int, UniformAWQ4BitMLP], SharedQuantLUTPool]:
    """Builds a hardware-honest Fused Uniform 4-Bit AWQ (`g=64`) reference stack."""
    decoder_layers = resolve_decoder_layers(model)
    lut_pool = SharedQuantLUTPool(device=torch.device(device), dtype=dtype)
    q4_mlps: Dict[int, UniformAWQ4BitMLP] = {}
    dev = torch.device(device)

    for idx in sorted(layer_indices):
        w_dict = snapshots[idx]
        orig_mlp = get_target_mlp(model, idx)
        act_fn = getattr(orig_mlp, "act_fn", nn.GELU(approximate="tanh"))

        Wg = w_dict["gate_proj"].to(dev, torch.float32)
        Wu = w_dict["up_proj"].to(dev, torch.float32)
        Wd = w_dict["down_proj"].to(dev, torch.float32)
        _, d_model = Wg.shape

        X_orig = torch.cat([teacher_x_logic[idx], teacher_x_fact[idx]], dim=0).to(dev, torch.float32)
        s_in = X_orig.abs().mean(dim=0).clamp_min(1e-5).pow(0.5)

        W_gu = torch.cat([Wg, Wu], dim=0).contiguous()
        p_gu, mn_gu, sc_gu, _ = quantize_and_pack_uint4_group(
            W_gu, s_in=s_in, group_size=group_size, target_device=device, scale_dtype=dtype
        )
        pd_T, mnd_T, scd_T, _ = quantize_and_pack_uint4_group(
            Wd.T.contiguous(), s_in=None, group_size=group_size, target_device=device, scale_dtype=dtype
        )

        mod = UniformAWQ4BitMLP(
            p_gu_u8=p_gu,
            mn_gu=mn_gu,
            sc_gu=sc_gu,
            p_down_T_u8=pd_T,
            mn_down_T=mnd_T,
            sc_down_T=scd_T,
            s_in=s_in.to(device=device, dtype=dtype),
            d_model=d_model,
            group_size=group_size,
            lut_pool=lut_pool,
            act_fn=act_fn,
        )
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(orig_mlp, proj_name, None)
            if proj is not None and hasattr(proj, "weight"):
                proj.weight = nn.Parameter(torch.empty(0, dtype=dtype), requires_grad=False)
        decoder_layers[idx].mlp = mod
        q4_mlps[idx] = mod

        del Wg, Wu, Wd, W_gu, X_orig
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return q4_mlps, lut_pool
