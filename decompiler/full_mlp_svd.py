"""Upgrade 2.1 (Phase A Hardware-Honest): Full-MLP Tri-Matrix Decompilation & Spectral Page Clustering."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.calibration import TokenizedSplit
from decompiler.contrastive_svd import _safe_cholesky, compute_covariance_matrix
from decompiler.hooks import (
    _EarlyExitAfterHook,
    get_target_down_proj,
    get_target_mlp,
    resolve_decoder_layers,
)
from runtime.paged_mlp_mmu import ClusteredPageRouter, PagedNeuralMLP


@dataclass
class FullMLPLayerStats:
    """Diagnostics recorded when decompiling a full 3-matrix MLP block at Layer L."""

    layer_idx: int
    d_ffn: int
    rank: int
    num_pages: int
    page_size: int
    gate_energy_retained: float
    up_energy_retained: float
    down_energy_retained: float
    post_correction_rel_error: float


class MultiLayerMLPInputAndActCollector:
    """Captures pre-MLP input X^(L) and intermediate GeGLU activation H^(L) across target layers."""

    def __init__(
        self,
        model: nn.Module,
        layer_indices: List[int],
        max_tokens: Optional[int] = None,
        capture_x: bool = True,
        capture_h: bool = True,
    ) -> None:
        self.model = model
        self.layer_indices = sorted(layer_indices)
        self.max_tokens = max_tokens
        self.max_layer = max(self.layer_indices)
        self.capture_x = capture_x
        self.capture_h = capture_h

        self.x_buffers: Dict[int, List[torch.Tensor]] = {idx: [] for idx in self.layer_indices}
        self.h_buffers: Dict[int, List[torch.Tensor]] = {idx: [] for idx in self.layer_indices}
        self.x_counts: Dict[int, int] = {idx: 0 for idx in self.layer_indices}
        self.h_counts: Dict[int, int] = {idx: 0 for idx in self.layer_indices}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._active_mask: Optional[torch.Tensor] = None

    def set_batch_mask(self, attention_mask: Optional[torch.Tensor]) -> None:
        self._active_mask = attention_mask

    def _flatten_masked(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 3 and self._active_mask is not None:
            mask = self._active_mask.to(device=t.device, dtype=torch.bool)
            return t[mask]
        return t.reshape(-1, t.shape[-1])

    def _make_mlp_pre_hook(self, layer_idx: int):
        def _pre_hook(module: nn.Module, args: Tuple[torch.Tensor, ...]) -> None:
            if not self.capture_x or not args:
                return
            x = args[0].detach()
            flat_x = self._flatten_masked(x)
            rem = (
                (self.max_tokens - self.x_counts[layer_idx])
                if self.max_tokens is not None
                else flat_x.shape[0]
            )
            if rem > 0:
                slice_x = flat_x[:rem].to(device="cpu", dtype=torch.bfloat16)
                self.x_buffers[layer_idx].append(slice_x)
                self.x_counts[layer_idx] += int(slice_x.shape[0])

        return _pre_hook

    def _make_down_hook(self, layer_idx: int):
        def _hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if self.capture_h and inputs:
                h = inputs[0].detach()
                flat_h = self._flatten_masked(h)
                h_cap = min(self.max_tokens, 2048) if self.max_tokens is not None else flat_h.shape[0]
                rem = h_cap - self.h_counts[layer_idx]
                if rem > 0:
                    slice_h = flat_h[:rem].to(device="cpu", dtype=torch.bfloat16)
                    self.h_buffers[layer_idx].append(slice_h)
                    self.h_counts[layer_idx] += int(slice_h.shape[0])

            if layer_idx == self.max_layer:
                raise _EarlyExitAfterHook()

        return _hook

    def __enter__(self) -> "MultiLayerMLPInputAndActCollector":
        for idx in self.layer_indices:
            mlp = get_target_mlp(self.model, idx)
            if self.capture_x:
                self._handles.append(mlp.register_forward_pre_hook(self._make_mlp_pre_hook(idx)))
            down_proj = get_target_down_proj(self.model, idx)
            self._handles.append(down_proj.register_forward_hook(self._make_down_hook(idx)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_x_activations(self) -> Dict[int, torch.Tensor]:
        return {idx: torch.cat(self.x_buffers[idx], dim=0) for idx in self.layer_indices}

    def get_h_activations(self) -> Dict[int, torch.Tensor]:
        return {idx: torch.cat(self.h_buffers[idx], dim=0) for idx in self.layer_indices}


@torch.no_grad()
def capture_teacher_full_mlp_traces(
    model: nn.Module,
    dataset_split: TokenizedSplit,
    layer_indices: List[int],
    batch_size: int = 4,
    device: str = "cuda:0",
    max_tokens: Optional[int] = 8192,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Captures unmodified teacher pre-MLP inputs X_orig^(L) and GeGLU states H_orig^(L) in one pass."""
    model.eval()
    num_seqs = dataset_split.num_sequences

    with MultiLayerMLPInputAndActCollector(
        model, layer_indices, max_tokens=max_tokens, capture_x=True, capture_h=True
    ) as collector:
        for start in range(0, num_seqs, batch_size):
            end = min(start + batch_size, num_seqs)
            batch_ids = dataset_split.input_ids[start:end].to(device)
            batch_mask = dataset_split.attention_mask[start:end].to(device)
            collector.set_batch_mask(batch_mask)

            try:
                model(input_ids=batch_ids, attention_mask=batch_mask, use_cache=False)
            except _EarlyExitAfterHook:
                pass

            if max_tokens is not None and all(
                collector.x_counts[idx] >= max_tokens for idx in layer_indices
            ):
                break

    return collector.get_x_activations(), collector.get_h_activations()


@torch.no_grad()
def capture_single_layer_mlp_input(
    model: nn.Module,
    dataset_split: TokenizedSplit,
    layer_idx: int,
    batch_size: int = 4,
    device: str = "cuda:0",
    max_tokens: int = 4096,
) -> torch.Tensor:
    """Captures drifted pre-MLP input X_actual^(L) entering Layer L from converted upstream layers."""
    model.eval()
    num_seqs = dataset_split.num_sequences
    mlp = get_target_mlp(model, layer_idx)
    buffers: List[torch.Tensor] = []
    count = 0
    active_mask: Optional[torch.Tensor] = None

    def _pre_hook(module: nn.Module, args: Tuple[torch.Tensor, ...]) -> None:
        nonlocal count
        if not args:
            return
        x = args[0].detach()
        if x.dim() == 3 and active_mask is not None:
            flat_x = x[active_mask.to(device=x.device, dtype=torch.bool)]
        else:
            flat_x = x.reshape(-1, x.shape[-1])
        rem = max_tokens - count
        if rem > 0:
            s_x = flat_x[:rem].to(device="cpu", dtype=torch.bfloat16)
            buffers.append(s_x)
            count += int(s_x.shape[0])
        raise _EarlyExitAfterHook()

    handle = mlp.register_forward_pre_hook(_pre_hook)
    try:
        for start in range(0, num_seqs, batch_size):
            end = min(start + batch_size, num_seqs)
            batch_ids = dataset_split.input_ids[start:end].to(device)
            batch_mask = dataset_split.attention_mask[start:end].to(device)
            active_mask = batch_mask
            try:
                model(input_ids=batch_ids, attention_mask=batch_mask, use_cache=False)
            except _EarlyExitAfterHook:
                pass
            if count >= max_tokens:
                break
    finally:
        handle.remove()

    return torch.cat(buffers, dim=0)


@torch.no_grad()
def cluster_neurons_into_knowledge_pages(
    W_down_fp32: torch.Tensor,
    H_orig_logic: torch.Tensor,
    H_orig_fact: torch.Tensor,
    num_pages: int = 64,
    displace_fact_cols: int = 0,
    compute_device: str = "cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Groups d_ffn neurons into P=64 contiguous Knowledge Pages via spectral co-activation clustering.

    Uses top-2 spectral directions of centered/normalized token co-activations so neurons that fire
    together on the same semantic contexts are packed into the same contiguous page of size C = d_ffn // P,
    with NO artificial boost onto Pages 0..3.
    """
    _, d_ffn = W_down_fp32.shape
    page_size = d_ffn // num_pages

    hl = H_orig_logic[:2048].to(compute_device, torch.float32)
    dl_col = (hl * hl).mean(dim=0) + 0.1 * hl.abs().amax(dim=0).square()
    del hl

    hf = H_orig_fact[:2048].to(compute_device, torch.float32)
    df_col = (hf * hf).mean(dim=0)
    col_w2 = (W_down_fp32 * W_down_fp32).sum(dim=0)

    # Spectral angular co-activation sorting across [T, d_ffn]
    sample_t = min(1024, hf.shape[0])
    h_act = hf[:sample_t].abs()
    h_centered = h_act - h_act.mean(dim=0, keepdim=True)
    h_normed = F.normalize(h_centered, dim=0)  # [T, d_ffn]
    # Fast low-rank SVD of [T, d_ffn] to extract 2D spectral co-activation coordinates per neuron
    _, _, v_spec = torch.svd_lowrank(h_normed, q=min(6, sample_t, d_ffn), niter=2)
    angles = torch.atan2(v_spec[:, 1], v_spec[:, 0])  # [-pi, pi] across all d_ffn neurons

    if displace_fact_cols > 0:
        logic_silent_mask = dl_col <= torch.quantile(dl_col, 0.24)
        fact_selectivity = torch.where(
            logic_silent_mask,
            df_col * col_w2 * (df_col / dl_col.clamp_min(1e-6)),
            torch.zeros_like(df_col),
        )
        k_disp = min(displace_fact_cols, 4 * page_size, d_ffn)
        _, raw_fact_cols = torch.topk(fact_selectivity, k=k_disp, sorted=True)
        is_fact = torch.zeros(d_ffn, dtype=torch.bool, device=compute_device)
        is_fact[raw_fact_cols] = True
        remaining_cols = torch.nonzero(~is_fact, as_tuple=False).squeeze(-1)
        _, rem_order = torch.sort(angles[remaining_cols])
        perm = torch.cat([raw_fact_cols, remaining_cols[rem_order]], dim=0)
        fact_cols_perm = torch.arange(0, k_disp, device="cpu", dtype=torch.long)
    else:
        _, perm = torch.sort(angles)
        fact_cols_perm = torch.empty(0, device="cpu", dtype=torch.long)

    inv_perm = torch.argsort(perm)
    del hf, h_act, h_centered, h_normed

    # Unbiased per-page residual energy prior (zero synthetic boost on Pages 0..3)
    combined_energy = (0.5 * dl_col + 0.5 * df_col) * col_w2
    page_prior_scores = combined_energy[perm].reshape(num_pages, page_size).mean(dim=1)
    return perm, inv_perm, fact_cols_perm, page_prior_scores


@torch.no_grad()
def decompose_input_whitened_projection(
    W_perm_fp32: torch.Tensor,
    sigma_x_whitening: torch.Tensor,
    rank: int,
    fact_cols_perm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Decomposes W_gate or W_up in shared input space Sigma_X via Cholesky-whitened SVD."""
    L_x = _safe_cholesky(sigma_x_whitening)
    L_x_inv = torch.linalg.inv(L_x)

    M = W_perm_fp32 @ L_x  # [d_ffn, d_model]
    gram = M.T @ M  # [d_model, d_model]
    gram = 0.5 * (gram + gram.T)
    eigvals, eigvecs = torch.linalg.eigh(gram)
    eigvals = eigvals.flip(0).clamp_min(1e-12)
    V = eigvecs.flip(1)
    S = torch.sqrt(eigvals)

    total_energy = float(eigvals.sum().clamp_min(1e-12).item())
    retained_energy = float(eigvals[:rank].sum().item()) / total_energy

    V_r = V[:, :rank]  # [d_model, r]
    S_r = S[:rank]  # [r]
    sqrt_S_r = torch.sqrt(S_r)
    inv_sqrt_S_r = 1.0 / sqrt_S_r.clamp_min(1e-8)

    A_cpu_fp32 = (M @ V_r) * inv_sqrt_S_r.unsqueeze(0)  # [d_ffn, r]
    B_cpu_fp32 = (V_r * sqrt_S_r.unsqueeze(0)).T @ L_x_inv  # [r, d_model]

    if fact_cols_perm.numel() > 0:
        A_cpu_fp32[fact_cols_perm.to(A_cpu_fp32.device), :] = 0.0

    W_ram_fp32 = W_perm_fp32 - (A_cpu_fp32 @ B_cpu_fp32)  # [d_ffn, d_model]
    return A_cpu_fp32, B_cpu_fp32, W_ram_fp32, retained_energy


@torch.no_grad()
def _simulate_sparse_geglu_and_routed_pages(
    x_tokens: torch.Tensor,
    A_gate_fp32: torch.Tensor,
    B_gate_fp32: torch.Tensor,
    W_gate_ram_fp32: torch.Tensor,
    A_up_fp32: torch.Tensor,
    B_up_fp32: torch.Tensor,
    W_up_ram_fp32: torch.Tensor,
    act_fn: nn.Module,
    router: ClusteredPageRouter,
    page_size: int,
    max_pages: int,
    seq_chunk: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulates the EXACT honest sparse `vm_paged` GeGLU activations (strictly `max_pages` per sequence)."""
    n_tok, d_model = x_tokens.shape
    d_ffn = A_gate_fp32.shape[0]
    g_out = (x_tokens @ B_gate_fp32.T) @ A_gate_fp32.T
    u_out = (x_tokens @ B_up_fp32.T) @ A_up_fp32.T

    page_arange = torch.arange(page_size, device=x_tokens.device, dtype=torch.long)
    num_chunks = (n_tok + seq_chunk - 1) // seq_chunk
    routed_pages_list: List[torch.Tensor] = []

    for c_idx in range(num_chunks):
        s = c_idx * seq_chunk
        e = min(n_tok, s + seq_chunk)
        x_seq = x_tokens[s:e]  # [S_c, d_model]
        top_p = router.select_pages(x_seq, max_pages_override=max_pages)  # [K_p]
        routed_pages_list.append(top_p)
        cols = ((top_p.unsqueeze(1) * page_size) + page_arange.unsqueeze(0)).reshape(-1)
        g_out[s:e].index_add_(1, cols, x_seq @ W_gate_ram_fp32.index_select(0, cols).T)
        u_out[s:e].index_add_(1, cols, x_seq @ W_up_ram_fp32.index_select(0, cols).T)

    h_sparse = act_fn(g_out) * u_out
    routed_pages = torch.stack(routed_pages_list, dim=0)  # [num_chunks, K_p]
    return h_sparse, routed_pages


@torch.no_grad()
def decompose_full_mlp_layer(
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    act_fn: nn.Module,
    X_orig_logic: torch.Tensor,
    X_orig_fact: torch.Tensor,
    H_orig_logic: torch.Tensor,
    H_orig_fact: torch.Tensor,
    X_actual_logic: Optional[torch.Tensor] = None,
    X_actual_fact: Optional[torch.Tensor] = None,
    rank_ratio: float = 0.25,
    num_pages: int = 64,
    max_pages: int = 4,
    displace_fact_cols: int = 0,
    lambda_reg: float = 1e-4,
    closed_loop_ridge: float = 1e-2,
    compute_device: str = "cuda:0",
    target_device: str = "cuda:0",
    target_dtype: torch.dtype = torch.bfloat16,
    layer_idx: int = 18,
    shared_prefetch_stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[PagedNeuralMLP, FullMLPLayerStats]:
    """Decompiles gate_proj, up_proj, and down_proj into CPU Cores + 64 Honest Clustered RAM Pages."""
    d_model, d_ffn = W_down.shape
    rank = max(1, int(rank_ratio * min(d_model, d_ffn)))
    page_size = d_ffn // num_pages

    W_gate_fp32 = W_gate.to(compute_device, torch.float32)
    W_up_fp32 = W_up.to(compute_device, torch.float32)
    W_down_fp32 = W_down.to(compute_device, torch.float32)

    # 1. Offline Spectral Co-Activation Page Clustering
    perm, _, fact_cols_perm, page_prior_scores = cluster_neurons_into_knowledge_pages(
        W_down_fp32=W_down_fp32,
        H_orig_logic=H_orig_logic,
        H_orig_fact=H_orig_fact,
        num_pages=num_pages,
        displace_fact_cols=displace_fact_cols,
        compute_device=compute_device,
    )

    W_gate_perm = W_gate_fp32[perm, :].contiguous()  # [d_ffn, d_model]
    W_up_perm = W_up_fp32[perm, :].contiguous()  # [d_ffn, d_model]
    W_down_perm = W_down_fp32[:, perm].contiguous()  # [d_model, d_ffn]

    hl_orig_perm = H_orig_logic[:2048].to(compute_device, torch.float32)[:, perm]
    hf_orig_perm = H_orig_fact[:2048].to(compute_device, torch.float32)[:, perm]

    cpu_logic_bias: Optional[torch.Tensor] = None
    fact_cols_dev = fact_cols_perm.to(compute_device)
    if fact_cols_perm.numel() > 0:
        mean_hl = hl_orig_perm.mean(dim=0)
        missing_mean_y = W_down_perm[:, fact_cols_dev] @ mean_hl[fact_cols_dev]
        cpu_logic_bias = missing_mean_y.to(device="cpu", dtype=target_dtype).contiguous()

    # 2. Shared Input Cholesky Whitening for gate_proj and up_proj
    x_act_l = X_actual_logic if X_actual_logic is not None else X_orig_logic
    x_act_f = X_actual_fact if X_actual_fact is not None else X_orig_fact

    sigma_x_l = compute_covariance_matrix(x_act_l, lambda_reg=lambda_reg, compute_device=compute_device)
    sigma_x_f = compute_covariance_matrix(x_act_f, lambda_reg=lambda_reg, compute_device=compute_device)
    rx = torch.diagonal(sigma_x_l).clamp_min(lambda_reg) / torch.diagonal(sigma_x_f).clamp_min(lambda_reg)
    rx = torch.clamp(rx, min=0.55, max=1.80)
    rx_scale = torch.sqrt(rx / rx.mean().clamp_min(1e-6))
    sigma_x_whitening = sigma_x_l * rx_scale.unsqueeze(1) * rx_scale.unsqueeze(0)
    sigma_x_whitening = 0.5 * (sigma_x_whitening + sigma_x_whitening.T)
    dx_idx = torch.arange(d_model, device=compute_device)
    sigma_x_whitening[dx_idx, dx_idx] += lambda_reg
    del sigma_x_l, sigma_x_f

    A_gate_fp32, B_gate_fp32, W_gate_ram_fp32, gate_energy = decompose_input_whitened_projection(
        W_perm_fp32=W_gate_perm,
        sigma_x_whitening=sigma_x_whitening,
        rank=rank,
        fact_cols_perm=fact_cols_perm,
    )
    A_up_fp32, B_up_fp32, W_up_ram_fp32, up_energy = decompose_input_whitened_projection(
        W_perm_fp32=W_up_perm,
        sigma_x_whitening=sigma_x_whitening,
        rank=rank,
        fact_cols_perm=fact_cols_perm,
    )
    del sigma_x_whitening

    # 3. Build Router & Simulate TRUE Sparse-Paged GeGLU Activations (`max_pages` only, zero dense cheat)
    init_col_norms = torch.linalg.norm(W_down_perm, dim=0)
    router = ClusteredPageRouter.from_clustered_mlp_weights(
        W_gate_perm=W_gate_ram_fp32,
        W_up_perm=W_up_ram_fp32,
        down_ram_col_norms_perm=init_col_norms,
        page_prior_scores=page_prior_scores,
        num_pages=num_pages,
        max_pages=max_pages,
        target_device=compute_device,
        target_dtype=target_dtype,
    )

    sample_l = min(2048, x_act_l.shape[0], hl_orig_perm.shape[0])
    sample_f = min(2048, x_act_f.shape[0], hf_orig_perm.shape[0])
    xl_sub = x_act_l[:sample_l].to(compute_device, torch.float32)
    xf_sub = x_act_f[:sample_f].to(compute_device, torch.float32)

    hl_paged, pages_l = _simulate_sparse_geglu_and_routed_pages(
        xl_sub,
        A_gate_fp32,
        B_gate_fp32,
        W_gate_ram_fp32,
        A_up_fp32,
        B_up_fp32,
        W_up_ram_fp32,
        act_fn,
        router,
        page_size=page_size,
        max_pages=max_pages,
    )
    hf_paged, pages_f = _simulate_sparse_geglu_and_routed_pages(
        xf_sub,
        A_gate_fp32,
        B_gate_fp32,
        W_gate_ram_fp32,
        A_up_fp32,
        B_up_fp32,
        W_up_ram_fp32,
        act_fn,
        router,
        page_size=page_size,
        max_pages=max_pages,
    )

    sigma_h_l = compute_covariance_matrix(hl_paged, lambda_reg=lambda_reg, compute_device=compute_device)
    sigma_h_f = compute_covariance_matrix(hf_paged, lambda_reg=lambda_reg, compute_device=compute_device)
    rh = torch.diagonal(sigma_h_l).clamp_min(lambda_reg) / torch.diagonal(sigma_h_f).clamp_min(lambda_reg)
    rh = torch.clamp(rh, min=0.55, max=1.80)
    rh_scale = torch.sqrt(rh / rh.mean().clamp_min(1e-6))
    sigma_h_whitening = sigma_h_l * rh_scale.unsqueeze(1) * rx_scale.new_ones(1) * rh_scale.unsqueeze(0)
    sigma_h_whitening = 0.5 * (sigma_h_whitening + sigma_h_whitening.T)
    dh_idx = torch.arange(d_ffn, device=compute_device)
    mean_dh = float(torch.diagonal(sigma_h_whitening).abs().mean().clamp_min(1.0).item())
    sigma_h_whitening[dh_idx, dh_idx] += max(lambda_reg, mean_dh * 1e-4)
    del sigma_h_l, sigma_h_f

    gram_down = W_down_perm @ sigma_h_whitening @ W_down_perm.T
    gram_down = 0.5 * (gram_down + gram_down.T)
    eigvals_d, eigvecs_d = torch.linalg.eigh(gram_down)
    eigvals_d = eigvals_d.flip(0).clamp_min(1e-12)
    U_d = eigvecs_d.flip(1)
    S_d = torch.sqrt(eigvals_d)
    down_energy = float(eigvals_d[:rank].sum().item()) / float(eigvals_d.sum().clamp_min(1e-12).item())

    U_dr = U_d[:, :rank]
    S_dr = S_d[:rank]
    sqrt_S_dr = torch.sqrt(S_dr)
    inv_sqrt_S_dr = 1.0 / sqrt_S_dr.clamp_min(1e-8)

    A_down_fp32 = U_dr * sqrt_S_dr.unsqueeze(0)  # [d_model, r]
    B_down_fp32 = (U_dr * inv_sqrt_S_dr.unsqueeze(0)).T @ W_down_perm  # [r, d_ffn]
    if fact_cols_perm.numel() > 0:
        B_down_fp32[:, fact_cols_dev] = 0.0

    W_down_ram_fp32 = W_down_perm - (A_down_fp32 @ B_down_fp32)  # [d_model, d_ffn]
    del sigma_h_whitening

    # Update router using exact down_ram residual column norms
    down_ram_col_norms = torch.linalg.norm(W_down_ram_fp32, dim=0)
    ram_page_energy = (
        ((hl_paged.square().mean(dim=0) + hf_paged.square().mean(dim=0)) * 0.5 * down_ram_col_norms.square())
        .reshape(num_pages, page_size)
        .mean(dim=1)
    )
    router = ClusteredPageRouter.from_clustered_mlp_weights(
        W_gate_perm=W_gate_ram_fp32,
        W_up_perm=W_up_ram_fp32,
        down_ram_col_norms_perm=down_ram_col_norms,
        page_prior_scores=ram_page_energy,
        num_pages=num_pages,
        max_pages=max_pages,
        target_device=target_device,
        target_dtype=target_dtype,
    )

    # 4. Honest Closed-Loop Drift Correction on A_down Using STRICTLY `max_pages` Routed Pages
    h_orig_mix = torch.cat([hl_orig_perm[:sample_l], hf_orig_perm[:sample_f]], dim=0)
    h_paged_mix = torch.cat([hl_paged, hf_paged], dim=0)
    all_pages = torch.cat([pages_l, pages_f], dim=0)  # [num_chunks_total, K_p]

    y_teacher = h_orig_mix @ W_down_perm.T  # [N_mix, d_model]
    z_latent = h_paged_mix @ B_down_fp32.T  # [N_mix, r]

    # Compute exact sparse y_ram_paged using ONLY the `max_pages` routed per sequence chunk (60 pages = 0)
    y_ram_paged = torch.zeros_like(y_teacher)
    page_arange = torch.arange(page_size, device=compute_device, dtype=torch.long)
    seq_chunk = 256
    offset = 0
    for chunk_i in range(pages_l.shape[0]):
        s = offset
        e = min(sample_l, s + seq_chunk)
        cols = ((pages_l[chunk_i].unsqueeze(1) * page_size) + page_arange.unsqueeze(0)).reshape(-1)
        y_ram_paged[s:e] = hl_paged[s:e].index_select(1, cols) @ W_down_ram_fp32.index_select(1, cols).T
        offset = e
    offset_f = 0
    for chunk_i in range(pages_f.shape[0]):
        s = sample_l + offset_f
        e = sample_l + min(sample_f, offset_f + seq_chunk)
        cols = ((pages_f[chunk_i].unsqueeze(1) * page_size) + page_arange.unsqueeze(0)).reshape(-1)
        y_ram_paged[s:e] = (
            hf_paged[offset_f : offset_f + (e - s)].index_select(1, cols)
            @ W_down_ram_fp32.index_select(1, cols).T
        )
        offset_f += e - s

    y_vm_init = (z_latent @ A_down_fp32.T) + y_ram_paged
    residual_target = y_teacher - y_vm_init

    ztz = z_latent.T @ z_latent
    reg_val = closed_loop_ridge * float(torch.trace(ztz).item()) / max(1, rank)
    r_idx = torch.arange(rank, device=compute_device)
    ztz[r_idx, r_idx] += max(reg_val, 1e-4)
    delta_A_T = torch.linalg.solve(ztz, z_latent.T @ residual_target)
    A_down_closed_fp32 = A_down_fp32 + 0.85 * delta_A_T.T

    y_vm_post = (z_latent[sample_l:] @ A_down_closed_fp32.T) + y_ram_paged[sample_l:]
    y_targ_f = hf_orig_perm[:sample_f] @ W_down_perm.T
    post_err = float(
        (torch.linalg.norm(y_targ_f - y_vm_post) / torch.linalg.norm(y_targ_f).clamp_min(1e-12)).item()
    )

    # 5. Pack W_gate_ram, W_up_ram, W_down_ram.T into P=64 Contiguous Pinned Host RAM Pages [P, 3, C, d_model]
    gate_pages = W_gate_ram_fp32.reshape(num_pages, page_size, d_model)
    up_pages = W_up_ram_fp32.reshape(num_pages, page_size, d_model)
    down_pages = W_down_ram_fp32.T.contiguous().reshape(num_pages, page_size, d_model)

    W_mlp_ram_pages = (
        torch.stack([gate_pages, up_pages, down_pages], dim=1)
        .to(device="cpu", dtype=target_dtype)
        .contiguous()
    )
    if torch.cuda.is_available():
        try:
            W_mlp_ram_pages = W_mlp_ram_pages.pin_memory()
        except Exception:
            pass

    paged_mlp = PagedNeuralMLP(
        A_gate_cpu=A_gate_fp32.to(device=target_device, dtype=target_dtype).contiguous(),
        B_gate_cpu=B_gate_fp32.to(device=target_device, dtype=target_dtype).contiguous(),
        A_up_cpu=A_up_fp32.to(device=target_device, dtype=target_dtype).contiguous(),
        B_up_cpu=B_up_fp32.to(device=target_device, dtype=target_dtype).contiguous(),
        A_down_cpu=A_down_fp32.to(device=target_device, dtype=target_dtype).contiguous(),
        B_down_cpu=B_down_fp32.to(device=target_device, dtype=target_dtype).contiguous(),
        W_mlp_ram_pages_pinned=W_mlp_ram_pages,
        act_fn=act_fn,
        router=router,
        num_pages=num_pages,
        max_pages=max_pages,
        mode="vm_paged",
        paging_strategy="sync_clustered",
        prefetch_stream=shared_prefetch_stream,
        A_down_closed_loop=A_down_closed_fp32.to(device=target_device, dtype=target_dtype).contiguous(),
        displaced_fact_cols_cpu=fact_cols_perm if fact_cols_perm.numel() > 0 else None,
        cpu_logic_bias=cpu_logic_bias,
    )

    stats = FullMLPLayerStats(
        layer_idx=layer_idx,
        d_ffn=d_ffn,
        rank=rank,
        num_pages=num_pages,
        page_size=page_size,
        gate_energy_retained=gate_energy,
        up_energy_retained=up_energy,
        down_energy_retained=down_energy,
        post_correction_rel_error=post_err,
    )
    return paged_mlp, stats


@torch.no_grad()
def decompile_full_mlp_cascade_stack(
    model: nn.Module,
    snapshots: Dict[int, Dict[str, torch.Tensor]],
    teacher_x_logic: Dict[int, torch.Tensor],
    teacher_x_fact: Dict[int, torch.Tensor],
    teacher_h_logic: Dict[int, torch.Tensor],
    teacher_h_fact: Dict[int, torch.Tensor],
    logic_calib_split: TokenizedSplit,
    fact_calib_split: TokenizedSplit,
    layer_indices: List[int],
    rank_ratio: float = 0.25,
    num_pages: int = 64,
    max_pages: int = 4,
    displace_fact_cols: int = 0,
    lambda_reg: float = 1e-4,
    closed_loop_ridge: float = 1e-2,
    batch_size: int = 4,
    max_calib_tokens: int = 4096,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Dict[int, PagedNeuralMLP], List[FullMLPLayerStats]]:
    """Sequentially decompiles full MLP blocks into hardware-honest PagedNeuralMLP modules."""
    decoder_layers = resolve_decoder_layers(model)
    paged_mlps: Dict[int, PagedNeuralMLP] = {}
    stats_list: List[FullMLPLayerStats] = []
    shared_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    sorted_layers = sorted(layer_indices)
    for step_i, idx in enumerate(sorted_layers):
        w_dict = snapshots[idx]
        orig_mlp = get_target_mlp(model, idx)
        act_fn = getattr(orig_mlp, "act_fn", nn.GELU(approximate="tanh"))

        if step_i == 0:
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

        paged_mlp, st = decompose_full_mlp_layer(
            W_gate=w_dict["gate_proj"],
            W_up=w_dict["up_proj"],
            W_down=w_dict["down_proj"],
            act_fn=act_fn,
            X_orig_logic=teacher_x_logic[idx],
            X_orig_fact=teacher_x_fact[idx],
            H_orig_logic=teacher_h_logic[idx],
            H_orig_fact=teacher_h_fact[idx],
            X_actual_logic=x_act_l,
            X_actual_fact=x_act_f,
            rank_ratio=rank_ratio,
            num_pages=num_pages,
            max_pages=max_pages,
            displace_fact_cols=displace_fact_cols,
            lambda_reg=lambda_reg,
            closed_loop_ridge=closed_loop_ridge,
            compute_device=device,
            target_device=device,
            target_dtype=dtype,
            layer_idx=idx,
            shared_prefetch_stream=shared_stream,
        )

        orig_mlp.to("cpu")
        decoder_layers[idx].mlp = paged_mlp
        paged_mlp.attach_decoder_layer_pre_hook(decoder_layers[idx])
        paged_mlps[idx] = paged_mlp
        stats_list.append(st)

        del x_act_l, x_act_f, orig_mlp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return paged_mlps, stats_list
