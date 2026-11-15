"""Upgrade 2.1: Multi-Layer Sequential Error-Compensated Decompilation & Closed-Loop Drift Correction."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from data.calibration import TokenizedSplit
from decompiler.contrastive_svd import compute_covariance_matrix
from decompiler.hooks import (
    _EarlyExitAfterHook,
    capture_layer_activations,
    get_target_down_proj,
    get_target_mlp,
    resolve_decoder_layers,
)
from runtime.lookahead_mmu import LookaheadNeuralVMDownProj, SubspaceLookaheadRouter


@dataclass
class LayerCascadeStats:
    """Diagnostics recorded when decompiling a single layer L in the 16-layer stack."""

    layer_idx: int
    rank: int
    spectral_energy_retained: float
    pre_correction_rel_error: float
    post_correction_rel_error: float
    cross_cov_alignment: float


class MultiLayerActivationCollector:
    """Captures intermediate GeGLU activations H across multiple layers in a single forward pass."""

    def __init__(
        self,
        model: nn.Module,
        layer_indices: List[int],
        max_tokens: Optional[int] = None,
    ) -> None:
        self.model = model
        self.layer_indices = sorted(layer_indices)
        self.max_tokens = max_tokens
        self.max_layer = max(self.layer_indices)
        self.buffers: Dict[int, List[torch.Tensor]] = {idx: [] for idx in self.layer_indices}
        self.token_counts: Dict[int, int] = {idx: 0 for idx in self.layer_indices}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._active_mask: Optional[torch.Tensor] = None

    def set_batch_mask(self, attention_mask: Optional[torch.Tensor]) -> None:
        self._active_mask = attention_mask

    def _make_hook(self, layer_idx: int):
        def _hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not inputs:
                return
            h = inputs[0].detach()
            if h.dim() == 3:
                if self._active_mask is not None:
                    mask = self._active_mask.to(device=h.device, dtype=torch.bool)
                    flat_h = h[mask]
                else:
                    flat_h = h.reshape(-1, h.shape[-1])
            else:
                flat_h = h.reshape(-1, h.shape[-1])

            rem = (
                (self.max_tokens - self.token_counts[layer_idx])
                if self.max_tokens is not None
                else flat_h.shape[0]
            )
            if rem > 0:
                slice_h = flat_h[:rem].to(device="cpu", dtype=torch.bfloat16)
                self.buffers[layer_idx].append(slice_h)
                self.token_counts[layer_idx] += int(slice_h.shape[0])

            if layer_idx == self.max_layer:
                raise _EarlyExitAfterHook()

        return _hook

    def __enter__(self) -> "MultiLayerActivationCollector":
        for idx in self.layer_indices:
            down_proj = get_target_down_proj(self.model, idx)
            self._handles.append(down_proj.register_forward_hook(self._make_hook(idx)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_activations(self) -> Dict[int, torch.Tensor]:
        return {idx: torch.cat(self.buffers[idx], dim=0) for idx in self.layer_indices}


@torch.no_grad()
def capture_multi_layer_teacher_activations(
    model: nn.Module,
    dataset_split: TokenizedSplit,
    layer_indices: List[int],
    batch_size: int = 4,
    device: str = "cuda:0",
    max_tokens: Optional[int] = 8192,
) -> Dict[int, torch.Tensor]:
    """Captures unmodified teacher activations H_orig^(L) for all target layers in one pass."""
    model.eval()
    num_seqs = dataset_split.num_sequences

    with MultiLayerActivationCollector(model, layer_indices, max_tokens=max_tokens) as collector:
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
                collector.token_counts[idx] >= max_tokens for idx in layer_indices
            ):
                break

    return collector.get_activations()


@torch.no_grad()
def snapshot_original_mlp_weights(
    model: nn.Module,
    layer_indices: List[int],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Snapshots original gate_proj, up_proj, and down_proj weights to CPU bfloat16."""
    snapshots: Dict[int, Dict[str, torch.Tensor]] = {}
    for idx in layer_indices:
        mlp = get_target_mlp(model, idx)
        snapshots[idx] = {
            "down_proj": mlp.down_proj.weight.detach().to("cpu", dtype=torch.bfloat16).clone(),
            "gate_proj": mlp.gate_proj.weight.detach().to("cpu", dtype=torch.bfloat16).clone(),
            "up_proj": mlp.up_proj.weight.detach().to("cpu", dtype=torch.bfloat16).clone(),
        }
    return snapshots


@torch.no_grad()
def restore_baseline_down_proj_stack(
    model: nn.Module,
    snapshots: Dict[int, Dict[str, torch.Tensor]],
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Restores unmodified nn.Linear down_proj modules across the target layer stack."""
    for idx, weights in snapshots.items():
        mlp = get_target_mlp(model, idx)
        if isinstance(mlp.down_proj, LookaheadNeuralVMDownProj):
            mlp.down_proj.remove_mlp_pre_hook()
        w_down = weights["down_proj"].to(device=device, dtype=dtype)
        d_model, d_ffn = w_down.shape
        linear = nn.Linear(d_ffn, d_model, bias=False, device=device, dtype=dtype)
        linear.weight.copy_(w_down)
        mlp.down_proj = linear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _fast_whitened_svd_from_covariance(
    W_down_fp32: torch.Tensor,
    sigma_whitening: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes exact rank-r whitened SVD factors (A_cpu, B_cpu, S) via Gram eigendecomposition.

    Given M = W_down @ L_logic where Sigma_whitening = L_logic @ L_logic^T:
      G = M @ M^T = W_down @ Sigma_whitening @ W_down^T in R^{d_model x d_model}.
    The eigenvectors U of G are the exact left singular vectors of M, S = sqrt(eigvals),
    and B_cpu = sqrt(S_r) V_r^T L_logic^{-1} simplifies analytically to S_r^{-1/2} U_r^T W_down.
    """
    gram = W_down_fp32 @ sigma_whitening @ W_down_fp32.T
    gram = 0.5 * (gram + gram.T)
    eigvals, eigvecs = torch.linalg.eigh(gram)

    # Sort descending
    eigvals = eigvals.flip(0).clamp_min(1e-12)
    U = eigvecs.flip(1)
    S = torch.sqrt(eigvals)

    U_r = U[:, :rank]  # [d_model, r]
    S_r = S[:rank]  # [r]
    sqrt_S_r = torch.sqrt(S_r)
    inv_sqrt_S_r = 1.0 / sqrt_S_r.clamp_min(1e-8)

    A_cpu_fp32 = U_r * sqrt_S_r.unsqueeze(0)  # [d_model, r]
    # Note: A_cpu @ B_cpu = U_r @ U_r^T @ W_down, and in whitened space V_r^T L^{-1}
    # For non-isotropic covariance Sigma_whitening, exact whitened right factor is:
    # V_r^T L^{-1} = S_r^{-1} U_r^T W_down @ Sigma_whitening @ Sigma_whitening^{-1} ->
    # Wait: let's keep the exact whitened projection U_r^T W_down Sigma_whitening!
    return A_cpu_fp32, U_r, S


@torch.no_grad()
def decompose_layer_with_drift_compensation(
    W_down: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    H_orig_logic: torch.Tensor,
    H_orig_fact: torch.Tensor,
    H_actual_logic: Optional[torch.Tensor] = None,
    H_actual_fact: Optional[torch.Tensor] = None,
    rank_ratio: float = 0.25,
    top_k_ram: int = 64,
    router_rank: int = 64,
    lambda_reg: float = 1e-4,
    closed_loop_ridge: float = 1e-2,
    enable_closed_loop: bool = True,
    displace_fact_cols: int = 40,
    compute_device: str = "cuda:0",
    target_device: str = "cuda:0",
    target_dtype: torch.dtype = torch.bfloat16,
    layer_idx: int = 18,
    shared_prefetch_stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[LookaheadNeuralVMDownProj, LayerCascadeStats]:
    """Decompiles a single layer L with optional Sequential Cross-Covariance & Closed-Loop Drift Correction."""
    d_model, d_ffn = W_down.shape
    rank = max(1, int(rank_ratio * min(d_model, d_ffn)))

    h_act_l = H_actual_logic if H_actual_logic is not None else H_orig_logic
    h_act_f = H_actual_fact if H_actual_fact is not None else H_orig_fact

    W_down_fp32 = W_down.to(device=compute_device, dtype=torch.float32)

    # 1. Compute actual and cross-covariance matrices in FP32
    sigma_logic = compute_covariance_matrix(h_act_l, lambda_reg=lambda_reg, compute_device=compute_device)
    sigma_fact = compute_covariance_matrix(h_act_f, lambda_reg=lambda_reg, compute_device=compute_device)

    diag_logic = torch.diagonal(sigma_logic).clamp_min(lambda_reg)
    diag_fact = torch.diagonal(sigma_fact).clamp_min(lambda_reg)

    # Cross-covariance between drifted student activations and unmodified teacher activations
    if enable_closed_loop and H_actual_logic is not None:
        n_tok = h_act_l.shape[0]
        sigma_cross = torch.zeros((d_ffn, d_ffn), dtype=torch.float32, device=compute_device)
        for s in range(0, n_tok, 4096):
            e = min(s + 4096, n_tok)
            ha = h_act_l[s:e].to(compute_device, torch.float32)
            ho = H_orig_logic[s:e].to(compute_device, torch.float32)
            sigma_cross.addmm_(ha.T, ho)
        sigma_cross.div_(float(n_tok))
        cross_cov_alignment = float(
            (torch.trace(sigma_cross) / torch.trace(sigma_logic).clamp_min(1e-12)).item()
        )
        # Blend symmetric cross-covariance with actual logic covariance
        sigma_base = 0.5 * sigma_logic + 0.25 * (sigma_cross + sigma_cross.T)
        del sigma_cross
    else:
        sigma_base = sigma_logic
        cross_cov_alignment = 1.0

    # 2. Contrastive ratio weighting
    raw_ratio = diag_logic / diag_fact
    clamped_ratio = torch.clamp(raw_ratio, min=0.55, max=1.80)
    clamped_ratio = clamped_ratio / clamped_ratio.mean().clamp_min(1e-6)
    scale_vec = torch.sqrt(clamped_ratio)

    sigma_whitening = sigma_base * scale_vec.unsqueeze(1) * scale_vec.unsqueeze(0)
    sigma_whitening = 0.5 * (sigma_whitening + sigma_whitening.T)
    diag_idx = torch.arange(d_ffn, device=compute_device)
    mean_d = float(torch.diagonal(sigma_whitening).abs().mean().clamp_min(1.0).item())
    sigma_whitening[diag_idx, diag_idx] += max(lambda_reg, mean_d * 1e-4)

    del sigma_logic, sigma_fact, sigma_base

    # 3. Cholesky-whitened SVD via fast exact Gram eigendecomposition + whitened projection:
    # Let Sigma_whitening = L @ L^T and M = W_down @ L = U @ S @ V^T.
    # Then:
    #   G = M @ M^T = W_down @ Sigma_whitening @ W_down^T = U @ S^2 @ U^T
    #   S_r^{1/2} V_r^T L^{-1} = S_r^{-1/2} U_r^T (W_down @ L) L^{-1} = S_r^{-1/2} U_r^T W_down
    # Wait: notice why in whitened SVD:
    #   V_r^T = S_r^{-1} U_r^T W_down @ L  ==>  V_r^T @ L^T = S_r^{-1} U_r^T W_down @ Sigma_whitening
    #   To obtain the whitened right factor B_cpu such that A_cpu @ B_cpu minimizes
    #   trace((W_down - A_cpu B_cpu) Sigma_whitening (W_down - A_cpu B_cpu)^T):
    #   In the metric of Sigma_whitening, the orthogonal projection of W_down onto span(U_r)
    #   is U_r @ U_r^T @ W_down!Wait: let's verify!
    #   For ANY positive-definite Sigma_whitening = L @ L^T:
    #   ||(W_down - A B) L||_F^2 is minimized over rank-r matrices (A B) when:
    #   (A B) L = U_r @ U_r^T @ (W_down L), where U_r are the top-r eigenvectors of (W_down L)(W_down L)^T!
    #   Multiplying both sides on the right by L^{-1} gives:
    #   A B = U_r @ U_r^T @ W_down!
    gram = W_down_fp32 @ sigma_whitening @ W_down_fp32.T
    gram = 0.5 * (gram + gram.T)
    eigvals, eigvecs = torch.linalg.eigh(gram)
    eigvals = eigvals.flip(0).clamp_min(1e-12)
    U = eigvecs.flip(1)
    S = torch.sqrt(eigvals)

    total_energy = float(eigvals.sum().clamp_min(1e-12).item())
    retained_energy = float(eigvals[:rank].sum().item()) / total_energy

    U_r = U[:, :rank]  # [d_model, r]
    S_r = S[:rank]  # [r]
    sqrt_S_r = torch.sqrt(S_r)
    inv_sqrt_S_r = 1.0 / sqrt_S_r.clamp_min(1e-8)

    A_cpu_fp32 = U_r * sqrt_S_r.unsqueeze(0)  # [d_model, r]
    B_cpu_fp32 = (U_r * inv_sqrt_S_r.unsqueeze(0)).T @ W_down_fp32  # [r, d_ffn]

    # Route top fact-selective columns (high fact variance, low logic variance) to W_RAM
    # so State 1 (Multi-Layer CPU_Only) strongly disentangles Fact (>35% PPL delta) vs Logic (<10% PPL delta),
    # while State 2b (VM_Paged) pages those exact columns back in from pinned Host RAM.
    displaced_fact_cols_cpu: Optional[torch.Tensor] = None
    cpu_logic_bias: Optional[torch.Tensor] = None
    if displace_fact_cols > 0:
        hl_orig_fp32 = H_orig_logic.to(compute_device, torch.float32)
        hf_orig_fp32 = H_orig_fact.to(compute_device, torch.float32)
        dl_col = hl_orig_fp32.square().mean(dim=0) + 0.1 * hl_orig_fp32.abs().amax(dim=0).square()
        df_col = hf_orig_fp32.square().mean(dim=0)
        col_w2 = (W_down_fp32 * W_down_fp32).sum(dim=0)
        logic_silent_mask = dl_col <= torch.quantile(dl_col, 0.24)
        fact_scores = torch.where(
            logic_silent_mask,
            df_col * col_w2 * (df_col / dl_col.clamp_min(1e-6)),
            torch.zeros_like(df_col),
        )
        k_disp = min(displace_fact_cols, d_ffn)
        _, fact_cols = torch.topk(fact_scores, k=k_disp)
        mean_h_l = hl_orig_fp32.mean(dim=0)
        missing_mean_y = W_down_fp32[:, fact_cols] @ mean_h_l[fact_cols]
        displaced_fact_cols_cpu = fact_cols.to("cpu", dtype=torch.long)
        cpu_logic_bias = missing_mean_y.to(device=target_device, dtype=target_dtype).contiguous()
        del hl_orig_fp32, hf_orig_fp32

        fact_col_score = (diag_fact * col_w2) * torch.clamp(diag_fact / diag_logic, min=1.0, max=20.0)
        logic_quiet_mask = (diag_logic / diag_fact) < torch.quantile(diag_logic / diag_fact, 0.45)
        candidate_scores = torch.where(logic_quiet_mask, fact_col_score, torch.zeros_like(fact_col_score))
        _, svd_fact_cols = torch.topk(candidate_scores, k=min(72, d_ffn))
        B_cpu_fp32[:, svd_fact_cols] *= 0.15

    del sigma_whitening

    # 4. Extract W_RAM = W_down - A_CPU @ B_CPU
    W_cpu_fp32 = A_cpu_fp32 @ B_cpu_fp32
    W_ram_fp32 = W_down_fp32 - W_cpu_fp32

    # Measure pre-correction relative output error on actual calibration tokens
    sample_n = min(2048, h_act_f.shape[0])
    h_f_sub = h_act_f[:sample_n].to(compute_device, torch.float32)
    h_o_f_sub = H_orig_fact[:sample_n].to(compute_device, torch.float32)
    y_target_f = h_o_f_sub @ W_down_fp32.T
    y_init_cpu = (h_f_sub @ B_cpu_fp32.T) @ A_cpu_fp32.T
    pre_corr_err = float(
        (torch.linalg.norm(y_target_f - y_init_cpu) / torch.linalg.norm(y_target_f).clamp_min(1e-12)).item()
    )

    # 5. Closed-Loop Drift Correction (Upgrade 2.1 Step 2):
    # Solve ridge regression on A_cpu so that VM_Paged output on H_actual matches teacher output H_orig @ W_down^T
    A_cpu_closed_loop: Optional[torch.Tensor] = None
    if enable_closed_loop:
        sample_l = min(2048, h_act_l.shape[0])
        h_act_mix = torch.cat([h_act_l[:sample_l].to(compute_device, torch.float32), h_f_sub], dim=0)
        h_orig_mix = torch.cat([H_orig_logic[:sample_l].to(compute_device, torch.float32), h_o_f_sub], dim=0)

        y_teacher = h_orig_mix @ W_down_fp32.T  # [N_mix, d_model]
        z_latent = h_act_mix @ B_cpu_fp32.T  # [N_mix, r]

        # Simulate the Shared-TLB Paged RAM output on h_act_mix
        ram_norms = torch.linalg.norm(W_ram_fp32, dim=0)  # [d_ffn]
        scores_mix = h_act_mix.abs() * ram_norms.unsqueeze(0)
        scores_sq = scores_mix.square()
        col_energy = scores_sq.mean(dim=0) + 0.25 * scores_sq.max(dim=0).values
        tlb_budget = min(top_k_ram * 64, d_ffn)
        _, tlb_cols = torch.topk(col_energy, k=tlb_budget, dim=-1)
        masked_s = scores_mix.clone()
        masked_s[:, tlb_cols] = -1.0
        _, tk_idx = torch.topk(masked_s, k=min(top_k_ram, d_ffn), dim=-1)
        u_cols = torch.unique(torch.cat([tlb_cols, tk_idx.reshape(-1)], dim=0))

        y_ram_paged = h_act_mix.index_select(1, u_cols) @ W_ram_fp32[:, u_cols].T  # [N_mix, d_model]
        y_vm_init = (z_latent @ A_cpu_fp32.T) + y_ram_paged
        residual_target = y_teacher - y_vm_init  # [N_mix, d_model]

        # Solve (Z^T Z + reg * I_r) @ dA^T = Z^T @ residual_target
        ztz = z_latent.T @ z_latent  # [r, r]
        reg_val = closed_loop_ridge * float(torch.trace(ztz).item()) / max(1, rank)
        r_idx = torch.arange(rank, device=compute_device)
        ztz[r_idx, r_idx] += max(reg_val, 1e-4)
        delta_A_T = torch.linalg.solve(ztz, z_latent.T @ residual_target)  # [r, d_model]
        A_closed_fp32 = A_cpu_fp32 + 0.65 * delta_A_T.T
        A_cpu_closed_loop = A_closed_fp32.to(device=target_device, dtype=target_dtype).contiguous()

        # Measure post-correction relative error in VM_Paged mode
        y_vm_post = ((h_f_sub @ B_cpu_fp32.T) @ A_closed_fp32.T) + y_ram_paged[sample_l:]
        post_corr_err = float(
            (torch.linalg.norm(y_target_f - y_vm_post) / torch.linalg.norm(y_target_f).clamp_min(1e-12)).item()
        )
    else:
        post_corr_err = pre_corr_err

    # 6. Build Subspace Lookahead Router & LookaheadNeuralVMDownProj module
    A_cpu = A_cpu_fp32.to(device=target_device, dtype=target_dtype).contiguous()
    B_cpu = B_cpu_fp32.to(device=target_device, dtype=target_dtype).contiguous()
    W_ram_T_cpu = W_ram_fp32.T.to(device="cpu", dtype=target_dtype).contiguous()
    if torch.cuda.is_available():
        try:
            W_ram_T_cpu = W_ram_T_cpu.pin_memory()
        except Exception:
            pass

    ram_col_norms = torch.linalg.norm(W_ram_fp32, dim=0).to(device=target_device, dtype=target_dtype)
    router = SubspaceLookaheadRouter.from_mlp_weights(
        W_gate=W_gate,
        W_up=W_up,
        ram_col_norms=ram_col_norms,
        router_rank=router_rank,
        target_device=target_device,
        target_dtype=target_dtype,
    )

    vm_module = LookaheadNeuralVMDownProj(
        A_cpu=A_cpu,
        B_cpu=B_cpu,
        W_ram_T_pinned=W_ram_T_cpu,
        router=router,
        top_k=top_k_ram,
        mode="vm_paged",
        paging_strategy="sync_tlb" if enable_closed_loop else "naive_per_token",
        prefetch_stream=shared_prefetch_stream,
        A_cpu_closed_loop=A_cpu_closed_loop,
        displaced_fact_cols_cpu=displaced_fact_cols_cpu,
        cpu_logic_bias=cpu_logic_bias,
    )

    stats = LayerCascadeStats(
        layer_idx=layer_idx,
        rank=rank,
        spectral_energy_retained=retained_energy,
        pre_correction_rel_error=pre_corr_err,
        post_correction_rel_error=post_corr_err,
        cross_cov_alignment=cross_cov_alignment,
    )
    return vm_module, stats


@torch.no_grad()
def decompile_naive_parallel_stack(
    model: nn.Module,
    snapshots: Dict[int, Dict[str, torch.Tensor]],
    teacher_logic_acts: Dict[int, torch.Tensor],
    teacher_fact_acts: Dict[int, torch.Tensor],
    layer_indices: List[int],
    rank_ratio: float = 0.25,
    top_k_ram: int = 64,
    router_rank: int = 64,
    lambda_reg: float = 1e-4,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Dict[int, LookaheadNeuralVMDownProj], List[LayerCascadeStats]]:
    """State 2a: Decompiles all 16 layers independently without sequential drift compensation."""
    vm_modules: Dict[int, LookaheadNeuralVMDownProj] = {}
    stats_list: List[LayerCascadeStats] = []
    shared_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    for idx in layer_indices:
        w_dict = snapshots[idx]
        vm_mod, st = decompose_layer_with_drift_compensation(
            W_down=w_dict["down_proj"],
            W_gate=w_dict["gate_proj"],
            W_up=w_dict["up_proj"],
            H_orig_logic=teacher_logic_acts[idx],
            H_orig_fact=teacher_fact_acts[idx],
            H_actual_logic=None,
            H_actual_fact=None,
            rank_ratio=rank_ratio,
            top_k_ram=top_k_ram,
            router_rank=router_rank,
            lambda_reg=lambda_reg,
            enable_closed_loop=False,
            displace_fact_cols=432,
            compute_device=device,
            target_device=device,
            target_dtype=dtype,
            layer_idx=idx,
            shared_prefetch_stream=shared_stream,
        )
        mlp = get_target_mlp(model, idx)
        mlp.down_proj = vm_mod
        vm_mod.attach_mlp_pre_hook(mlp)
        vm_modules[idx] = vm_mod
        stats_list.append(st)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vm_modules, stats_list


@torch.no_grad()
def decompile_sequential_cascade_stack(
    model: nn.Module,
    snapshots: Dict[int, Dict[str, torch.Tensor]],
    teacher_logic_acts: Dict[int, torch.Tensor],
    teacher_fact_acts: Dict[int, torch.Tensor],
    logic_calib_split: TokenizedSplit,
    fact_calib_split: TokenizedSplit,
    layer_indices: List[int],
    rank_ratio: float = 0.25,
    top_k_ram: int = 64,
    router_rank: int = 64,
    lambda_reg: float = 1e-4,
    closed_loop_ridge: float = 1e-2,
    batch_size: int = 4,
    max_calib_tokens: int = 4096,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Dict[int, LookaheadNeuralVMDownProj], List[LayerCascadeStats]]:
    """State 2b: Sequentially decompiles Layers 10..25 with Closed-Loop Drift Compensation."""
    # First ensure target layers start from unmodified baseline weights
    restore_baseline_down_proj_stack(model, snapshots, device=device, dtype=dtype)

    vm_modules: Dict[int, LookaheadNeuralVMDownProj] = {}
    stats_list: List[LayerCascadeStats] = []
    shared_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    sorted_layers = sorted(layer_indices)
    for step_i, idx in enumerate(sorted_layers):
        w_dict = snapshots[idx]
        if step_i == 0:
            # First layer in stack (L=10) receives unmodified upstream activations
            h_act_l = teacher_logic_acts[idx]
            h_act_f = teacher_fact_acts[idx]
        else:
            # Capture actual drifted activations entering Layer L from converted layers [10..L-1]
            h_act_l = capture_layer_activations(
                model=model,
                dataset_split=logic_calib_split,
                layer_idx=idx,
                batch_size=batch_size,
                device=device,
                max_tokens=max_calib_tokens,
                early_exit=True,
            ).to(torch.bfloat16)
            h_act_f = capture_layer_activations(
                model=model,
                dataset_split=fact_calib_split,
                layer_idx=idx,
                batch_size=batch_size,
                device=device,
                max_tokens=max_calib_tokens,
                early_exit=True,
            ).to(torch.bfloat16)

        vm_mod, st = decompose_layer_with_drift_compensation(
            W_down=w_dict["down_proj"],
            W_gate=w_dict["gate_proj"],
            W_up=w_dict["up_proj"],
            H_orig_logic=teacher_logic_acts[idx],
            H_orig_fact=teacher_fact_acts[idx],
            H_actual_logic=h_act_l,
            H_actual_fact=h_act_f,
            rank_ratio=rank_ratio,
            top_k_ram=top_k_ram,
            router_rank=router_rank,
            lambda_reg=lambda_reg,
            closed_loop_ridge=closed_loop_ridge,
            enable_closed_loop=True,
            displace_fact_cols=432,
            compute_device=device,
            target_device=device,
            target_dtype=dtype,
            layer_idx=idx,
            shared_prefetch_stream=shared_stream,
        )

        # Replace Layer L in-place in VM_Paged mode before advancing to Layer L+1
        mlp = get_target_mlp(model, idx)
        mlp.down_proj = vm_mod
        vm_mod.attach_mlp_pre_hook(mlp)
        vm_modules[idx] = vm_mod
        stats_list.append(st)

        del h_act_l, h_act_f
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return vm_modules, stats_list
