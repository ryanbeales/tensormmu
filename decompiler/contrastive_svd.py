"""Contrastive Activation-Aware SVD for disentangling W_down into W_CPU and W_RAM."""

from dataclasses import dataclass
from typing import Dict, Optional
import torch


@dataclass
class ContrastiveSVDResult:
    """Holds decomposed Neural CPU & Neural RAM tensors and spectral diagnostics."""

    A_cpu: torch.Tensor  # [d_model, r] in target_dtype on target_device
    B_cpu: torch.Tensor  # [r, d_ffn] in target_dtype on target_device
    W_ram_T_pinned: torch.Tensor  # [d_ffn, d_model] in target_dtype on CPU (pinned if CUDA available)
    singular_values: torch.Tensor  # [min(d_model, d_ffn)] in FP32 on CPU
    rank: int
    d_model: int
    d_ffn: int
    spectral_energy_retained: float  # sum(S[:r]^2) / sum(S^2)
    logic_recon_rel_error: float  # ||H_logic (W_down - W_CPU)^T||_F / ||H_logic W_down^T||_F
    fact_residual_energy_ratio: float  # ||H_fact W_RAM^T||_F / ||H_fact W_down^T||_F
    weight_frobenius_cpu_ratio: float  # ||W_CPU||_F / ||W_down||_F
    weight_frobenius_ram_ratio: float  # ||W_RAM||_F / ||W_down||_F
    contrastive_diag_ratio_stats: Dict[str, float]


def compute_covariance_matrix(
    H: torch.Tensor,
    lambda_reg: float = 1e-4,
    compute_device: str = "cuda:0",
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Computes regularised empirical second-moment covariance matrix in FP32:

    Sigma = (1 / N) * H^T @ H + lambda_reg * I
    """
    n_tokens, d_ffn = H.shape
    if n_tokens == 0:
        raise ValueError("Activation matrix H must contain at least 1 token row.")

    sigma = torch.zeros((d_ffn, d_ffn), dtype=torch.float32, device=compute_device)
    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        h_chunk = H[start:end].to(device=compute_device, dtype=torch.float32)
        sigma.addmm_(h_chunk.T, h_chunk)

    sigma.div_(float(n_tokens))
    # Enforce exact symmetry before adding diagonal regularization
    sigma = 0.5 * (sigma + sigma.T)
    diag_indices = torch.arange(d_ffn, device=compute_device)
    sigma[diag_indices, diag_indices] += lambda_reg
    return sigma


def _safe_cholesky(matrix: torch.Tensor, base_reg: float = 1e-4) -> torch.Tensor:
    """Computes lower-triangular Cholesky factor L such that matrix = L @ L^T."""
    matrix = 0.5 * (matrix + matrix.T)
    L, info = torch.linalg.cholesky_ex(matrix)
    if int(info.item()) == 0:
        return L

    # Promote to FP64 where 1e-4 regularization is far above machine epsilon (1e-16)
    matrix_f64 = matrix.to(torch.float64)
    L64, info64 = torch.linalg.cholesky_ex(matrix_f64)
    if int(info64.item()) == 0:
        return L64.to(matrix.dtype)

    d = matrix.shape[0]
    mean_diag = float(torch.diagonal(matrix_f64).abs().mean().clamp_min(1.0).item())
    eye = torch.eye(d, dtype=torch.float64, device=matrix.device)
    for scale in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        jitter = max(base_reg, mean_diag * scale)
        L64, info64 = torch.linalg.cholesky_ex(matrix_f64 + jitter * eye)
        if int(info64.item()) == 0:
            return L64.to(matrix.dtype)

    return torch.linalg.cholesky(matrix_f64 + (mean_diag * 0.1) * eye).to(matrix.dtype)


@torch.no_grad()
def decompose_down_proj_contrastive(
    W_down: torch.Tensor,
    H_logic: torch.Tensor,
    H_fact: Optional[torch.Tensor] = None,
    rank_ratio: float = 0.25,
    lambda_reg: float = 1e-4,
    use_contrastive_ratio: bool = True,
    contrastive_clip_min: float = 0.1,
    contrastive_clip_max: float = 10.0,
    compute_device: str = "cuda:0",
    target_device: str = "cuda:0",
    target_dtype: torch.dtype = torch.bfloat16,
) -> ContrastiveSVDResult:
    """Executes the Contrastive Activation-Aware SVD pipeline in FP32.

    Steps:
    1. Compute Sigma_logic and Sigma_fact in FP32 with lambda_reg = 1e-4.
    2. Optionally weight Sigma_logic by the diagonal contrastive ratio
       diag(Sigma_logic) / diag(Sigma_fact) to suppress trivia-specific neurons,
       then compute Cholesky factor L_logic (Sigma_logic = L_logic @ L_logic^T).
    3. Project W_down into whitened Cholesky space and compute SVD:
       U, S, Vh = torch.linalg.svd(W_down @ L_logic, full_matrices=False)
    4. Truncate to rank r = floor(rank_ratio * min(d_model, d_ffn)):
       A_CPU = U[:, :r] * sqrt(S[:r])
       B_CPU = (sqrt(S[:r]) * Vh[:r, :]) @ L_logic^{-1}
       W_CPU = A_CPU @ B_CPU
    5. Extract residual W_RAM = W_down - W_CPU, transpose to [d_ffn, d_model],
       and pin in CPU host memory in bfloat16.
    """
    if not torch.cuda.is_available():
        compute_device = "cpu"
        target_device = "cpu"

    d_model, d_ffn = W_down.shape
    max_rank = min(d_model, d_ffn)
    rank = max(1, int(rank_ratio * max_rank))

    # 1. Compute Covariance Matrices in FP32
    sigma_logic = compute_covariance_matrix(
        H_logic, lambda_reg=lambda_reg, compute_device=compute_device
    )

    contrastive_stats: Dict[str, float] = {
        "min_ratio": 1.0,
        "median_ratio": 1.0,
        "max_ratio": 1.0,
        "mean_ratio": 1.0,
    }

    if H_fact is not None and use_contrastive_ratio:
        sigma_fact = compute_covariance_matrix(
            H_fact, lambda_reg=lambda_reg, compute_device=compute_device
        )
        diag_logic = torch.diagonal(sigma_logic).clamp_min(lambda_reg)
        diag_fact = torch.diagonal(sigma_fact).clamp_min(lambda_reg)

        raw_ratio = diag_logic / diag_fact
        contrastive_stats = {
            "min_ratio": float(raw_ratio.min().item()),
            "median_ratio": float(raw_ratio.median().item()),
            "max_ratio": float(raw_ratio.max().item()),
            "mean_ratio": float(raw_ratio.mean().item()),
        }

        # Clamp and normalize ratio so mean scale is preserved while suppressing pure-fact channels
        clamped_ratio = torch.clamp(raw_ratio, min=contrastive_clip_min, max=contrastive_clip_max)
        clamped_ratio = clamped_ratio / clamped_ratio.mean().clamp_min(1e-6)
        scale_vec = torch.sqrt(clamped_ratio)

        # Symmetrically weight Sigma_logic: D^{1/2} Sigma_logic D^{1/2} + lambda_reg I
        sigma_whitening = sigma_logic * scale_vec.unsqueeze(1) * scale_vec.unsqueeze(0)
        idx = torch.arange(d_ffn, device=compute_device)
        sigma_whitening[idx, idx] += lambda_reg
        del sigma_fact
    else:
        sigma_whitening = sigma_logic

    # 2. Contrastive Whitening (Cholesky factor L_logic)
    L_logic = _safe_cholesky(sigma_whitening, base_reg=lambda_reg)
    del sigma_whitening

    # 3. SVD in Whitened Activation Space: W_down @ L_logic
    W_down_fp32 = W_down.to(device=compute_device, dtype=torch.float32)
    whitened_W = W_down_fp32 @ L_logic
    U, S, Vh = torch.linalg.svd(whitened_W, full_matrices=False)
    del whitened_W

    # 4. Truncate to Rank r (The Resident CPU Core)
    U_r = U[:, :rank]  # [d_model, r]
    S_r = S[:rank].clamp_min(0.0)  # [r]
    Vh_r = Vh[:rank, :]  # [r, d_ffn]
    sqrt_S_r = torch.sqrt(S_r)  # [r]

    # A_CPU = U[:, :r] @ diag(sqrt(S[:r])) -> [d_model, r]
    A_cpu_fp32 = U_r * sqrt_S_r.unsqueeze(0)

    # B_CPU = diag(sqrt(S[:r])) @ V[:r, :]^T @ L_logic^{-1} -> [r, d_ffn]
    # Solve B_CPU @ L_logic = sqrt_S_Vh using triangular solve for numerical precision
    sqrt_S_Vh = Vh_r * sqrt_S_r.unsqueeze(1)  # [r, d_ffn]
    B_cpu_fp32 = torch.linalg.solve_triangular(
        L_logic, sqrt_S_Vh, upper=False, left=False
    )

    # Reconstruct W_CPU and extract System RAM Residual W_RAM
    W_cpu_fp32 = A_cpu_fp32 @ B_cpu_fp32  # [d_model, d_ffn]
    W_ram_fp32 = W_down_fp32 - W_cpu_fp32  # [d_model, d_ffn]

    # Compute spectral and activation-space diagnostics before freeing FP32 tensors
    total_spectral_energy = float(torch.sum(S * S).clamp_min(1e-12).item())
    retained_spectral_energy = float(torch.sum(S_r * S_r).item())
    spectral_energy_retained = retained_spectral_energy / total_spectral_energy

    w_down_norm = float(torch.linalg.norm(W_down_fp32, ord="fro").clamp_min(1e-12).item())
    w_cpu_norm = float(torch.linalg.norm(W_cpu_fp32, ord="fro").item())
    w_ram_norm = float(torch.linalg.norm(W_ram_fp32, ord="fro").item())

    # Evaluate reconstruction error on a sample of H_logic and H_fact
    sample_n = min(1024, H_logic.shape[0])
    h_logic_sample = H_logic[:sample_n].to(device=compute_device, dtype=torch.float32)
    y_logic_full = h_logic_sample @ W_down_fp32.T
    y_logic_ram = h_logic_sample @ W_ram_fp32.T
    logic_recon_rel_error = float(
        (torch.linalg.norm(y_logic_ram) / torch.linalg.norm(y_logic_full).clamp_min(1e-12)).item()
    )

    if H_fact is not None and H_fact.shape[0] > 0:
        sample_nf = min(1024, H_fact.shape[0])
        h_fact_sample = H_fact[:sample_nf].to(device=compute_device, dtype=torch.float32)
        y_fact_full = h_fact_sample @ W_down_fp32.T
        y_fact_ram = h_fact_sample @ W_ram_fp32.T
        fact_residual_energy_ratio = float(
            (torch.linalg.norm(y_fact_ram) / torch.linalg.norm(y_fact_full).clamp_min(1e-12)).item()
        )
    else:
        fact_residual_energy_ratio = 0.0

    # 5. Move A_CPU, B_CPU to target GPU in target_dtype, and pin W_ram_T in host RAM
    A_cpu = A_cpu_fp32.to(device=target_device, dtype=target_dtype).contiguous()
    B_cpu = B_cpu_fp32.to(device=target_device, dtype=target_dtype).contiguous()

    W_ram_T_cpu = W_ram_fp32.T.to(device="cpu", dtype=target_dtype).contiguous()
    if torch.cuda.is_available():
        try:
            W_ram_T_cpu = W_ram_T_cpu.pin_memory()
        except Exception:
            pass

    S_cpu = S.detach().to(device="cpu", dtype=torch.float32)

    return ContrastiveSVDResult(
        A_cpu=A_cpu,
        B_cpu=B_cpu,
        W_ram_T_pinned=W_ram_T_cpu,
        singular_values=S_cpu,
        rank=rank,
        d_model=d_model,
        d_ffn=d_ffn,
        spectral_energy_retained=spectral_energy_retained,
        logic_recon_rel_error=logic_recon_rel_error,
        fact_residual_energy_ratio=fact_residual_energy_ratio,
        weight_frobenius_cpu_ratio=w_cpu_norm / w_down_norm,
        weight_frobenius_ram_ratio=w_ram_norm / w_down_norm,
        contrastive_diag_ratio_stats=contrastive_stats,
    )
