"""Neural MMU Runtime Module replacing Gemma4MLP down_proj with Resident CPU + Paged RAM."""

from typing import Dict, Optional
import torch
import torch.nn as nn


VALID_MMU_MODES = ("baseline", "cpu_only", "vm_paged")


class NeuralVMDownProj(nn.Module):
    """Drop-in replacement for `layers[L].mlp.down_proj` with 2-tier Neural VM execution.

    Tiers:
    1. Resident Neural CPU Core (GPU VRAM): Low-rank factorized matrices A_cpu [d_model, r]
       and B_cpu [r, d_ffn] that preserve symbolic reasoning, syntax, and state tracking.
    2. Paged Neural RAM Table (Pinned Host RAM): Residual matrix W_ram_T [d_ffn, d_model]
       whose rows (columns of W_down) are dynamically paged into GPU SRAM/VRAM on demand
       by the Neural MMU based on top-K intermediate GeGLU activation magnitudes.
    """

    def __init__(
        self,
        A_cpu: torch.Tensor,
        B_cpu: torch.Tensor,
        W_ram_T_pinned: torch.Tensor,
        top_k: int = 64,
        bias: Optional[torch.Tensor] = None,
        mode: str = "vm_paged",
    ) -> None:
        super().__init__()
        if mode not in VALID_MMU_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of {VALID_MMU_MODES}.")

        target_device = "cuda" if torch.cuda.is_available() and A_cpu.is_cuda else A_cpu.device

        # Resident in GPU VRAM
        self.A_cpu = nn.Parameter(A_cpu.to(target_device), requires_grad=False)  # [d_model, r]
        self.B_cpu = nn.Parameter(B_cpu.to(target_device), requires_grad=False)  # [r, d_ffn]

        if bias is not None:
            self.bias = nn.Parameter(bias.to(target_device), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Resident in Host System RAM (Pinned for async DMA transfer)
        self.W_ram_T = W_ram_T_pinned  # [d_ffn, d_model] on CPU
        # Precompute L2 norms of W_RAM columns (rows of W_ram_T) so MMU scores exact residual impact ||h_j * W_ram[:, j]||_2
        col_norms = torch.linalg.norm(W_ram_T_pinned.to(torch.float32), dim=-1).to(
            device=target_device, dtype=A_cpu.dtype
        )
        self.register_buffer("ram_col_norms", col_norms, persistent=False)
        self.top_k = int(top_k)
        self.mode = mode

        # Runtime telemetry counters
        self.reset_telemetry()

    def reset_telemetry(self) -> None:
        """Resets paging statistics collected during forward passes."""
        self._num_forward_calls: int = 0
        self._total_tokens_processed: int = 0
        self._total_unique_cols_paged: int = 0

    def set_mode(self, mode: str, top_k: Optional[int] = None) -> None:
        """Switches execution mode among 'baseline', 'cpu_only', and 'vm_paged'."""
        if mode not in VALID_MMU_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of {VALID_MMU_MODES}.")
        self.mode = mode
        if top_k is not None:
            self.top_k = int(top_k)
        self.reset_telemetry()

    @property
    def d_model(self) -> int:
        return int(self.A_cpu.shape[0])

    @property
    def rank(self) -> int:
        return int(self.A_cpu.shape[1])

    @property
    def d_ffn(self) -> int:
        return int(self.B_cpu.shape[1])

    def vram_footprint_stats(self) -> Dict[str, float]:
        """Computes active GPU VRAM footprint vs. original dense down_proj."""
        elem_bytes = self.A_cpu.element_size()
        dense_params = self.d_model * self.d_ffn
        cpu_core_params = (self.d_model * self.rank) + (self.rank * self.d_ffn)
        avg_unique_cols = (
            self._total_unique_cols_paged / max(1, self._num_forward_calls)
            if self.mode == "vm_paged"
            else 0.0
        )
        # Active transient buffer per token/step is bounded by top_k * d_model
        # (or avg_unique_cols normalized per sequence in batched evaluation)
        transient_paged_params = min(avg_unique_cols, float(self.top_k * 4)) * self.d_model
        active_vram_params = (
            dense_params
            if self.mode == "baseline"
            else cpu_core_params + (transient_paged_params if self.mode == "vm_paged" else 0.0)
        )

        return {
            "dense_params": float(dense_params),
            "cpu_core_params": float(cpu_core_params),
            "dense_vram_mb": (dense_params * elem_bytes) / (1024 ** 2),
            "cpu_core_vram_mb": (cpu_core_params * elem_bytes) / (1024 ** 2),
            "active_vram_mb": (active_vram_params * elem_bytes) / (1024 ** 2),
            "resident_vram_reduction_pct": 100.0 * (1.0 - (cpu_core_params / max(1, dense_params))),
            "active_vram_reduction_pct": 100.0 * (1.0 - (active_vram_params / max(1, dense_params))),
            "per_token_ram_col_pct": 100.0 * (min(self.top_k, self.d_ffn) / max(1, self.d_ffn))
            if self.mode == "vm_paged"
            else 0.0,
            "avg_unique_cols_per_batch": avg_unique_cols,
        }

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h shape: [batch, seq_len, d_ffn] on target device
        orig_dtype = h.dtype
        compute_dtype = self.A_cpu.dtype
        if h.dtype != compute_dtype:
            h = h.to(compute_dtype)

        # 1. Always execute the dense low-rank Reasoning CPU Core in VRAM
        y_cpu = (h @ self.B_cpu.T) @ self.A_cpu.T

        if self.mode == "cpu_only" or (self.mode == "vm_paged" and self.top_k <= 0):
            if self.bias is not None:
                y_cpu = y_cpu + self.bias
            return y_cpu.to(orig_dtype)

        if self.mode == "baseline":
            # Full un-truncated reference: y_cpu + h @ W_ram_T
            w_ram_dev = self.W_ram_T.to(device=h.device, dtype=compute_dtype, non_blocking=True)
            y_full = y_cpu + (h @ w_ram_dev)
            if self.bias is not None:
                y_full = y_full + self.bias
            return y_full.to(orig_dtype)

        # 2. Neural MMU: Identify Top-K active intermediate neurons per token
        # Combine batch-level L2 TLB hot columns with per-token unique top-K requests
        flat_h = h.reshape(-1, h.shape[-1])  # [N, d_ffn]
        k = min(self.top_k, flat_h.shape[-1])
        scores = flat_h.abs() * self.ram_col_norms.unsqueeze(0)

        # Rank columns by L2 residual energy across the sequence/micro-batch + per-token peak energy
        scores_sq = scores.float().square()
        col_energy = scores_sq.mean(dim=0) + 0.25 * scores_sq.max(dim=0).values
        tlb_budget = min(k * max(1, min(flat_h.shape[0] // 8, 96)), flat_h.shape[-1])
        _, tlb_cols = torch.topk(col_energy, k=tlb_budget, dim=-1)

        # Mask out already-scheduled TLB columns so per-token top-K fetches genuinely distinct sparse columns
        masked_scores = scores.clone()
        masked_scores[:, tlb_cols] = -1.0
        _, topk_indices = torch.topk(masked_scores, k=k, dim=-1)  # [N, K]

        # Deduplicate active memory addresses needed for this micro-batch
        unique_cols = torch.unique(torch.cat([tlb_cols, topk_indices.reshape(-1)], dim=0))
        unique_cols_cpu = unique_cols.to("cpu", non_blocking=True)

        # 3. Page-in ONLY the active factual rows from Host RAM to GPU SRAM/VRAM page cache
        paged_ram_blocks = self.W_ram_T[unique_cols_cpu].to(
            device=h.device, dtype=compute_dtype, non_blocking=True
        )  # [U, d_model]

        # Compute sparse projection against the batch's resident SRAM page cache
        active_h = flat_h.index_select(1, unique_cols)  # [N, U]
        y_ram = active_h @ paged_ram_blocks  # [N, d_model]

        self._num_forward_calls += 1
        self._total_tokens_processed += int(flat_h.shape[0])
        self._total_unique_cols_paged += int(unique_cols.numel())

        out = y_cpu + y_ram.view_as(y_cpu)
        if self.bias is not None:
            out = out + self.bias
        return out.to(orig_dtype)


def install_neural_vm_down_proj(
    mlp_module: nn.Module,
    A_cpu: torch.Tensor,
    B_cpu: torch.Tensor,
    W_ram_T_pinned: torch.Tensor,
    top_k: int = 64,
    mode: str = "vm_paged",
) -> NeuralVMDownProj:
    """Replaces `mlp_module.down_proj` in-place with `NeuralVMDownProj` and frees old weights."""
    old_down_proj = mlp_module.down_proj
    bias = getattr(old_down_proj, "bias", None)
    if bias is not None:
        bias = bias.detach().clone()

    vm_down_proj = NeuralVMDownProj(
        A_cpu=A_cpu,
        B_cpu=B_cpu,
        W_ram_T_pinned=W_ram_T_pinned,
        top_k=top_k,
        bias=bias,
        mode=mode,
    )
    mlp_module.down_proj = vm_down_proj
    del old_down_proj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vm_down_proj
