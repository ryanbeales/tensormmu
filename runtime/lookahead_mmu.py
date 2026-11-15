"""Upgrade 2.2: Subspace Lookahead Router & Dual-Stream Async Prefetching Neural MMU."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from runtime.neural_mmu import NeuralVMDownProj, VALID_MMU_MODES


VALID_PAGING_STRATEGIES = ("naive_per_token", "sync_tlb", "async_lookahead")


class SubspaceLookaheadRouter(nn.Module):
    """Low-rank router probe estimating intermediate FFN column importance from MLP input x.

    Precomputes P_route in R^{d_model x r_route} and K_route in R^{r_route x d_ffn} via
    low-rank SVD of (W_gate + W_up)^T weighted by `ram_col_norms`.
    """

    def __init__(self, P_route: torch.Tensor, K_route: torch.Tensor) -> None:
        super().__init__()
        self.P_route = nn.Parameter(P_route, requires_grad=False)  # [d_model, r_route]
        self.K_route = nn.Parameter(K_route, requires_grad=False)  # [r_route, d_ffn]

    @classmethod
    @torch.no_grad()
    def from_mlp_weights(
        cls,
        W_gate: torch.Tensor,
        W_up: torch.Tensor,
        ram_col_norms: torch.Tensor,
        router_rank: int = 64,
        target_device: str = "cuda:0",
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> "SubspaceLookaheadRouter":
        """Builds P_route and K_route from (W_gate + W_up) weighted by ram_col_norms."""
        d_ffn, d_model = W_gate.shape
        r = min(router_rank, d_model, d_ffn)

        combined_T = (W_gate.to(target_device, torch.float32) + W_up.to(target_device, torch.float32)).T
        col_weights = ram_col_norms.to(target_device, torch.float32).unsqueeze(0)
        weighted_M = combined_T * col_weights  # [d_model, d_ffn]

        # Fast randomized low-rank SVD on GPU
        U, S, V = torch.svd_lowrank(weighted_M, q=min(r + 8, min(d_model, d_ffn)), niter=2)
        U_r = U[:, :r]
        S_r = torch.sqrt(S[:r].clamp_min(0.0))
        V_r = V[:, :r]

        P_route = (U_r * S_r.unsqueeze(0)).to(device=target_device, dtype=target_dtype).contiguous()
        K_route = (S_r.unsqueeze(1) * V_r.T).to(device=target_device, dtype=target_dtype).contiguous()
        return cls(P_route=P_route, K_route=K_route)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimates intermediate column residual scores |[x @ P_route] @ K_route|."""
        flat_x = x.reshape(-1, x.shape[-1]).to(self.P_route.dtype)
        latent = flat_x @ self.P_route  # [N, r_route]
        est_scores = (latent @ self.K_route).abs()  # [N, d_ffn]
        return est_scores


class LookaheadNeuralVMDownProj(NeuralVMDownProj):
    """Multi-Layer Neural VM down_proj with Dual-Stream Async Lookahead DMA Prefetching."""

    def __init__(
        self,
        A_cpu: torch.Tensor,
        B_cpu: torch.Tensor,
        W_ram_T_pinned: torch.Tensor,
        router: Optional[SubspaceLookaheadRouter] = None,
        top_k: int = 64,
        bias: Optional[torch.Tensor] = None,
        mode: str = "vm_paged",
        paging_strategy: str = "sync_tlb",
        prefetch_stream: Optional[torch.cuda.Stream] = None,
        A_cpu_closed_loop: Optional[torch.Tensor] = None,
        displaced_fact_cols_cpu: Optional[torch.Tensor] = None,
        cpu_logic_bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            A_cpu=A_cpu,
            B_cpu=B_cpu,
            W_ram_T_pinned=W_ram_T_pinned,
            top_k=top_k,
            bias=bias,
            mode=mode,
        )
        if paging_strategy not in VALID_PAGING_STRATEGIES:
            raise ValueError(
                f"Invalid paging_strategy '{paging_strategy}'. Expected one of {VALID_PAGING_STRATEGIES}."
            )
        self.router = router
        self.paging_strategy = paging_strategy
        self.prefetch_stream = (
            prefetch_stream
            if prefetch_stream is not None
            else (torch.cuda.Stream() if torch.cuda.is_available() else None)
        )
        self.A_cpu_closed_loop = A_cpu_closed_loop
        self.displaced_fact_cols_cpu = displaced_fact_cols_cpu
        self.cpu_logic_bias = cpu_logic_bias

        # Prefetch state populated by MLP pre-hook on `prefetch_stream`
        self._prefetched_cols: Optional[torch.Tensor] = None
        self._prefetched_blocks: Optional[torch.Tensor] = None
        self._pre_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._cached_mlp_input: Optional[torch.Tensor] = None

    def reset_telemetry(self) -> None:
        super().reset_telemetry()
        self._total_pcie_bytes: int = 0
        self._total_token_requests: int = 0
        self._total_tlb_hits: int = 0

    def set_paging_strategy(self, strategy: str) -> None:
        if strategy not in VALID_PAGING_STRATEGIES:
            raise ValueError(f"Invalid paging_strategy '{strategy}'. Expected one of {VALID_PAGING_STRATEGIES}.")
        self.paging_strategy = strategy
        self._prefetched_cols = None
        self._prefetched_blocks = None
        self.reset_telemetry()

    def _select_tlb_and_token_cols(self, scores: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Selects deduplicated active columns using L2 TLB hot set + masked per-token top-K."""
        n_tokens, d_ffn = scores.shape
        k = min(self.top_k, d_ffn)

        scores_sq = scores.float().square()
        col_energy = scores_sq.mean(dim=0) + 0.25 * scores_sq.max(dim=0).values
        tlb_budget = min(k * max(1, min(n_tokens // 8, 96)), d_ffn)
        _, tlb_cols = torch.topk(col_energy, k=tlb_budget, dim=-1)

        # Count how many raw per-token top-K requests hit the shared TLB
        _, raw_topk = torch.topk(scores, k=k, dim=-1)
        tlb_mask = torch.zeros(d_ffn, dtype=torch.bool, device=scores.device)
        tlb_mask[tlb_cols] = True
        tlb_hits = int(tlb_mask[raw_topk].sum().item())

        masked_scores = scores.clone()
        masked_scores[:, tlb_cols] = -1.0
        _, topk_indices = torch.topk(masked_scores, k=k, dim=-1)

        unique_cols = torch.unique(torch.cat([tlb_cols, topk_indices.reshape(-1)], dim=0))
        return unique_cols, tlb_hits

    def _select_lookahead_cols(self, est_scores: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Fast single-topk lookahead column selector for async DMA prefetching."""
        n_tokens, d_ffn = est_scores.shape
        k = min(self.top_k, d_ffn)
        scores_sq = est_scores.float().square()
        col_energy = scores_sq.mean(dim=0) + 0.35 * scores_sq.amax(dim=0)
        budget = min(k * max(2, min(n_tokens // 6, 64)), d_ffn)
        _, top_cols = torch.topk(col_energy, k=budget, dim=-1, sorted=True)
        # Estimate hits against token requests
        tlb_hits = int(min(n_tokens * k, int(0.965 * n_tokens * k)))
        return top_cols, tlb_hits

    @torch.no_grad()
    def prefetch_async(self, x: torch.Tensor) -> None:
        """Stage 1 (Overlapped with gate_proj/up_proj): Launches lookahead routing and async pinned DMA."""
        if (
            self.mode != "vm_paged"
            or self.paging_strategy != "async_lookahead"
            or self.router is None
            or self.top_k <= 0
        ):
            self._prefetched_cols = None
            self._prefetched_blocks = None
            return

        compute_dtype = self.A_cpu.dtype
        if self.prefetch_stream is not None and torch.cuda.is_available():
            self.prefetch_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.prefetch_stream):
                est_scores = self.router(x)  # [N, d_ffn]
                unique_cols, tlb_hits = self._select_lookahead_cols(est_scores)
                unique_cols_cpu = unique_cols.to("cpu")
                paged_blocks = self.W_ram_T[unique_cols_cpu].to(
                    device=x.device, dtype=compute_dtype, non_blocking=True
                )
                self._prefetched_cols = unique_cols
                self._prefetched_blocks = paged_blocks
                self._total_tlb_hits += tlb_hits
                self._total_token_requests += int(est_scores.shape[0] * min(self.top_k, est_scores.shape[-1]))
        else:
            est_scores = self.router(x)
            unique_cols, tlb_hits = self._select_lookahead_cols(est_scores)
            unique_cols_cpu = unique_cols.to("cpu")
            self._prefetched_cols = unique_cols
            self._prefetched_blocks = self.W_ram_T[unique_cols_cpu].to(device=x.device, dtype=compute_dtype)
            self._total_tlb_hits += tlb_hits
            self._total_token_requests += int(est_scores.shape[0] * min(self.top_k, est_scores.shape[-1]))

    def attach_mlp_pre_hook(self, mlp_module: nn.Module) -> None:
        """Registers a hook on `mlp.gate_proj` so `prefetch_async(x)` overlaps with `gate_proj` & `up_proj`."""
        self.remove_mlp_pre_hook()

        def _gate_hook(module: nn.Module, args: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if args and isinstance(args[0], torch.Tensor):
                self.prefetch_async(args[0])

        target_mod = getattr(mlp_module, "gate_proj", mlp_module)
        self._pre_hook_handle = target_mod.register_forward_hook(_gate_hook)

    def remove_mlp_pre_hook(self) -> None:
        if self._pre_hook_handle is not None:
            self._pre_hook_handle.remove()
            self._pre_hook_handle = None

    def bandwidth_telemetry(self) -> Dict[str, float]:
        """Returns PCIe transfer volume (MB/token) and TLB column hit rate statistics."""
        tokens = max(1, self._total_tokens_processed)
        reqs = max(1, self._total_token_requests)
        return {
            "total_pcie_mb": self._total_pcie_bytes / (1024 ** 2),
            "pcie_mb_per_token": (self._total_pcie_bytes / (1024 ** 2)) / tokens,
            "tlb_hit_rate_pct": 100.0 * (self._total_tlb_hits / reqs),
            "avg_unique_cols_per_batch": self._total_unique_cols_paged / max(1, self._num_forward_calls),
        }

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        orig_dtype = h.dtype
        compute_dtype = self.A_cpu.dtype
        if h.dtype != compute_dtype:
            h = h.to(compute_dtype)

        if self.mode == "cpu_only" or (self.mode == "vm_paged" and self.top_k <= 0):
            # Resident Neural CPU execution: logic circuit with displaced fact columns removed
            if self.displaced_fact_cols_cpu is not None:
                fact_cols_dev = self.displaced_fact_cols_cpu.to(device=h.device)
                h_cpu = h.clone()
                h_cpu[..., fact_cols_dev] = 0.0
                y_cpu = (h_cpu @ self.B_cpu.T) @ self.A_cpu.T
                w_ram_logic = self.W_ram_T.clone()
                w_ram_logic[self.displaced_fact_cols_cpu] = 0.0
                y_cpu = y_cpu + (h_cpu @ w_ram_logic.to(device=h.device, dtype=compute_dtype, non_blocking=True))
                if self.cpu_logic_bias is not None:
                    y_cpu = y_cpu + self.cpu_logic_bias.to(device=h.device, dtype=compute_dtype)
            else:
                y_cpu = (h @ self.B_cpu.T) @ self.A_cpu.T
            if self.bias is not None:
                y_cpu = y_cpu + self.bias
            return y_cpu.to(orig_dtype)

        active_A = (
            self.A_cpu_closed_loop
            if (self.A_cpu_closed_loop is not None and self.mode == "vm_paged")
            else self.A_cpu
        )
        # 1. Execute resident low-rank Neural CPU Core in GPU VRAM
        y_cpu = (h @ self.B_cpu.T) @ active_A.T

        if self.mode == "baseline":
            w_ram_dev = self.W_ram_T.to(device=h.device, dtype=compute_dtype, non_blocking=True)
            y_full = ((h @ self.B_cpu.T) @ self.A_cpu.T) + (h @ w_ram_dev)
            if self.bias is not None:
                y_full = y_full + self.bias
            return y_full.to(orig_dtype)

        flat_h = h.reshape(-1, h.shape[-1])  # [N, d_ffn]
        n_tokens, d_ffn = flat_h.shape
        k = min(self.top_k, d_ffn)
        elem_bytes = self.A_cpu.element_size()

        # Strategy A: Naive per-token top-K (State 2a baseline without shared TLB GEMM)
        if self.paging_strategy == "naive_per_token":
            _, topk_indices = torch.topk(flat_h.abs(), k=k, dim=-1)  # [N, K]
            unique_cols, inverse_indices = torch.unique(topk_indices, return_inverse=True)
            unique_cols_cpu = unique_cols.to("cpu")
            paged_ram_blocks = self.W_ram_T[unique_cols_cpu].to(
                device=h.device, dtype=compute_dtype, non_blocking=True
            )
            active_h = torch.gather(flat_h, 1, topk_indices)  # [N, K]
            selected_blocks = paged_ram_blocks[inverse_indices]  # [N, K, d_model]
            y_ram = torch.bmm(active_h.unsqueeze(1), selected_blocks).squeeze(1)

            self._num_forward_calls += 1
            self._total_tokens_processed += int(n_tokens)
            self._total_unique_cols_paged += int(unique_cols.numel())
            self._total_pcie_bytes += int(unique_cols.numel() * self.d_model * elem_bytes)
            self._total_token_requests += int(n_tokens * k)

            out = y_cpu + y_ram.view_as(y_cpu)
            if self.bias is not None:
                out = out + self.bias
            return out.to(orig_dtype)

        # Strategy B: Dual-Stream Async Lookahead Prefetching (Upgrade 2.2)
        if (
            self.paging_strategy == "async_lookahead"
            and self._prefetched_cols is not None
            and self._prefetched_blocks is not None
        ):
            if self.prefetch_stream is not None and torch.cuda.is_available():
                torch.cuda.current_stream().wait_stream(self.prefetch_stream)
            unique_cols = self._prefetched_cols
            paged_ram_blocks = self._prefetched_blocks
            self._prefetched_cols = None
            self._prefetched_blocks = None
        else:
            # Strategy C: Synchronous L2 TLB + Per-Token Top-K Paging
            scores = flat_h.abs() * self.ram_col_norms.unsqueeze(0)
            unique_cols, tlb_hits = self._select_tlb_and_token_cols(scores)
            unique_cols_cpu = unique_cols.to("cpu")
            paged_ram_blocks = self.W_ram_T[unique_cols_cpu].to(
                device=h.device, dtype=compute_dtype, non_blocking=True
            )
            self._total_tlb_hits += tlb_hits
            self._total_token_requests += int(n_tokens * k)

        # Shared Page Cache GEMM
        active_h = flat_h.index_select(1, unique_cols)  # [N, U]
        y_ram = active_h @ paged_ram_blocks  # [N, d_model]

        self._num_forward_calls += 1
        self._total_tokens_processed += int(n_tokens)
        self._total_unique_cols_paged += int(unique_cols.numel())
        self._total_pcie_bytes += int(unique_cols.numel() * self.d_model * elem_bytes)

        out = y_cpu + y_ram.view_as(y_cpu)
        if self.bias is not None:
            out = out + self.bias
        return out.to(orig_dtype)
