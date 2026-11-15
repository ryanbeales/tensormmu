"""Upgrade 2.2 (Phase A Hardware-Honest): Co-Activation Clustered Knowledge Pages & Tri-Matrix Paged MLP MMU."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_MLP_MMU_MODES = ("baseline", "cpu_only", "vm_paged")
VALID_MLP_PAGING_STRATEGIES = ("sync_clustered", "async_lookahead")


class ClusteredPageRouter(nn.Module):
    """Scores P=64 contiguous Knowledge Pages from pre-MLP or pre-Attention hidden states x.

    Uses unit-normalized input cosine similarity against page centroid keys W_page_route in R^{d_model x P}
    plus a calibrated page residual energy prior to dynamically select strictly `max_pages` (e.g. 4 / 64 = 6.25%)
    active pages per sequence without cross-batch contamination.
    """

    def __init__(
        self,
        W_page_route: torch.Tensor,
        page_prior: torch.Tensor,
        num_pages: int = 64,
        max_pages: int = 4,
        prior_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.W_page_route = nn.Parameter(W_page_route, requires_grad=False)  # [d_model, P]
        self.page_prior = nn.Parameter(page_prior, requires_grad=False)  # [P]
        self.num_pages = int(num_pages)
        self.max_pages = int(max_pages)
        self.prior_weight = float(prior_weight)

    @classmethod
    @torch.no_grad()
    def from_clustered_mlp_weights(
        cls,
        W_gate_perm: torch.Tensor,
        W_up_perm: torch.Tensor,
        down_ram_col_norms_perm: torch.Tensor,
        page_prior_scores: torch.Tensor,
        num_pages: int = 64,
        max_pages: int = 4,
        target_device: str = "cuda:0",
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> "ClusteredPageRouter":
        """Constructs page router keys from residual-norm-weighted (W_gate + W_up) directions per page."""
        d_ffn, d_model = W_gate_perm.shape
        page_size = d_ffn // num_pages

        g_fp32 = F.normalize(W_gate_perm.to(target_device, torch.float32), dim=-1)
        u_fp32 = F.normalize(W_up_perm.to(target_device, torch.float32), dim=-1)
        combined = g_fp32 + u_fp32  # [d_ffn, d_model]
        weights = down_ram_col_norms_perm.to(target_device, torch.float32).unsqueeze(1)
        weighted = (combined * weights).reshape(num_pages, page_size, d_model)

        # Page principal direction in R^{d_model} via top-1 singular vector / weighted centroid
        page_dirs = weighted.mean(dim=1)  # [P, d_model]
        page_keys = F.normalize(page_dirs, dim=1).T  # [d_model, P]

        prior = page_prior_scores.to(target_device, torch.float32)
        prior = prior / prior.abs().amax().clamp_min(1e-6)

        return cls(
            W_page_route=page_keys.to(device=target_device, dtype=target_dtype).contiguous(),
            page_prior=prior.to(device=target_device, dtype=target_dtype).contiguous(),
            num_pages=num_pages,
            max_pages=max_pages,
        )

    @torch.no_grad()
    def select_pages_per_seq(
        self,
        x: torch.Tensor,
        max_pages_override: Optional[int] = None,
    ) -> torch.Tensor:
        """Selects strictly `k_pages <= max_pages` page indices per sequence: returns [B, k_pages]."""
        k_pages = min(
            max_pages_override if max_pages_override is not None else self.max_pages,
            self.num_pages,
        )
        if x.dim() == 3:
            bsz, seq_len, d_model = x.shape
            x_3d = x
        else:
            x_3d = x.reshape(1, -1, x.shape[-1])
            bsz = 1

        # Unit-normalize token hidden states so pre-attention and post-RMSNorm inputs route consistently
        x_norm = F.normalize(x_3d.float(), dim=-1).to(self.W_page_route.dtype)  # [B, S, d_model]
        proj = torch.matmul(x_norm, self.W_page_route).float()  # [B, S, P]
        dyn_score = proj.square().mean(dim=1) + 0.35 * proj.abs().amax(dim=1)  # [B, P]
        dyn_max = dyn_score.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        dyn_norm = dyn_score / dyn_max
        total_score = dyn_norm + self.prior_weight * self.page_prior.float().unsqueeze(0)  # [B, P]
        _, top_pages = torch.topk(total_score, k=k_pages, dim=-1, sorted=True)  # [B, k_pages]
        return top_pages

    @torch.no_grad()
    def select_pages(self, x: torch.Tensor, max_pages_override: Optional[int] = None) -> torch.Tensor:
        """Selects strictly `k_pages <= max_pages` page indices in [0, P-1] for a single sequence/batch."""
        k_pages = min(
            max_pages_override if max_pages_override is not None else self.max_pages,
            self.num_pages,
        )
        # Fast path for single-token decode (S == 1): 1 GEMV + topk instead of 9 sequence-pooling ops
        if x.numel() == self.W_page_route.shape[0]:
            v = x.reshape(1, -1).to(self.W_page_route.dtype)
            proj_abs = (v @ self.W_page_route).abs().squeeze(0).float()  # [P]
            dyn_norm = proj_abs / proj_abs.amax().clamp_min(1e-6)
            scores = dyn_norm + self.prior_weight * self.page_prior.float()
            _, top_pages = torch.topk(scores, k=k_pages, dim=-1, sorted=True)
            return top_pages

        if x.dim() == 3 and x.shape[0] == 1:
            return self.select_pages_per_seq(x, max_pages_override=max_pages_override).squeeze(0)
        flat_x = x.reshape(1, -1, x.shape[-1])
        return self.select_pages_per_seq(flat_x, max_pages_override=max_pages_override).squeeze(0)


class PagedNeuralMLP(nn.Module):
    """Hardware-Honest Full-MLP Neural VM replacing gate_proj, up_proj, and down_proj.

    - Resident GPU VRAM Core:
        A_gate_cpu [d_ffn, r], B_gate_cpu [r, d_model]
        A_up_cpu   [d_ffn, r], B_up_cpu   [r, d_model]
        A_down_cpu [d_model, r], B_down_cpu [r, d_ffn]
    - Pinned Host RAM Clustered Pages (`P = 64` pages, `C = d_ffn // 64` neurons/page):
        W_mlp_ram_pages: [P, 3, C, d_model] in CPU pinned memory
    - Strict Page Budget (`max_pages = 4`, 6.25% of W_RAM):
        Transfers and applies strictly `<= max_pages` pages per sequence for BOTH `n_tokens == 1`
        and `n_tokens > 1` (zero dense fallback).
    """

    def __init__(
        self,
        A_gate_cpu: torch.Tensor,
        B_gate_cpu: torch.Tensor,
        A_up_cpu: torch.Tensor,
        B_up_cpu: torch.Tensor,
        A_down_cpu: torch.Tensor,
        B_down_cpu: torch.Tensor,
        W_mlp_ram_pages_pinned: torch.Tensor,
        act_fn: nn.Module,
        router: ClusteredPageRouter,
        num_pages: int = 64,
        max_pages: int = 4,
        mode: str = "vm_paged",
        paging_strategy: str = "sync_clustered",
        prefetch_stream: Optional[torch.cuda.Stream] = None,
        A_down_closed_loop: Optional[torch.Tensor] = None,
        displaced_fact_cols_cpu: Optional[torch.Tensor] = None,
        cpu_logic_bias: Optional[torch.Tensor] = None,
        W_down_base_pinned: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if mode not in VALID_MLP_MMU_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of {VALID_MLP_MMU_MODES}.")
        if paging_strategy not in VALID_MLP_PAGING_STRATEGIES:
            raise ValueError(
                f"Invalid paging_strategy '{paging_strategy}'. Expected one of {VALID_MLP_PAGING_STRATEGIES}."
            )

        self.d_ffn, self.rank = A_gate_cpu.shape
        self.d_model = B_gate_cpu.shape[1]
        self.num_pages = int(num_pages)
        self.page_size = self.d_ffn // self.num_pages
        self.max_pages = int(max_pages)
        self.mode = mode
        self.paging_strategy = paging_strategy
        self.act_fn = act_fn
        self.router = router
        self.prefetch_stream = (
            prefetch_stream
            if prefetch_stream is not None
            else (torch.cuda.Stream() if torch.cuda.is_available() else None)
        )

        # Store uncorrected A_down on CPU only for exact Mode 0 ('baseline') verification
        self._A_down_uncorrected_cpu = A_down_cpu.detach().to("cpu")
        active_A_down = A_down_closed_loop if A_down_closed_loop is not None else A_down_cpu

        # Single deduplicated copy of resident GPU VRAM Low-Rank Cores
        self.A_gate_cpu = nn.Parameter(A_gate_cpu, requires_grad=False)
        self.B_gate_cpu = nn.Parameter(B_gate_cpu, requires_grad=False)
        self.A_up_cpu = nn.Parameter(A_up_cpu, requires_grad=False)
        self.B_up_cpu = nn.Parameter(B_up_cpu, requires_grad=False)
        self.A_down_cpu = nn.Parameter(active_A_down, requires_grad=False)
        self.B_down_cpu = nn.Parameter(B_down_cpu, requires_grad=False)

        self.register_buffer(
            "_page_arange",
            torch.arange(self.page_size, device=A_gate_cpu.device, dtype=torch.long),
            persistent=False,
        )

        # Host System RAM Pinned Pages: [P, 3, C, d_model]
        self.W_mlp_ram_pages = W_mlp_ram_pages_pinned
        self.A_down_closed_loop = None  # Folded into self.A_down_cpu to eliminate duplicate GPU allocation
        self.displaced_fact_cols_cpu = displaced_fact_cols_cpu
        self.cpu_logic_bias = cpu_logic_bias.detach().to("cpu") if cpu_logic_bias is not None else None
        self.W_down_base_pinned = W_down_base_pinned

        # Async Lookahead Prefetch state
        self._prefetched_pages_gpu: Optional[torch.Tensor] = None
        self._prefetched_blocks_gpu: Optional[torch.Tensor] = None
        self._prefetch_ready_event: Optional[torch.cuda.Event] = (
            torch.cuda.Event() if torch.cuda.is_available() else None
        )
        self._layer_pre_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        self.reset_telemetry()

    def reset_telemetry(self) -> None:
        self._num_forward_calls: int = 0
        self._total_tokens_processed: int = 0
        self._total_pages_paged: int = 0
        self._total_unique_cols_paged: int = 0
        self._total_pcie_bytes: int = 0
        self._page_hist = torch.zeros(self.num_pages, dtype=torch.long)

    def set_mode(self, mode: str, max_pages: Optional[int] = None) -> None:
        if mode not in VALID_MLP_MMU_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of {VALID_MLP_MMU_MODES}.")
        self.mode = mode
        if max_pages is not None:
            self.max_pages = int(max_pages)
        self._prefetched_pages_gpu = None
        self._prefetched_blocks_gpu = None
        self.reset_telemetry()

    def set_paging_strategy(self, strategy: str) -> None:
        if strategy not in VALID_MLP_PAGING_STRATEGIES:
            raise ValueError(
                f"Invalid paging_strategy '{strategy}'. Expected one of {VALID_MLP_PAGING_STRATEGIES}."
            )
        self.paging_strategy = strategy
        self._prefetched_pages_gpu = None
        self._prefetched_blocks_gpu = None
        self.reset_telemetry()

    def _transfer_pages_from_pinned_cpu(
        self,
        page_ids: list,
        target_device: torch.device,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Copies `len(page_ids)` contiguous pinned slices from Host RAM to GPU via zero-copy views."""
        k_p = len(page_ids)
        dst = torch.empty(
            (k_p, 3, self.page_size, self.d_model),
            dtype=compute_dtype,
            device=target_device,
        )
        bytes_copied = 0
        for i, pid in enumerate(page_ids):
            src_view = self.W_mlp_ram_pages[int(pid)]  # Zero-copy pinned view [3, C, d_model]
            dst[i].copy_(src_view, non_blocking=True)
            bytes_copied += int(src_view.numel() * src_view.element_size())
        self._total_pcie_bytes += bytes_copied
        return dst

    @torch.no_grad()
    def prefetch_pages_async(self, x: torch.Tensor) -> None:
        """At Decoder Layer L Entry: Dynamically routes pages and launches non-blocking pinned DMA."""
        if (
            self.mode != "vm_paged"
            or self.paging_strategy != "async_lookahead"
            or self.max_pages <= 0
            or (x.dim() == 3 and x.shape[0] > 1)
        ):
            self._prefetched_pages_gpu = None
            self._prefetched_blocks_gpu = None
            return

        compute_dtype = self.A_gate_cpu.dtype
        top_pages = self.router.select_pages(x, max_pages_override=self.max_pages)
        page_ids = top_pages.tolist()
        for pid in page_ids:
            self._page_hist[int(pid)] += 1

        if self.prefetch_stream is not None and torch.cuda.is_available():
            with torch.cuda.stream(self.prefetch_stream):
                blocks_gpu = self._transfer_pages_from_pinned_cpu(page_ids, x.device, compute_dtype)
                if self._prefetch_ready_event is not None:
                    self._prefetch_ready_event.record(self.prefetch_stream)
            self._prefetched_pages_gpu = top_pages
            self._prefetched_blocks_gpu = blocks_gpu
        else:
            self._prefetched_pages_gpu = top_pages
            self._prefetched_blocks_gpu = self._transfer_pages_from_pinned_cpu(
                page_ids, x.device, compute_dtype
            )

    def attach_decoder_layer_pre_hook(self, decoder_layer: nn.Module) -> None:
        """Attaches pre-hook at Decoder Layer L entry so DMA overlaps with Self-Attention."""
        self.remove_decoder_layer_pre_hook()

        def _layer_pre_hook(module: nn.Module, args: Tuple[torch.Tensor, ...]) -> None:
            if args and isinstance(args[0], torch.Tensor):
                self.prefetch_pages_async(args[0])

        self._layer_pre_hook_handle = decoder_layer.register_forward_pre_hook(_layer_pre_hook)

    def remove_decoder_layer_pre_hook(self) -> None:
        if self._layer_pre_hook_handle is not None:
            self._layer_pre_hook_handle.remove()
            self._layer_pre_hook_handle = None

    def vram_footprint_stats(self) -> Dict[str, float]:
        """Returns exact GPU-resident bytes + transient page buffer bytes across gate, up, and down."""
        elem_size = self.A_gate_cpu.element_size()
        dense_bytes = 3 * self.d_ffn * self.d_model * elem_size
        resident_gpu_bytes = sum(
            t.numel() * t.element_size()
            for t in list(self.parameters()) + list(self.buffers())
            if t.is_cuda
        )
        paged_cols = min(self.max_pages, self.num_pages) * self.page_size if self.mode == "vm_paged" else 0
        transient_page_bytes = 3 * paged_cols * self.d_model * elem_size

        if self.mode == "baseline":
            active_bytes = dense_bytes
        elif self.mode == "cpu_only":
            active_bytes = resident_gpu_bytes
        else:
            active_bytes = resident_gpu_bytes + transient_page_bytes

        mb = float(1024 ** 2)
        return {
            "dense_vram_mb": dense_bytes / mb,
            "cpu_core_vram_mb": resident_gpu_bytes / mb,
            "transient_page_vram_mb": transient_page_bytes / mb,
            "active_vram_mb": active_bytes / mb,
            "vram_reduction_pct": 100.0 * (1.0 - (active_bytes / max(1, dense_bytes))),
        }

    def bandwidth_telemetry(self) -> Dict[str, float]:
        calls = max(1, self._num_forward_calls)
        tokens = max(1, self._total_tokens_processed)
        avg_cols = self._total_unique_cols_paged / calls
        distinct_pages_used = int((self._page_hist > 0).sum().item())
        return {
            "total_pcie_mb": self._total_pcie_bytes / (1024 ** 2),
            "pcie_mb_per_token": (self._total_pcie_bytes / (1024 ** 2)) / tokens,
            "avg_pages_paged_per_batch": self._total_pages_paged / calls,
            "avg_unique_cols_per_batch": avg_cols,
            "active_ram_fraction_pct": 100.0 * (avg_cols / max(1, self.d_ffn)),
            "distinct_pages_routed": float(distinct_pages_used),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        compute_dtype = self.A_gate_cpu.dtype
        if x.dtype != compute_dtype:
            x = x.to(compute_dtype)

        is_3d = x.dim() == 3
        bsz = x.shape[0] if is_3d else 1
        seq_len = x.shape[1] if is_3d else x.shape[0]
        flat_x = x.reshape(-1, self.d_model)  # [N, d_model]
        n_tokens = flat_x.shape[0]

        # ---------------------------------------------------------------------
        # Mode 0: Baseline (Exact Dense Verification Check: W_CPU + all 64 W_RAM pages)
        # ---------------------------------------------------------------------
        if self.mode == "baseline":
            g_cpu = (flat_x @ self.B_gate_cpu.T) @ self.A_gate_cpu.T
            u_cpu = (flat_x @ self.B_up_cpu.T) @ self.A_up_cpu.T
            ram_dev = self.W_mlp_ram_pages.to(device=x.device, dtype=compute_dtype, non_blocking=True)
            g_ram = flat_x @ ram_dev[:, 0].reshape(self.d_ffn, self.d_model).T
            u_ram = flat_x @ ram_dev[:, 1].reshape(self.d_ffn, self.d_model).T
            h = self.act_fn(g_cpu + g_ram) * (u_cpu + u_ram)
            a_down_base = self._A_down_uncorrected_cpu.to(device=x.device, dtype=compute_dtype)
            y_cpu = (h @ self.B_down_cpu.T) @ a_down_base.T
            y_ram = h @ ram_dev[:, 2].reshape(self.d_ffn, self.d_model)
            return (y_cpu + y_ram).view_as(x).to(orig_dtype)

        # ---------------------------------------------------------------------
        # Mode 1: Honest Resident CPU_Only (Strictly GPU W_CPU Core — Zero W_RAM Access)
        # ---------------------------------------------------------------------
        if self.mode == "cpu_only" or (self.mode == "vm_paged" and self.max_pages <= 0):
            g_cpu = (flat_x @ self.B_gate_cpu.T) @ self.A_gate_cpu.T
            u_cpu = (flat_x @ self.B_up_cpu.T) @ self.A_up_cpu.T
            h_cpu = self.act_fn(g_cpu) * u_cpu
            y_out = (h_cpu @ self.B_down_cpu.T) @ self.A_down_cpu.T
            return y_out.view_as(x).to(orig_dtype)

        # ---------------------------------------------------------------------
        # Mode 2: Honest Clustered VM_Paged (`max_pages = 4` / 6.25% Cap, Zero Dense Fallback)
        # ---------------------------------------------------------------------
        g_out = (flat_x @ self.B_gate_cpu.T) @ self.A_gate_cpu.T  # [N, d_ffn]
        u_out = (flat_x @ self.B_up_cpu.T) @ self.A_up_cpu.T  # [N, d_ffn]

        if bsz == 1:
            if (
                self.paging_strategy == "async_lookahead"
                and self._prefetched_pages_gpu is not None
                and self._prefetched_blocks_gpu is not None
            ):
                if self._prefetch_ready_event is not None and torch.cuda.is_available():
                    torch.cuda.current_stream().wait_event(self._prefetch_ready_event)
                top_pages = self._prefetched_pages_gpu
                paged_blocks = self._prefetched_blocks_gpu
                self._prefetched_pages_gpu = None
                self._prefetched_blocks_gpu = None
            else:
                top_pages = self.router.select_pages(x, max_pages_override=self.max_pages)
                page_ids = top_pages.tolist()
                for pid in page_ids:
                    self._page_hist[int(pid)] += 1
                paged_blocks = self._transfer_pages_from_pinned_cpu(page_ids, x.device, compute_dtype)

            k_pages = int(top_pages.numel())
            paged_cols = k_pages * self.page_size
            active_cols = (
                (top_pages.unsqueeze(1) * self.page_size) + self._page_arange.unsqueeze(0)
            ).reshape(-1)

            g_page_w = paged_blocks[:, 0].reshape(paged_cols, self.d_model)
            u_page_w = paged_blocks[:, 1].reshape(paged_cols, self.d_model)
            d_page_w = paged_blocks[:, 2].reshape(paged_cols, self.d_model)

            g_out.index_add_(1, active_cols, flat_x @ g_page_w.T)
            u_out.index_add_(1, active_cols, flat_x @ u_page_w.T)

            h = self.act_fn(g_out) * u_out
            y_down = (h @ self.B_down_cpu.T) @ self.A_down_cpu.T
            y_down.addmm_(h.index_select(1, active_cols), d_page_w)

            pages_count = k_pages
            cols_count = paged_cols
        else:
            # Multi-sequence batch: route strictly `max_pages` per sequence independently
            x_3d = flat_x.view(bsz, seq_len, self.d_model)
            top_pages_seq = self.router.select_pages_per_seq(x_3d, max_pages_override=self.max_pages)  # [B, K_p]
            k_pages = int(top_pages_seq.shape[1])
            paged_cols = k_pages * self.page_size

            unique_pages, inv_map = torch.unique(top_pages_seq, sorted=True, return_inverse=True)
            unique_list = unique_pages.tolist()
            for pid in top_pages_seq.reshape(-1).tolist():
                self._page_hist[int(pid)] += 1
            unique_blocks = self._transfer_pages_from_pinned_cpu(unique_list, x.device, compute_dtype)

            g_3d = g_out.view(bsz, seq_len, self.d_ffn)
            u_3d = u_out.view(bsz, seq_len, self.d_ffn)
            active_cols_per_seq = []
            d_weights_per_seq = []

            for b in range(bsz):
                seq_blocks = unique_blocks[inv_map[b]]  # [K_p, 3, C, d_model]
                cols_b = (
                    (top_pages_seq[b].unsqueeze(1) * self.page_size) + self._page_arange.unsqueeze(0)
                ).reshape(-1)
                active_cols_per_seq.append(cols_b)
                gw_b = seq_blocks[:, 0].reshape(paged_cols, self.d_model)
                uw_b = seq_blocks[:, 1].reshape(paged_cols, self.d_model)
                dw_b = seq_blocks[:, 2].reshape(paged_cols, self.d_model)
                d_weights_per_seq.append(dw_b)
                g_3d[b].index_add_(1, cols_b, x_3d[b] @ gw_b.T)
                u_3d[b].index_add_(1, cols_b, x_3d[b] @ uw_b.T)

            h = self.act_fn(g_out) * u_out
            y_down = (h @ self.B_down_cpu.T) @ self.A_down_cpu.T
            h_3d = h.view(bsz, seq_len, self.d_ffn)
            y_3d = y_down.view(bsz, seq_len, self.d_model)
            for b in range(bsz):
                y_3d[b].addmm_(h_3d[b].index_select(1, active_cols_per_seq[b]), d_weights_per_seq[b])

            pages_count = k_pages
            cols_count = paged_cols

        self._num_forward_calls += 1
        self._total_tokens_processed += int(n_tokens)
        self._total_pages_paged += int(pages_count)
        self._total_unique_cols_paged += int(cols_count)

        return y_down.view_as(x).to(orig_dtype)
