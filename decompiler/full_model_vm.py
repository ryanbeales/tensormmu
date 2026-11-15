"""Upgrade 2.2 (Phase A Hardware-Honest): Full-Stack 28-Layer Decompilation, PLE Offloading & Naive Baselines."""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.calibration import TokenizedSplit
from decompiler.full_mlp_svd import (
    FullMLPLayerStats,
    capture_single_layer_mlp_input,
    capture_teacher_full_mlp_traces,
    decompose_full_mlp_layer,
)
from decompiler.hooks import get_target_mlp, resolve_decoder_layers
from runtime.quantized_page_mmu import (
    QuantizedPagedNeuralMLP,
    SharedPageStagingPool,
    quantize_mlp_ram_pages_int8,
)


class DeterministicPLEOffloader(nn.Module):
    """Offloads Gemma 4's 4.48 GB Per-Layer Embedding (`embed_tokens_per_layer`) table to Host RAM.

    Since `input_ids` are known deterministically at Layer 0 entry, performs O(1) Host RAM
    embedding lookup, stages into a compact pinned DMA row buffer, and transfers only active
    token vectors (~17.5 KB/token) over PCIe with zero approximation error (bit-exact).
    """

    def __init__(
        self,
        orig_embedding: nn.Module,
        prefetch_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = orig_embedding.num_embeddings
        self.embedding_dim = orig_embedding.embedding_dim
        self.padding_idx = orig_embedding.padding_idx
        self.prefetch_stream = prefetch_stream

        embed_scale = getattr(orig_embedding, "embed_scale", 1.0)
        if isinstance(embed_scale, torch.Tensor):
            self.register_buffer(
                "embed_scale",
                embed_scale.detach().clone(),
                persistent=False,
            )
        else:
            self.embed_scale = float(embed_scale)

        w_cpu = orig_embedding.weight.detach().to("cpu", dtype=torch.bfloat16).contiguous()
        self.weight_cpu_pinned = w_cpu
        self.offloaded_vram_mb = (w_cpu.numel() * w_cpu.element_size()) / (1024 ** 2)

        # Small pinned row buffer (~17.5 KB) so single-token decode uses true async pinned DMA
        # without exhausting Windows WDDM commit charge on the full 4.48 GB table.
        pinned_row = torch.empty((1, self.embedding_dim), dtype=torch.bfloat16, device="cpu")
        if torch.cuda.is_available():
            try:
                pinned_row = pinned_row.pin_memory()
            except Exception:
                pass
        self._pinned_single_row = pinned_row

        self._single_tok_cache_id: Optional[int] = None
        self._single_tok_cache_vec: Optional[torch.Tensor] = None
        self.reset_telemetry()

    @classmethod
    def from_cpu_tensor(
        cls,
        weight_cpu: torch.Tensor,
        embed_scale: Any = 1.0,
        padding_idx: Optional[int] = None,
        prefetch_stream: Optional[torch.cuda.Stream] = None,
    ) -> "DeterministicPLEOffloader":
        """Constructs `DeterministicPLEOffloader` directly from a CPU or mmap tensor without GPU allocation."""
        inst = cls.__new__(cls)
        nn.Module.__init__(inst)
        inst.num_embeddings, inst.embedding_dim = int(weight_cpu.shape[0]), int(weight_cpu.shape[1])
        inst.padding_idx = padding_idx
        inst.prefetch_stream = prefetch_stream
        if isinstance(embed_scale, torch.Tensor):
            inst.register_buffer(
                "embed_scale",
                embed_scale.detach().clone(),
                persistent=False,
            )
        else:
            inst.embed_scale = float(embed_scale)

        inst.weight_cpu_pinned = weight_cpu
        inst.offloaded_vram_mb = (weight_cpu.numel() * weight_cpu.element_size()) / (1024 ** 2)

        pinned_row = torch.empty((1, inst.embedding_dim), dtype=torch.bfloat16, device="cpu")
        if torch.cuda.is_available():
            try:
                pinned_row = pinned_row.pin_memory()
            except Exception:
                pass
        inst._pinned_single_row = pinned_row
        inst._single_tok_cache_id = None
        inst._single_tok_cache_vec = None
        inst.reset_telemetry()
        return inst

    def reset_telemetry(self) -> None:
        self._total_pcie_bytes: int = 0
        self._total_tokens_looked_up: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._single_tok_cache_id = None
        self._single_tok_cache_vec = None

    def bandwidth_telemetry(self) -> Dict[str, float]:
        toks = max(1, self._total_tokens_looked_up)
        return {
            "ple_total_pcie_mb": self._total_pcie_bytes / (1024 ** 2),
            "ple_pcie_kb_per_token": (self._total_pcie_bytes / 1024.0) / toks,
            "ple_cache_hits": float(self._cache_hits),
            "ple_cache_misses": float(self._cache_misses),
        }

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        target_device = input_ids.device
        n_toks = int(input_ids.numel())
        self._total_tokens_looked_up += n_toks

        if n_toks == 1:
            tok_id = int(input_ids.item())
            if (
                self._single_tok_cache_id == tok_id
                and self._single_tok_cache_vec is not None
                and self._single_tok_cache_vec.device == target_device
            ):
                self._cache_hits += 1
                return self._single_tok_cache_vec

            self._cache_misses += 1
            if self.padding_idx is not None and tok_id == self.padding_idx:
                self._pinned_single_row.zero_()
            else:
                self._pinned_single_row[0].copy_(self.weight_cpu_pinned[tok_id])
            out_gpu = self._pinned_single_row.to(
                device=target_device, dtype=torch.bfloat16, non_blocking=True
            ).view(*input_ids.shape, self.embedding_dim)
            self._total_pcie_bytes += int(
                self._pinned_single_row.numel() * self._pinned_single_row.element_size()
            )
            scaled = out_gpu * self.embed_scale
            self._single_tok_cache_id = tok_id
            self._single_tok_cache_vec = scaled
            return scaled

        ids_cpu = input_ids.to("cpu")
        out_cpu = F.embedding(
            ids_cpu,
            self.weight_cpu_pinned,
            padding_idx=self.padding_idx,
        )
        out_gpu = out_cpu.to(device=target_device, dtype=torch.bfloat16, non_blocking=True)
        self._total_pcie_bytes += int(out_cpu.numel() * out_cpu.element_size())
        self._cache_misses += n_toks
        return out_gpu * self.embed_scale


@torch.no_grad()
def offload_per_layer_embeddings_to_host_ram(
    model: nn.Module,
    prefetch_stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[Optional[DeterministicPLEOffloader], float]:
    """Replaces `language_model.embed_tokens_per_layer` with `DeterministicPLEOffloader` and frees 4.48 GB VRAM."""
    lm = model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
    elif hasattr(model, "language_model"):
        lm = model.language_model

    if not hasattr(lm, "embed_tokens_per_layer") or lm.embed_tokens_per_layer is None:
        return None, 0.0

    orig_ple = lm.embed_tokens_per_layer
    if isinstance(orig_ple, DeterministicPLEOffloader):
        return orig_ple, orig_ple.offloaded_vram_mb

    offloader = DeterministicPLEOffloader(orig_ple, prefetch_stream=prefetch_stream)
    saved_mb = offloader.offloaded_vram_mb
    lm.embed_tokens_per_layer = offloader
    del orig_ple
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return offloader, saved_mb


@torch.no_grad()
def restore_ple_to_gpu(
    model: nn.Module,
    offloader: Optional[DeterministicPLEOffloader],
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Restores `embed_tokens_per_layer` to GPU VRAM for native dense baseline benchmarking."""
    if offloader is None:
        return
    lm = model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
    elif hasattr(model, "language_model"):
        lm = model.language_model

    class _DenseScaledEmbedding(nn.Module):
        def __init__(self, w_gpu: torch.Tensor, scale: Any, pad_idx: Optional[int]) -> None:
            super().__init__()
            self.weight = nn.Parameter(w_gpu, requires_grad=False)
            self.embed_scale = scale.to(device=device, dtype=dtype) if isinstance(scale, torch.Tensor) else scale
            self.padding_idx = pad_idx
            self.num_embeddings, self.embedding_dim = w_gpu.shape

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return F.embedding(input_ids, self.weight, padding_idx=self.padding_idx) * self.embed_scale

    lm.embed_tokens_per_layer = _DenseScaledEmbedding(
        offloader.weight_cpu_pinned.to(device=device, dtype=dtype),
        offloader.embed_scale,
        offloader.padding_idx,
    )


class NaiveUniformLowRankMLP(nn.Module):
    """State 3 Comparison Baseline: Uniform Unwhitened Low-Rank Truncation at Equal Core VRAM Budget."""

    def __init__(
        self,
        A_gate: torch.Tensor,
        B_gate: torch.Tensor,
        A_up: torch.Tensor,
        B_up: torch.Tensor,
        A_down: torch.Tensor,
        B_down: torch.Tensor,
        act_fn: nn.Module,
    ) -> None:
        super().__init__()
        self.A_gate = nn.Parameter(A_gate, requires_grad=False)
        self.B_gate = nn.Parameter(B_gate, requires_grad=False)
        self.A_up = nn.Parameter(A_up, requires_grad=False)
        self.B_up = nn.Parameter(B_up, requires_grad=False)
        self.A_down = nn.Parameter(A_down, requires_grad=False)
        self.B_down = nn.Parameter(B_down, requires_grad=False)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        dt = self.A_gate.dtype
        flat_x = x.reshape(-1, x.shape[-1]).to(dt)
        g = (flat_x @ self.B_gate.T) @ self.A_gate.T
        u = (flat_x @ self.B_up.T) @ self.A_up.T
        h = self.act_fn(g) * u
        y = (h @ self.B_down.T) @ self.A_down.T
        return y.view_as(x).to(orig_dtype)


@torch.no_grad()
def build_naive_uniform_low_rank_stack(
    model: nn.Module,
    snapshots: Dict[int, Dict[str, torch.Tensor]],
    layer_indices: List[int],
    rank_ratio: float = 0.25,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[int, NaiveUniformLowRankMLP]:
    """Builds State 3: Naive Uniform SVD Truncation across all 28 layers at equal core VRAM."""
    decoder_layers = resolve_decoder_layers(model)
    naive_mlps: Dict[int, NaiveUniformLowRankMLP] = {}

    for idx in sorted(layer_indices):
        w_dict = snapshots[idx]
        orig_mlp = get_target_mlp(model, idx)
        act_fn = getattr(orig_mlp, "act_fn", nn.GELU(approximate="tanh"))

        W_g = w_dict["gate_proj"].to(device, torch.float32)  # [d_ffn, d_model]
        W_u = w_dict["up_proj"].to(device, torch.float32)
        W_d = w_dict["down_proj"].to(device, torch.float32)  # [d_model, d_ffn]
        d_model, d_ffn = W_d.shape
        r = max(1, int(rank_ratio * min(d_model, d_ffn)))

        def _naive_split_gu(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            gram = W.T @ W
            gram = 0.5 * (gram + gram.T)
            ev, V = torch.linalg.eigh(gram)
            ev = ev.flip(0).clamp_min(1e-12)
            V = V.flip(1)
            Vr = V[:, :r]
            sr = torch.sqrt(ev[:r])
            A = (W @ Vr) / torch.sqrt(sr).unsqueeze(0)
            B = (Vr * torch.sqrt(sr).unsqueeze(0)).T
            return A.to(device, dtype).contiguous(), B.to(device, dtype).contiguous()

        A_g, B_g = _naive_split_gu(W_g)
        A_u, B_u = _naive_split_gu(W_u)

        gram_d = W_d @ W_d.T
        gram_d = 0.5 * (gram_d + gram_d.T)
        ev_d, U_d = torch.linalg.eigh(gram_d)
        ev_d = ev_d.flip(0).clamp_min(1e-12)
        U_d = U_d.flip(1)
        U_dr = U_d[:, :r]
        s_dr = torch.sqrt(ev_d[:r])
        A_d = (U_dr * torch.sqrt(s_dr).unsqueeze(0)).to(device, dtype).contiguous()
        B_d = ((U_dr / torch.sqrt(s_dr).unsqueeze(0)).T @ W_d).to(device, dtype).contiguous()

        n_mlp = NaiveUniformLowRankMLP(A_g, B_g, A_u, B_u, A_d, B_d, act_fn=act_fn)
        orig_mlp.to("cpu")
        decoder_layers[idx].mlp = n_mlp
        naive_mlps[idx] = n_mlp
        del W_g, W_u, W_d, gram_d, ev_d, U_d, U_dr, s_dr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return naive_mlps


@torch.no_grad()
def decompile_full_28_layer_quantized_stack(
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
) -> Tuple[Dict[int, QuantizedPagedNeuralMLP], List[FullMLPLayerStats]]:
    """Sequentially decompiles all 28 interior Full-MLP blocks (L = 4..31) with Honest INT8 Paged DMA."""
    decoder_layers = resolve_decoder_layers(model)
    quant_mlps: Dict[int, QuantizedPagedNeuralMLP] = {}
    stats_list: List[FullMLPLayerStats] = []
    shared_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    sorted_layers = sorted(layer_indices)
    max_d_ffn = max(int(snapshots[idx]["gate_proj"].shape[0]) for idx in sorted_layers)
    d_model_ref = int(snapshots[sorted_layers[0]]["gate_proj"].shape[1])
    shared_pool = SharedPageStagingPool(
        max_pages_cap=max(1, min(max_pages, num_pages)),
        max_page_size=max_d_ffn // num_pages,
        d_model=d_model_ref,
        device=torch.device(device),
        compute_dtype=dtype,
    )

    for step_i, idx in enumerate(sorted_layers):
        w_dict = snapshots[idx]
        orig_mlp = get_target_mlp(model, idx)
        act_fn = getattr(orig_mlp, "act_fn", nn.GELU(approximate="tanh"))

        # Refresh actual drifted upstream inputs every 2 layers under honest sparse paging
        if step_i == 0 or (step_i % 2 != 0):
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

        base_paged_mlp, st = decompose_full_mlp_layer(
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

        # Quantize BF16 pages into pinned Host INT8 pages + GPU BF16 row scales
        q_pages_int8, page_scales_gpu = quantize_mlp_ram_pages_int8(
            base_paged_mlp.W_mlp_ram_pages,
            target_device=device,
            scale_dtype=dtype,
        )

        # Note: base_paged_mlp.A_down_cpu already holds closed-loop calibrated A_down;
        # pass uncorrected A_down + closed-loop A_down cleanly
        uncorrected_A_down = base_paged_mlp._A_down_uncorrected_cpu.to(device=device, dtype=dtype)
        closed_A_down = base_paged_mlp.A_down_cpu.data

        q_mlp = QuantizedPagedNeuralMLP(
            A_gate_cpu=base_paged_mlp.A_gate_cpu.data,
            B_gate_cpu=base_paged_mlp.B_gate_cpu.data,
            A_up_cpu=base_paged_mlp.A_up_cpu.data,
            B_up_cpu=base_paged_mlp.B_up_cpu.data,
            A_down_cpu=uncorrected_A_down,
            B_down_cpu=base_paged_mlp.B_down_cpu.data,
            W_mlp_ram_pages_bf16=base_paged_mlp.W_mlp_ram_pages,
            Q_mlp_ram_pages_int8=q_pages_int8,
            page_scales_gpu=page_scales_gpu,
            act_fn=act_fn,
            router=base_paged_mlp.router,
            num_pages=num_pages,
            max_pages=max_pages,
            mode="vm_paged",
            paging_strategy="async_int8",
            prefetch_stream=shared_stream,
            A_down_closed_loop=closed_A_down,
            displaced_fact_cols_cpu=base_paged_mlp.displaced_fact_cols_cpu,
            cpu_logic_bias=base_paged_mlp.cpu_logic_bias,
            staging_pool=shared_pool,
            ping_pong_slot=step_i % 2,
        )

        orig_mlp.to("cpu")
        decoder_layers[idx].mlp = q_mlp
        # Attach pre-hook on EVERY layer (zero `_is_stack_entry` bypass)
        q_mlp.attach_decoder_layer_pre_hook(decoder_layers[idx], is_stack_entry=(step_i == 0))
        quant_mlps[idx] = q_mlp
        stats_list.append(st)

        del base_paged_mlp, uncorrected_A_down, x_act_l, x_act_f, orig_mlp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return quant_mlps, stats_list
