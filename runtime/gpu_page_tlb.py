"""Phase B Runtime: Fused Bit-Tiered Resident Core + Shared Batched GPU VRAM LRU Page Cache (TLB).

Key Hardware Optimizations:
  1. Fused `gate + up` (`gu`) projections across Tier 1 (`INT8` Hot Core), Tier 0 (`3-Bit Base8-int16`
     Cold Skeleton), and `UniformAWQ4BitMLP`, cutting CUDA kernel launches by ~45% and doubling GEMV occupancy.
  2. `SharedBatchTLBController` for `speculative_tlb` mode: all 28 layers write their requested page IDs
     into a single shared `[num_layers, decode_req_pages]` GPU tensor with ZERO per-layer `.item()` or
     `synchronize()` stalls. At the final layer, one 448-byte async D2H copy stages requests to pinned CPU
     memory during `lm_head` execution, and Layer 0 of the next token drains misses in a single pass.
  3. Physical GPU TLB buffer release/re-allocation between `core_only` and `vm_paged` modes so
     `torch.cuda.memory_allocated()` measures exact hardware bytes in every state.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from runtime.phase_c_kernels import (
    MAX_FUSED_DECODE_TOKENS,
    PhaseCLayerLauncher,
    detect_act_kind,
)

VALID_PHASE_B_MODES = ("core_only", "vm_paged")
VALID_TLB_STRATEGIES = ("sync_tlb", "speculative_tlb", "amortized_spec_tlb", "stateless_dma")


def choose_base8_group_size(d_in: int) -> int:
    """Selects a fine-grained group size divisible by 5 (~65-85) that minimizes padding for `d_in`."""
    for g in (65, 80, 85, 75, 70, 60, 90):
        if d_in % g == 0:
            return g
    best_g = 65
    best_pad = (65 - (d_in % 65)) % 65
    for g in (60, 65, 70, 75, 80, 85):
        pad = (g - (d_in % g)) % g
        if pad < best_pad:
            best_pad = pad
            best_g = g
    return best_g


class SharedQuantLUTPool(nn.Module):
    """Holds shared L1/L2-cached BF16 lookup tables across all layers (~322 KB total GPU VRAM)."""

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        idx8 = torch.arange(32768, dtype=torch.int32, device=device)
        lut_base8 = torch.stack(
            [
                (idx8 & 7).to(dtype),
                ((idx8 >> 3) & 7).to(dtype),
                ((idx8 >> 6) & 7).to(dtype),
                ((idx8 >> 9) & 7).to(dtype),
                ((idx8 >> 12) & 7).to(dtype),
            ],
            dim=-1,
        ).contiguous()

        idx4 = torch.arange(256, dtype=torch.int32, device=device)
        lut_int4 = torch.stack(
            [((idx4 & 0x0F) - 8).to(dtype), (((idx4 >> 4) & 0x0F) - 8).to(dtype)], dim=-1
        ).contiguous()
        lut_uint4 = torch.stack(
            [(idx4 & 0x0F).to(dtype), ((idx4 >> 4) & 0x0F).to(dtype)], dim=-1
        ).contiguous()

        self.register_buffer("lut_base8", lut_base8, persistent=False)
        self.register_buffer("lut_int4", lut_int4, persistent=False)
        self.register_buffer("lut_uint4", lut_uint4, persistent=False)

    @property
    def total_vram_bytes(self) -> int:
        return int(
            self.lut_base8.numel() * self.lut_base8.element_size()
            + self.lut_int4.numel() * self.lut_int4.element_size()
            + self.lut_uint4.numel() * self.lut_uint4.element_size()
        )


@torch.no_grad()
def quantize_and_pack_3bit_base8(
    w_fp32: torch.Tensor,
    s_in: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    target_device: str = "cuda:0",
    scale_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Quantizes `[rows, cols]` to asymmetric 3-bit and packs 5 codes per `torch.int16` (`0.40 B/w`)."""
    rows, cols = w_fp32.shape
    g = group_size if group_size is not None else choose_base8_group_size(cols)
    if g % 5 != 0:
        raise ValueError(f"group_size ({g}) must be divisible by 5.")

    w_work = w_fp32 * s_in.unsqueeze(0) if s_in is not None else w_fp32
    pad_cols = (g - (cols % g)) % g
    w_pad = F.pad(w_work, (0, pad_cols), value=0.0) if pad_cols > 0 else w_work
    d_pad = w_pad.shape[1]
    n_groups = d_pad // g

    wg = w_pad.view(rows, n_groups, g)
    mn0 = wg.amin(dim=-1, keepdim=True)
    mx0 = wg.amax(dim=-1, keepdim=True)
    mid = 0.5 * (mn0 + mx0)
    half0 = 0.5 * (mx0 - mn0)

    # Optimal MSE clipping search across 3-bit grid (prevents single outlier weight from wasting 8-level range)
    best_mse = torch.full((rows, n_groups, 1), float("inf"), device=w_fp32.device, dtype=torch.float32)
    wmin_bf = mn0.to(scale_dtype)
    scale_bf = ((mx0 - mn0) / 7.0).clamp_min(1e-8).to(scale_dtype)
    for clip_alpha in (0.84, 0.89, 0.93, 0.97, 1.00):
        cand_mn = (mid - half0 * clip_alpha).to(scale_dtype)
        cand_sc = ((2.0 * half0 * clip_alpha) / 7.0).clamp_min(1e-8).to(scale_dtype)
        cand_q = torch.clamp(torch.round((wg - cand_mn.float()) / cand_sc.float()), 0.0, 7.0)
        cand_dq = (cand_q.to(scale_dtype) * cand_sc + cand_mn).float()
        cand_mse = (wg - cand_dq).square().mean(dim=-1, keepdim=True)
        better = cand_mse < best_mse
        best_mse = torch.where(better, cand_mse, best_mse)
        wmin_bf = torch.where(better, cand_mn, wmin_bf)
        scale_bf = torch.where(better, cand_sc, scale_bf)

    q = torch.clamp(torch.round((wg - wmin_bf.float()) / scale_bf.float()), 0.0, 7.0).to(torch.int32)
    q5 = q.view(rows, d_pad // 5, 5)
    packed_i32 = (
        q5[..., 0]
        | (q5[..., 1] << 3)
        | (q5[..., 2] << 6)
        | (q5[..., 3] << 9)
        | (q5[..., 4] << 12)
    )
    packed_i16_gpu = packed_i32.to(device=target_device, dtype=torch.int16).contiguous()
    wmin_gpu = wmin_bf.to(device=target_device).contiguous()
    scale_gpu = scale_bf.to(device=target_device).contiguous()

    deq_pad_fp32 = (q.to(scale_dtype) * scale_bf + wmin_bf).float().view(rows, d_pad)[:, :cols]
    deq_unscaled_fp32 = (deq_pad_fp32 / s_in.unsqueeze(0)) if s_in is not None else deq_pad_fp32
    return packed_i16_gpu, wmin_gpu, scale_gpu, deq_unscaled_fp32, g


def unpack_3bit_base8_to_bf16(
    packed_i16: torch.Tensor,
    wmin: torch.Tensor,
    scale: torch.Tensor,
    lut_base8: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    rows = packed_i16.shape[0]
    n_groups = wmin.shape[1]
    d_pad = n_groups * group_size
    q_bf16 = F.embedding(packed_i16.to(torch.int32), lut_base8).view(rows, n_groups, group_size)
    return (q_bf16 * scale + wmin).view(rows, d_pad)


@torch.no_grad()
def quantize_int8_rowwise(
    w_fp32: torch.Tensor,
    s_in: Optional[torch.Tensor] = None,
    target_device: str = "cuda:0",
    scale_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    w_work = w_fp32 * s_in.unsqueeze(0) if s_in is not None else w_fp32
    sc_bf = (w_work.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-8).to(scale_dtype)
    q_i8 = torch.clamp(torch.round(w_work / sc_bf.float()), -127.0, 127.0).to(torch.int8)
    return q_i8.to(device=target_device).contiguous(), sc_bf.to(device=target_device).contiguous()


@torch.no_grad()
def quantize_and_pack_int4_pages_pinned(
    dw_pages_fp32: torch.Tensor,
    s_in_gu: Optional[torch.Tensor] = None,
    target_device: str = "cuda:0",
    scale_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Packs `[P, 3, psize, d_model]` (`0=gate, 1=up, 2=down_T`) into pinned `uint8` INT4 pages."""
    P, three, psize, d_model = dw_pages_fp32.shape
    assert three == 3
    work = dw_pages_fp32.clone()
    if s_in_gu is not None:
        s_view = s_in_gu.view(1, 1, d_model)
        work[:, 0] *= s_view
        work[:, 1] *= s_view

    if d_model % 2 != 0:
        work = F.pad(work, (0, 1), value=0.0)
    d_even = work.shape[-1]

    sc_bf = (work.abs().amax(dim=-1, keepdim=True) / 7.5).clamp_min(1e-8).to(scale_dtype)
    q_s4 = torch.clamp(torch.round(work / sc_bf.float()), -8.0, 7.0).to(torch.int32)
    u4 = (q_s4 + 8).clamp(0, 15).to(torch.uint8)
    u4_pair = u4.view(P, 3, psize, d_even // 2, 2)
    packed_u8 = (u4_pair[..., 0] | (u4_pair[..., 1] << 4)).to("cpu").contiguous()

    if torch.cuda.is_available():
        try:
            packed_u8 = packed_u8.pin_memory()
        except Exception:
            pass

    scales_gpu = sc_bf.to(device=target_device).contiguous()
    return packed_u8, scales_gpu


def unpack_int4_pages_to_bf16(
    packed_u8: torch.Tensor,
    scales: torch.Tensor,
    lut_int4: torch.Tensor,
    d_model: int,
) -> torch.Tensor:
    prefix = packed_u8.shape[:-1]
    d_even = packed_u8.shape[-1] * 2
    deq = F.embedding(packed_u8.to(torch.int32), lut_int4).view(*prefix, d_even) * scales
    return deq[..., :d_model]


class GPUPageTLBCache(nn.Module):
    """Persistent GPU VRAM LRU Page Cache (TLB) for Cold Specialist Precision-Upgrade Pages."""

    def __init__(
        self,
        num_pages: int,
        tlb_slots: int,
        page_size: int,
        d_model: int,
        q4_pages_cpu_pinned: torch.Tensor,
        page_scales_gpu: torch.Tensor,
        device: torch.device,
        scale_dtype: torch.dtype = torch.bfloat16,
        dma_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.num_pages = int(num_pages)
        self.tlb_slots = int(min(tlb_slots, num_pages))
        self.page_size = int(page_size)
        self.d_model = int(d_model)
        self.d_even_half = (self.d_model + 1) // 2
        self.device = device
        self.scale_dtype = scale_dtype
        self.dma_stream = dma_stream

        self.q4_pages_cpu_pinned = q4_pages_cpu_pinned
        self.bytes_per_page = int(
            q4_pages_cpu_pinned[0].numel() * q4_pages_cpu_pinned.element_size()
        )

        self.register_buffer("page_scales_gpu", page_scales_gpu, persistent=False)
        self.register_buffer(
            "tlb_packed_q4",
            torch.zeros(
                (self.tlb_slots, 3, self.page_size, self.d_even_half),
                dtype=torch.uint8,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "tlb_scales",
            torch.zeros((self.tlb_slots, 3, self.page_size, 1), dtype=scale_dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "tlb_col_indices",
            torch.zeros((self.tlb_slots * self.page_size,), dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer(
            "gpu_page_in_tlb",
            torch.zeros((self.num_pages,), dtype=torch.bool, device=device),
            persistent=False,
        )

        self._dma_event: Optional[torch.cuda.Event] = (
            torch.cuda.Event() if (torch.cuda.is_available() and device.type == "cuda") else None
        )
        self._arange_psize = torch.arange(self.page_size, dtype=torch.long, device=device)
        self._all_page_col_indices = (
            torch.arange(self.num_pages, dtype=torch.long, device=device).unsqueeze(1) * self.page_size
            + self._arange_psize.unsqueeze(0)
        ).contiguous()
        self._host_is_pinned: bool = bool(
            torch.cuda.is_available()
            and device.type == "cuda"
            and q4_pages_cpu_pinned.device.type == "cpu"
            and q4_pages_cpu_pinned.is_pinned()
        )
        self._is_allocated = True
        self.clear_cache()
        self.reset_telemetry()

    def release_gpu_slots(self) -> None:
        """Physically frees GPU TLB buffers in `core_only` mode so CUDA memory reflects exact Core bytes."""
        if not self._is_allocated:
            return
        self.clear_cache()
        self.tlb_packed_q4 = torch.empty((0,), dtype=torch.uint8, device=self.device)
        self.tlb_scales = torch.empty((0,), dtype=self.scale_dtype, device=self.device)
        self.tlb_col_indices = torch.empty((0,), dtype=torch.long, device=self.device)
        self._is_allocated = False

    def allocate_gpu_slots(self) -> None:
        """Re-allocates GPU TLB buffers when entering `vm_paged` mode."""
        if self._is_allocated:
            return
        self.tlb_packed_q4 = torch.zeros(
            (self.tlb_slots, 3, self.page_size, self.d_even_half),
            dtype=torch.uint8,
            device=self.device,
        )
        self.tlb_scales = torch.zeros(
            (self.tlb_slots, 3, self.page_size, 1),
            dtype=self.scale_dtype,
            device=self.device,
        )
        self.tlb_col_indices = torch.zeros(
            (self.tlb_slots * self.page_size,), dtype=torch.long, device=self.device
        )
        self._is_allocated = True
        self.clear_cache()

    @property
    def total_gpu_vram_bytes(self) -> int:
        if not self._is_allocated:
            return int(
                self.page_scales_gpu.numel() * self.page_scales_gpu.element_size()
                + self.gpu_page_in_tlb.numel() * self.gpu_page_in_tlb.element_size()
            )
        total = 0
        for buf in (
            self.page_scales_gpu,
            self.tlb_packed_q4,
            self.tlb_scales,
            self.tlb_col_indices,
            self.gpu_page_in_tlb,
        ):
            total += int(buf.numel() * buf.element_size())
        return total

    def clear_cache(self) -> None:
        self._slot_to_page: List[int] = [-1] * self.tlb_slots
        self._page_to_slot: Dict[int, int] = {}
        self._slot_last_used: List[int] = [0] * self.tlb_slots
        self._access_clock: int = 0
        self._num_valid: int = 0
        self.gpu_page_in_tlb.zero_()

    def reset_telemetry(self) -> None:
        self._pcie_bytes: int = 0
        self._page_hits: int = 0
        self._page_misses: int = 0
        self._all_hit_fastpaths: int = 0
        self._distinct_pages: Set[int] = set()

    def telemetry(self) -> Dict[str, float]:
        tot_req = max(1, self._page_hits + self._page_misses)
        return {
            "pcie_bytes": float(self._pcie_bytes),
            "page_hits": float(self._page_hits),
            "page_misses": float(self._page_misses),
            "hit_rate_pct": 100.0 * (float(self._page_hits) / float(tot_req)),
            "all_hit_fastpaths": float(self._all_hit_fastpaths),
            "distinct_pages": float(len(self._distinct_pages)),
        }

    def _choose_victim_slot(self, protected_pages: Set[int]) -> int:
        if self._num_valid < self.tlb_slots:
            return self._num_valid
        best_slot = 0
        oldest_tick = 1 << 60
        for s in range(self.tlb_slots):
            p = self._slot_to_page[s]
            if p in protected_pages:
                continue
            if self._slot_last_used[s] < oldest_tick:
                oldest_tick = self._slot_last_used[s]
                best_slot = s
        return best_slot

    def _load_page_into_slot(
        self,
        slot: int,
        page_id: int,
        stream: Optional[torch.cuda.Stream],
        src_override: Optional[torch.Tensor] = None,
        _already_on_stream: bool = False,
    ) -> None:
        old_p = self._slot_to_page[slot]
        if old_p >= 0 and old_p in self._page_to_slot:
            del self._page_to_slot[old_p]
            self.gpu_page_in_tlb[old_p] = False

        src_q4 = src_override if src_override is not None else self.q4_pages_cpu_pinned[page_id]
        col_slice = self.tlb_col_indices[slot * self.page_size : (slot + 1) * self.page_size]
        src_cols = self._all_page_col_indices[page_id]

        if stream is not None and self.device.type == "cuda" and not _already_on_stream:
            with torch.cuda.stream(stream):
                self.tlb_packed_q4[slot].copy_(src_q4, non_blocking=True)
                self.tlb_scales[slot].copy_(self.page_scales_gpu[page_id], non_blocking=True)
                col_slice.copy_(src_cols, non_blocking=True)
        else:
            self.tlb_packed_q4[slot].copy_(src_q4, non_blocking=True)
            self.tlb_scales[slot].copy_(self.page_scales_gpu[page_id], non_blocking=True)
            col_slice.copy_(src_cols, non_blocking=True)

        self._slot_to_page[slot] = page_id
        self._page_to_slot[page_id] = slot
        self.gpu_page_in_tlb[page_id] = True
        self._access_clock += 1
        self._slot_last_used[slot] = self._access_clock
        if slot >= self._num_valid:
            self._num_valid = slot + 1
        self._pcie_bytes += self.bytes_per_page
        self._distinct_pages.add(page_id)

    def find_misses_and_touch_hits(self, page_ids: List[int]) -> Tuple[List[int], Set[int]]:
        """Updates LRU clocks for TLB hits and returns `(misses, protected_set)`."""
        protected = set(page_ids)
        misses: List[int] = []
        for p in page_ids:
            p_int = int(p)
            if p_int in self._page_to_slot:
                self._page_hits += 1
                self._access_clock += 1
                self._slot_last_used[self._page_to_slot[p_int]] = self._access_clock
            else:
                self._page_misses += 1
                misses.append(p_int)
        return misses, protected

    def service_page_list(
        self, page_ids: List[int], stream: Optional[torch.cuda.Stream] = None
    ) -> bool:
        """Updates LRU state and loads any missed pages on `stream`. Returns True if any miss occurred."""
        misses, protected = self.find_misses_and_touch_hits(page_ids)
        if not misses:
            return False

        for p_int in misses:
            victim = self._choose_victim_slot(protected)
            self._load_page_into_slot(victim, p_int, stream=stream)
            protected.add(p_int)
        return True

    def request_pages_sync_tlb(
        self, req_pages_gpu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        k_req = int(req_pages_gpu.numel())
        if self.tlb_slots <= 0 or k_req <= 0:
            return self.tlb_packed_q4[:0], self.tlb_scales[:0], self.tlb_col_indices[:0], 0

        req_list = [int(x) for x in req_pages_gpu.tolist()]
        if self._num_valid == self.tlb_slots and all(p in self._page_to_slot for p in req_list):
            self._page_hits += k_req
            self._all_hit_fastpaths += 1
            for p in req_list:
                self._access_clock += 1
                self._slot_last_used[self._page_to_slot[p]] = self._access_clock
            return (
                self.tlb_packed_q4[: self._num_valid],
                self.tlb_scales[: self._num_valid],
                self.tlb_col_indices[: self._num_valid * self.page_size],
                self._num_valid,
            )

        self.service_page_list(req_list, stream=None)
        n_act = self._num_valid
        return (
            self.tlb_packed_q4[:n_act],
            self.tlb_scales[:n_act],
            self.tlb_col_indices[: n_act * self.page_size],
            n_act,
        )

    def request_pages_stateless_dma(
        self, req_pages_gpu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        req_list = [int(x) for x in req_pages_gpu.tolist()]
        k_req = min(len(req_list), self.tlb_slots)
        for slot in range(k_req):
            p_int = req_list[slot]
            self.tlb_packed_q4[slot].copy_(self.q4_pages_cpu_pinned[p_int], non_blocking=True)
            self.tlb_scales[slot].copy_(self.page_scales_gpu[p_int], non_blocking=True)
            self.tlb_col_indices[slot * self.page_size : (slot + 1) * self.page_size].copy_(
                self._all_page_col_indices[p_int], non_blocking=True
            )
            self._page_misses += 1
            self._pcie_bytes += self.bytes_per_page
            self._distinct_pages.add(p_int)
        return (
            self.tlb_packed_q4[:k_req],
            self.tlb_scales[:k_req],
            self.tlb_col_indices[: k_req * self.page_size],
            k_req,
        )


class SharedBatchTLBController:
    """Zero-Stall Batched GPU TLB Controller across all compiled layers (`1` async D2H copy per step).

    Supports both `speculative_tlb` (`M=1`) and `amortized_spec_tlb` (`1 <= M <= 8` multi-token
    verification steps), plus a double-buffered 32-page Pinned Host DMA Staging Ring so unpinned
    `nvme_mmap` / Windows pageable pages stream across PCIe Gen4 asynchronously at full DMA bandwidth
    while preserving near-zero OS physical RAM footprint (`~75 MB` pinned ring vs `14+ GB` full pin).
    """

    STAGING_RING_SLOTS: int = 32
    STAGING_HALF_SLOTS: int = 16

    def __init__(
        self,
        num_layers: int,
        decode_req_pages: int,
        device: torch.device,
        dma_stream: Optional[torch.cuda.Stream] = None,
        enable_pinned_staging: bool = False,
    ) -> None:
        self.num_layers = int(num_layers)
        self.decode_req_pages = int(decode_req_pages)
        self.device = device
        self.dma_stream = dma_stream
        self.enable_pinned_staging: bool = bool(enable_pinned_staging)

        self.gpu_req_matrix = torch.zeros(
            (self.num_layers, self.decode_req_pages), dtype=torch.int32, device=device
        )
        cpu_req = torch.zeros(
            (self.num_layers, self.decode_req_pages), dtype=torch.int32, device="cpu"
        )
        if torch.cuda.is_available():
            try:
                cpu_req = cpu_req.pin_memory()
            except Exception:
                pass
        self.cpu_pinned_req_matrix = cpu_req
        self.caches: List[Optional[GPUPageTLBCache]] = [None] * self.num_layers
        self.d2h_event: Optional[torch.cuda.Event] = (
            torch.cuda.Event() if (torch.cuda.is_available() and device.type == "cuda") else None
        )
        self.h2d_event: Optional[torch.cuda.Event] = (
            torch.cuda.Event() if (torch.cuda.is_available() and device.type == "cuda") else None
        )
        self._ring_events: List[Optional[torch.cuda.Event]] = (
            [torch.cuda.Event(), torch.cuda.Event()]
            if (torch.cuda.is_available() and device.type == "cuda")
            else [None, None]
        )
        self._stage_q4_pinned: Optional[torch.Tensor] = None
        self._stage_is_pinned: bool = False
        self._stage_init_attempted: bool = False
        self.has_pending_batch: bool = False
        self._last_h2d_recorded: bool = False

    def _ensure_pinned_staging_ring(self) -> Optional[torch.Tensor]:
        if self._stage_init_attempted:
            return self._stage_q4_pinned if self._stage_is_pinned else None
        self._stage_init_attempted = True
        if not (torch.cuda.is_available() and self.device.type == "cuda"):
            return None
        max_bytes = max((c.bytes_per_page for c in self.caches if c is not None), default=0)
        if max_bytes <= 0:
            return None
        try:
            buf = torch.empty((self.STAGING_RING_SLOTS, max_bytes), dtype=torch.uint8, device="cpu").pin_memory()
            if buf.is_pinned():
                self._stage_q4_pinned = buf
                self._stage_is_pinned = True
                return buf
        except Exception:
            pass
        self._stage_q4_pinned = None
        self._stage_is_pinned = False
        return None

    def register_layer_cache(self, slot_idx: int, cache: GPUPageTLBCache) -> None:
        self.caches[slot_idx] = cache

    def clear_pending(self) -> None:
        self.has_pending_batch = False

    def record_layer_request(self, slot_idx: int, req_pages_gpu: torch.Tensor) -> None:
        k = min(self.decode_req_pages, int(req_pages_gpu.numel()))
        self.gpu_req_matrix[slot_idx, :k].copy_(req_pages_gpu[:k].to(torch.int32))
        if slot_idx == self.num_layers - 1:
            if self.dma_stream is not None and self.device.type == "cuda":
                cur = torch.cuda.current_stream(self.device)
                self.dma_stream.wait_stream(cur)
                with torch.cuda.stream(self.dma_stream):
                    self.cpu_pinned_req_matrix.copy_(self.gpu_req_matrix, non_blocking=True)
                    if self.d2h_event is not None:
                        self.d2h_event.record(self.dma_stream)
            else:
                self.cpu_pinned_req_matrix.copy_(self.gpu_req_matrix)
            self.has_pending_batch = True

    def drain_and_prefetch_all(self) -> None:
        if not self.has_pending_batch:
            return
        if self.d2h_event is not None and self.device.type == "cuda":
            self.d2h_event.synchronize()
        if self._last_h2d_recorded and self.h2d_event is not None and self.device.type == "cuda":
            self.h2d_event.synchronize()
            self._last_h2d_recorded = False
        self.has_pending_batch = False

        # Fast bulk conversion of `[L, k_req]` int32 CPU tensor to nested Python ints (1 C++ call!)
        all_reqs: List[List[int]] = self.cpu_pinned_req_matrix.tolist()
        any_miss = False
        stream = self.dma_stream if (self.dma_stream is not None and self.device.type == "cuda") else None
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(self.device))

        if stream is not None:
            ring_half_in_flight = [False, False]
            st_idx = 0
            with torch.cuda.stream(stream):
                for s_idx, cache in enumerate(self.caches):
                    if cache is None or not cache._is_allocated:
                        continue
                    misses, protected = cache.find_misses_and_touch_hits(all_reqs[s_idx])
                    if not misses:
                        continue
                    any_miss = True
                    stage_ring: Optional[torch.Tensor] = None
                    if self.enable_pinned_staging and not cache._host_is_pinned:
                        stage_ring = self._ensure_pinned_staging_ring()

                    page_shape = cache.q4_pages_cpu_pinned[0].shape
                    n_bytes = cache.bytes_per_page
                    for p_int in misses:
                        victim = cache._choose_victim_slot(protected)
                        protected.add(p_int)
                        if stage_ring is not None:
                            slot_in_ring = st_idx & 31
                            half_idx = slot_in_ring >> 4
                            offset_in_half = slot_in_ring & 15
                            if offset_in_half == 0 and ring_half_in_flight[half_idx]:
                                ev = self._ring_events[half_idx]
                                if ev is not None:
                                    ev.synchronize()
                                ring_half_in_flight[half_idx] = False

                            stage_slot = stage_ring[slot_in_ring, :n_bytes].view(page_shape)
                            stage_slot.copy_(cache.q4_pages_cpu_pinned[p_int], non_blocking=False)
                            cache._load_page_into_slot(
                                victim, p_int, stream=stream, src_override=stage_slot, _already_on_stream=True
                            )
                            if offset_in_half == 15:
                                ev = self._ring_events[half_idx]
                                if ev is not None:
                                    ev.record(stream)
                                ring_half_in_flight[half_idx] = True
                            st_idx += 1
                        else:
                            cache._load_page_into_slot(
                                victim, p_int, stream=stream, _already_on_stream=True
                            )
        else:
            for s_idx, cache in enumerate(self.caches):
                if cache is None or not cache._is_allocated:
                    continue
                missed = cache.service_page_list(all_reqs[s_idx], stream=None)
                if missed:
                    any_miss = True

        if any_miss and stream is not None and self.h2d_event is not None:
            self.h2d_event.record(stream)
            self._last_h2d_recorded = True
            torch.cuda.current_stream(self.device).wait_event(self.h2d_event)


class BitTieredTLBMLP(nn.Module):
    """Phase B Neural VM MLP with Fused `gate+up` Projections and Batched/LRU GPU Page TLB."""

    def __init__(
        self,
        # Tier 1: Fused Hot Generalist INT8 Core (`gu` is `[2 * n_hot, d_model]`, `down` is `[d_model, n_hot]`)
        q_gu_hot_i8: torch.Tensor,
        sc_gu_hot: torch.Tensor,
        q_down_hot_i8: torch.Tensor,
        sc_down_hot: torch.Tensor,
        s_mid_hot: torch.Tensor,
        # Tier 0: Fused Cold Specialist 3-Bit Base8 Skeleton (`gu` is `[2 * n_cold, d_pad // 5]`, `down` is `[d_model, n_cold_pad // 5]`)
        p_gu_cold_i16: torch.Tensor,
        mn_gu_cold: torch.Tensor,
        sc_gu_cold: torch.Tensor,
        p_down_cold_i16: torch.Tensor,
        mn_down_cold: torch.Tensor,
        sc_down_cold: torch.Tensor,
        s_mid_cold: torch.Tensor,
        gu_group_size: int,
        down_group_size: int,
        # AWQ channel scale (`[d_model]`) + Post-GeGLU self-router column weights (`[n_cold]`)
        s_in: torch.Tensor,
        col_err_weight: torch.Tensor,
        # Closed-Loop Rank-r Drift & Quantization Bridge
        U_bridge: torch.Tensor,
        V_bridge: torch.Tensor,
        # Tier 2: Persistent GPU LRU Page Cache (TLB) + Shared LUT Pool + Batch Controller
        tlb_cache: GPUPageTLBCache,
        lut_pool: SharedQuantLUTPool,
        act_fn: nn.Module,
        num_pages: int = 64,
        decode_req_pages: int = 4,
        mode: str = "vm_paged",
        tlb_strategy: str = "sync_tlb",
        batch_controller: Optional[SharedBatchTLBController] = None,
        controller_slot_idx: int = 0,
        use_phase_c_kernels: bool = True,
        _pre_inverted_scales: bool = False,
    ) -> None:
        super().__init__()
        if mode not in VALID_PHASE_B_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Expected {VALID_PHASE_B_MODES}.")
        if tlb_strategy not in VALID_TLB_STRATEGIES:
            raise ValueError(f"Invalid tlb_strategy '{tlb_strategy}'. Expected {VALID_TLB_STRATEGIES}.")

        self.d_model = int(q_down_hot_i8.shape[0])
        self.n_hot = int(q_down_hot_i8.shape[1])
        self.n_cold = int(s_mid_cold.numel())
        self.d_ffn = self.n_hot + self.n_cold
        self.gu_group_size = int(gu_group_size)
        self.down_group_size = int(down_group_size)
        self.d_pad = int(mn_gu_cold.shape[1] * self.gu_group_size)
        self.pad_len = self.d_pad - self.d_model
        self.n_cold_pad = int(mn_down_cold.shape[1] * self.down_group_size)
        self.cold_pad_len = self.n_cold_pad - self.n_cold

        self.num_pages = int(num_pages)
        self.page_size = self.n_cold // self.num_pages
        self.decode_req_pages = int(decode_req_pages)
        self.mode = mode
        self.tlb_strategy = tlb_strategy
        self.act_fn = act_fn
        self.tlb_cache = tlb_cache
        self.lut_pool = lut_pool
        self.batch_controller = batch_controller
        self.controller_slot_idx = int(controller_slot_idx)

        self.register_buffer("q_gu_hot_i8", q_gu_hot_i8, persistent=False)
        self.register_buffer("sc_gu_hot", sc_gu_hot, persistent=False)
        self.register_buffer("q_down_hot_i8", q_down_hot_i8, persistent=False)
        self.register_buffer("sc_down_hot", sc_down_hot, persistent=False)
        inv_hot = (
            s_mid_hot.to(sc_gu_hot.dtype).contiguous()
            if _pre_inverted_scales
            else (1.0 / s_mid_hot.float()).to(sc_gu_hot.dtype).contiguous()
        )
        self.register_buffer("inv_s_mid_hot", inv_hot, persistent=False)

        self.register_buffer("p_gu_cold_i16", p_gu_cold_i16, persistent=False)
        self.register_buffer("mn_gu_cold", mn_gu_cold, persistent=False)
        self.register_buffer("sc_gu_cold", sc_gu_cold, persistent=False)
        self.register_buffer("p_down_cold_i16", p_down_cold_i16, persistent=False)
        self.register_buffer("mn_down_cold", mn_down_cold, persistent=False)
        self.register_buffer("sc_down_cold", sc_down_cold, persistent=False)
        inv_cold = (
            s_mid_cold.to(sc_gu_hot.dtype).contiguous()
            if _pre_inverted_scales
            else (1.0 / s_mid_cold.float()).to(sc_gu_hot.dtype).contiguous()
        )
        self.register_buffer("inv_s_mid_cold", inv_cold, persistent=False)

        inv_in = (
            s_in.to(sc_gu_hot.dtype).contiguous()
            if _pre_inverted_scales
            else (1.0 / s_in.float()).to(sc_gu_hot.dtype).contiguous()
        )
        self.register_buffer("inv_s_in", inv_in, persistent=False)
        self.register_buffer("col_err_weight", col_err_weight.to(sc_gu_hot.dtype).contiguous(), persistent=False)
        self.register_buffer("U_bridge", U_bridge.contiguous(), persistent=False)
        self.register_buffer("V_bridge", V_bridge.contiguous(), persistent=False)

        self.use_phase_c_kernels: bool = bool(use_phase_c_kernels)
        self._act_kind: int = detect_act_kind(act_fn)
        self._phase_c_launcher: Optional[PhaseCLayerLauncher] = None
        self._total_tokens: int = 0

    def export_serialized_core_and_pages(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Exports GPU core buffers, CPU INT4 page tensor, and layer metadata for `.nvm` serialization."""
        core_tensors = {
            "q_gu_hot_i8": self.q_gu_hot_i8.detach().cpu().contiguous(),
            "sc_gu_hot": self.sc_gu_hot.detach().cpu().contiguous(),
            "q_down_hot_i8": self.q_down_hot_i8.detach().cpu().contiguous(),
            "sc_down_hot": self.sc_down_hot.detach().cpu().contiguous(),
            "inv_s_mid_hot": self.inv_s_mid_hot.detach().cpu().contiguous(),
            "p_gu_cold_i16": self.p_gu_cold_i16.detach().cpu().contiguous(),
            "mn_gu_cold": self.mn_gu_cold.detach().cpu().contiguous(),
            "sc_gu_cold": self.sc_gu_cold.detach().cpu().contiguous(),
            "p_down_cold_i16": self.p_down_cold_i16.detach().cpu().contiguous(),
            "mn_down_cold": self.mn_down_cold.detach().cpu().contiguous(),
            "sc_down_cold": self.sc_down_cold.detach().cpu().contiguous(),
            "inv_s_mid_cold": self.inv_s_mid_cold.detach().cpu().contiguous(),
            "inv_s_in": self.inv_s_in.detach().cpu().contiguous(),
            "col_err_weight": self.col_err_weight.detach().cpu().contiguous(),
            "U_bridge": self.U_bridge.detach().cpu().contiguous(),
            "V_bridge": self.V_bridge.detach().cpu().contiguous(),
            "page_scales_gpu": self.tlb_cache.page_scales_gpu.detach().cpu().contiguous(),
        }
        page_tensors = {
            "q4_pages": self.tlb_cache.q4_pages_cpu_pinned.detach().cpu().contiguous(),
        }
        meta = {
            "d_model": self.d_model,
            "n_hot": self.n_hot,
            "n_cold": self.n_cold,
            "d_ffn": self.d_ffn,
            "gu_group_size": self.gu_group_size,
            "down_group_size": self.down_group_size,
            "num_pages": self.num_pages,
            "page_size": self.page_size,
            "tlb_slots": self.tlb_cache.tlb_slots,
            "decode_req_pages": self.decode_req_pages,
            "act_kind": self._act_kind,
        }
        return core_tensors, page_tensors, meta

    @classmethod
    def from_serialized_buffers(
        cls,
        core_tensors: Dict[str, torch.Tensor],
        q4_pages_host: torch.Tensor,
        meta: Dict[str, Any],
        lut_pool: SharedQuantLUTPool,
        act_fn: nn.Module,
        device: torch.device,
        mode: str = "vm_paged",
        tlb_strategy: str = "speculative_tlb",
        tlb_slots_override: Optional[int] = None,
        decode_req_pages_override: Optional[int] = None,
        shared_dma_stream: Optional[torch.cuda.Stream] = None,
        batch_controller: Optional[SharedBatchTLBController] = None,
        controller_slot_idx: int = 0,
        use_phase_c_kernels: bool = True,
    ) -> "BitTieredTLBMLP":
        """Reconstructs `BitTieredTLBMLP` directly from `.nvm` core & page buffers with zero dense overhead."""
        eff_slots = int(tlb_slots_override) if tlb_slots_override is not None else int(meta["tlb_slots"])
        eff_req = (
            int(decode_req_pages_override)
            if decode_req_pages_override is not None
            else int(meta["decode_req_pages"])
        )
        page_scales_gpu = core_tensors["page_scales_gpu"].to(device=device, non_blocking=True).contiguous()
        tlb_cache = GPUPageTLBCache(
            num_pages=int(meta["num_pages"]),
            tlb_slots=min(eff_slots, int(meta["num_pages"])),
            page_size=int(meta["page_size"]),
            d_model=int(meta["d_model"]),
            q4_pages_cpu_pinned=q4_pages_host,
            page_scales_gpu=page_scales_gpu,
            device=device,
            scale_dtype=page_scales_gpu.dtype,
            dma_stream=shared_dma_stream,
        )
        if mode == "core_only":
            tlb_cache.release_gpu_slots()
        if batch_controller is not None:
            batch_controller.register_layer_cache(controller_slot_idx, tlb_cache)

        mlp = cls(
            q_gu_hot_i8=core_tensors["q_gu_hot_i8"].to(device=device, non_blocking=True).contiguous(),
            sc_gu_hot=core_tensors["sc_gu_hot"].to(device=device, non_blocking=True).contiguous(),
            q_down_hot_i8=core_tensors["q_down_hot_i8"].to(device=device, non_blocking=True).contiguous(),
            sc_down_hot=core_tensors["sc_down_hot"].to(device=device, non_blocking=True).contiguous(),
            s_mid_hot=core_tensors["inv_s_mid_hot"].to(device=device, non_blocking=True).contiguous(),
            p_gu_cold_i16=core_tensors["p_gu_cold_i16"].to(device=device, non_blocking=True).contiguous(),
            mn_gu_cold=core_tensors["mn_gu_cold"].to(device=device, non_blocking=True).contiguous(),
            sc_gu_cold=core_tensors["sc_gu_cold"].to(device=device, non_blocking=True).contiguous(),
            p_down_cold_i16=core_tensors["p_down_cold_i16"].to(device=device, non_blocking=True).contiguous(),
            mn_down_cold=core_tensors["mn_down_cold"].to(device=device, non_blocking=True).contiguous(),
            sc_down_cold=core_tensors["sc_down_cold"].to(device=device, non_blocking=True).contiguous(),
            s_mid_cold=core_tensors["inv_s_mid_cold"].to(device=device, non_blocking=True).contiguous(),
            gu_group_size=int(meta["gu_group_size"]),
            down_group_size=int(meta["down_group_size"]),
            s_in=core_tensors["inv_s_in"].to(device=device, non_blocking=True).contiguous(),
            col_err_weight=core_tensors["col_err_weight"].to(device=device, non_blocking=True).contiguous(),
            U_bridge=core_tensors["U_bridge"].to(device=device, non_blocking=True).contiguous(),
            V_bridge=core_tensors["V_bridge"].to(device=device, non_blocking=True).contiguous(),
            tlb_cache=tlb_cache,
            lut_pool=lut_pool,
            act_fn=act_fn,
            num_pages=int(meta["num_pages"]),
            decode_req_pages=eff_req,
            mode=mode,
            tlb_strategy=tlb_strategy,
            batch_controller=batch_controller,
            controller_slot_idx=controller_slot_idx,
            use_phase_c_kernels=use_phase_c_kernels,
            _pre_inverted_scales=True,
        )
        return mlp

    def set_phase_c_enabled(self, enabled: bool) -> None:
        self.use_phase_c_kernels = bool(enabled)
        if (
            self.use_phase_c_kernels
            and self._phase_c_launcher is None
            and self.q_gu_hot_i8.is_cuda
            and self.sc_gu_hot.dtype == torch.bfloat16
            and self._act_kind >= 0
            and self.num_pages <= 256
        ):
            self._phase_c_launcher = PhaseCLayerLauncher(self, self._act_kind)

    def set_mode(self, mode: str) -> None:
        if mode not in VALID_PHASE_B_MODES:
            raise ValueError(f"Invalid mode '{mode}'.")
        self.mode = mode
        if mode == "core_only":
            self.tlb_cache.release_gpu_slots()
        else:
            self.tlb_cache.allocate_gpu_slots()

    def set_tlb_strategy(self, strategy: str) -> None:
        if strategy not in VALID_TLB_STRATEGIES:
            raise ValueError(f"Invalid tlb_strategy '{strategy}'.")
        self.tlb_strategy = strategy
        if self.batch_controller is not None:
            self.batch_controller.clear_pending()

    def reset_telemetry(self, clear_tlb: bool = False) -> None:
        self._total_tokens = 0
        self.tlb_cache.reset_telemetry()
        if clear_tlb:
            self.tlb_cache.clear_cache()
            if self._phase_c_launcher is not None:
                self._phase_c_launcher.ws.zero_page_scores()
            if self.batch_controller is not None and self.controller_slot_idx == 0:
                self.batch_controller.clear_pending()

    def vram_footprint_stats(self) -> Dict[str, float]:
        mb = float(1024 ** 2)
        dense_bytes = 3 * self.d_ffn * self.d_model * 2

        hot_bytes = sum(
            int(t.numel() * t.element_size())
            for t in (
                self.q_gu_hot_i8,
                self.sc_gu_hot,
                self.q_down_hot_i8,
                self.sc_down_hot,
                self.inv_s_mid_hot,
            )
        )
        cold_base_bytes = sum(
            int(t.numel() * t.element_size())
            for t in (
                self.p_gu_cold_i16,
                self.mn_gu_cold,
                self.sc_gu_cold,
                self.p_down_cold_i16,
                self.mn_down_cold,
                self.sc_down_cold,
                self.inv_s_mid_cold,
                self.inv_s_in,
                self.col_err_weight,
            )
        )
        bridge_bytes = int(
            self.U_bridge.numel() * self.U_bridge.element_size()
            + self.V_bridge.numel() * self.V_bridge.element_size()
        )
        core_bytes = hot_bytes + cold_base_bytes + bridge_bytes
        tlb_bytes = self.tlb_cache.total_gpu_vram_bytes if self.mode == "vm_paged" else 0
        host_pinned_bytes = int(
            self.tlb_cache.q4_pages_cpu_pinned.numel()
            * self.tlb_cache.q4_pages_cpu_pinned.element_size()
        )

        return {
            "dense_vram_mb": dense_bytes / mb,
            "hot_core_mb": hot_bytes / mb,
            "cold_base_mb": cold_base_bytes / mb,
            "bridge_mb": bridge_bytes / mb,
            "core_total_mb": core_bytes / mb,
            "tlb_cache_mb": tlb_bytes / mb,
            "active_total_mb": (core_bytes + tlb_bytes) / mb,
            "host_ram_pages_mb": host_pinned_bytes / mb,
        }

    def bandwidth_telemetry(self) -> Dict[str, float]:
        t = self.tlb_cache.telemetry()
        toks = max(1, self._total_tokens)
        pcie_mb = t["pcie_bytes"] / float(1024 ** 2)
        return {
            "total_pcie_mb": pcie_mb,
            "pcie_mb_per_token": pcie_mb / float(toks),
            "tlb_hit_rate_pct": t["hit_rate_pct"],
            "tlb_hits": t["page_hits"],
            "tlb_misses": t["page_misses"],
            "all_hit_fastpaths": t["all_hit_fastpaths"],
            "distinct_pages": t["distinct_pages"],
        }

    def _select_top_pages_from_h(self, h_cold: torch.Tensor, k_pages: int) -> torch.Tensor:
        n_tok = h_cold.shape[0]
        weighted = (h_cold.abs() * self.col_err_weight.unsqueeze(0)).view(
            n_tok, self.num_pages, self.page_size
        )
        per_tok_page = weighted.sum(dim=-1)
        agg = per_tok_page[0] if n_tok == 1 else (per_tok_page.mean(dim=0) + 0.35 * per_tok_page.amax(dim=0))
        k_eff = max(1, min(int(k_pages), self.num_pages))
        _, top_idx = torch.topk(agg, k=k_eff, largest=True, sorted=False)
        return top_idx

    def _apply_tlb_upgrade_window(
        self,
        x_scaled_win: torch.Tensor,
        g_cold_win: torch.Tensor,
        u_cold_win: torch.Tensor,
        h_cold_win: torch.Tensor,
        is_prefill_window: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        tlb_cap = self.tlb_cache.tlb_slots
        if tlb_cap <= 0:
            return h_cold_win, None

        if self.tlb_strategy == "stateless_dma":
            req = self._select_top_pages_from_h(h_cold_win, self.decode_req_pages)
            q4_pack, q4_sc, cols, n_act = self.tlb_cache.request_pages_stateless_dma(req)
        elif (
            (
                (self.tlb_strategy == "speculative_tlb" and not is_prefill_window)
                or (
                    self.tlb_strategy == "amortized_spec_tlb"
                    and int(h_cold_win.shape[0]) <= MAX_FUSED_DECODE_TOKENS
                )
            )
            and self.batch_controller is not None
        ):
            # At Layer 0 of Token t, drain Token t-1's 448-byte request matrix in ONE pass!
            if self.controller_slot_idx == 0:
                self.batch_controller.drain_and_prefetch_all()

            if self.tlb_cache._num_valid == 0:
                req_boot = self._select_top_pages_from_h(h_cold_win, tlb_cap)
                q4_pack, q4_sc, cols, n_act = self.tlb_cache.request_pages_sync_tlb(req_boot)
            else:
                req = self._select_top_pages_from_h(h_cold_win, self.decode_req_pages)
                self.batch_controller.record_layer_request(self.controller_slot_idx, req)
                n_act = self.tlb_cache._num_valid
                q4_pack = self.tlb_cache.tlb_packed_q4[:n_act]
                q4_sc = self.tlb_cache.tlb_scales[:n_act]
                cols = self.tlb_cache.tlb_col_indices[: n_act * self.page_size]
        else:
            k_req = tlb_cap if (is_prefill_window or self.tlb_cache._num_valid < tlb_cap) else self.decode_req_pages
            req = self._select_top_pages_from_h(h_cold_win, k_req)
            q4_pack, q4_sc, cols, n_act = self.tlb_cache.request_pages_sync_tlb(req)

        if n_act <= 0:
            return h_cold_win, None

        dw_bf16 = unpack_int4_pages_to_bf16(q4_pack, q4_sc, self.lut_pool.lut_int4, self.d_model)
        c_act = n_act * self.page_size
        # Fused `gate + up` TLB residual projection `[2 * c_act, d_model]`!
        dw_gu = torch.cat(
            [dw_bf16[:, 0].reshape(c_act, self.d_model), dw_bf16[:, 1].reshape(c_act, self.d_model)],
            dim=0,
        )
        dw_down_T = dw_bf16[:, 2].reshape(c_act, self.d_model)

        d_gu = x_scaled_win @ dw_gu.T
        dg, du = d_gu.chunk(2, dim=-1)
        g_new = g_cold_win[:, cols] + dg
        u_new = u_cold_win[:, cols] + du
        h_new = self.act_fn(g_new) * u_new

        h_upgraded = h_cold_win.clone()
        h_upgraded[:, cols] = h_new
        y_dw_down = h_new @ dw_down_T
        return h_upgraded, y_dw_down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        dt = self.sc_gu_hot.dtype

        flat_x = x.reshape(-1, self.d_model)
        n_tokens = int(flat_x.shape[0])
        self._total_tokens += n_tokens

        # Phase C Register-Fused NVRTC CUDA Fast-Path for Autoregressive Decode (`1 <= M <= 8`)
        is_3d_multi_seq = len(orig_shape) == 3 and orig_shape[0] > 1 and orig_shape[1] > 1
        if (
            self.use_phase_c_kernels
            and 1 <= n_tokens <= MAX_FUSED_DECODE_TOKENS
            and not is_3d_multi_seq
            and flat_x.is_cuda
            and dt == torch.bfloat16
            and self._act_kind >= 0
            and self.num_pages <= 256
        ):
            if self._phase_c_launcher is None:
                self._phase_c_launcher = PhaseCLayerLauncher(self, self._act_kind)
            flat_bf16 = flat_x if flat_x.dtype == torch.bfloat16 else flat_x.to(torch.bfloat16)
            return self._phase_c_launcher.forward_decode(flat_bf16, orig_shape, orig_dtype)

        flat_x = flat_x.to(dt)
        x_scaled = flat_x * self.inv_s_in
        x_scaled_pad = F.pad(x_scaled, (0, self.pad_len), value=0.0) if self.pad_len > 0 else x_scaled

        # 1. Tier 1: Fused Hot INT8 Generalist Core (AWQ-scaled on both input and mid GeGLU channels)
        w_gu_hot = self.q_gu_hot_i8.to(dt) * self.sc_gu_hot
        gu_hot = x_scaled @ w_gu_hot.T
        g_hot, u_hot = gu_hot.chunk(2, dim=-1)
        h_hot = (self.act_fn(g_hot) * u_hot) * self.inv_s_mid_hot
        w_down_hot = self.q_down_hot_i8.to(dt) * self.sc_down_hot
        y_hot = h_hot @ w_down_hot.T

        # 2. Tier 0: Fused Cold 3-Bit Base8 Resident Skeleton
        w_gu_cold_pad = unpack_3bit_base8_to_bf16(
            self.p_gu_cold_i16,
            self.mn_gu_cold,
            self.sc_gu_cold,
            self.lut_pool.lut_base8,
            self.gu_group_size,
        )
        gu_cold = x_scaled_pad @ w_gu_cold_pad.T
        del w_gu_cold_pad
        g_cold, u_cold = gu_cold.chunk(2, dim=-1)
        h_cold = self.act_fn(g_cold) * u_cold

        # 3. Tier 2: Persistent GPU LRU Page Cache (TLB) Precision Upgrade
        y_dw_down_extra: Optional[torch.Tensor] = None
        if self.mode == "vm_paged" and self.tlb_cache.tlb_slots > 0:
            is_3d_multi_seq = len(orig_shape) == 3 and orig_shape[0] > 1 and orig_shape[1] > 1
            is_prefill_step = (orig_shape[1] > 1) if len(orig_shape) == 3 else (n_tokens > 1)

            if not is_3d_multi_seq:
                h_cold, y_dw_down_extra = self._apply_tlb_upgrade_window(
                    x_scaled, g_cold, u_cold, h_cold, is_prefill_window=is_prefill_step
                )
            else:
                seq_len = int(orig_shape[1])
                h_chunks: List[torch.Tensor] = []
                y_extra_chunks: List[torch.Tensor] = []
                for s in range(0, n_tokens, seq_len):
                    e = min(n_tokens, s + seq_len)
                    hc_w, yd_w = self._apply_tlb_upgrade_window(
                        x_scaled[s:e],
                        g_cold[s:e],
                        u_cold[s:e],
                        h_cold[s:e],
                        is_prefill_window=True,
                    )
                    h_chunks.append(hc_w)
                    if yd_w is not None:
                        y_extra_chunks.append(yd_w)
                h_cold = torch.cat(h_chunks, dim=0)
                if y_extra_chunks:
                    y_dw_down_extra = torch.cat(y_extra_chunks, dim=0)

        h_cold_scaled = h_cold * self.inv_s_mid_cold
        if self.cold_pad_len > 0:
            h_cold_scaled = F.pad(h_cold_scaled, (0, self.cold_pad_len), value=0.0)

        w_down_cold_pad = unpack_3bit_base8_to_bf16(
            self.p_down_cold_i16,
            self.mn_down_cold,
            self.sc_down_cold,
            self.lut_pool.lut_base8,
            self.down_group_size,
        )
        y_cold = h_cold_scaled @ w_down_cold_pad.T
        del w_down_cold_pad

        if y_dw_down_extra is not None:
            y_cold = y_cold + y_dw_down_extra

        # 4. Closed-Loop Rank-64 Drift & Residual Bridge
        y_bridge = (flat_x @ self.U_bridge) @ self.V_bridge
        out = y_hot + y_cold + y_bridge
        return out.view(*orig_shape).to(orig_dtype)


@torch.no_grad()
def quantize_and_pack_uint4_group(
    w_fp32: torch.Tensor,
    s_in: Optional[torch.Tensor] = None,
    group_size: int = 64,
    target_device: str = "cuda:0",
    scale_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    rows, cols = w_fp32.shape
    w_work = w_fp32 * s_in.unsqueeze(0) if s_in is not None else w_fp32
    pad_cols = (group_size - (cols % group_size)) % group_size
    w_pad = F.pad(w_work, (0, pad_cols), value=0.0) if pad_cols > 0 else w_work
    d_pad = w_pad.shape[1]
    n_groups = d_pad // group_size

    wg = w_pad.view(rows, n_groups, group_size)
    wmin_bf = wg.amin(dim=-1, keepdim=True).to(scale_dtype)
    wmax_bf = wg.amax(dim=-1, keepdim=True).to(scale_dtype)
    scale_bf = ((wmax_bf.float() - wmin_bf.float()) / 15.0).clamp_min(1e-8).to(scale_dtype)

    q = torch.clamp(torch.round((wg - wmin_bf.float()) / scale_bf.float()), 0.0, 15.0).to(torch.uint8)
    q_pair = q.view(rows, d_pad // 2, 2)
    packed_u8 = (q_pair[..., 0] | (q_pair[..., 1] << 4)).to(device=target_device).contiguous()
    return (
        packed_u8,
        wmin_bf.to(device=target_device).contiguous(),
        scale_bf.to(device=target_device).contiguous(),
        d_pad,
    )


class UniformAWQ4BitMLP(nn.Module):
    """Hardware-Honest Fused Static Uniform 4-Bit AWQ (`g=64`) Reference Baseline."""

    def __init__(
        self,
        p_gu_u8: torch.Tensor,
        mn_gu: torch.Tensor,
        sc_gu: torch.Tensor,
        p_down_T_u8: torch.Tensor,
        mn_down_T: torch.Tensor,
        sc_down_T: torch.Tensor,
        s_in: torch.Tensor,
        d_model: int,
        group_size: int,
        lut_pool: SharedQuantLUTPool,
        act_fn: nn.Module,
    ) -> None:
        super().__init__()
        self.d_ffn = int(p_down_T_u8.shape[0])
        self.d_model = int(d_model)
        self.group_size = int(group_size)
        self.d_pad = int(p_down_T_u8.shape[1] * 2)
        self.pad_len = self.d_pad - self.d_model
        self.lut_pool = lut_pool
        self.act_fn = act_fn

        self.register_buffer("p_gu_u8", p_gu_u8, persistent=False)
        self.register_buffer("mn_gu", mn_gu, persistent=False)
        self.register_buffer("sc_gu", sc_gu, persistent=False)
        self.register_buffer("p_down_T_u8", p_down_T_u8, persistent=False)
        self.register_buffer("mn_down_T", mn_down_T, persistent=False)
        self.register_buffer("sc_down_T", sc_down_T, persistent=False)
        self.register_buffer("inv_s_in", (1.0 / s_in.float()).to(sc_gu.dtype).contiguous(), persistent=False)

    def vram_footprint_mb(self) -> float:
        tot = sum(
            int(b.numel() * b.element_size())
            for b in (
                self.p_gu_u8,
                self.mn_gu,
                self.sc_gu,
                self.p_down_T_u8,
                self.mn_down_T,
                self.sc_down_T,
                self.inv_s_in,
            )
        )
        return tot / float(1024 ** 2)

    def _unpack(self, p_u8: torch.Tensor, mn: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
        rows = p_u8.shape[0]
        n_groups = mn.shape[1]
        q_bf = F.embedding(p_u8.to(torch.int32), self.lut_pool.lut_uint4).view(rows, n_groups, self.group_size)
        return (q_bf * sc + mn).view(rows, self.d_pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        dt = self.sc_gu.dtype
        flat_x = x.reshape(-1, self.d_model).to(dt)
        x_s = flat_x * self.inv_s_in
        x_sp = F.pad(x_s, (0, self.pad_len), value=0.0) if self.pad_len > 0 else x_s

        w_gu = self._unpack(self.p_gu_u8, self.mn_gu, self.sc_gu)
        gu = x_sp @ w_gu.T
        del w_gu
        g, u = gu.chunk(2, dim=-1)
        h = self.act_fn(g) * u
        wd_T = self._unpack(self.p_down_T_u8, self.mn_down_T, self.sc_down_T)
        y = h @ wd_T[:, : self.d_model]
        del wd_T
        return y.view(*orig_shape).to(orig_dtype)
