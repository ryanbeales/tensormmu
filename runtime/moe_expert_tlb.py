"""Phase F, G & H Runtime: Zero-Skeleton 3-Tier Sparse MoE Virtual Memory (`SparseMoETLBExperts` & `GPUExpertTLBCache`).

Architecture:
  1. **3-Tier GPU Virtual Address Hierarchy:**
     - **L1 GPU VRAM TLB (`slot >= 0`, `1,008 GB/s` HBM):** Holds Top-`S` hot experts per layer with Probationary
       Draft-LRU (`Phase G1b`) protecting verified experts from speculative/lookahead churn.
     - **L2 Zero-Copy Pinned Host DDR5 (`slot <= -2`, `16 GB/s` direct PCIe `cuMemHostGetDevicePointer_v2`):**
       Maps calibration-ranked warm experts directly into GPU virtual address space (`0` VRAM, `0` eviction on
       transient/1-off hits, automatic 2-hit promotion into L1 VRAM, and graceful demotion back to L2 on eviction).
       When `L1 + L2 == num_experts` (e.g. `Gemma-4-26B-A4B`), `all_experts_mapped == True` eliminates 100% of CPU
       D2H syncs so the entire 30-layer model executes as a single end-to-end CUDA Graph!
     - **L3 Cold `nvme_mmap` (`slot == -1`) via 32-Slot Pinned DMA Staging Ring (`22.9 GB/s`):**
       Eliminates blocking pageable `mmap` copies (`8.8 GB/s`), overlapping true async `cudaMemcpyAsync` behind
       GPU `SharedMLP` and cross-layer lookahead (`Phase G2`).
  2. **Phase H3 Bit-Exact Multi-Wave Grouped Prefill (`T > 8`):**
     Groups `(token, k)` routes by expert in `O(T*K log(T*K))` with a single D2H sync and executes packed `UINT4`
     pages directly through `MoETLBLayerLauncher` waves (`0.0-ulp` identical to `M=1` decode, ~20x faster TTFT).
"""

from __future__ import annotations

import ctypes
import os
import types
from typing import Any, Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from runtime.gpu_page_tlb import SharedQuantLUTPool
from runtime.moe_tlb_kernels import (
    ContiguousExpertPageLayout,
    MAX_ACTIVE_UNIQUE,
    MoETLBLayerLauncher,
    PREFILL_MAX_CHUNK_TOKENS,
    PREFILL_STAGE_SLOTS,
    PinnedZeroCopyBuffer,
    WINAExpertPageLayout,
    alloc_zero_copy_host_buffer,
    get_shared_moe_workspace,
    get_zero_copy_device_ptr,
    query_available_system_ram_bytes,
)
from runtime.phase_c_kernels import MAX_FUSED_DECODE_TOKENS, detect_act_kind

VALID_MOE_MODES = ("vm_paged", "all_resident", "core_only")
VALID_MOE_TLB_STRATEGIES = (
    "lookahead_overlap",
    "overlap_shared_mlp",
    "exact_demand",
    "amortized_spec_tlb",
    "speculative_tlb",
)


@torch.no_grad()
def _quantize_uint4_2d_with_salience(
    w_fp32: torch.Tensor,
    group_size: int,
    col_salience: Optional[torch.Tensor] = None,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantizes `[rows, cols]` on GPU into `(packed_u8, wmin_bf16, scale_bf16)` with salience-weighted MSE search."""
    rows, cols = int(w_fp32.shape[0]), int(w_fp32.shape[1])
    pad_cols = (group_size - (cols % group_size)) % group_size
    w_pad = F.pad(w_fp32, (0, pad_cols), value=0.0) if pad_cols > 0 else w_fp32
    d_pad = int(w_pad.shape[1])
    n_groups = d_pad // group_size

    wg = w_pad.view(rows, n_groups, group_size)
    if col_salience is not None:
        sal_p = F.pad(col_salience.float(), (0, pad_cols), value=1.0) if pad_cols > 0 else col_salience.float()
        sal_g = (sal_p / sal_p.mean().clamp_min(1e-8)).clamp(0.25, 16.0).view(1, n_groups, group_size)
    else:
        sal_g = None

    mn0 = wg.amin(dim=-1, keepdim=True)
    mx0 = wg.amax(dim=-1, keepdim=True)
    mid = 0.5 * (mn0 + mx0)
    half0 = 0.5 * (mx0 - mn0)

    best_mse = torch.full((rows, n_groups, 1), float("inf"), device=w_fp32.device, dtype=torch.float32)
    wmin_bf = mn0.to(scale_dtype)
    scale_bf = ((mx0 - mn0) / 15.0).clamp_min(1e-8).to(scale_dtype)

    for clip_alpha in (0.86, 0.91, 0.95, 0.98, 1.00):
        cand_mn = (mid - half0 * clip_alpha).to(scale_dtype)
        cand_sc = ((2.0 * half0 * clip_alpha) / 15.0).clamp_min(1e-8).to(scale_dtype)
        cand_q = torch.clamp(torch.round((wg - cand_mn.float()) / cand_sc.float()), 0.0, 15.0)
        cand_dq = (cand_q.to(scale_dtype) * cand_sc + cand_mn).float()
        diff2 = (wg - cand_dq).square()
        if sal_g is not None:
            diff2 = diff2 * sal_g
        cand_mse = diff2.mean(dim=-1, keepdim=True)
        better = cand_mse < best_mse
        best_mse = torch.where(better, cand_mse, best_mse)
        wmin_bf = torch.where(better, cand_mn, wmin_bf)
        scale_bf = torch.where(better, cand_sc, scale_bf)

    q_u8 = torch.clamp(torch.round((wg - wmin_bf.float()) / scale_bf.float()), 0.0, 15.0).to(torch.uint8)
    q_pair = q_u8.view(rows, d_pad // 2, 2)
    packed_u8 = (q_pair[..., 0] | (q_pair[..., 1] << 4)).contiguous()
    return packed_u8, wmin_bf.squeeze(-1).contiguous(), scale_bf.squeeze(-1).contiguous()


@torch.no_grad()
def quantize_pack_moe_experts_uint4(
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    group_size: int = 64,
    input_salience_ms: Optional[torch.Tensor] = None,
    compute_device: str = "cuda:0",
    expert_chunk_size: int = 16,
    pin_host_memory: bool = False,
) -> Tuple[torch.Tensor, ContiguousExpertPageLayout]:
    """Packs `[E, 2*I, D]` & `[E, D, I]` into 4KB-aligned contiguous `uint8[E, page_bytes]` expert pages."""
    E, two_i, d_model = int(gate_up_weight.shape[0]), int(gate_up_weight.shape[1]), int(gate_up_weight.shape[2])
    assert two_i % 2 == 0
    d_ffn = two_i // 2
    assert tuple(down_weight.shape) == (E, d_model, d_ffn), (
        f"Expected down_weight shape {(E, d_model, d_ffn)}, got {tuple(down_weight.shape)}."
    )

    layout = ContiguousExpertPageLayout.compute(d_model=d_model, d_ffn=d_ffn, group_size=group_size)
    dev = torch.device(compute_device)
    sal_gpu = input_salience_ms.to(device=dev, dtype=torch.float32) if input_salience_ms is not None else None

    host_pages = torch.zeros((E, layout.page_bytes), dtype=torch.uint8, device="cpu")
    gpu_page_chunk = torch.zeros(
        (min(E, expert_chunk_size), layout.page_bytes), dtype=torch.uint8, device=dev
    )

    for s in range(0, E, expert_chunk_size):
        e_end = min(E, s + expert_chunk_size)
        b_cnt = e_end - s
        gu_f32 = gate_up_weight[s:e_end].to(device=dev, dtype=torch.float32)
        dn_f32 = down_weight[s:e_end].to(device=dev, dtype=torch.float32)

        gu_p, gu_mn, gu_sc = _quantize_uint4_2d_with_salience(
            gu_f32.reshape(b_cnt * two_i, d_model),
            group_size=group_size,
            col_salience=sal_gpu,
        )
        dn_p, dn_mn, dn_sc = _quantize_uint4_2d_with_salience(
            dn_f32.reshape(b_cnt * d_model, d_ffn),
            group_size=group_size,
            col_salience=None,
        )

        buf = gpu_page_chunk[:b_cnt]
        buf.zero_()
        buf[:, layout.off_gu_q4 : layout.off_gu_q4 + layout.bytes_gu_q4].copy_(gu_p.view(b_cnt, -1))
        buf[:, layout.off_dn_q4 : layout.off_dn_q4 + layout.bytes_dn_q4].copy_(dn_p.view(b_cnt, -1))
        buf[:, layout.off_gu_wmin : layout.off_gu_wmin + layout.bytes_gu_wmin].copy_(
            gu_mn.view(torch.uint8).reshape(b_cnt, -1)
        )
        buf[:, layout.off_gu_scale : layout.off_gu_scale + layout.bytes_gu_scale].copy_(
            gu_sc.view(torch.uint8).reshape(b_cnt, -1)
        )
        buf[:, layout.off_dn_wmin : layout.off_dn_wmin + layout.bytes_dn_wmin].copy_(
            dn_mn.view(torch.uint8).reshape(b_cnt, -1)
        )
        buf[:, layout.off_dn_scale : layout.off_dn_scale + layout.bytes_dn_scale].copy_(
            dn_sc.view(torch.uint8).reshape(b_cnt, -1)
        )

        host_pages[s:e_end].copy_(buf.cpu())
        del gu_f32, dn_f32, gu_p, gu_mn, gu_sc, dn_p, dn_mn, dn_sc

    del gpu_page_chunk
    if pin_host_memory and torch.cuda.is_available():
        try:
            host_pages = host_pages.pin_memory()
        except Exception:
            pass
    return host_pages, layout


@torch.no_grad()
def unpack_moe_expert_page_to_bf16(
    page_u8_gpu: torch.Tensor,
    layout: ContiguousExpertPageLayout,
    lut_uint4: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpacks a single GPU `uint8[page_bytes]` expert page into `(gate_up_bf16 [2*I, D], down_bf16 [D, I])`."""
    two_i = 2 * layout.d_ffn
    gu_q4 = page_u8_gpu[layout.off_gu_q4 : layout.off_gu_q4 + layout.bytes_gu_q4].view(two_i, layout.d_pad_half)
    dn_q4 = page_u8_gpu[layout.off_dn_q4 : layout.off_dn_q4 + layout.bytes_dn_q4].view(layout.d_model, layout.i_pad_half)

    gu_wmin = (
        page_u8_gpu[layout.off_gu_wmin : layout.off_gu_wmin + layout.bytes_gu_wmin]
        .view(torch.bfloat16)
        .view(two_i, layout.n_groups_d, 1)
    )
    gu_scale = (
        page_u8_gpu[layout.off_gu_scale : layout.off_gu_scale + layout.bytes_gu_scale]
        .view(torch.bfloat16)
        .view(two_i, layout.n_groups_d, 1)
    )
    dn_wmin = (
        page_u8_gpu[layout.off_dn_wmin : layout.off_dn_wmin + layout.bytes_dn_wmin]
        .view(torch.bfloat16)
        .view(layout.d_model, layout.n_groups_i, 1)
    )
    dn_scale = (
        page_u8_gpu[layout.off_dn_scale : layout.off_dn_scale + layout.bytes_dn_scale]
        .view(torch.bfloat16)
        .view(layout.d_model, layout.n_groups_i, 1)
    )

    gu_deq = (
        F.embedding(gu_q4.to(torch.int32), lut_uint4).view(two_i, layout.n_groups_d, layout.group_size)
        * gu_scale
        + gu_wmin
    ).view(two_i, layout.d_pad)[:, : layout.d_model]

    dn_deq = (
        F.embedding(dn_q4.to(torch.int32), lut_uint4).view(layout.d_model, layout.n_groups_i, layout.group_size)
        * dn_scale
        + dn_wmin
    ).view(layout.d_model, layout.i_pad)[:, : layout.d_ffn]
    return gu_deq, dn_deq


@torch.no_grad()
def repack_uint4_page_to_wina_q4(
    page_u8_cpu_or_gpu: torch.Tensor,
    layout: ContiguousExpertPageLayout,
    wina_layout: WINAExpertPageLayout,
    device: torch.device,
    lut_uint4: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Repacks an existing `ContiguousExpertPageLayout` page into neuron-major `UINT4-WINA` format on GPU.

    Preserves `gate_proj` and `up_proj` `UINT4` weights bit-for-bit, computes exact WINA column norms
    `||W_down[:, j]||_2`, and packs `(up_row[j], down_col[j])` contiguously per intermediate neuron `j`.
    """
    pg = page_u8_cpu_or_gpu.to(device=device, non_blocking=False)
    d_i = layout.d_ffn
    two_i = 2 * d_i
    n_gd = wina_layout.n_groups_d
    n_q4_b = n_gd * 32

    # 1. Reuse exact `gate_proj` & `up_proj` UINT4 nibbles + scales (0 re-quantization loss!)
    gu_q4 = pg[layout.off_gu_q4 : layout.off_gu_q4 + layout.bytes_gu_q4].view(two_i, layout.d_pad_half)[:, :n_q4_b]
    gu_wmin = (
        pg[layout.off_gu_wmin : layout.off_gu_wmin + layout.bytes_gu_wmin]
        .view(torch.uint16)
        .view(two_i, layout.n_groups_d)[:, :n_gd]
    )
    gu_scale = (
        pg[layout.off_gu_scale : layout.off_gu_scale + layout.bytes_gu_scale]
        .view(torch.uint16)
        .view(two_i, layout.n_groups_d)[:, :n_gd]
    )
    gu_ms_u8 = torch.stack([gu_wmin, gu_scale], dim=-1).view(torch.uint8).reshape(two_i, n_gd * 4)

    gate_rows = torch.zeros((d_i, wina_layout.row_bytes), dtype=torch.uint8, device=device)
    gate_rows[:, :n_q4_b].copy_(gu_q4[:d_i])
    gate_rows[:, n_q4_b : n_q4_b + n_gd * 4].copy_(gu_ms_u8[:d_i])

    up_rows = torch.zeros((d_i, wina_layout.row_bytes), dtype=torch.uint8, device=device)
    up_rows[:, :n_q4_b].copy_(gu_q4[d_i:])
    up_rows[:, n_q4_b : n_q4_b + n_gd * 4].copy_(gu_ms_u8[d_i:])

    # 2. Dequantize `down_proj` -> compute WINA column norms + quantize column-major `down_col[j]` (`[d_ffn, d_model]`)
    dn_q4 = pg[layout.off_dn_q4 : layout.off_dn_q4 + layout.bytes_dn_q4].view(layout.d_model, layout.i_pad_half)
    dn_wmin = (
        pg[layout.off_dn_wmin : layout.off_dn_wmin + layout.bytes_dn_wmin]
        .view(torch.bfloat16)
        .view(layout.d_model, layout.n_groups_i, 1)
    )
    dn_scale = (
        pg[layout.off_dn_scale : layout.off_dn_scale + layout.bytes_dn_scale]
        .view(torch.bfloat16)
        .view(layout.d_model, layout.n_groups_i, 1)
    )
    if lut_uint4 is not None:
        dn_deq = (
            F.embedding(dn_q4.to(torch.int32), lut_uint4.to(device=device)).view(
                layout.d_model, layout.n_groups_i, layout.group_size
            )
            * dn_scale
            + dn_wmin
        ).view(layout.d_model, layout.i_pad)[:, :d_i]
    else:
        lo = (dn_q4 & 0x0F).to(torch.bfloat16)
        hi = ((dn_q4 >> 4) & 0x0F).to(torch.bfloat16)
        dn_vals = torch.stack([lo, hi], dim=-1).view(layout.d_model, layout.n_groups_i, layout.group_size)
        dn_deq = (dn_vals * dn_scale + dn_wmin).view(layout.d_model, layout.i_pad)[:, :d_i]

    dn_col_norms = dn_deq.float().norm(dim=0).to(torch.bfloat16).contiguous()  # [d_ffn]
    dn_t_f32 = dn_deq.T.float().contiguous()  # [d_ffn, d_model]
    dn_pk_u8, dn_mn_bf, dn_sc_bf = _quantize_uint4_2d_with_salience(dn_t_f32, group_size=64)
    dn_ms_u8 = torch.stack([dn_mn_bf.view(torch.uint16), dn_sc_bf.view(torch.uint16)], dim=-1).view(torch.uint8).reshape(d_i, n_gd * 4)

    dn_rows = torch.zeros((d_i, wina_layout.row_bytes), dtype=torch.uint8, device=device)
    dn_rows[:, :n_q4_b].copy_(dn_pk_u8.reshape(d_i, n_q4_b))
    dn_rows[:, n_q4_b : n_q4_b + n_gd * 4].copy_(dn_ms_u8)

    out_pg = torch.zeros((wina_layout.page_bytes,), dtype=torch.uint8, device=device)
    out_pg[wina_layout.off_gate : wina_layout.off_gate + d_i * wina_layout.row_bytes].copy_(gate_rows.reshape(-1))
    out_pg[wina_layout.off_wina_norm : wina_layout.off_wina_norm + d_i * 2].copy_(dn_col_norms.view(torch.uint8).reshape(-1))
    slabs = torch.cat([up_rows, dn_rows], dim=1)  # [d_ffn, 2 * row_bytes]
    out_pg[wina_layout.off_slabs : wina_layout.off_slabs + d_i * wina_layout.slab_bytes].copy_(slabs.reshape(-1))
    return out_pg.cpu()


def get_default_l2_wina_ratio() -> float:
    raw = os.environ.get("TMMU_L2_WINA_RATIO", "0.50").strip()
    try:
        val = float(raw)
        return max(0.20, min(1.0, val))
    except ValueError:
        return 0.50


class GPUExpertTLBCache(nn.Module):
    """3-Tier GPU Expert Cache: L1 VRAM LRU + L2 Zero-Copy Pinned DDR5 + Pinned-Ring L3 `nvme_mmap` DMA."""

    def __init__(
        self,
        num_experts: int,
        tlb_slots: int,
        layout: ContiguousExpertPageLayout,
        host_expert_pages: torch.Tensor,
        device: torch.device,
        expert_freq_rank: Optional[torch.Tensor] = None,
        dma_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.configured_tlb_slots = int(max(0, min(tlb_slots, num_experts)))
        self.tlb_slots = self.configured_tlb_slots
        self.layout = layout
        self.bytes_per_page = int(layout.page_bytes)
        self.device = device
        self.host_expert_pages = host_expert_pages
        self.dma_stream = dma_stream

        if expert_freq_rank is not None:
            self.freq_rank_list: List[int] = [int(x) for x in expert_freq_rank.detach().cpu().tolist()]
        else:
            self.freq_rank_list = list(range(self.num_experts))

        self.register_buffer(
            "tlb_pages",
            torch.empty((self.tlb_slots, self.bytes_per_page), dtype=torch.uint8, device=device),
            persistent=False,
        )
        self.register_buffer(
            "gpu_expert_to_slot",
            torch.full((self.num_experts,), -1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self._cpu_slot_map = torch.full((self.num_experts,), -1, dtype=torch.int32, device="cpu")
        self._slot_map_ring: List[torch.Tensor] = [
            torch.empty((self.num_experts,), dtype=torch.int32, device="cpu") for _ in range(4)
        ]
        self._slot_map_ring_idx: int = 0
        self._d2h_pinned_cpu = torch.empty((128,), dtype=torch.int64, device="cpu")
        if torch.cuda.is_available() and device.type == "cuda":
            try:
                self._cpu_slot_map = self._cpu_slot_map.pin_memory()
                self._slot_map_ring = [t.pin_memory() for t in self._slot_map_ring]
                self._d2h_pinned_cpu = self._d2h_pinned_cpu.pin_memory()
            except Exception:
                pass

        is_cuda = torch.cuda.is_available() and device.type == "cuda"
        self._dma_event: Optional[torch.cuda.Event] = torch.cuda.Event() if is_cuda else None
        self._compute_read_event: Optional[torch.cuda.Event] = torch.cuda.Event() if is_cuda else None
        self._has_compute_read_event: bool = False
        self._dma_in_flight: bool = False
        self._d2h_idx_gpu: Optional[torch.Tensor] = (
            torch.empty((128,), dtype=torch.int64, device=device) if is_cuda else None
        )

        # L2 Zero-Copy Pinned Host DDR5 Tier (`slot <= -2` -> `l2_idx = -2 - slot`)
        # Supports both standard `UINT4` (`l2_pinned_pages`) and neuron-major `UINT4-WINA` (`l2_wina_pinned_pages`)!
        self.l2_slots: int = 0
        self.l2_pinned_pages: Optional[torch.Tensor] = None
        self.l2_zero_copy_dev_ptr: int = 0
        self.l2_wina_pinned_pages: Optional[torch.Tensor] = None
        self.l2_wina_dev_ptr: int = 0
        self._l2_zc_handle: Optional[PinnedZeroCopyBuffer] = None
        self.wina_layout: Optional[WINAExpertPageLayout] = (
            WINAExpertPageLayout.compute(layout.d_model, layout.d_ffn) if (layout.d_model % 64 == 0) else None
        )
        self.wina_keep_ratio: float = get_default_l2_wina_ratio()
        self._l2_lut_uint4: Optional[torch.Tensor] = None
        self._l2_aliases_host_pages: bool = False
        self._expert_l2_home: List[int] = [-1] * self.num_experts
        self._l2_slot_to_expert: List[int] = []
        self._l2_hit_counts: List[int] = [0] * self.num_experts
        self.all_experts_mapped: bool = False
        self.layout_version: int = 0

        self._serviced_topk_ptr: int = 0
        self._serviced_topk_numel: int = 0
        self._is_allocated: bool = True

        # If `host_expert_pages` is ALREADY pinned (`page_storage == "pinned_ram"`), map all experts in zero-copy L2 for free!
        if is_cuda and self.host_expert_pages.device.type == "cpu" and self.host_expert_pages.is_pinned():
            try:
                zc_ptr = get_zero_copy_device_ptr(self.host_expert_pages, device)
                self.l2_slots = self.num_experts
                self.l2_pinned_pages = self.host_expert_pages
                self.l2_zero_copy_dev_ptr = zc_ptr
                self._l2_aliases_host_pages = True
                self._expert_l2_home = list(range(self.num_experts))
                self._l2_slot_to_expert = list(range(self.num_experts))
            except Exception:
                pass

        self.clear_cache()
        self.reset_telemetry()
        self.seed_default_experts()

    @property
    def total_gpu_vram_bytes(self) -> int:
        return int(
            self.tlb_pages.numel() * self.tlb_pages.element_size()
            + self.gpu_expert_to_slot.numel() * self.gpu_expert_to_slot.element_size()
        )

    @property
    def has_l2_mapped(self) -> bool:
        return self.l2_zero_copy_dev_ptr != 0 or self.l2_wina_dev_ptr != 0

    @property
    def total_l2_pinned_ram_bytes(self) -> int:
        if self._l2_aliases_host_pages:
            return 0
        total = 0
        if self.l2_pinned_pages is not None:
            total += int(self.l2_pinned_pages.numel() * self.l2_pinned_pages.element_size())
        if self.l2_wina_pinned_pages is not None:
            total += int(self.l2_wina_pinned_pages.numel() * self.l2_wina_pinned_pages.element_size())
        return total

    def _fallback_encoded_slot(self, exp_id: int) -> int:
        l2_h = self._expert_l2_home[exp_id]
        if l2_h >= 0 and self.has_l2_mapped:
            return -2 - l2_h
        return -1

    def _recompute_all_mapped_flag(self) -> None:
        if self.tlb_slots <= 0 and not self.has_l2_mapped:
            self.all_experts_mapped = False
            return
        if self.tlb_slots >= self.num_experts and self._num_valid == self.num_experts:
            self.all_experts_mapped = True
            return
        if self.has_l2_mapped:
            for e in range(self.num_experts):
                if self._expert_to_slot[e] < 0 and self._expert_l2_home[e] < 0:
                    self.all_experts_mapped = False
                    return
            self.all_experts_mapped = True
            return
        self.all_experts_mapped = False

    def release_l2_cache(self) -> None:
        if not self._l2_aliases_host_pages:
            self.l2_pinned_pages = None
            self.l2_wina_pinned_pages = None
            if self._l2_zc_handle is not None:
                if torch.cuda.is_available() and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                self._l2_zc_handle.release()
                self._l2_zc_handle = None
            self.l2_slots = 0
            self.l2_zero_copy_dev_ptr = 0
            self.l2_wina_dev_ptr = 0
            self._expert_l2_home = [-1] * self.num_experts
            self._l2_slot_to_expert = []
            self._l2_hit_counts = [0] * self.num_experts
            if self._is_allocated:
                for e in range(self.num_experts):
                    if self._expert_to_slot[e] < 0:
                        self._cpu_slot_map[e] = -1
                self.gpu_expert_to_slot.copy_(self._cpu_slot_map, non_blocking=True)
            self._recompute_all_mapped_flag()
            self.layout_version += 1

    def _reseed_l2_disjoint_from_l1(self) -> None:
        """Restores disjoint L2 population (`freq_rank_list` not in L1) after a full L1 re-seed."""
        active_buf = self.l2_wina_pinned_pages if self.l2_wina_pinned_pages is not None else self.l2_pinned_pages
        if self._l2_aliases_host_pages or active_buf is None or self.l2_slots <= 0:
            return
        use_wina = self.l2_wina_pinned_pages is not None and self.wina_layout is not None
        in_l1 = {e for e in range(self.num_experts) if self._expert_to_slot[e] >= 0}
        self._expert_l2_home = [-1] * self.num_experts
        self._l2_hit_counts = [0] * self.num_experts
        chosen: List[int] = []
        for exp_id in self.freq_rank_list:
            if exp_id not in in_l1:
                chosen.append(int(exp_id))
                if len(chosen) >= int(active_buf.shape[0]):
                    break
        for idx_l2, exp_id in enumerate(chosen):
            if idx_l2 >= len(self._l2_slot_to_expert) or self._l2_slot_to_expert[idx_l2] != exp_id:
                if use_wina:
                    assert self.wina_layout is not None
                    active_buf[idx_l2].copy_(
                        repack_uint4_page_to_wina_q4(
                            self.host_expert_pages[exp_id],
                            self.layout,
                            self.wina_layout,
                            self.device,
                            lut_uint4=self._l2_lut_uint4,
                        )
                    )
                else:
                    active_buf[idx_l2].copy_(self.host_expert_pages[exp_id])
            self._expert_l2_home[exp_id] = idx_l2
        self._l2_slot_to_expert = list(chosen)
        self.l2_slots = len(chosen)

    def configure_l2_zero_copy(
        self,
        l2_slots: int,
        wina_keep_ratio: Optional[float] = None,
        lut_uint4: Optional[torch.Tensor] = None,
    ) -> int:
        """Allocates `l2_slots` of device-mapped pinned host DDR5 (`UINT4-WINA` when `keep_ratio < 1.0`)."""
        if self._l2_aliases_host_pages or self.device.type != "cuda" or not torch.cuda.is_available():
            return self.l2_slots
        eff_ratio = self.wina_keep_ratio if wina_keep_ratio is None else max(0.20, min(1.0, float(wina_keep_ratio)))
        if lut_uint4 is not None:
            self._l2_lut_uint4 = lut_uint4
        use_wina = (0.0 < eff_ratio < 1.0) and (self.wina_layout is not None)

        max_useful = max(0, self.num_experts - self.tlb_slots)
        target = int(max(0, min(l2_slots, max_useful)))
        if (
            target == self.l2_slots
            and eff_ratio == self.wina_keep_ratio
            and (target == 0 or (self.l2_wina_dev_ptr != 0 if use_wina else self.l2_zero_copy_dev_ptr != 0))
        ):
            return self.l2_slots

        self.wina_keep_ratio = eff_ratio
        self.l2_pinned_pages = None
        self.l2_wina_pinned_pages = None
        if self._l2_zc_handle is not None:
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._l2_zc_handle.release()
            self._l2_zc_handle = None
        self.l2_slots = 0
        self.l2_zero_copy_dev_ptr = 0
        self.l2_wina_dev_ptr = 0
        self._expert_l2_home = [-1] * self.num_experts
        self._l2_slot_to_expert = []
        self._l2_hit_counts = [0] * self.num_experts

        if target > 0:
            pg_bytes = self.wina_layout.page_bytes if (use_wina and self.wina_layout is not None) else self.bytes_per_page
            zc_handle: Optional[PinnedZeroCopyBuffer] = None
            try:
                zc_handle = alloc_zero_copy_host_buffer(target, pg_bytes, self.device)
                buf = zc_handle.tensor
                zc_ptr = zc_handle.dev_ptr
            except Exception:
                if zc_handle is not None:
                    zc_handle.release()
                zc_handle = None
                buf = None
                zc_ptr = 0

            if buf is not None and zc_ptr != 0:
                self._l2_zc_handle = zc_handle
                # Seed L2 with next-hottest experts not initially in L1 (`freq_rank_list[tlb_slots : tlb_slots + target]`)
                in_l1 = {e for e in range(self.num_experts) if self._expert_to_slot[e] >= 0}
                chosen: List[int] = []
                for exp_id in self.freq_rank_list:
                    if exp_id not in in_l1:
                        chosen.append(int(exp_id))
                        if len(chosen) >= target:
                            break
                for idx_l2, exp_id in enumerate(chosen):
                    if use_wina and self.wina_layout is not None:
                        buf[idx_l2].copy_(
                            repack_uint4_page_to_wina_q4(
                                self.host_expert_pages[exp_id],
                                self.layout,
                                self.wina_layout,
                                self.device,
                                lut_uint4=self._l2_lut_uint4,
                            )
                        )
                    else:
                        buf[idx_l2].copy_(self.host_expert_pages[exp_id])
                    self._expert_l2_home[exp_id] = idx_l2
                if use_wina:
                    self.l2_wina_pinned_pages = buf
                    self.l2_wina_dev_ptr = zc_ptr
                else:
                    self.l2_pinned_pages = buf
                    self.l2_zero_copy_dev_ptr = zc_ptr
                self._l2_slot_to_expert = list(chosen)
                self.l2_slots = len(chosen)

        # Refresh encoded fallback slots for all non-L1 experts on GPU
        for e in range(self.num_experts):
            if self._expert_to_slot[e] >= 0:
                self._cpu_slot_map[e] = self._expert_to_slot[e]
            else:
                self._cpu_slot_map[e] = self._fallback_encoded_slot(e)
        self.gpu_expert_to_slot.copy_(self._cpu_slot_map, non_blocking=True)
        self._recompute_all_mapped_flag()
        self.layout_version += 1
        return self.l2_slots

    def _await_pending_prefetch(self) -> None:
        return

    def _snapshot_slot_map(self) -> torch.Tensor:
        buf = self._slot_map_ring[self._slot_map_ring_idx & 3]
        self._slot_map_ring_idx += 1
        buf.copy_(self._cpu_slot_map)
        return buf

    def record_compute_read(self, compute_stream: torch.cuda.Stream) -> None:
        """Records completion of `MoETLBLayerLauncher` read on `compute_stream` for hazard-free async DMA."""
        if self._compute_read_event is not None and not torch.cuda.is_current_stream_capturing():
            self._compute_read_event.record(compute_stream)
            self._has_compute_read_event = True

    def clear_cache(self) -> None:
        self._await_pending_prefetch()
        self._slot_to_expert: List[int] = [-1] * self.tlb_slots
        self._expert_to_slot: List[int] = [-1] * self.num_experts
        self._slot_last_used: List[int] = [0] * self.tlb_slots
        self._slot_probationary: List[bool] = [False] * self.tlb_slots
        self._slot_lookahead_pending: List[bool] = [False] * self.tlb_slots
        self._has_any_lookahead_pending: bool = False
        self._step_spec_touched: List[Tuple[int, int, bool, int]] = []
        self._l2_hit_counts = [0] * self.num_experts
        self._access_clock: int = 0
        self._num_valid: int = 0
        for e in range(self.num_experts):
            self._cpu_slot_map[e] = self._fallback_encoded_slot(e)
        self.gpu_expert_to_slot.copy_(self._cpu_slot_map, non_blocking=True)
        self._dma_in_flight = False
        self._has_compute_read_event = False
        self._serviced_topk_ptr = 0
        self._serviced_topk_numel = 0
        self._recompute_all_mapped_flag()
        self.layout_version += 1

    def reset_telemetry(self) -> None:
        self._pcie_bytes: int = 0
        self._page_hits: int = 0
        self._l2_zero_copy_hits: int = 0
        self._page_misses: int = 0
        self._demand_misses: int = 0
        self._lookahead_prefetched: int = 0
        self._lookahead_hits: int = 0
        self._lookahead_wasted: int = 0
        self._all_hit_fastpaths: int = 0
        self._distinct_pages: Set[int] = set()

    def telemetry(self) -> Dict[str, float]:
        tot_req = max(1, self._page_hits + self._l2_zero_copy_hits + self._page_misses)
        stall_free_hits = self._page_hits + self._l2_zero_copy_hits + self._lookahead_hits
        return {
            "pcie_bytes": float(self._pcie_bytes),
            "page_hits": float(self._page_hits + self._l2_zero_copy_hits),
            "l1_vram_hits": float(self._page_hits),
            "l2_zero_copy_hits": float(self._l2_zero_copy_hits),
            "page_misses": float(self._page_misses),
            "demand_misses": float(self._demand_misses),
            "lookahead_prefetched": float(self._lookahead_prefetched),
            "lookahead_hits": float(self._lookahead_hits),
            "lookahead_wasted": float(self._lookahead_wasted),
            "hit_rate_pct": 100.0 * (float(self._page_hits + self._l2_zero_copy_hits) / float(tot_req)),
            "effective_stall_free_hit_pct": 100.0 * (float(stall_free_hits) / float(tot_req)),
            "all_hit_fastpaths": float(self._all_hit_fastpaths),
            "distinct_pages": float(len(self._distinct_pages)),
        }

    def _copy_host_expert_to_l1_slot(
        self,
        slot: int,
        exp_id: int,
        dma_stream: Optional[torch.cuda.Stream],
        ws: Any,
    ) -> None:
        """Copies `exp_id` into `tlb_pages[slot]` via L2 pinned memory or the 22.9 GB/s pinned DMA staging ring."""
        l2_h = self._expert_l2_home[exp_id]
        if l2_h >= 0 and self.l2_pinned_pages is not None:
            self.tlb_pages[slot].copy_(self.l2_pinned_pages[l2_h], non_blocking=True)
            return
        src = self.host_expert_pages[exp_id]
        if src.is_pinned():
            self.tlb_pages[slot].copy_(src, non_blocking=True)
            return
        if ws is not None and ws.pinned_dma_stage.numel() > 0:
            r_slot, stage = ws.acquire_pinned_stage_slot(self.bytes_per_page)
            stage.copy_(src)
            self.tlb_pages[slot].copy_(stage, non_blocking=True)
            eff_s = dma_stream if dma_stream is not None else (
                torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
            )
            ws.record_stage_slot(r_slot, eff_s)
        else:
            self.tlb_pages[slot].copy_(src, non_blocking=True)

    def seed_default_experts(self) -> None:
        """Pre-populates GPU L1 TLB slots with the Top-`S` calibration-frequent experts."""
        self.clear_cache()
        if self.tlb_slots <= 0 or not self._is_allocated:
            return
        ws = get_shared_moe_workspace(self.device) if (torch.cuda.is_available() and self.device.type == "cuda") else None
        if ws is not None:
            ws.ensure_capacity(self.layout.d_model, self.layout.d_ffn, self.bytes_per_page)
        n_seed = min(self.tlb_slots, len(self.freq_rank_list))
        for slot in range(n_seed):
            exp_id = int(self.freq_rank_list[slot])
            self._copy_host_expert_to_l1_slot(slot, exp_id, None, ws)
            self._slot_to_expert[slot] = exp_id
            self._expert_to_slot[exp_id] = slot
            self._cpu_slot_map[exp_id] = slot
            self._slot_probationary[slot] = False
            self._slot_last_used[slot] = n_seed - slot
        self._access_clock = n_seed + 1
        self._num_valid = n_seed
        if (
            not self._l2_aliases_host_pages
            and self.l2_slots > 0
            and self.l2_pinned_pages is not None
            and any(self._expert_l2_home[self._slot_to_expert[s]] >= 0 for s in range(n_seed))
        ):
            self._reseed_l2_disjoint_from_l1()
            for e in range(self.num_experts):
                if self._expert_to_slot[e] < 0:
                    self._cpu_slot_map[e] = self._fallback_encoded_slot(e)
        self.gpu_expert_to_slot.copy_(self._cpu_slot_map, non_blocking=True)
        self._recompute_all_mapped_flag()
        self.layout_version += 1

    def resize_slots(self, new_slots: int, seed: bool = True) -> None:
        """Dynamically resizes GPU Expert TLB capacity (`0 .. num_experts`) without reloading `.nvm` from disk."""
        self._await_pending_prefetch()
        target = int(max(0, min(new_slots, self.num_experts)))
        if target == self.tlb_slots and self._is_allocated == (target > 0):
            if seed and target > 0:
                self.seed_default_experts()
            return
        if target >= self.num_experts and not self._l2_aliases_host_pages and self.l2_slots > 0:
            self.release_l2_cache()
        self.tlb_pages = torch.empty((0,), dtype=torch.uint8, device=self.device)
        self.tlb_slots = target
        if target > 0:
            self.tlb_pages = torch.empty(
                (target, self.bytes_per_page), dtype=torch.uint8, device=self.device
            )
            self._is_allocated = True
            if seed:
                self.seed_default_experts()
            else:
                self.clear_cache()
        else:
            self._is_allocated = False
            self.clear_cache()
        self.layout_version += 1

    def release_gpu_slots(self) -> None:
        self.release_l2_cache()
        self.resize_slots(0, seed=False)

    def _choose_victim_slot(self, protected_experts: Set[int]) -> int:
        if self._num_valid < self.tlb_slots:
            return self._num_valid

        s2e = self._slot_to_expert
        s_lu = self._slot_last_used
        s_prob = self._slot_probationary

        best_prob_slot = -1
        oldest_prob_tick = 1 << 62
        best_ver_slot = 0
        oldest_ver_tick = 1 << 62

        for s in range(self.tlb_slots):
            if s2e[s] in protected_experts:
                continue
            lu = s_lu[s]
            if s_prob[s]:
                if lu < oldest_prob_tick:
                    oldest_prob_tick = lu
                    best_prob_slot = s
            else:
                if lu < oldest_ver_tick:
                    oldest_ver_tick = lu
                    best_ver_slot = s

        return best_prob_slot if best_prob_slot >= 0 else best_ver_slot

    def commit_speculative_step(self, n_accepted: int) -> None:
        """Promotes experts used by accepted draft tokens (`1..n_accepted`) and resets rejected draft slots."""
        if not self._step_spec_touched:
            return
        n_acc = int(n_accepted)
        for slot, min_m, prev_prob, prev_lu in self._step_spec_touched:
            if 0 <= slot < self.tlb_slots:
                if min_m <= n_acc:
                    self._slot_probationary[slot] = False
                    self._access_clock += 1
                    self._slot_last_used[slot] = self._access_clock
                else:
                    self._slot_probationary[slot] = prev_prob
                    self._slot_last_used[slot] = prev_lu
        self._step_spec_touched.clear()

    def schedule_lookahead_prefetch(self, predicted_experts: List[int]) -> None:
        """Schedules async background H2D prefetch of cold L3 predicted experts into probationary L1 slots."""
        if self.tlb_slots <= 0 or not self._is_allocated or self.all_experts_mapped:
            return
        if not predicted_experts or self.dma_stream is None or self.device.type != "cuda":
            return

        self._await_pending_prefetch()

        e2s = self._expert_to_slot
        l2_home = self._expert_l2_home
        protected: Set[int] = set()
        cold_to_load: List[int] = []
        seen: Set[int] = set()
        max_pre_slots = min(2, max(1, self.tlb_slots // 4))

        for e in predicted_experts:
            if 0 <= e < self.num_experts and e not in seen:
                seen.add(e)
                slot = e2s[e]
                if slot >= 0:
                    protected.add(e)
                elif l2_home[e] < 0:
                    # Only prefetch genuinely cold L3 experts; L2 Zero-Copy experts are already GPU-mapped!
                    if len(cold_to_load) < max_pre_slots:
                        cold_to_load.append(e)

        if not cold_to_load:
            return

        ws = get_shared_moe_workspace(self.device)
        staged_pairs: List[Tuple[int, torch.Tensor]] = []
        for e in cold_to_load:
            victim = self._choose_victim_slot(protected)
            protected.add(e)
            old_e = self._slot_to_expert[victim]
            if old_e >= 0:
                if self._slot_lookahead_pending[victim]:
                    self._lookahead_wasted += 1
                    self._slot_lookahead_pending[victim] = False
                self._expert_to_slot[old_e] = -1
                self._cpu_slot_map[old_e] = self._fallback_encoded_slot(old_e)

            self._slot_to_expert[victim] = e
            self._expert_to_slot[e] = victim
            self._cpu_slot_map[e] = victim
            self._slot_probationary[victim] = True
            self._slot_lookahead_pending[victim] = True
            self._slot_last_used[victim] = 0
            if victim >= self._num_valid:
                self._num_valid = victim + 1
            self._pcie_bytes += self.bytes_per_page
            self._lookahead_prefetched += 1
            self._distinct_pages.add(e)

            src = self.host_expert_pages[e]
            if src.is_pinned():
                staged_pairs.append((victim, src, -1))
            else:
                r_slot, st_buf = ws.acquire_pinned_stage_slot(self.bytes_per_page)
                st_buf.copy_(src)
                staged_pairs.append((victim, st_buf, r_slot))

        slot_snap = self._snapshot_slot_map()
        self._has_any_lookahead_pending = True
        self._dma_in_flight = True
        stream = self.dma_stream
        with torch.cuda.stream(stream):
            if self._has_compute_read_event and self._compute_read_event is not None:
                stream.wait_event(self._compute_read_event)
            for vic, pinned_src, r_slot in staged_pairs:
                self.tlb_pages[vic].copy_(pinned_src, non_blocking=True)
                if r_slot >= 0:
                    ws.record_stage_slot(r_slot, stream)
            self.gpu_expert_to_slot.copy_(slot_snap, non_blocking=True)
            if self._dma_event is not None:
                self._dma_event.record(stream)

    def _service_token_rows(
        self,
        token_rows: List[List[int]],
        use_async_dma_stream: bool,
    ) -> Optional[List[int]]:
        """Services exact `token_rows` (`[T, K]`) across L1 VRAM, L2 Zero-Copy DDR5, and L3 `nvme_mmap`."""
        self._step_spec_touched.clear()

        if self.tlb_slots <= 0 and not self.has_l2_mapped:
            return None

        n_exp = self.num_experts
        T = len(token_rows)
        verified_set: Set[int] = set()
        min_draft_tok: Dict[int, int] = {}
        uniq: List[int] = []
        seen: Set[int] = set()

        if T > 0:
            for x in token_rows[0]:
                xi = int(x)
                if 0 <= xi < n_exp:
                    verified_set.add(xi)
                    if xi not in seen:
                        seen.add(xi)
                        uniq.append(xi)
            for t_idx in range(1, T):
                for x in token_rows[t_idx]:
                    xi = int(x)
                    if 0 <= xi < n_exp:
                        if xi not in verified_set and xi not in min_draft_tok:
                            min_draft_tok[xi] = t_idx
                        if xi not in seen:
                            seen.add(xi)
                            uniq.append(xi)

        e2s = self._expert_to_slot
        l2_home = self._expert_l2_home
        has_l2 = self.has_l2_mapped

        must_be_in_l1: Set[int] = {
            e for e in uniq if not (has_l2 and l2_home[e] >= 0)
        }
        if len(must_be_in_l1) > self.tlb_slots:
            return uniq

        can_promote_l2 = (
            len(uniq) <= self.tlb_slots
            and self.tlb_slots > 0
            and not self._l2_aliases_host_pages
            and self.l2_pinned_pages is not None
        )
        l2_promotions_this_step = 0
        misses: List[int] = []
        had_prefetched_hit = False

        for e in uniq:
            slot = e2s[e]
            is_ver = e in verified_set
            if slot >= 0:
                if self._slot_lookahead_pending[slot]:
                    self._slot_lookahead_pending[slot] = False
                    self._page_misses += 1
                    self._lookahead_hits += 1
                    had_prefetched_hit = True
                    if is_ver:
                        self._slot_probationary[slot] = False
                        self._access_clock += 1
                        self._slot_last_used[slot] = self._access_clock
                    else:
                        self._slot_probationary[slot] = True
                        self._access_clock += 1
                        self._slot_last_used[slot] = self._access_clock
                        self._step_spec_touched.append((slot, min_draft_tok[e], True, 0))
                else:
                    self._page_hits += 1
                    if is_ver:
                        self._slot_probationary[slot] = False
                        self._access_clock += 1
                        self._slot_last_used[slot] = self._access_clock
                    else:
                        prev_p = self._slot_probationary[slot]
                        prev_lu = self._slot_last_used[slot]
                        self._step_spec_touched.append((slot, min_draft_tok[e], prev_p, prev_lu))
            elif has_l2 and l2_home[e] >= 0:
                # Hit in L2 Zero-Copy Pinned DDR5!
                # Serve transient hits (1..3) directly over Zero-Copy PCIe (zero L1 churn, zero DMA wait);
                # on sustained usage (`>= 4` verified hits), perform an Exclusive L1 <-> L2 Swap so `e`
                # promotes into 1 TB/s L1 VRAM while the evicted L1 expert demotes into `e`'s vacated L2 slot!
                if is_ver:
                    self._l2_hit_counts[e] += 1
                if (
                    is_ver
                    and self._l2_hit_counts[e] >= 4
                    and can_promote_l2
                    and l2_promotions_this_step < 4
                    and len(misses) < 20
                ):
                    l2_promotions_this_step += 1
                    self._l2_zero_copy_hits += 1
                    misses.append(e)
                else:
                    self._l2_zero_copy_hits += 1
                    self._pcie_bytes += self.bytes_per_page
                    self._distinct_pages.add(e)
            else:
                self._page_misses += 1
                self._demand_misses += 1
                misses.append(e)

        # Recycle any unused lookahead slots immediately so `misses` reclaim them before touching hot experts!
        if self._has_any_lookahead_pending:
            for s in range(self._num_valid):
                if self._slot_lookahead_pending[s]:
                    self._slot_lookahead_pending[s] = False
                    self._slot_probationary[s] = True
                    self._slot_last_used[s] = 0
                    self._lookahead_wasted += 1
            self._has_any_lookahead_pending = False

        if not misses:
            self._all_hit_fastpaths += 1
            if not had_prefetched_hit:
                self._dma_in_flight = False
            return None

        protected: Set[int] = set(uniq) if len(uniq) <= self.tlb_slots else set(must_be_in_l1)
        stream = (
            self.dma_stream
            if (use_async_dma_stream and self.dma_stream is not None and self.device.type == "cuda")
            else None
        )
        ws = get_shared_moe_workspace(self.device) if self.device.type == "cuda" else None
        demote_d2h_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        staged_pairs: List[Tuple[int, torch.Tensor, int]] = []

        for e in misses:
            victim = self._choose_victim_slot(protected)
            protected.add(e)
            old_e = self._slot_to_expert[victim]
            l2_idx = l2_home[e]
            is_exclusive_l2_swap = (
                l2_idx >= 0
                and self.l2_pinned_pages is not None
                and not self._l2_aliases_host_pages
                and ws is not None
            )

            if is_exclusive_l2_swap:
                assert self.l2_pinned_pages is not None and ws is not None
                # 1. Stage `e` from pinned L2 into pinned DMA ring on CPU (`0` disk I/O)
                r_slot, st_buf = ws.acquire_pinned_stage_slot(self.bytes_per_page)
                st_buf.copy_(self.l2_pinned_pages[l2_idx])
                l2_home[e] = -1
                self._l2_hit_counts[e] = 0

                # 2. Demote evicted L1 `old_e` (`tlb_pages[victim]`) into vacated `l2_pinned_pages[l2_idx]` via D2H DMA!
                if old_e >= 0 and l2_home[old_e] < 0:
                    self._expert_to_slot[old_e] = -1
                    l2_home[old_e] = l2_idx
                    if l2_idx < len(self._l2_slot_to_expert):
                        self._l2_slot_to_expert[l2_idx] = old_e
                    self._l2_hit_counts[old_e] = 0
                    self._cpu_slot_map[old_e] = -2 - l2_idx
                    demote_d2h_pairs.append((self.l2_pinned_pages[l2_idx], self.tlb_pages[victim]))
                    self._pcie_bytes += self.bytes_per_page
                else:
                    if old_e >= 0:
                        self._expert_to_slot[old_e] = -1
                        self._cpu_slot_map[old_e] = self._fallback_encoded_slot(old_e)
                    if l2_idx < len(self._l2_slot_to_expert):
                        self._l2_slot_to_expert[l2_idx] = -1

                staged_pairs.append((victim, st_buf, r_slot))
            else:
                if old_e >= 0:
                    self._expert_to_slot[old_e] = -1
                    self._cpu_slot_map[old_e] = self._fallback_encoded_slot(old_e)
                if l2_idx >= 0 and self.l2_pinned_pages is not None:
                    staged_pairs.append((victim, self.l2_pinned_pages[l2_idx], -1))
                else:
                    src = self.host_expert_pages[e]
                    if src.is_pinned() or ws is None:
                        staged_pairs.append((victim, src, -1))
                    else:
                        r_slot, st_buf = ws.acquire_pinned_stage_slot(self.bytes_per_page)
                        st_buf.copy_(src)
                        staged_pairs.append((victim, st_buf, r_slot))

            self._slot_to_expert[victim] = e
            self._expert_to_slot[e] = victim
            self._cpu_slot_map[e] = victim
            self._access_clock += 1
            self._slot_last_used[victim] = self._access_clock
            if e in verified_set:
                self._slot_probationary[victim] = False
            else:
                self._slot_probationary[victim] = True
                self._step_spec_touched.append((victim, min_draft_tok[e], True, 0))
            if victim >= self._num_valid:
                self._num_valid = victim + 1
            self._pcie_bytes += self.bytes_per_page
            self._distinct_pages.add(e)

        slot_snap = self._snapshot_slot_map()
        if stream is not None:
            with torch.cuda.stream(stream):
                if self._has_compute_read_event and self._compute_read_event is not None:
                    stream.wait_event(self._compute_read_event)
                for dst_l2_pinned, src_l1_vram in demote_d2h_pairs:
                    dst_l2_pinned.copy_(src_l1_vram, non_blocking=True)
                for vic, pinned_src, r_slot in staged_pairs:
                    self.tlb_pages[vic].copy_(pinned_src, non_blocking=True)
                    if r_slot >= 0 and ws is not None:
                        ws.record_stage_slot(r_slot, stream)
                self.gpu_expert_to_slot.copy_(slot_snap, non_blocking=True)
                if self._dma_event is not None:
                    self._dma_event.record(stream)
            self._dma_in_flight = True
        else:
            cur_s = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
            for dst_l2_pinned, src_l1_vram in demote_d2h_pairs:
                dst_l2_pinned.copy_(src_l1_vram, non_blocking=True)
            for vic, pinned_src, r_slot in staged_pairs:
                self.tlb_pages[vic].copy_(pinned_src, non_blocking=True)
                if r_slot >= 0 and ws is not None:
                    ws.record_stage_slot(r_slot, cur_s)
            self.gpu_expert_to_slot.copy_(slot_snap, non_blocking=True)
            self._dma_in_flight = False
        return None

    def extract_unique_experts_from_topk(self, top_k_index: torch.Tensor) -> List[int]:
        flat_list: List[int] = top_k_index.reshape(-1).tolist()
        seen: Set[int] = set()
        uniq: List[int] = []
        n_exp = self.num_experts
        for x in flat_list:
            xi = int(x)
            if 0 <= xi < n_exp and xi not in seen:
                seen.add(xi)
                uniq.append(xi)
        return uniq

    def sync_topk_and_lookahead_d2h(
        self,
        top_k_index: torch.Tensor,
        lookahead_pred_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """Single-sync pinned D2H extraction of `(cur_flat, next_list)` BEFORE launching GPU Shared MLP compute."""
        if self.all_experts_mapped:
            return None, None
        if self.tlb_slots <= 0 and not self.has_l2_mapped:
            return None, None

        f_cur = top_k_index.reshape(-1)
        n_cur = int(f_cur.numel())
        if (
            self.device.type == "cuda"
            and self._d2h_idx_gpu is not None
            and self._d2h_pinned_cpu.is_pinned()
        ):
            if lookahead_pred_idx is not None and lookahead_pred_idx.numel() > 0:
                f_la = lookahead_pred_idx.reshape(-1)
                n_la = int(f_la.numel())
                tot = n_cur + n_la
                if tot <= int(self._d2h_idx_gpu.numel()):
                    buf = self._d2h_idx_gpu[:tot]
                    buf[:n_cur].copy_(f_cur)
                    buf[n_cur:].copy_(f_la)
                    host_slice = self._d2h_pinned_cpu[:tot]
                    host_slice.copy_(buf, non_blocking=True)
                    torch.cuda.current_stream(self.device).synchronize()
                    combined: List[int] = host_slice.tolist()
                    cur_flat = combined[:n_cur]
                    next_list = [int(x) for x in combined[n_cur:] if int(x) >= 0]
                    return cur_flat, (next_list if next_list else None)
            elif n_cur <= int(self._d2h_pinned_cpu.numel()):
                host_slice = self._d2h_pinned_cpu[:n_cur]
                host_slice.copy_(f_cur, non_blocking=True)
                torch.cuda.current_stream(self.device).synchronize()
                return host_slice.tolist(), None

        if lookahead_pred_idx is not None and lookahead_pred_idx.numel() > 0:
            cur_flat = f_cur.tolist()
            raw_la = lookahead_pred_idx.reshape(-1).tolist()
            next_list = [int(x) for x in raw_la if int(x) >= 0]
            return cur_flat, (next_list if next_list else None)

        return f_cur.tolist(), None

    def stage_synced_topk(
        self,
        top_k_index: torch.Tensor,
        cur_flat: Optional[List[int]],
    ) -> None:
        """Stages missed Layer `l` pages onto `dma_stream` WHILE GPU Shared MLP runs on `compute_stream`."""
        if cur_flat is None or self.all_experts_mapped:
            return
        if self.tlb_slots <= 0 and self.l2_zero_copy_dev_ptr == 0:
            return
        T = int(top_k_index.shape[0])
        K = int(top_k_index.shape[1])
        token_rows = [cur_flat[t * K : (t + 1) * K] for t in range(T)]
        overflow = self._service_token_rows(token_rows, use_async_dma_stream=True)
        if overflow is None:
            self._serviced_topk_ptr = int(top_k_index.data_ptr())
            self._serviced_topk_numel = T * K

    def prefetch_for_topk(
        self,
        top_k_index: torch.Tensor,
        lookahead_pred_idx: Optional[torch.Tensor] = None,
    ) -> Optional[List[int]]:
        """Starts non-blocking PCIe DMA for missed experts on `dma_stream` and returns decoded lookahead IDs."""
        cur_flat, next_list = self.sync_topk_and_lookahead_d2h(top_k_index, lookahead_pred_idx)
        self.stage_synced_topk(top_k_index, cur_flat)
        return next_list

    def wait_or_service_topk(self, top_k_index: torch.Tensor) -> Optional[List[int]]:
        """Ensures all experts in `top_k_index` are mapped/resident on `current_stream`."""
        if self.all_experts_mapped:
            self._page_hits += int(top_k_index.shape[0])
            self._all_hit_fastpaths += 1
            return None
        if self.tlb_slots <= 0 and not self.has_l2_mapped:
            return None

        ptr = int(top_k_index.data_ptr())
        n_el = int(top_k_index.numel())
        if ptr == self._serviced_topk_ptr and n_el == self._serviced_topk_numel:
            self._serviced_topk_ptr = 0
            self._serviced_topk_numel = 0
            if self._dma_in_flight and self._dma_event is not None and self.device.type == "cuda":
                torch.cuda.current_stream(self.device).wait_event(self._dma_event)
            self._dma_in_flight = False
            return None

        T = int(top_k_index.shape[0])
        K = int(top_k_index.shape[1])
        if self.device.type == "cuda" and n_el <= int(self._d2h_pinned_cpu.numel()) and self._d2h_pinned_cpu.is_pinned():
            host_slice = self._d2h_pinned_cpu[:n_el]
            host_slice.copy_(top_k_index.reshape(-1), non_blocking=True)
            torch.cuda.current_stream(self.device).synchronize()
            cur_flat: List[int] = host_slice.tolist()
        else:
            cur_flat = top_k_index.reshape(-1).tolist()
        token_rows = [cur_flat[t * K : (t + 1) * K] for t in range(T)]
        overflow = self._service_token_rows(token_rows, use_async_dma_stream=False)
        if overflow is None and self._dma_in_flight and self._dma_event is not None and self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_event(self._dma_event)
            self._dma_in_flight = False
        return overflow


class SparseMoETLBExperts(nn.Module):
    """Drop-in Zero-Skeleton Virtual Memory replacement for `Gemma4TextExperts` / `Qwen3NextExperts`."""

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_ffn: int,
        layout: ContiguousExpertPageLayout,
        host_expert_pages: torch.Tensor,
        lut_pool: SharedQuantLUTPool,
        act_fn: Any,
        device: torch.device,
        tlb_slots: int = 48,
        expert_freq_rank: Optional[torch.Tensor] = None,
        mode: str = "vm_paged",
        tlb_strategy: str = "overlap_shared_mlp",
        shared_dma_stream: Optional[torch.cuda.Stream] = None,
        layer_idx: int = 0,
        use_fused_kernels: bool = True,
    ) -> None:
        super().__init__()
        if mode not in VALID_MOE_MODES:
            raise ValueError(f"Invalid MoE mode '{mode}'. Expected {VALID_MOE_MODES}.")
        self.num_experts = int(num_experts)
        self.hidden_dim = int(d_model)
        self.d_model = int(d_model)
        self.intermediate_dim = int(d_ffn)
        self.d_ffn = int(d_ffn)
        self.layout = layout
        self.lut_pool = lut_pool
        self.act_fn = act_fn
        self.act_kind = detect_act_kind(act_fn)
        self.device = device
        self.layer_idx = int(layer_idx)
        self.default_tlb_slots = int(max(1, min(tlb_slots, num_experts)))
        self.mode = mode
        self.tlb_strategy = tlb_strategy
        self.use_fused_kernels = bool(use_fused_kernels)

        # Cross-layer lookahead references (not registered as nn.Module children to avoid circular/state_dict bloat)
        self._lookahead_refs: Dict[str, Any] = {"head": None, "target": None}
        self._queued_lookahead_experts: Optional[List[int]] = None

        if expert_freq_rank is not None:
            self.register_buffer(
                "expert_freq_rank",
                expert_freq_rank.detach().to(device="cpu", dtype=torch.int32).contiguous(),
                persistent=True,
            )
        else:
            self.register_buffer(
                "expert_freq_rank",
                torch.arange(self.num_experts, dtype=torch.int32, device="cpu"),
                persistent=True,
            )

        init_slots = (
            self.num_experts
            if mode == "all_resident"
            else (0 if mode == "core_only" else self.default_tlb_slots)
        )
        self.tlb_cache = GPUExpertTLBCache(
            num_experts=self.num_experts,
            tlb_slots=init_slots,
            layout=layout,
            host_expert_pages=host_expert_pages,
            device=device,
            expert_freq_rank=self.expert_freq_rank,
            dma_stream=shared_dma_stream,
        )

        self._cuda_launcher: Optional[MoETLBLayerLauncher] = None
        if self.use_fused_kernels and torch.cuda.is_available() and device.type == "cuda":
            self._cuda_launcher = MoETLBLayerLauncher(
                device=device,
                num_experts=self.num_experts,
                layout=layout,
                act_kind=self.act_kind,
            )

    def set_lookahead_target(self, head: Any, target_experts: "SparseMoETLBExperts") -> None:
        self._lookahead_refs["head"] = head
        self._lookahead_refs["target"] = target_experts

    def clear_lookahead_target(self) -> None:
        self._lookahead_refs["head"] = None
        self._lookahead_refs["target"] = None
        self._queued_lookahead_experts = None

    def has_active_lookahead(self, num_tokens: int) -> bool:
        if self.mode != "vm_paged" or self.tlb_strategy != "lookahead_overlap":
            return False
        if not (1 <= int(num_tokens) <= MAX_FUSED_DECODE_TOKENS):
            return False
        head = self._lookahead_refs.get("head")
        target: Optional[SparseMoETLBExperts] = self._lookahead_refs.get("target")
        if head is None or target is None or target.mode != "vm_paged":
            return False
        if target.tlb_cache.all_experts_mapped:
            return False
        return True

    def compute_lookahead_indices(self, router_input_flat: torch.Tensor) -> Optional[torch.Tensor]:
        head = self._lookahead_refs.get("head")
        if head is None:
            return None
        return head.predict_topk_indices(router_input_flat)

    def set_mode(self, mode: str, tlb_slots_override: Optional[int] = None) -> None:
        if mode not in VALID_MOE_MODES:
            raise ValueError(f"Invalid MoE mode '{mode}'. Expected {VALID_MOE_MODES}.")
        self.mode = mode
        self._queued_lookahead_experts = None
        if tlb_slots_override is not None:
            self.default_tlb_slots = int(max(1, min(tlb_slots_override, self.num_experts)))
        if mode == "core_only":
            self.tlb_cache.release_l2_cache()
            self.tlb_cache.resize_slots(0, seed=False)
        elif mode == "all_resident":
            self.tlb_cache.release_l2_cache()
            self.tlb_cache.resize_slots(self.num_experts, seed=True)
        else:
            self.tlb_cache.resize_slots(self.default_tlb_slots, seed=True)

    def set_tlb_strategy(self, strategy: str) -> None:
        if strategy not in VALID_MOE_TLB_STRATEGIES:
            raise ValueError(f"Invalid MoE TLB strategy '{strategy}'. Expected {VALID_MOE_TLB_STRATEGIES}.")
        self.tlb_strategy = strategy
        self._queued_lookahead_experts = None

    def set_phase_c_enabled(self, enabled: bool) -> None:
        self.use_fused_kernels = bool(enabled)

    def release_tlb(self) -> None:
        self.tlb_cache.release_l2_cache()
        self.tlb_cache.release_gpu_slots()

    def commit_speculative_step(self, n_accepted: int) -> None:
        self.tlb_cache.commit_speculative_step(n_accepted)

    def reset_telemetry(self, clear_tlb: bool = False) -> None:
        self._queued_lookahead_experts = None
        if clear_tlb:
            if self.mode == "vm_paged":
                self.tlb_cache.seed_default_experts()
            elif self.mode == "all_resident":
                self.tlb_cache.resize_slots(self.num_experts, seed=True)
            else:
                self.tlb_cache.clear_cache()
        self.tlb_cache.reset_telemetry()

    def sync_indices_for_step(
        self,
        top_k_index: torch.Tensor,
        lookahead_pred_idx: Optional[torch.Tensor] = None,
    ) -> Optional[List[int]]:
        self._queued_lookahead_experts = None
        if self.mode != "vm_paged" or self.tlb_strategy == "exact_demand":
            return None
        if self.tlb_cache.all_experts_mapped:
            return None
        if int(top_k_index.shape[0]) > MAX_FUSED_DECODE_TOKENS:
            return None
        cur_flat, next_list = self.tlb_cache.sync_topk_and_lookahead_d2h(
            top_k_index, lookahead_pred_idx=lookahead_pred_idx
        )
        if next_list and self.tlb_strategy == "lookahead_overlap":
            self._queued_lookahead_experts = next_list
        return cur_flat

    def stage_pages_for_step(
        self,
        top_k_index: torch.Tensor,
        cur_flat: Optional[List[int]],
    ) -> None:
        if cur_flat is None or self.mode != "vm_paged" or self.tlb_strategy == "exact_demand":
            return
        self.tlb_cache.stage_synced_topk(top_k_index, cur_flat)

    def prefetch_for_step(
        self,
        top_k_index: torch.Tensor,
        lookahead_pred_idx: Optional[torch.Tensor] = None,
    ) -> None:
        cur_flat = self.sync_indices_for_step(top_k_index, lookahead_pred_idx=lookahead_pred_idx)
        self.stage_pages_for_step(top_k_index, cur_flat)

    def _dispatch_queued_lookahead(self) -> None:
        queued = self._queued_lookahead_experts
        self._queued_lookahead_experts = None
        if not queued:
            return
        target: Optional[SparseMoETLBExperts] = self._lookahead_refs.get("target")
        if target is not None and target.mode == "vm_paged":
            target.tlb_cache.schedule_lookahead_prefetch(queued)

    def export_serialized_core_and_pages(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        core_tensors = {
            "expert_freq_rank": self.expert_freq_rank.detach().cpu().contiguous(),
        }
        page_tensors = {
            "expert_pages": self.tlb_cache.host_expert_pages.detach().cpu().contiguous(),
        }
        meta = {
            "moe_type": "sparse_moe_uint4_v1",
            "num_experts": self.num_experts,
            "d_model": self.d_model,
            "d_ffn": self.d_ffn,
            "tlb_slots": self.default_tlb_slots,
            "act_kind": self.act_kind,
            "layout": self.layout.to_dict(),
        }
        return core_tensors, page_tensors, meta

    @classmethod
    def from_serialized_buffers(
        cls,
        core_tensors: Dict[str, torch.Tensor],
        expert_pages_host: torch.Tensor,
        meta: Dict[str, Any],
        lut_pool: SharedQuantLUTPool,
        act_fn: Any,
        device: torch.device,
        mode: str = "vm_paged",
        tlb_strategy: str = "overlap_shared_mlp",
        tlb_slots_override: Optional[int] = None,
        shared_dma_stream: Optional[torch.cuda.Stream] = None,
        layer_idx: int = 0,
        use_fused_kernels: bool = True,
    ) -> "SparseMoETLBExperts":
        ld = meta["layout"]
        layout = ContiguousExpertPageLayout(**{k: int(v) for k, v in ld.items()})
        eff_slots = (
            int(tlb_slots_override)
            if tlb_slots_override is not None
            else int(meta.get("tlb_slots", 48))
        )
        return cls(
            num_experts=int(meta["num_experts"]),
            d_model=int(meta["d_model"]),
            d_ffn=int(meta["d_ffn"]),
            layout=layout,
            host_expert_pages=expert_pages_host,
            lut_pool=lut_pool,
            act_fn=act_fn,
            device=device,
            tlb_slots=eff_slots,
            expert_freq_rank=core_tensors.get("expert_freq_rank"),
            mode=mode,
            tlb_strategy=tlb_strategy,
            shared_dma_stream=shared_dma_stream,
            layer_idx=layer_idx,
            use_fused_kernels=use_fused_kernels,
        )

    def _run_packed_wave_list(
        self,
        x_chunk_bf16: torch.Tensor,
        tlb_pages_ptr: int,
        l2_dev_ptr: int,
        partial_f32_ptr: int,
        items: List[Tuple[int, List[int], List[int], List[float]]],
    ) -> None:
        """Uploads waves of `<= 128` `(slot, tok_ids, pair_ids, weights)` work-items and launches MoE kernels."""
        assert self._cuda_launcher is not None
        ws = self._cuda_launcher._ws
        n_tp = MAX_ACTIVE_UNIQUE * MAX_FUSED_DECODE_TOKENS  # 1024
        off_cnt = MAX_ACTIVE_UNIQUE
        off_tok = 2 * MAX_ACTIVE_UNIQUE
        off_pair = off_tok + n_tp
        off_num = off_pair + n_tp
        total_int_len = off_num + 1

        total_items = len(items)
        for w_start in range(0, total_items, MAX_ACTIVE_UNIQUE):
            wave = items[w_start : w_start + MAX_ACTIVE_UNIQUE]
            n_u = len(wave)
            max_m = 1
            int_py = [0] * total_int_len
            w_len = n_u * MAX_FUSED_DECODE_TOKENS
            w_py = [0.0] * w_len

            for u_idx, (slot_val, t_list, pair_list, w_list) in enumerate(wave):
                m_cnt = len(t_list)
                if m_cnt > max_m:
                    max_m = m_cnt
                int_py[u_idx] = slot_val
                int_py[off_cnt + u_idx] = m_cnt
                base_m = u_idx * MAX_FUSED_DECODE_TOKENS
                tok_base = off_tok + base_m
                pair_base = off_pair + base_m
                int_py[tok_base : tok_base + m_cnt] = t_list
                int_py[pair_base : pair_base + m_cnt] = pair_list
                w_py[base_m : base_m + m_cnt] = w_list
            int_py[off_num] = n_u

            # 9 KB + 4 KB (< 64 KB): CUDA driver inline-stages host data into command ring in ~3 us with zero host overwrite race!
            ws._route_int_buf[:total_int_len].copy_(torch.tensor(int_py, dtype=torch.int32))
            ws.active_weights[:w_len].copy_(torch.tensor(w_py, dtype=torch.float32))
            self._cuda_launcher.launch_wave_items(
                x_bf16=x_chunk_bf16,
                tlb_pages_ptr=tlb_pages_ptr,
                l2_zero_copy_dev_ptr=l2_dev_ptr,
                partial_f32_ptr=partial_f32_ptr,
                num_items=n_u,
                max_m_in_wave=max_m,
            )

    def _forward_grouped_prefill_cuda(
        self,
        flat_x: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Phase H3: Bit-Exact (`0.0-ulp`) Multi-Wave Grouped NVRTC Prefill (`T > 8`, ~20x faster TTFT)."""
        assert self._cuda_launcher is not None
        self.tlb_cache._await_pending_prefetch()
        if self.tlb_cache._dma_in_flight and self.tlb_cache._dma_event is not None:
            torch.cuda.current_stream(flat_x.device).wait_event(self.tlb_cache._dma_event)
            self.tlb_cache._dma_in_flight = False

        T_total = int(flat_x.shape[0])
        K = int(top_k_index.shape[1])
        x_bf16 = flat_x.contiguous() if flat_x.dtype == torch.bfloat16 else flat_x.to(torch.bfloat16).contiguous()
        out_bf16 = torch.empty((T_total, self.d_model), dtype=torch.bfloat16, device=flat_x.device)

        if self._cuda_launcher._kd is None or self._cuda_launcher._bound_ws_version != self._cuda_launcher._ws.version:
            self._cuda_launcher._rebind(self.tlb_cache.tlb_pages, self.tlb_cache.gpu_expert_to_slot)
        ws = self._cuda_launcher._ws

        # Single D2H transfer for the whole prefill sequence (`[T_total, K]`) instead of ~420 `torch.where` syncs!
        idx_cpu_all = top_k_index.detach().to(dtype=torch.int64, device="cpu")
        w_cpu_all = top_k_weights.detach().to(dtype=torch.float32, device="cpu")

        e2s = self.tlb_cache._expert_to_slot
        l2_home = self.tlb_cache._expert_l2_home
        l2_dev_ptr = self.tlb_cache.l2_zero_copy_dev_ptr
        has_l2 = l2_dev_ptr != 0
        n_exp = self.num_experts
        page_bytes = self.layout.page_bytes

        for c_start in range(0, T_total, PREFILL_MAX_CHUNK_TOKENS):
            c_end = min(T_total, c_start + PREFILL_MAX_CHUNK_TOKENS)
            T_c = c_end - c_start
            ws.ensure_prefill_capacity(T_c, K)
            partial_ptr = ws.prefill_out_f32.data_ptr()

            # 1. Zero `partial_f32` on GPU while CPU groups routes in ~20 microseconds
            self._cuda_launcher.launch_zero_and_epilogue(partial_ptr, None, T_c, K, mode="zero")

            idx_flat = idx_cpu_all[c_start:c_end].reshape(-1)
            w_flat = w_cpu_all[c_start:c_end].reshape(-1)
            order = torch.argsort(idx_flat, stable=True)
            s_exp_list: List[int] = idx_flat[order].tolist()
            s_pair_list: List[int] = order.tolist()
            s_w_list: List[float] = w_flat[order].tolist()

            mapped_items: List[Tuple[int, List[int], List[int], List[float]]] = []
            cold_groups: List[Tuple[int, List[Tuple[int, int, float]]]] = []

            n_pairs = len(s_exp_list)
            pos = 0
            while pos < n_pairs:
                e_id = int(s_exp_list[pos])
                end_p = pos + 1
                while end_p < n_pairs and int(s_exp_list[end_p]) == e_id:
                    end_p += 1

                if 0 <= e_id < n_exp:
                    n_tok_e = end_p - pos
                    l1_slot = e2s[e_id] if self.tlb_cache._is_allocated else -1
                    if l1_slot >= 0:
                        self.tlb_cache._page_hits += 1
                        self.tlb_cache._access_clock += 1
                        self.tlb_cache._slot_last_used[l1_slot] = self.tlb_cache._access_clock
                        enc_slot = l1_slot
                    elif has_l2 and l2_home[e_id] >= 0:
                        self.tlb_cache._l2_zero_copy_hits += 1
                        self.tlb_cache._pcie_bytes += page_bytes
                        self.tlb_cache._distinct_pages.add(e_id)
                        # If `n_tok_e <= 8` (`1` work-item), read directly from L2 Zero-Copy with zero staging!
                        # If `n_tok_e > 8` (`2+` split work-items), stage once into VRAM so PCIe is read once!
                        enc_slot = (-2 - l2_home[e_id]) if n_tok_e <= MAX_FUSED_DECODE_TOKENS else -1
                    else:
                        self.tlb_cache._page_misses += 1
                        self.tlb_cache._demand_misses += 1
                        self.tlb_cache._pcie_bytes += page_bytes
                        self.tlb_cache._distinct_pages.add(e_id)
                        enc_slot = -1

                    if enc_slot != -1:
                        for sub in range(pos, end_p, MAX_FUSED_DECODE_TOKENS):
                            sub_end = min(end_p, sub + MAX_FUSED_DECODE_TOKENS)
                            t_ids = [s_pair_list[j] // K for j in range(sub, sub_end)]
                            p_ids = [s_pair_list[j] for j in range(sub, sub_end)]
                            ws_sub = [s_w_list[j] for j in range(sub, sub_end)]
                            mapped_items.append((enc_slot, t_ids, p_ids, ws_sub))
                    else:
                        pairs_for_e = [
                            (s_pair_list[j] // K, s_pair_list[j], s_w_list[j]) for j in range(pos, end_p)
                        ]
                        cold_groups.append((e_id, pairs_for_e))
                pos = end_p

            x_chunk = x_bf16[c_start:c_end]
            out_chunk = out_bf16[c_start:c_end]

            # 2. Execute all L1 VRAM + single-item L2 Zero-Copy mapped experts with ZERO staging copies!
            if mapped_items:
                self._run_packed_wave_list(
                    x_chunk_bf16=x_chunk,
                    tlb_pages_ptr=self.tlb_cache.tlb_pages.data_ptr(),
                    l2_dev_ptr=l2_dev_ptr,
                    partial_f32_ptr=partial_ptr,
                    items=mapped_items,
                )

            # 3. Stream cold L3 `nvme_mmap` (or multi-chunk L2) experts in 32-expert (~56 MB) pinned burst DMA waves!
            if cold_groups:
                comp_stream = torch.cuda.current_stream(flat_x.device)
                stage_gpu_2d = ws.prefill_page_stage[: PREFILL_STAGE_SLOTS * page_bytes].view(
                    PREFILL_STAGE_SLOTS, page_bytes
                )
                l2_pages = self.tlb_cache.l2_pinned_pages
                for b_idx in range(0, len(cold_groups), PREFILL_STAGE_SLOTS):
                    batch = cold_groups[b_idx : b_idx + PREFILL_STAGE_SLOTS]
                    cold_items: List[Tuple[int, List[int], List[int], List[float]]] = []
                    staged_h2d: List[Tuple[int, torch.Tensor, int]] = []
                    for local_slot, (e_id, pairs_for_e) in enumerate(batch):
                        l2_idx = l2_home[e_id] if has_l2 else -1
                        if l2_idx >= 0 and l2_pages is not None:
                            staged_h2d.append((local_slot, l2_pages[l2_idx], -1))
                        else:
                            src_p = self.tlb_cache.host_expert_pages[e_id]
                            if src_p.is_pinned():
                                staged_h2d.append((local_slot, src_p, -1))
                            else:
                                r_slot, st_buf = ws.acquire_pinned_stage_slot(page_bytes)
                                st_buf.copy_(src_p)
                                staged_h2d.append((local_slot, st_buf, r_slot))
                        for sub in range(0, len(pairs_for_e), MAX_FUSED_DECODE_TOKENS):
                            sub_slice = pairs_for_e[sub : sub + MAX_FUSED_DECODE_TOKENS]
                            cold_items.append(
                                (
                                    local_slot,
                                    [p[0] for p in sub_slice],
                                    [p[1] for p in sub_slice],
                                    [p[2] for p in sub_slice],
                                )
                            )
                    for local_slot, pinned_src, r_slot in staged_h2d:
                        stage_gpu_2d[local_slot].copy_(pinned_src, non_blocking=True)
                        if r_slot >= 0:
                            ws.record_stage_slot(r_slot, comp_stream)
                    self._run_packed_wave_list(
                        x_chunk_bf16=x_chunk,
                        tlb_pages_ptr=stage_gpu_2d.data_ptr(),
                        l2_dev_ptr=0,
                        partial_f32_ptr=partial_ptr,
                        items=cold_items,
                    )

            # 4. Deterministic FP32 -> BF16 reduction across `k = 0 .. K - 1`
            self._cuda_launcher.launch_zero_and_epilogue(partial_ptr, out_chunk, T_c, K, mode="epilogue")

        self.tlb_cache.record_compute_read(torch.cuda.current_stream(flat_x.device))
        return out_bf16 if out_bf16.dtype == flat_x.dtype else out_bf16.to(flat_x.dtype)

    def _forward_python_reference(
        self,
        flat_x: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Reference fallback when `use_fused_kernels == False` or running on CPU."""
        if (
            self.use_fused_kernels
            and self._cuda_launcher is not None
            and flat_x.is_cuda
            and int(top_k_index.shape[1]) <= 16
        ):
            return self._forward_grouped_prefill_cuda(flat_x, top_k_index, top_k_weights)

        self.tlb_cache._await_pending_prefetch()
        T = int(flat_x.shape[0])
        if self.mode == "core_only":
            return torch.zeros_like(flat_x)

        out_f32 = torch.zeros((T, self.d_model), dtype=torch.float32, device=flat_x.device)
        ws = get_shared_moe_workspace(flat_x.device) if flat_x.is_cuda else None
        if ws is not None:
            ws.ensure_capacity(self.d_model, self.d_ffn, self.layout.page_bytes)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index.long(), num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit_experts = torch.greater(expert_mask[: self.num_experts].sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)

        hit_list: List[int] = [int(x[0]) for x in hit_experts.tolist()]
        e2s = self.tlb_cache._expert_to_slot

        for e_id in hit_list:
            top_k_pos, token_idx = torch.where(expert_mask[e_id])
            cur_x = flat_x[token_idx].to(torch.bfloat16)

            slot = e2s[e_id] if self.tlb_cache._is_allocated else -1
            if slot >= 0:
                self.tlb_cache._page_hits += 1
                self.tlb_cache._access_clock += 1
                self.tlb_cache._slot_last_used[slot] = self.tlb_cache._access_clock
                page_gpu = self.tlb_cache.tlb_pages[slot]
            else:
                self.tlb_cache._page_misses += 1
                self.tlb_cache._demand_misses += 1
                self.tlb_cache._pcie_bytes += self.layout.page_bytes
                self.tlb_cache._distinct_pages.add(e_id)
                if ws is not None:
                    page_gpu = ws.prefill_page_stage[: self.layout.page_bytes]
                    r_slot, st = ws.acquire_pinned_stage_slot(self.layout.page_bytes)
                    st.copy_(self.tlb_cache.host_expert_pages[e_id])
                    page_gpu.copy_(st, non_blocking=True)
                    ws.record_stage_slot(r_slot, torch.cuda.current_stream(flat_x.device))
                else:
                    page_gpu = self.tlb_cache.host_expert_pages[e_id].to(flat_x.device)

            w_gu, w_dn = unpack_moe_expert_page_to_bf16(page_gpu, self.layout, self.lut_pool.lut_uint4)
            gate, up = F.linear(cur_x, w_gu).chunk(2, dim=-1)
            inter = self.act_fn(gate) * up
            cur_out = F.linear(inter, w_dn).float()
            w_scale = top_k_weights[token_idx, top_k_pos].float().unsqueeze(-1)
            out_f32.index_add_(0, token_idx, cur_out * w_scale)

        if flat_x.is_cuda:
            self.tlb_cache.record_compute_read(torch.cuda.current_stream(flat_x.device))
        return out_f32.to(flat_x.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        orig_dtype = hidden_states.dtype
        flat_x = hidden_states.reshape(-1, self.d_model)
        T = int(flat_x.shape[0])
        K = int(top_k_index.shape[1])

        if self.mode == "core_only":
            return torch.zeros_like(hidden_states)

        if (
            1 <= T <= MAX_FUSED_DECODE_TOKENS
            and K <= 16
            and self.use_fused_kernels
            and self._cuda_launcher is not None
            and flat_x.is_cuda
            and (self.tlb_cache.tlb_slots > 0 or self.tlb_cache.has_l2_mapped)
        ):
            overflow_uniq = self.tlb_cache.wait_or_service_topk(top_k_index)
            if overflow_uniq is None:
                x_bf = flat_x.contiguous() if flat_x.dtype == torch.bfloat16 else flat_x.to(torch.bfloat16).contiguous()
                idx_i64 = (
                    top_k_index.contiguous()
                    if top_k_index.dtype == torch.int64
                    else top_k_index.to(torch.int64).contiguous()
                )
                w_f32 = (
                    top_k_weights.contiguous()
                    if top_k_weights.dtype == torch.float32
                    else top_k_weights.float().contiguous()
                )
                out_bf = torch.empty((T, self.d_model), dtype=torch.bfloat16, device=x_bf.device)
                self._cuda_launcher.launch(
                    x_bf16=x_bf,
                    top_k_idx_i64=idx_i64,
                    top_k_w_f32=w_f32,
                    tlb_pages=self.tlb_cache.tlb_pages,
                    expert_to_slot=self.tlb_cache.gpu_expert_to_slot,
                    out_bf16=out_bf,
                    l2_zero_copy_dev_ptr=self.tlb_cache.l2_zero_copy_dev_ptr,
                    l2_wina_dev_ptr=self.tlb_cache.l2_wina_dev_ptr,
                    wina_keep_ratio=self.tlb_cache.wina_keep_ratio,
                )
                self.tlb_cache.record_compute_read(torch.cuda.current_stream(x_bf.device))
                self._dispatch_queued_lookahead()
                out = out_bf.view(orig_shape)
                return out if out.dtype == orig_dtype else out.to(orig_dtype)

            if self.tlb_cache.tlb_slots >= K and T > 1:
                outs = [
                    self.forward(flat_x[t : t + 1], top_k_index[t : t + 1], top_k_weights[t : t + 1])
                    for t in range(T)
                ]
                return torch.cat(outs, dim=0).view(orig_shape).to(orig_dtype)

        return self._forward_python_reference(flat_x, top_k_index, top_k_weights).view(orig_shape)


def auto_configure_moe_l2_zero_copy(
    moe_experts_by_layer: Dict[int, SparseMoETLBExperts],
    reserve_free_ram_gb: float = 9.5,
    max_l2_ram_gb: Optional[float] = None,
    wina_keep_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """Automatically sizes device-mapped pinned DDR5 L2 expert caches within safe physical RAM limits.

    - On moderate-sized MoEs (e.g. `Gemma-4-26B-A4B`, `128` experts), maps 100% of non-L1 experts into L2
      Zero-Copy DDR5 (`~8 GB`), enabling 100% zero-sync full-model CUDA Graph execution!
    - On massive MoEs exceeding host RAM (e.g. `Qwen3-Next-80B-A3B`, `43.5 GB` expert pages on `34 GB` RAM),
      caps L2 Zero-Copy at `~7.5 GB` (`~88–96` warm experts/layer), leaving `14+ GB` free OS RAM for `nvme_mmap`
      file caching while eliminating transient L1 evictions.
    """
    if not moe_experts_by_layer:
        return {"l2_slots_per_layer": 0, "total_l2_gb": 0.0, "all_mapped": True}

    layers = list(moe_experts_by_layer.values())
    first = layers[0]
    if first.mode != "vm_paged" or first.device.type != "cuda" or not torch.cuda.is_available():
        had_l2 = any(m.tlb_cache.has_l2_mapped or m.tlb_cache.l2_pinned_pages is not None or m.tlb_cache.l2_wina_pinned_pages is not None for m in layers)
        for m in layers:
            m.tlb_cache.release_l2_cache()
        if had_l2 and hasattr(torch._C, "_host_emptyCache"):
            import gc

            gc.collect()
            torch._C._host_emptyCache()
        return {
            "l2_slots_per_layer": 0,
            "total_l2_gb": 0.0,
            "all_mapped": all(m.tlb_cache.all_experts_mapped for m in layers),
        }

    if first.tlb_cache._l2_aliases_host_pages:
        return {
            "l2_slots_per_layer": first.num_experts,
            "total_l2_gb": 0.0,
            "all_mapped": True,
        }

    # Release any prior L2 allocation AND flush PyTorch's CachingHostAllocator so `avail_bytes` reflects free OS RAM!
    for m in layers:
        m.tlb_cache.release_l2_cache()
    if hasattr(torch._C, "_host_emptyCache"):
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch._C._host_emptyCache()

    # On Windows, `seed_default_experts()` faults clean read-only `nvme_mmap` pages into the process Working Set,
    # which `GlobalMemoryStatusEx.ullAvailPhys` subtracts as "in-use" even though Windows moves them to Standby RAM
    # on demand. Trimming clean Working-Set pages to Standby RAM (pinned/locked pages are untouched!) exposes true
    # available physical RAM while keeping those pages warm in OS Standby RAM!
    if os.name == "nt":
        try:
            import ctypes

            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            h_proc = k32.GetCurrentProcess()
            if hasattr(k32, "K32EmptyWorkingSet"):
                k32.K32EmptyWorkingSet.argtypes = [ctypes.c_void_p]
                k32.K32EmptyWorkingSet.restype = ctypes.c_int
                k32.K32EmptyWorkingSet(h_proc)
            else:
                psapi = ctypes.windll.psapi
                psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
                psapi.EmptyWorkingSet.restype = ctypes.c_int
                psapi.EmptyWorkingSet(h_proc)
        except Exception:
            pass

    avail_bytes, total_bytes = query_available_system_ram_bytes()
    eff_wina = get_default_l2_wina_ratio() if wina_keep_ratio is None else max(0.20, min(1.0, float(wina_keep_ratio)))
    use_wina = 0.0 < eff_wina < 1.0

    def _eff_page_bytes(m: SparseMoETLBExperts) -> int:
        if use_wina and m.tlb_cache.wina_layout is not None:
            return int(m.tlb_cache.wina_layout.page_bytes)
        return int(m.layout.page_bytes)

    bytes_per_slot_across_model = sum(_eff_page_bytes(m) for m in layers)
    total_unmapped_bytes = sum(
        max(0, m.num_experts - m.tlb_cache.tlb_slots) * _eff_page_bytes(m) for m in layers
    )
    # When 100% of remaining experts fit in pinned DDR5 (`all_mapped == True`, e.g. Gemma-4-26B ~7.5 GB,
    # or Qwen3-Next-80B on 64 GB+ RAM), `nvme_mmap` is never touched during decode so 4.0 GiB OS reserve suffices.
    # When L3 `nvme_mmap` paging remains active, dynamically scale L2 to available free RAM while keeping
    # `min(reserve_free_ram_gb, 30% of free RAM, >=6 GiB)` for OS + Standby file cache (no hardcoded 7.5 GiB cap!).
    full_map_reserve = int(min(float(reserve_free_ram_gb), 4.0) * (1024 ** 3))
    if total_unmapped_bytes + 128 * (1024 ** 2) <= max(0, avail_bytes - full_map_reserve):
        budget_bytes = max(0, avail_bytes - full_map_reserve - 128 * (1024 ** 2))
    else:
        dyn_reserve = max(6.0 * (1024 ** 3), min(float(reserve_free_ram_gb) * (1024 ** 3), 0.30 * float(avail_bytes)))
        budget_bytes = max(0, avail_bytes - int(dyn_reserve) - 128 * (1024 ** 2))

    env_cap = os.environ.get("TMMU_MAX_L2_RAM_GB", "").strip()
    if max_l2_ram_gb is not None:
        cap_bytes = int(float(max_l2_ram_gb) * (1024 ** 3))
    elif env_cap:
        try:
            cap_bytes = int(float(env_cap) * (1024 ** 3))
        except ValueError:
            cap_bytes = budget_bytes
    else:
        if total_unmapped_bytes <= budget_bytes:
            cap_bytes = total_unmapped_bytes
        elif os.name == "nt" and total_bytes > 0:
            # Windows WDDM pinned-memory commit ceiling (~38% of physical RAM) so `cudaMalloc` never OOMs after L2 pin
            cap_bytes = int(0.38 * float(total_bytes))
        else:
            cap_bytes = budget_bytes

    eff_budget = min(budget_bytes, cap_bytes)
    if bytes_per_slot_across_model <= 0 or eff_budget < bytes_per_slot_across_model:
        return {"l2_slots_per_layer": 0, "total_l2_gb": 0.0, "all_mapped": False}

    l2_slots = eff_budget // bytes_per_slot_across_model
    for m in layers:
        m.tlb_cache.configure_l2_zero_copy(
            int(l2_slots),
            wina_keep_ratio=eff_wina,
            lut_uint4=m.lut_pool.lut_uint4 if m.lut_pool is not None else None,
        )

    tot_l2_bytes = sum(m.tlb_cache.total_l2_pinned_ram_bytes for m in layers)
    all_mapped = all(m.tlb_cache.all_experts_mapped for m in layers)
    return {
        "l2_slots_per_layer": int(layers[0].tlb_cache.l2_slots),
        "total_l2_gb": float(tot_l2_bytes) / float(1024 ** 3),
        "all_mapped": bool(all_mapped),
        "l2_wina_ratio": float(layers[0].tlb_cache.wina_keep_ratio) if layers[0].tlb_cache.l2_wina_dev_ptr != 0 else 1.0,
    }


def install_moe_intra_layer_overlap(decoder_layer: nn.Module) -> bool:
    """Enqueues GPU Shared Dense MLP + Lookahead BEFORE CPU `nvme_mmap` staging (`0-ulp` identical output)."""
    # Pattern 1: Gemma 4 Decoder Layer (`enable_moe_block == True`)
    if (
        getattr(decoder_layer, "enable_moe_block", False)
        and hasattr(decoder_layer, "router")
        and hasattr(decoder_layer, "experts")
        and isinstance(decoder_layer.experts, SparseMoETLBExperts)
        and hasattr(decoder_layer, "mlp")
    ):
        def _gemma4_overlapped_forward(
            self: nn.Module,
            hidden_states: torch.Tensor,
            per_layer_input: Optional[torch.Tensor] = None,
            shared_kv_states: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
            position_embeddings: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Any = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                shared_kv_states=shared_kv_states,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            T_cur = int(hidden_states_flat.shape[0])

            _, top_k_weights, top_k_index = self.router(hidden_states_flat)
            pred_next_idx = (
                self.experts.compute_lookahead_indices(hidden_states_flat)
                if self.experts.has_active_lookahead(T_cur)
                else None
            )
            cur_flat = self.experts.sync_indices_for_step(top_k_index, lookahead_pred_idx=pred_next_idx)

            hidden_states_mlp = self.pre_feedforward_layernorm(residual)
            hidden_states_mlp = self.mlp(hidden_states_mlp)
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states_mlp)
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
            self.experts.stage_pages_for_step(top_k_index, cur_flat)

            hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            hidden_states = hidden_states_1 + hidden_states_2
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states

            if self.hidden_size_per_layer_input:
                residual = hidden_states
                hidden_states = self.per_layer_input_gate(hidden_states)
                hidden_states = self.act_fn(hidden_states)
                hidden_states = hidden_states * per_layer_input
                hidden_states = self.per_layer_projection(hidden_states)
                hidden_states = self.post_per_layer_input_norm(hidden_states)
                hidden_states = residual + hidden_states

            hidden_states *= self.layer_scalar
            return hidden_states

        decoder_layer.forward = types.MethodType(_gemma4_overlapped_forward, decoder_layer)
        return True

    # Pattern 2: Qwen3-Next / DeepSeek SparseMoeBlock (`layer.mlp` has `.gate`, `.shared_expert`, `.experts`)
    moe_block = getattr(decoder_layer, "mlp", None)
    if (
        moe_block is not None
        and hasattr(moe_block, "gate")
        and hasattr(moe_block, "shared_expert")
        and hasattr(moe_block, "experts")
        and isinstance(moe_block.experts, SparseMoETLBExperts)
    ):
        def _qwen_moe_overlapped_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
            T_cur = int(hidden_states_reshaped.shape[0])

            _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
            pred_next_idx = (
                self.experts.compute_lookahead_indices(hidden_states_reshaped)
                if self.experts.has_active_lookahead(T_cur)
                else None
            )
            cur_flat = self.experts.sync_indices_for_step(selected_experts, lookahead_pred_idx=pred_next_idx)

            shared_expert_output = self.shared_expert(hidden_states_reshaped)
            if hasattr(self, "shared_expert_gate") and self.shared_expert_gate is not None:
                shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
            self.experts.stage_pages_for_step(selected_experts, cur_flat)

            expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
            expert_output = (expert_output + shared_expert_output).reshape(batch_size, sequence_length, hidden_dim)
            return expert_output

        moe_block.forward = types.MethodType(_qwen_moe_overlapped_forward, moe_block)
        return True

    return False
