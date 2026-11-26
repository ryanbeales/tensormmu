"""Phase F, G & H Runtime: Register-Fused NVRTC CUDA C++ Kernels for Zero-Skeleton Sparse MoE Virtual Memory.

Executes routed `UINT4` group-wise experts directly out of:
  - **L1 GPU VRAM TLB** (`slot >= 0` -> `tlb_pages + slot * page_bytes`, `1,008 GB/s` HBM), AND
  - **L2 Zero-Copy Pinned Host DDR5** (`slot <= -2` -> `l2_pages + (-2 - slot) * page_bytes`, direct PCIe 4.0 x16
    `cuMemHostGetDevicePointer_v2` reads with `0` VRAM footprint, `0` DMA copy, and `0` L1 eviction!),
with zero unpacked BF16 HBM allocations, Unique-Expert Multi-Token Amortization (`1 <= T <= 8`), and
Bit-Exact Multi-Wave Grouped Prefill (`T > 8`, eliminating ~360k Python kernel launches per prompt):
  1. `moe_pack_routes_kernel`: Deterministically groups `(token, k)` router selections by unique active expert `u`
     on GPU and zeroes the FP32 accumulation workspace.
  2. `moe_uint4_gate_up_act_kernel`: Reads each active expert's `gate_up` `UINT4` weights ONCE from L1 VRAM or
     L2 Zero-Copy DDR5 via coalesced 32-bit `__ldg`, evaluating up to `MAX_M = 8` tokens routed to that expert in
     registers and fusing `GeGLU` (`gelu_tanh`) or `SwiGLU` (`silu`).
  3. `moe_uint4_down_accum_kernel`: Reads each active expert's `down_proj` `UINT4` weights ONCE from L1/L2,
     scales by exact FP32 `top_k_weights`, and writes `partial_f32[pair_id, d_model]`.
  4. `moe_f32_to_bf16_kernel`: Deterministically reduces `partial_f32[T, K, d_model]` across `k = 0..K-1` to BF16.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

from runtime.phase_c_kernels import MAX_FUSED_DECODE_TOKENS

MAX_MOE_EXPERTS = 1024
MAX_MOE_TOPK = 16
MAX_ACTIVE_UNIQUE = MAX_FUSED_DECODE_TOKENS * MAX_MOE_TOPK  # 128 work-items per wave
OS_PAGE_ALIGN_BYTES = 4096
SUBTENSOR_ALIGN_BYTES = 256

DMA_STAGE_RING_SLOTS = 32
PREFILL_STAGE_SLOTS = 32
PREFILL_MAX_CHUNK_TOKENS = 256


def _align_up(val: int, align: int) -> int:
    return (int(val) + align - 1) // align * align


@dataclass(frozen=True)
class ContiguousExpertPageLayout:
    """Byte offsets for a single 4KB-aligned contiguous `UINT4` (`group_size`) MoE expert page."""

    d_model: int
    d_ffn: int
    group_size: int
    d_pad: int
    d_pad_half: int
    n_groups_d: int
    i_pad: int
    i_pad_half: int
    n_groups_i: int
    off_gu_q4: int
    bytes_gu_q4: int
    off_dn_q4: int
    bytes_dn_q4: int
    off_gu_wmin: int
    bytes_gu_wmin: int
    off_gu_scale: int
    bytes_gu_scale: int
    off_dn_wmin: int
    bytes_dn_wmin: int
    off_dn_scale: int
    bytes_dn_scale: int
    page_bytes: int

    @classmethod
    def compute(cls, d_model: int, d_ffn: int, group_size: int = 64) -> "ContiguousExpertPageLayout":
        if group_size <= 0 or (group_size % 2) != 0:
            raise ValueError(f"group_size must be positive and even, got {group_size}.")
        d_m = int(d_model)
        d_i = int(d_ffn)
        g = int(group_size)

        d_pad = _align_up(d_m, g)
        d_pad_half = d_pad // 2
        n_groups_d = d_pad // g

        i_pad = _align_up(d_i, g)
        i_pad_half = i_pad // 2
        n_groups_i = i_pad // g

        off_gu_q4 = 0
        bytes_gu_q4 = (2 * d_i) * d_pad_half

        off_dn_q4 = _align_up(off_gu_q4 + bytes_gu_q4, SUBTENSOR_ALIGN_BYTES)
        bytes_dn_q4 = d_m * i_pad_half

        off_gu_wmin = _align_up(off_dn_q4 + bytes_dn_q4, SUBTENSOR_ALIGN_BYTES)
        bytes_gu_wmin = (2 * d_i) * n_groups_d * 2  # bf16

        off_gu_scale = _align_up(off_gu_wmin + bytes_gu_wmin, SUBTENSOR_ALIGN_BYTES)
        bytes_gu_scale = (2 * d_i) * n_groups_d * 2  # bf16

        off_dn_wmin = _align_up(off_gu_scale + bytes_gu_scale, SUBTENSOR_ALIGN_BYTES)
        bytes_dn_wmin = d_m * n_groups_i * 2  # bf16

        off_dn_scale = _align_up(off_dn_wmin + bytes_dn_wmin, SUBTENSOR_ALIGN_BYTES)
        bytes_dn_scale = d_m * n_groups_i * 2  # bf16

        page_bytes = _align_up(off_dn_scale + bytes_dn_scale, OS_PAGE_ALIGN_BYTES)
        return cls(
            d_model=d_m,
            d_ffn=d_i,
            group_size=g,
            d_pad=d_pad,
            d_pad_half=d_pad_half,
            n_groups_d=n_groups_d,
            i_pad=i_pad,
            i_pad_half=i_pad_half,
            n_groups_i=n_groups_i,
            off_gu_q4=off_gu_q4,
            bytes_gu_q4=bytes_gu_q4,
            off_dn_q4=off_dn_q4,
            bytes_dn_q4=bytes_dn_q4,
            off_gu_wmin=off_gu_wmin,
            bytes_gu_wmin=bytes_gu_wmin,
            off_gu_scale=off_gu_scale,
            bytes_gu_scale=bytes_gu_scale,
            off_dn_wmin=off_dn_wmin,
            bytes_dn_wmin=bytes_dn_wmin,
            off_dn_scale=off_dn_scale,
            bytes_dn_scale=bytes_dn_scale,
            page_bytes=page_bytes,
        )

    def to_dict(self) -> Dict[str, int]:
        return {k: int(getattr(self, k)) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class WINAExpertPageLayout:
    """4KB-aligned neuron-major `UINT4-WINA` L2 Zero-Copy expert page layout (`gate` + `wina_norms` + `(up_j, down_j)`)."""

    d_model: int
    d_ffn: int
    n_groups_d: int
    row_bytes: int
    slab_bytes: int
    off_gate: int
    off_wina_norm: int
    off_slabs: int
    page_bytes: int

    @classmethod
    def compute(cls, d_model: int, d_ffn: int) -> "WINAExpertPageLayout":
        d_m = int(d_model)
        d_i = int(d_ffn)
        assert d_m % 64 == 0, f"WINAExpertPageLayout requires d_model % 64 == 0, got {d_m}"
        n_groups = d_m // 64
        # Per-row layout: [n_groups * 32 B of UINT4] + [n_groups * 4 B of packed (wmin_bf16, scale_bf16)]
        row_b = _align_up(n_groups * 36, SUBTENSOR_ALIGN_BYTES)
        slab_b = 2 * row_b  # [up_row[j], down_col[j]] contiguous per intermediate neuron j
        off_gate = 0
        off_wina = _align_up(d_i * row_b, SUBTENSOR_ALIGN_BYTES)
        off_slabs = _align_up(off_wina + d_i * 2, SUBTENSOR_ALIGN_BYTES)
        page_b = _align_up(off_slabs + d_i * slab_b, OS_PAGE_ALIGN_BYTES)
        return cls(
            d_model=d_m,
            d_ffn=d_i,
            n_groups_d=n_groups,
            row_bytes=row_b,
            slab_bytes=slab_b,
            off_gate=off_gate,
            off_wina_norm=off_wina,
            off_slabs=off_slabs,
            page_bytes=page_b,
        )


MOE_TLB_CUDA_SOURCE = r"""
typedef unsigned short bf16_t;
typedef unsigned char uint8_t;

__device__ __forceinline__ float bf16_to_f32(bf16_t u) {
    return __uint_as_float(((unsigned int)u) << 16);
}

__device__ __forceinline__ bf16_t f32_to_bf16(float f) {
    unsigned int x = __float_as_uint(f);
    if ((x & 0x7fffffffU) > 0x7f800000U) return (bf16_t)0x7fc0U;
    unsigned int lsb = (x >> 16) & 1U;
    x += 0x7fffU + lsb;
    return (bf16_t)(x >> 16);
}

__device__ __forceinline__ float apply_moe_act(float x, int act_kind) {
    if (act_kind == 2) {
        return x / (1.0f + __expf(-x));
    } else if (act_kind == 1) {
        return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
    } else {
        float x3 = x * x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x3);
        return 0.5f * x * (1.0f + tanhf(inner));
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffU, v, offset);
    }
    return v;
}

#define MAX_M 8
#define MAX_EXPERTS 1024
#define TILE_K 1024

extern "C" __global__ void moe_zero_f32_kernel(
    float* __restrict__ buf,
    int total_elems
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elems) {
        buf[idx] = 0.0f;
    }
}

extern "C" __global__ void moe_pack_routes_kernel(
    const long long* __restrict__ top_k_index,  // [T * K]
    const float* __restrict__ top_k_weights,    // [T * K]
    const int* __restrict__ expert_to_slot,     // [num_experts]: >=0 L1 VRAM, <=-2 L2 Zero-Copy, -1 unmapped
    int* __restrict__ active_slots,             // [MAX_ACTIVE]
    int* __restrict__ active_counts,            // [MAX_ACTIVE]
    int* __restrict__ active_tok_ids,           // [MAX_ACTIVE * MAX_M]
    int* __restrict__ active_pair_ids,          // [MAX_ACTIVE * MAX_M]
    float* __restrict__ active_weights,         // [MAX_ACTIVE * MAX_M]
    int* __restrict__ num_unique_out,           // [1]
    float* __restrict__ partial_f32,            // [T * K * d_model]
    int T,
    int K,
    int num_experts,
    int d_model
) {
    __shared__ short s_u_idx[MAX_EXPERTS];
    __shared__ short s_cnt[MAX_EXPERTS];

    const int tid = threadIdx.x;
    const int total_partial = T * K * d_model;
    for (int i = tid; i < total_partial; i += blockDim.x) {
        partial_f32[i] = 0.0f;
    }

    const int n_exp = (num_experts <= MAX_EXPERTS) ? num_experts : MAX_EXPERTS;
    for (int e = tid; e < n_exp; e += blockDim.x) {
        s_u_idx[e] = -1;
        s_cnt[e] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        int num_u = 0;
        const int total_pairs = T * K;
        for (int idx = 0; idx < total_pairs; ++idx) {
            const int t = idx / K;
            const int e = (int)top_k_index[idx];
            if (e < 0 || e >= n_exp) continue;
            const int slot = expert_to_slot[e];
            if (slot == -1) continue;
            const float w = top_k_weights[idx];

            int u = (int)s_u_idx[e];
            if (u < 0) {
                u = num_u++;
                s_u_idx[e] = (short)u;
                active_slots[u] = slot;
            }
            int pos = (int)s_cnt[e];
            if (pos > 0 && active_tok_ids[u * MAX_M + pos - 1] == t) {
                active_weights[u * MAX_M + pos - 1] += w;
            } else if (pos < MAX_M) {
                active_tok_ids[u * MAX_M + pos] = t;
                active_pair_ids[u * MAX_M + pos] = idx;
                active_weights[u * MAX_M + pos] = w;
                s_cnt[e] = (short)(pos + 1);
                active_counts[u] = pos + 1;
            }
        }
        *num_unique_out = num_u;
    }
}

extern "C" __global__ void moe_uint4_gate_up_act_kernel(
    const bf16_t* __restrict__ x_in,            // [T, d_model]
    const uint8_t* __restrict__ tlb_pages,      // [tlb_slots, page_bytes] (L1 VRAM, slot >= 0)
    const uint8_t* __restrict__ l2_pages,       // [l2_slots, page_bytes]  (L2 Zero-Copy DDR5, slot <= -2)
    const int* __restrict__ active_slots,       // [MAX_ACTIVE]
    const int* __restrict__ active_counts,      // [MAX_ACTIVE]
    const int* __restrict__ active_tok_ids,     // [MAX_ACTIVE * MAX_M]
    const int* __restrict__ num_unique_ptr,     // [1]
    float* __restrict__ inter_f32,              // [MAX_ACTIVE * MAX_M * d_ffn]
    int d_model,
    int d_ffn,
    int d_pad_half,
    int n_groups_d,
    int group_size,
    int page_bytes,
    int off_gu_q4,
    int off_gu_wmin,
    int off_gu_scale,
    int act_kind
) {
    const int u = blockIdx.y;
    if (u >= __ldg(num_unique_ptr)) return;

    extern __shared__ float s_x[]; // [m_cnt * TILE_K]
    __shared__ int s_tok[MAX_M];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp_id;

    const int slot = __ldg(&active_slots[u]);
    const int m_cnt = __ldg(&active_counts[u]);
    if (slot == -1 || m_cnt <= 0) return;
    if (slot < -1 && l2_pages == nullptr) return;

    if (tid < m_cnt) {
        s_tok[tid] = __ldg(&active_tok_ids[u * MAX_M + tid]);
    }
    __syncthreads();

    float acc_g[MAX_M];
    float acc_u[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc_g[m] = 0.0f;
        acc_u[m] = 0.0f;
    }

    const uint8_t* page_ptr = (slot >= 0)
        ? (tlb_pages + (size_t)slot * (size_t)page_bytes)
        : (l2_pages + (size_t)(-2 - slot) * (size_t)page_bytes);
    const uint8_t* gu_q4 = page_ptr + off_gu_q4;
    const bf16_t* gu_wmin = (const bf16_t*)(page_ptr + off_gu_wmin);
    const bf16_t* gu_scale = (const bf16_t*)(page_ptr + off_gu_scale);

    const int d_pad = d_pad_half * 2;
    const int g_half = group_size >> 1;
    const int vec4_ok = (((d_pad_half & 3) == 0) && ((g_half & 3) == 0) && (g_half >= 4));
    const int n_tiles = (d_pad + TILE_K - 1) / TILE_K;

    for (int tile = 0; tile < n_tiles; ++tile) {
        const int base_k = tile * TILE_K;
        for (int idx = tid; idx < TILE_K; idx += blockDim.x) {
            const int col = base_k + idx;
            for (int m = 0; m < m_cnt; ++m) {
                s_x[m * TILE_K + idx] = (col < d_model)
                    ? bf16_to_f32(__ldg(&x_in[(size_t)s_tok[m] * d_model + col]))
                    : 0.0f;
            }
        }
        __syncthreads();

        if (row < d_ffn) {
            const int base_b = base_k >> 1;
            const int b_lim = ((base_b + (TILE_K >> 1)) < d_pad_half) ? (TILE_K >> 1) : (d_pad_half - base_b);
            const int up_row = d_ffn + row;

            const uint8_t* rp_g = &gu_q4[(size_t)row * d_pad_half + base_b];
            const uint8_t* rp_u = &gu_q4[(size_t)up_row * d_pad_half + base_b];
            const bf16_t* rmn_g = &gu_wmin[(size_t)row * n_groups_d];
            const bf16_t* rsc_g = &gu_scale[(size_t)row * n_groups_d];
            const bf16_t* rmn_u = &gu_wmin[(size_t)up_row * n_groups_d];
            const bf16_t* rsc_u = &gu_scale[(size_t)up_row * n_groups_d];

            if (vec4_ok) {
                int b_l = lane * 4;
                for (; b_l + 3 < b_lim; b_l += 128) {
                    const int grp = (base_b + b_l) / g_half;
                    const unsigned int pg = __ldg((const unsigned int*)&rp_g[b_l]);
                    const unsigned int pu = __ldg((const unsigned int*)&rp_u[b_l]);
                    const float sc_g = bf16_to_f32(__ldg(&rsc_g[grp]));
                    const float mn_g = bf16_to_f32(__ldg(&rmn_g[grp]));
                    const float sc_u = bf16_to_f32(__ldg(&rsc_u[grp]));
                    const float mn_u = bf16_to_f32(__ldg(&rmn_u[grp]));

                    const float gw0 = (float)((pg >>  0) & 0x0FU) * sc_g + mn_g;
                    const float gw1 = (float)((pg >>  4) & 0x0FU) * sc_g + mn_g;
                    const float gw2 = (float)((pg >>  8) & 0x0FU) * sc_g + mn_g;
                    const float gw3 = (float)((pg >> 12) & 0x0FU) * sc_g + mn_g;
                    const float gw4 = (float)((pg >> 16) & 0x0FU) * sc_g + mn_g;
                    const float gw5 = (float)((pg >> 20) & 0x0FU) * sc_g + mn_g;
                    const float gw6 = (float)((pg >> 24) & 0x0FU) * sc_g + mn_g;
                    const float gw7 = (float)((pg >> 28) & 0x0FU) * sc_g + mn_g;

                    const float uw0 = (float)((pu >>  0) & 0x0FU) * sc_u + mn_u;
                    const float uw1 = (float)((pu >>  4) & 0x0FU) * sc_u + mn_u;
                    const float uw2 = (float)((pu >>  8) & 0x0FU) * sc_u + mn_u;
                    const float uw3 = (float)((pu >> 12) & 0x0FU) * sc_u + mn_u;
                    const float uw4 = (float)((pu >> 16) & 0x0FU) * sc_u + mn_u;
                    const float uw5 = (float)((pu >> 20) & 0x0FU) * sc_u + mn_u;
                    const float uw6 = (float)((pu >> 24) & 0x0FU) * sc_u + mn_u;
                    const float uw7 = (float)((pu >> 28) & 0x0FU) * sc_u + mn_u;

                    const int x_idx = b_l << 1;
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < m_cnt) {
                            const float* sx = &s_x[m * TILE_K + x_idx];
                            acc_g[m] += gw0 * sx[0] + gw1 * sx[1] + gw2 * sx[2] + gw3 * sx[3]
                                      + gw4 * sx[4] + gw5 * sx[5] + gw6 * sx[6] + gw7 * sx[7];
                            acc_u[m] += uw0 * sx[0] + uw1 * sx[1] + uw2 * sx[2] + uw3 * sx[3]
                                      + uw4 * sx[4] + uw5 * sx[5] + uw6 * sx[6] + uw7 * sx[7];
                        }
                    }
                }
                for (int b_rem = (b_lim & ~3) + lane; b_rem < b_lim; b_rem += 32) {
                    const int grp = (base_b + b_rem) / g_half;
                    const uint8_t pg = __ldg(&rp_g[b_rem]);
                    const uint8_t pu = __ldg(&rp_u[b_rem]);
                    const float sc_g = bf16_to_f32(__ldg(&rsc_g[grp]));
                    const float mn_g = bf16_to_f32(__ldg(&rmn_g[grp]));
                    const float sc_u = bf16_to_f32(__ldg(&rsc_u[grp]));
                    const float mn_u = bf16_to_f32(__ldg(&rmn_u[grp]));

                    const float gw0 = (float)(pg & 0x0FU) * sc_g + mn_g;
                    const float gw1 = (float)((pg >> 4) & 0x0FU) * sc_g + mn_g;
                    const float uw0 = (float)(pu & 0x0FU) * sc_u + mn_u;
                    const float uw1 = (float)((pu >> 4) & 0x0FU) * sc_u + mn_u;
                    const int x_idx = b_rem << 1;
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < m_cnt) {
                            const float sx0 = s_x[m * TILE_K + x_idx + 0];
                            const float sx1 = s_x[m * TILE_K + x_idx + 1];
                            acc_g[m] += gw0 * sx0 + gw1 * sx1;
                            acc_u[m] += uw0 * sx0 + uw1 * sx1;
                        }
                    }
                }
            } else {
                for (int b_l = lane; b_l < b_lim; b_l += 32) {
                    const int grp = (base_b + b_l) / g_half;
                    const uint8_t pg = __ldg(&rp_g[b_l]);
                    const uint8_t pu = __ldg(&rp_u[b_l]);
                    const float sc_g = bf16_to_f32(__ldg(&rsc_g[grp]));
                    const float mn_g = bf16_to_f32(__ldg(&rmn_g[grp]));
                    const float sc_u = bf16_to_f32(__ldg(&rsc_u[grp]));
                    const float mn_u = bf16_to_f32(__ldg(&rmn_u[grp]));

                    const float gw0 = (float)(pg & 0x0FU) * sc_g + mn_g;
                    const float gw1 = (float)((pg >> 4) & 0x0FU) * sc_g + mn_g;
                    const float uw0 = (float)(pu & 0x0FU) * sc_u + mn_u;
                    const float uw1 = (float)((pu >> 4) & 0x0FU) * sc_u + mn_u;
                    const int x_idx = b_l << 1;
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < m_cnt) {
                            const float sx0 = s_x[m * TILE_K + x_idx + 0];
                            const float sx1 = s_x[m * TILE_K + x_idx + 1];
                            acc_g[m] += gw0 * sx0 + gw1 * sx1;
                            acc_u[m] += uw0 * sx0 + uw1 * sx1;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (row < d_ffn) {
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < m_cnt) {
                const float g_val = warp_sum(acc_g[m]);
                const float u_val = warp_sum(acc_u[m]);
                if (lane == 0) {
                    inter_f32[((size_t)u * MAX_M + m) * d_ffn + row] = apply_moe_act(g_val, act_kind) * u_val;
                }
            }
        }
    }
}

extern "C" __global__ void moe_uint4_down_accum_kernel(
    const float* __restrict__ inter_f32,        // [MAX_ACTIVE * MAX_M * d_ffn]
    const uint8_t* __restrict__ tlb_pages,      // [tlb_slots, page_bytes] (L1 VRAM, slot >= 0)
    const uint8_t* __restrict__ l2_pages,       // [l2_slots, page_bytes]  (L2 Zero-Copy DDR5, slot <= -2)
    const int* __restrict__ active_slots,       // [MAX_ACTIVE]
    const int* __restrict__ active_counts,      // [MAX_ACTIVE]
    const int* __restrict__ active_pair_ids,    // [MAX_ACTIVE * MAX_M]
    const float* __restrict__ active_weights,   // [MAX_ACTIVE * MAX_M]
    const int* __restrict__ num_unique_ptr,     // [1]
    float* __restrict__ partial_f32,            // [T * K * d_model]
    int d_model,
    int d_ffn,
    int i_pad_half,
    int n_groups_i,
    int group_size,
    int page_bytes,
    int off_dn_q4,
    int off_dn_wmin,
    int off_dn_scale
) {
    const int u = blockIdx.y;
    if (u >= __ldg(num_unique_ptr)) return;

    extern __shared__ float s_z[]; // [m_cnt * TILE_K]
    __shared__ int s_pair[MAX_M];
    __shared__ float s_w[MAX_M];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp_id;

    const int slot = __ldg(&active_slots[u]);
    const int m_cnt = __ldg(&active_counts[u]);
    if (slot == -1 || m_cnt <= 0) return;
    if (slot < -1 && l2_pages == nullptr) return;

    if (tid < m_cnt) {
        s_pair[tid] = __ldg(&active_pair_ids[u * MAX_M + tid]);
        s_w[tid] = __ldg(&active_weights[u * MAX_M + tid]);
    }
    __syncthreads();

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    const uint8_t* page_ptr = (slot >= 0)
        ? (tlb_pages + (size_t)slot * (size_t)page_bytes)
        : (l2_pages + (size_t)(-2 - slot) * (size_t)page_bytes);
    const uint8_t* dn_q4 = page_ptr + off_dn_q4;
    const bf16_t* dn_wmin = (const bf16_t*)(page_ptr + off_dn_wmin);
    const bf16_t* dn_scale = (const bf16_t*)(page_ptr + off_dn_scale);

    const int i_pad = i_pad_half * 2;
    const int g_half = group_size >> 1;
    const int vec4_ok = (((i_pad_half & 3) == 0) && ((g_half & 3) == 0) && (g_half >= 4));
    const int n_tiles = (i_pad + TILE_K - 1) / TILE_K;

    for (int tile = 0; tile < n_tiles; ++tile) {
        const int base_k = tile * TILE_K;
        for (int idx = tid; idx < TILE_K; idx += blockDim.x) {
            const int col = base_k + idx;
            for (int m = 0; m < m_cnt; ++m) {
                s_z[m * TILE_K + idx] = (col < d_ffn)
                    ? __ldg(&inter_f32[((size_t)u * MAX_M + m) * d_ffn + col])
                    : 0.0f;
            }
        }
        __syncthreads();

        if (row < d_model) {
            const int base_b = base_k >> 1;
            const int b_lim = ((base_b + (TILE_K >> 1)) < i_pad_half) ? (TILE_K >> 1) : (i_pad_half - base_b);
            const uint8_t* rp = &dn_q4[(size_t)row * i_pad_half + base_b];
            const bf16_t* rmn = &dn_wmin[(size_t)row * n_groups_i];
            const bf16_t* rsc = &dn_scale[(size_t)row * n_groups_i];

            if (vec4_ok) {
                int b_l = lane * 4;
                for (; b_l + 3 < b_lim; b_l += 128) {
                    const int grp = (base_b + b_l) / g_half;
                    const unsigned int p4 = __ldg((const unsigned int*)&rp[b_l]);
                    const float sc = bf16_to_f32(__ldg(&rsc[grp]));
                    const float mn = bf16_to_f32(__ldg(&rmn[grp]));

                    const float w0 = (float)((p4 >>  0) & 0x0FU) * sc + mn;
                    const float w1 = (float)((p4 >>  4) & 0x0FU) * sc + mn;
                    const float w2 = (float)((p4 >>  8) & 0x0FU) * sc + mn;
                    const float w3 = (float)((p4 >> 12) & 0x0FU) * sc + mn;
                    const float w4 = (float)((p4 >> 16) & 0x0FU) * sc + mn;
                    const float w5 = (float)((p4 >> 20) & 0x0FU) * sc + mn;
                    const float w6 = (float)((p4 >> 24) & 0x0FU) * sc + mn;
                    const float w7 = (float)((p4 >> 28) & 0x0FU) * sc + mn;
                    const int z_idx = b_l << 1;

                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < m_cnt) {
                            const float* sz = &s_z[m * TILE_K + z_idx];
                            acc[m] += w0 * sz[0] + w1 * sz[1] + w2 * sz[2] + w3 * sz[3]
                                    + w4 * sz[4] + w5 * sz[5] + w6 * sz[6] + w7 * sz[7];
                        }
                    }
                }
                for (int b_rem = (b_lim & ~3) + lane; b_rem < b_lim; b_rem += 32) {
                    const int grp = (base_b + b_rem) / g_half;
                    const uint8_t packed = __ldg(&rp[b_rem]);
                    const float sc = bf16_to_f32(__ldg(&rsc[grp]));
                    const float mn = bf16_to_f32(__ldg(&rmn[grp]));
                    const float w0 = (float)(packed & 0x0FU) * sc + mn;
                    const float w1 = (float)((packed >> 4) & 0x0FU) * sc + mn;
                    const int z_idx = b_rem << 1;
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < m_cnt) {
                            acc[m] += w0 * s_z[m * TILE_K + z_idx + 0] + w1 * s_z[m * TILE_K + z_idx + 1];
                        }
                    }
                }
            } else {
                for (int b_l = lane; b_l < b_lim; b_l += 32) {
                    const int grp = (base_b + b_l) / g_half;
                    const uint8_t packed = __ldg(&rp[b_l]);
                    const float sc = bf16_to_f32(__ldg(&rsc[grp]));
                    const float mn = bf16_to_f32(__ldg(&rmn[grp]));
                    const float w0 = (float)(packed & 0x0FU) * sc + mn;
                    const float w1 = (float)((packed >> 4) & 0x0FU) * sc + mn;
                    const int z_idx = b_l << 1;
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < m_cnt) {
                            acc[m] += w0 * s_z[m * TILE_K + z_idx + 0] + w1 * s_z[m * TILE_K + z_idx + 1];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (row < d_model) {
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < m_cnt) {
                const float s = warp_sum(acc[m]);
                if (lane == 0) {
                    partial_f32[(size_t)s_pair[m] * d_model + row] = s * s_w[m];
                }
            }
        }
    }
}

extern "C" __global__ void moe_f32_to_bf16_kernel(
    const float* __restrict__ partial_f32,  // [T, K, d_model]
    bf16_t* __restrict__ out_bf16,          // [T, d_model]
    int total_elems,
    int K,
    int d_model
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elems) {
        const int t = idx / d_model;
        const int col = idx - t * d_model;
        const float* base = partial_f32 + ((size_t)t * (size_t)K) * (size_t)d_model + col;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += __ldg(&base[(size_t)k * (size_t)d_model]);
        }
        out_bf16[idx] = f32_to_bf16(acc);
    }
}

// =============================================================================
// UINT4-WINA L2 Zero-Copy Kernels (Gate-First Conditional Neuron-Major Fetch)
// Splits active routes into:
//   - L1 VRAM hits (`slot >= 0`)   -> exact UINT4 `moe_uint4_gate_up_act_kernel` + `moe_uint4_down_accum_kernel`
//   - L2 DDR5 hits (`slot <= -2`)  -> `moe_wina_gate_q4_kernel` -> `moe_wina_select_top_kernel` -> `moe_wina_up_down_q4_kernel`
// Both accumulate into `partial_f32[pair_id, d]` so `moe_f32_to_bf16_kernel` sums L1 + L2 seamlessly!
// 100% Full CUDA Graph safe (`if (a >= *num_l2_ptr) return;` with zero CPU sync!).
// =============================================================================
extern "C" __global__ void moe_pack_routes_wina_kernel(
    const long long* __restrict__ top_k_index,  // [T * K]
    const float* __restrict__ top_k_weights,    // [T * K]
    const int* __restrict__ expert_to_slot,     // [num_experts]: >=0 L1 VRAM, <=-2 L2 WINA, -1 unmapped
    int* __restrict__ active_slots,             // [MAX_ACTIVE] (L1 only)
    int* __restrict__ active_counts,            // [MAX_ACTIVE]
    int* __restrict__ active_tok_ids,           // [MAX_ACTIVE * MAX_M]
    int* __restrict__ active_pair_ids,          // [MAX_ACTIVE * MAX_M]
    float* __restrict__ active_weights,         // [MAX_ACTIVE * MAX_M]
    int* __restrict__ num_unique_out,           // [1] (L1 unique count)
    int* __restrict__ wina_l2_slots,            // [MAX_ACTIVE] (l2_idx = -2 - slot)
    int* __restrict__ wina_l2_tok_ids,          // [MAX_ACTIVE]
    int* __restrict__ wina_l2_pair_ids,         // [MAX_ACTIVE]
    float* __restrict__ wina_l2_weights,        // [MAX_ACTIVE]
    int* __restrict__ wina_num_l2_out,          // [1] (L2 pair count)
    float* __restrict__ partial_f32,            // [T * K * d_model]
    int T,
    int K,
    int num_experts,
    int d_model
) {
    __shared__ short s_u_idx[MAX_EXPERTS];
    __shared__ short s_cnt[MAX_EXPERTS];

    const int tid = threadIdx.x;
    const int total_partial = T * K * d_model;
    for (int i = tid; i < total_partial; i += blockDim.x) {
        partial_f32[i] = 0.0f;
    }

    const int n_exp = (num_experts <= MAX_EXPERTS) ? num_experts : MAX_EXPERTS;
    for (int e = tid; e < n_exp; e += blockDim.x) {
        s_u_idx[e] = -1;
        s_cnt[e] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        int num_u = 0;
        int num_l2 = 0;
        const int total_pairs = T * K;
        for (int idx = 0; idx < total_pairs; ++idx) {
            const int t = idx / K;
            const int e = (int)top_k_index[idx];
            if (e < 0 || e >= n_exp) continue;
            const int slot = expert_to_slot[e];
            if (slot == -1) continue;
            const float w = top_k_weights[idx];

            if (slot <= -2) {
                if (num_l2 < 128) {
                    wina_l2_slots[num_l2] = -2 - slot;
                    wina_l2_tok_ids[num_l2] = t;
                    wina_l2_pair_ids[num_l2] = idx;
                    wina_l2_weights[num_l2] = w;
                    num_l2++;
                }
                continue;
            }

            int u = (int)s_u_idx[e];
            if (u < 0) {
                u = num_u++;
                s_u_idx[e] = (short)u;
                active_slots[u] = slot;
            }
            int pos = (int)s_cnt[e];
            if (pos > 0 && active_tok_ids[u * MAX_M + pos - 1] == t) {
                active_weights[u * MAX_M + pos - 1] += w;
            } else if (pos < MAX_M) {
                active_tok_ids[u * MAX_M + pos] = t;
                active_pair_ids[u * MAX_M + pos] = idx;
                active_weights[u * MAX_M + pos] = w;
                s_cnt[e] = (short)(pos + 1);
                active_counts[u] = pos + 1;
            }
        }
        *num_unique_out = num_u;
        *wina_num_l2_out = num_l2;
    }
}

__device__ __forceinline__ float wina_dot_q4_grp64_bf16(
    const uint4* __restrict__ q4_u4,
    unsigned int mn_sc_packed,
    const bf16_t* __restrict__ xp
) {
    const float mn = bf16_to_f32((bf16_t)(mn_sc_packed & 0xFFFFU));
    const float sc = bf16_to_f32((bf16_t)(mn_sc_packed >> 16));
    uint4 v0 = __ldg(&q4_u4[0]);
    uint4 v1 = __ldg(&q4_u4[1]);
    unsigned int words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    float qdot = 0.0f;
    float xsum = 0.0f;
    #pragma unroll
    for (int wi = 0; wi < 8; ++wi) {
        unsigned int w = words[wi];
        const bf16_t* xw = xp + (wi << 3);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float xv = bf16_to_f32(__ldg(&xw[k]));
            xsum += xv;
            qdot += (float)((w >> (k << 2)) & 0x0FU) * xv;
        }
    }
    return qdot * sc + xsum * mn;
}

extern "C" __global__ void moe_wina_gate_q4_kernel(
    const bf16_t* __restrict__ x_in,          // [T, d_model]
    const uint8_t* __restrict__ l2_wina_pages,// [l2_slots, page_bytes]
    const int* __restrict__ wina_l2_slots,    // [MAX_L2]
    const int* __restrict__ wina_l2_tok_ids,  // [MAX_L2]
    const int* __restrict__ wina_num_l2_ptr,  // [1]
    float* __restrict__ gate_act_out,         // [MAX_L2, d_ffn]
    float* __restrict__ wina_score_out,       // [MAX_L2, d_ffn]
    int d_model,
    int d_ffn,
    int n_groups_d,
    int row_bytes,
    int page_bytes,
    int off_gate,
    int off_wina_norm,
    int act_kind
) {
    const int a = blockIdx.y;
    if (a >= *wina_num_l2_ptr) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 4 + warp;
    if (row >= d_ffn) return;

    const int l2_idx = wina_l2_slots[a];
    const int t = wina_l2_tok_ids[a];
    const bf16_t* xt = x_in + (size_t)t * (size_t)d_model;
    const uint8_t* page = l2_wina_pages + (size_t)l2_idx * (size_t)page_bytes;
    const uint8_t* row_ptr = page + off_gate + (size_t)row * (size_t)row_bytes;
    const float dn_norm = bf16_to_f32(__ldg(&((const bf16_t*)(page + off_wina_norm))[row]));

    const uint4* q4_base = (const uint4*)row_ptr;
    const unsigned int* ms_base = (const unsigned int*)(row_ptr + n_groups_d * 32);
    float acc = 0.0f;
    for (int g = lane; g < n_groups_d; g += 32) {
        acc += wina_dot_q4_grp64_bf16(q4_base + g * 2, __ldg(&ms_base[g]), xt + (g << 6));
    }

    float g_val = warp_sum(acc);
    if (lane == 0) {
        float ga = apply_moe_act(g_val, act_kind);
        gate_act_out[(size_t)a * (size_t)d_ffn + row] = ga;
        wina_score_out[(size_t)a * (size_t)d_ffn + row] = fabsf(ga) * dn_norm;
    }
}

extern "C" __global__ void moe_wina_select_top_kernel(
    const float* __restrict__ wina_scores,    // [MAX_L2, d_ffn]
    const int* __restrict__ wina_num_l2_ptr,  // [1]
    int* __restrict__ active_neurons,         // [MAX_L2, k_keep]
    int d_ffn,
    int k_keep
) {
    const int a = blockIdx.x;
    if (a >= *wina_num_l2_ptr) return;

    const int tid = threadIdx.x;
    __shared__ float s_sc[2048];
    __shared__ float s_lo, s_hi, s_thr;
    __shared__ int s_cnt;

    const float* sc_row = wina_scores + (size_t)a * (size_t)d_ffn;
    float local_max = 0.0f;
    for (int j = tid; j < d_ffn; j += blockDim.x) {
        float v = sc_row[j];
        if (j < 2048) s_sc[j] = v;
        if (v > local_max) local_max = v;
    }
    if (tid == 0) { s_lo = 0.0f; s_hi = 0.0f; }
    __syncthreads();
    atomicMax((int*)&s_hi, __float_as_int(local_max));
    __syncthreads();

    for (int iter = 0; iter < 14; ++iter) {
        if (tid == 0) { s_thr = 0.5f * (s_lo + s_hi); s_cnt = 0; }
        __syncthreads();
        float thr = s_thr;
        int c = 0;
        for (int j = tid; j < d_ffn; j += blockDim.x) {
            float v = (j < 2048) ? s_sc[j] : sc_row[j];
            if (v >= thr) c++;
        }
        if (c > 0) atomicAdd(&s_cnt, c);
        __syncthreads();
        if (tid == 0) {
            if (s_cnt > k_keep) s_lo = thr;
            else s_hi = thr;
        }
        __syncthreads();
    }

    if (tid == 0) s_cnt = 0;
    __syncthreads();
    float final_thr = s_hi;
    int* out_row = active_neurons + (size_t)a * (size_t)k_keep;

    for (int j = tid; j < d_ffn; j += blockDim.x) {
        float v = (j < 2048) ? s_sc[j] : sc_row[j];
        if (v >= final_thr) {
            int pos = atomicAdd(&s_cnt, 1);
            if (pos < k_keep) out_row[pos] = j;
        }
    }
    __syncthreads();
    for (int j = tid; j < d_ffn; j += blockDim.x) {
        float v = (j < 2048) ? s_sc[j] : sc_row[j];
        if (v < final_thr && v >= s_lo) {
            int pos = atomicAdd(&s_cnt, 1);
            if (pos < k_keep) out_row[pos] = j;
        }
    }
    __syncthreads();
    for (int j = tid; j < k_keep; j += blockDim.x) {
        if (j >= s_cnt) out_row[j] = j;
    }
}

extern "C" __global__ void moe_wina_up_down_q4_kernel(
    const bf16_t* __restrict__ x_in,          // [T, d_model]
    const uint8_t* __restrict__ l2_wina_pages,// [l2_slots, page_bytes]
    const int* __restrict__ wina_l2_slots,    // [MAX_L2]
    const int* __restrict__ wina_l2_tok_ids,  // [MAX_L2]
    const int* __restrict__ wina_l2_pair_ids, // [MAX_L2]
    const float* __restrict__ wina_l2_weights,// [MAX_L2]
    const int* __restrict__ wina_num_l2_ptr,  // [1]
    const float* __restrict__ gate_act,       // [MAX_L2, d_ffn]
    const int* __restrict__ active_neurons,   // [MAX_L2, k_keep]
    float* __restrict__ partial_f32,          // [T * K * d_model] (each pair has its own row!)
    int d_model,
    int d_ffn,
    int k_keep,
    int n_groups_d,
    int row_bytes,
    int slab_bytes,
    int page_bytes,
    int off_slabs
) {
    const int a = blockIdx.y;
    if (a >= *wina_num_l2_ptr) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int slot_k = blockIdx.x * 4 + warp;
    if (slot_k >= k_keep) return;

    const int j = (k_keep < d_ffn) ? active_neurons[(size_t)a * (size_t)k_keep + slot_k] : slot_k;
    const int l2_idx = wina_l2_slots[a];
    const int t = wina_l2_tok_ids[a];
    const int pair_id = wina_l2_pair_ids[a];
    const bf16_t* xt = x_in + (size_t)t * (size_t)d_model;
    const uint8_t* page = l2_wina_pages + (size_t)l2_idx * (size_t)page_bytes;
    const uint8_t* slab = page + off_slabs + (size_t)j * (size_t)slab_bytes;

    const uint4* up_q4 = (const uint4*)slab;
    const unsigned int* up_ms = (const unsigned int*)(slab + n_groups_d * 32);
    float u_acc = 0.0f;
    for (int g = lane; g < n_groups_d; g += 32) {
        u_acc += wina_dot_q4_grp64_bf16(up_q4 + g * 2, __ldg(&up_ms[g]), xt + (g << 6));
    }

    float u_val = warp_sum(u_acc);
    u_val = __shfl_sync(0xffffffffU, u_val, 0);
    const float coeff = gate_act[(size_t)a * (size_t)d_ffn + j] * u_val * wina_l2_weights[a];

    const uint8_t* dn_ptr = slab + row_bytes;
    const uint4* dn_q4 = (const uint4*)dn_ptr;
    const unsigned int* dn_ms = (const unsigned int*)(dn_ptr + n_groups_d * 32);
    float* dst_row = partial_f32 + (size_t)pair_id * (size_t)d_model;

    for (int g = lane; g < n_groups_d; g += 32) {
        unsigned int mn_sc = __ldg(&dn_ms[g]);
        const float mn_c = bf16_to_f32((bf16_t)(mn_sc & 0xFFFFU)) * coeff;
        const float sc_c = bf16_to_f32((bf16_t)(mn_sc >> 16)) * coeff;
        uint4 v0 = __ldg(&dn_q4[g * 2]);
        uint4 v1 = __ldg(&dn_q4[g * 2 + 1]);
        unsigned int words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        float* yp = dst_row + (g << 6);
        #pragma unroll
        for (int wi = 0; wi < 8; ++wi) {
            unsigned int w = words[wi];
            float* yw = yp + (wi << 3);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                atomicAdd(&yw[k], (float)((w >> (k << 2)) & 0x0FU) * sc_c + mn_c);
            }
        }
    }
}
"""

_MOE_KERNELS_BY_DEVICE: Dict[int, Dict[str, Any]] = {}


def get_or_compile_moe_tlb_kernels(device: torch.device) -> Dict[str, Any]:
    dev_idx = 0 if device.index is None else int(device.index)
    if dev_idx in _MOE_KERNELS_BY_DEVICE:
        return _MOE_KERNELS_BY_DEVICE[dev_idx]

    from torch.cuda._utils import _cuda_load_module, _get_gpu_rtc_library, _get_gpu_runtime_library

    with torch.cuda.device(dev_idx):
        libnvrtc = _get_gpu_rtc_library()
        libcuda = _get_gpu_runtime_library()
        props = torch.cuda.get_device_properties(dev_idx)
        arch = f"--gpu-architecture=sm_{props.major}{props.minor}".encode("utf-8")
        opts = [arch, b"--use_fast_math", b"-default-device"]
        opts_arr = (ctypes.c_char_p * len(opts))(*opts)

        src = MOE_TLB_CUDA_SOURCE.encode("utf-8")
        prog = ctypes.c_void_p()
        libnvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"moe_tlb_kernels.cu", 0, None, None)
        res = libnvrtc.nvrtcCompileProgram(prog, len(opts), opts_arr)
        if res != 0:
            log_sz = ctypes.c_size_t()
            libnvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
            log_buf = ctypes.create_string_buffer(log_sz.value)
            libnvrtc.nvrtcGetProgramLog(prog, log_buf)
            libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            raise RuntimeError(f"MoE TLB NVRTC failed:\n{log_buf.value.decode('utf-8', errors='replace')}")

        cubin_sz = ctypes.c_size_t()
        libnvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(cubin_sz))
        cubin_buf = ctypes.create_string_buffer(cubin_sz.value)
        libnvrtc.nvrtcGetCUBIN(prog, cubin_buf)
        libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

        names = [
            "moe_zero_f32_kernel",
            "moe_pack_routes_kernel",
            "moe_uint4_gate_up_act_kernel",
            "moe_uint4_down_accum_kernel",
            "moe_f32_to_bf16_kernel",
            "moe_pack_routes_wina_kernel",
            "moe_wina_gate_q4_kernel",
            "moe_wina_select_top_kernel",
            "moe_wina_up_down_q4_kernel",
        ]
        kd = _cuda_load_module(cubin_buf.raw, names)
        kd["moe_uint4_gate_up_act_kernel"].set_shared_memory_config(65536)
        kd["moe_uint4_down_accum_kernel"].set_shared_memory_config(65536)
        kd["libcuda"] = libcuda

    _MOE_KERNELS_BY_DEVICE[dev_idx] = kd
    return kd


def get_zero_copy_device_ptr(pinned_host_tensor: torch.Tensor, device: torch.device) -> int:
    """Maps a `pin_memory=True` host tensor into the GPU's 64-bit UVA address space (`cuMemHostGetDevicePointer_v2`)."""
    if pinned_host_tensor.numel() == 0:
        return 0
    if not pinned_host_tensor.is_pinned():
        raise ValueError("get_zero_copy_device_ptr requires a pinned CPU tensor (is_pinned() == True).")
    dev_idx = 0 if device.index is None else int(device.index)
    kd = get_or_compile_moe_tlb_kernels(device)
    libcuda = kd["libcuda"]
    fn = getattr(libcuda, "cuMemHostGetDevicePointer_v2", None) or getattr(libcuda, "cuMemHostGetDevicePointer")
    with torch.cuda.device(dev_idx):
        dev_ptr = ctypes.c_uint64(0)
        res = fn(ctypes.byref(dev_ptr), ctypes.c_void_p(pinned_host_tensor.data_ptr()), ctypes.c_uint(0))
        if res != 0:
            raise RuntimeError(f"cuMemHostGetDevicePointer_v2 failed with CUDA driver error {res}")
        return int(dev_ptr.value)


def query_available_system_ram_bytes() -> Tuple[int, int]:
    """Returns `(avail_phys_bytes, total_phys_bytes)` on Windows or Linux."""
    try:
        if hasattr(ctypes, "windll"):
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = _MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(st)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullAvailPhys), int(st.ullTotalPhys)
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            vals: Dict[str, int] = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    vals[parts[0].strip()] = int(parts[1].strip().split()[0]) * 1024
            host_avail = vals.get("MemAvailable", 0)
            host_total = vals.get("MemTotal", 0)

            # Respect Docker / Kubernetes cgroup v2 & v1 container memory limits on Linux/AWS
            cg_limit = 0
            cg_used = 0
            try:
                if os.path.isfile("/sys/fs/cgroup/memory.max"):
                    raw_max = open("/sys/fs/cgroup/memory.max", "r", encoding="utf-8").read().strip()
                    if raw_max.isdigit():
                        cg_limit = int(raw_max)
                        if os.path.isfile("/sys/fs/cgroup/memory.current"):
                            cg_used = int(open("/sys/fs/cgroup/memory.current", "r", encoding="utf-8").read().strip())
                elif os.path.isfile("/sys/fs/cgroup/memory/memory.limit_in_bytes"):
                    raw_v1 = int(
                        open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r", encoding="utf-8").read().strip()
                    )
                    if 0 < raw_v1 < (1 << 60):
                        cg_limit = raw_v1
                        if os.path.isfile("/sys/fs/cgroup/memory/memory.usage_in_bytes"):
                            cg_used = int(
                                open("/sys/fs/cgroup/memory/memory.usage_in_bytes", "r", encoding="utf-8").read().strip()
                            )
            except Exception:
                cg_limit = 0

            if 0 < cg_limit < host_total:
                cg_avail = max(0, cg_limit - cg_used)
                return min(host_avail, cg_avail), cg_limit
            return host_avail, host_total
    except Exception:
        return 0, 0


class SharedMoEWorkspace:
    """Reusable per-GPU workspace + Pinned DMA Staging Ring shared across all MoE layers."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        # Pack all 6 int32 route-descriptor arrays into one contiguous GPU buffer so multi-wave prefill
        # can upload an entire 128-item wave in ONE 14 KB async H2D DMA transfer!
        # Layout in int32 elements:
        #   [0 .. 128):            active_slots    (128)
        #   [128 .. 256):          active_counts   (128)
        #   [256 .. 1280):         active_tok_ids  (1024)
        #   [1280 .. 2304):        active_pair_ids (1024)
        #   [2304 .. 2305):        num_unique      (1)
        # Total = 2305 int32s (9,220 B) + active_weights 1024 float32s (4,096 B)
        self._route_int_buf = torch.zeros((2308,), dtype=torch.int32, device=device)
        self.active_slots = self._route_int_buf[0:MAX_ACTIVE_UNIQUE]
        self.active_counts = self._route_int_buf[MAX_ACTIVE_UNIQUE : 2 * MAX_ACTIVE_UNIQUE]
        n_tp = MAX_ACTIVE_UNIQUE * MAX_FUSED_DECODE_TOKENS  # 1024
        self.active_tok_ids = self._route_int_buf[2 * MAX_ACTIVE_UNIQUE : 2 * MAX_ACTIVE_UNIQUE + n_tp]
        self.active_pair_ids = self._route_int_buf[2 * MAX_ACTIVE_UNIQUE + n_tp : 2 * MAX_ACTIVE_UNIQUE + 2 * n_tp]
        self.num_unique = self._route_int_buf[2 * MAX_ACTIVE_UNIQUE + 2 * n_tp : 2 * MAX_ACTIVE_UNIQUE + 2 * n_tp + 1]
        self.active_weights = torch.zeros((n_tp,), dtype=torch.float32, device=device)

        self.pinned_route_int = torch.zeros((2308,), dtype=torch.int32, device="cpu")
        self.pinned_route_w = torch.zeros((n_tp,), dtype=torch.float32, device="cpu")
        if torch.cuda.is_available() and device.type == "cuda":
            try:
                self.pinned_route_int = self.pinned_route_int.pin_memory()
                self.pinned_route_w = self.pinned_route_w.pin_memory()
            except Exception:
                pass

        self.max_d_model: int = 0
        self.max_d_ffn: int = 0
        self.max_page_bytes: int = 0
        self.inter_f32 = torch.empty((0,), dtype=torch.float32, device=device)
        self.out_f32 = torch.empty((0,), dtype=torch.float32, device=device)
        self.prefill_page_stage = torch.empty((0,), dtype=torch.uint8, device=device)
        self.prefill_out_f32 = torch.empty((0,), dtype=torch.float32, device=device)

        # Compact GPU scratch buffers for `UINT4-WINA` L2 Zero-Copy conditional neuron fetch (< 1.5 MiB)
        self.wina_l2_slots = torch.zeros((MAX_ACTIVE_UNIQUE,), dtype=torch.int32, device=device)
        self.wina_l2_tok_ids = torch.zeros((MAX_ACTIVE_UNIQUE,), dtype=torch.int32, device=device)
        self.wina_l2_pair_ids = torch.zeros((MAX_ACTIVE_UNIQUE,), dtype=torch.int32, device=device)
        self.wina_l2_weights = torch.zeros((MAX_ACTIVE_UNIQUE,), dtype=torch.float32, device=device)
        self.wina_num_l2 = torch.zeros((1,), dtype=torch.int32, device=device)
        self.wina_gate_act = torch.empty((0,), dtype=torch.float32, device=device)
        self.wina_scores = torch.empty((0,), dtype=torch.float32, device=device)
        self.wina_active_neurons = torch.empty((0,), dtype=torch.int32, device=device)

        # Shared 32-slot pinned host DMA staging ring (~56 MB on 80B) eliminating blocking pageable mmap copies!
        self.pinned_dma_stage = torch.empty((0, 0), dtype=torch.uint8, device="cpu")
        self._dma_ring_cursor: int = 0
        is_cuda = torch.cuda.is_available() and device.type == "cuda"
        self._slot_events: List[Optional[torch.cuda.Event]] = [
            torch.cuda.Event() if is_cuda else None for _ in range(DMA_STAGE_RING_SLOTS)
        ]
        self._slot_recorded: List[bool] = [False] * DMA_STAGE_RING_SLOTS
        self.version: int = 0

    def ensure_capacity(self, d_model: int, d_ffn: int, page_bytes: int = 0) -> None:
        changed = False
        if d_ffn > self.max_d_ffn:
            self.max_d_ffn = int(d_ffn)
            self.inter_f32 = torch.zeros(
                (MAX_ACTIVE_UNIQUE * MAX_FUSED_DECODE_TOKENS * self.max_d_ffn,),
                dtype=torch.float32,
                device=self.device,
            )
            self.wina_gate_act = torch.zeros((MAX_ACTIVE_UNIQUE * self.max_d_ffn,), dtype=torch.float32, device=self.device)
            self.wina_scores = torch.zeros((MAX_ACTIVE_UNIQUE * self.max_d_ffn,), dtype=torch.float32, device=self.device)
            self.wina_active_neurons = torch.zeros((MAX_ACTIVE_UNIQUE * self.max_d_ffn,), dtype=torch.int32, device=self.device)
            changed = True
        if d_model > self.max_d_model:
            self.max_d_model = int(d_model)
            self.out_f32 = torch.zeros(
                (MAX_FUSED_DECODE_TOKENS * MAX_MOE_TOPK * self.max_d_model,),
                dtype=torch.float32,
                device=self.device,
            )
            changed = True
        if page_bytes > self.max_page_bytes:
            self.max_page_bytes = int(page_bytes)
            stage_cpu = torch.empty((DMA_STAGE_RING_SLOTS, self.max_page_bytes), dtype=torch.uint8, device="cpu")
            if torch.cuda.is_available() and self.device.type == "cuda":
                try:
                    stage_cpu = stage_cpu.pin_memory()
                except Exception:
                    pass
            self.pinned_dma_stage = stage_cpu
            self._dma_ring_cursor = 0
            self._slot_recorded = [False] * DMA_STAGE_RING_SLOTS
            if int(self.prefill_page_stage.numel()) < self.max_page_bytes:
                self.prefill_page_stage = torch.empty((self.max_page_bytes,), dtype=torch.uint8, device=self.device)
        if changed:
            self.version += 1

    def ensure_prefill_capacity(self, chunk_tokens: int, top_k: int) -> None:
        need_out = int(chunk_tokens) * int(top_k) * int(self.max_d_model)
        if need_out > int(self.prefill_out_f32.numel()):
            self.prefill_out_f32 = torch.empty((need_out,), dtype=torch.float32, device=self.device)
        need_stage = PREFILL_STAGE_SLOTS * int(self.max_page_bytes)
        if need_stage > int(self.prefill_page_stage.numel()):
            self.prefill_page_stage = torch.empty((need_stage,), dtype=torch.uint8, device=self.device)

    def acquire_pinned_stage_slot(self, page_bytes: int) -> Tuple[int, torch.Tensor]:
        """Returns `(slot_idx, pinned_slice)` from the 32-slot ring, waiting if `slot_idx` H2D is still in flight."""
        idx = self._dma_ring_cursor
        if self._slot_recorded[idx] and self._slot_events[idx] is not None:
            self._slot_events[idx].synchronize()
            self._slot_recorded[idx] = False
        buf = self.pinned_dma_stage[idx, :page_bytes]
        self._dma_ring_cursor = (idx + 1) % DMA_STAGE_RING_SLOTS
        return idx, buf

    def record_stage_slot(self, slot_idx: int, stream: Optional[torch.cuda.Stream]) -> None:
        ev = self._slot_events[slot_idx]
        if ev is not None and stream is not None and not torch.cuda.is_current_stream_capturing():
            ev.record(stream)
            self._slot_recorded[slot_idx] = True

    def total_vram_bytes(self) -> int:
        total = 0
        for buf in (
            self._route_int_buf,
            self.active_weights,
            self.inter_f32,
            self.out_f32,
            self.prefill_page_stage,
            self.prefill_out_f32,
        ):
            total += int(buf.numel() * buf.element_size())
        return total


_MOE_WS_BY_DEVICE: Dict[int, SharedMoEWorkspace] = {}


def get_shared_moe_workspace(device: torch.device) -> SharedMoEWorkspace:
    dev_idx = 0 if device.index is None else int(device.index)
    if dev_idx not in _MOE_WS_BY_DEVICE:
        _MOE_WS_BY_DEVICE[dev_idx] = SharedMoEWorkspace(device)
    return _MOE_WS_BY_DEVICE[dev_idx]


class MoETLBLayerLauncher:
    """Zero-allocation `ctypes` launcher binding a layer's `SparseMoETLBExperts` to `SharedMoEWorkspace`."""

    def __init__(
        self,
        device: torch.device,
        num_experts: int,
        layout: ContiguousExpertPageLayout,
        act_kind: int,
    ) -> None:
        self.device = device
        self.num_experts = int(num_experts)
        self.layout = layout
        self.act_kind = int(act_kind)

        self._ws = get_shared_moe_workspace(device)
        self._ws.ensure_capacity(layout.d_model, layout.d_ffn, layout.page_bytes)
        self._bound_ws_version: int = -1
        self._bound_tlb_ptr: int = -1

        self._x_ptr = ctypes.c_void_p(0)
        self._topk_idx_ptr = ctypes.c_void_p(0)
        self._topk_w_ptr = ctypes.c_void_p(0)
        self._exp_to_slot_ptr = ctypes.c_void_p(0)
        self._tlb_ptr = ctypes.c_void_p(0)
        self._l2_ptr = ctypes.c_void_p(0)
        self._l2_wina_ptr = ctypes.c_void_p(0)
        self._partial_ptr = ctypes.c_void_p(0)
        self._out_bf16_ptr = ctypes.c_void_p(0)
        self._T = ctypes.c_int(1)
        self._K = ctypes.c_int(8)
        self._total_elems = ctypes.c_int(layout.d_model)
        self._zero_elems = ctypes.c_int(layout.d_model)
        self._wina_k_keep = ctypes.c_int(max(1, layout.d_ffn // 2))
        self.wina_layout: Optional[WINAExpertPageLayout] = (
            WINAExpertPageLayout.compute(layout.d_model, layout.d_ffn) if (layout.d_model % 64 == 0) else None
        )

        self._grid_gu_x = (layout.d_ffn + 3) // 4
        self._grid_dn_x = (layout.d_model + 3) // 4

        self._refs: List[Any] = []
        self._args_zero: Any = None
        self._args_pack: Any = None
        self._args_pack_wina: Any = None
        self._args_gu: Any = None
        self._args_dn: Any = None
        self._args_ep: Any = None
        self._args_wina_gate: Any = None
        self._args_wina_sel: Any = None
        self._args_wina_ud: Any = None
        self._kd: Optional[Dict[str, Any]] = None

    @staticmethod
    def _build_c_args(spec: List[Any], refs_sink: List[Any]) -> Any:
        arr = (ctypes.c_void_p * len(spec))()
        for i, item in enumerate(spec):
            if isinstance(item, torch.Tensor):
                p = ctypes.c_void_p(item.data_ptr())
                refs_sink.append(p)
                arr[i] = ctypes.cast(ctypes.byref(p), ctypes.c_void_p)
            elif isinstance(item, (ctypes.c_void_p, ctypes.c_int)):
                refs_sink.append(item)
                arr[i] = ctypes.cast(ctypes.byref(item), ctypes.c_void_p)
            elif isinstance(item, int):
                ci = ctypes.c_int(item)
                refs_sink.append(ci)
                arr[i] = ctypes.cast(ctypes.byref(ci), ctypes.c_void_p)
            else:
                raise TypeError(f"Unsupported kernel arg type: {type(item)}")
        return arr

    def _rebind(self, tlb_pages: torch.Tensor, expert_to_slot: torch.Tensor) -> None:
        self._ws.ensure_capacity(self.layout.d_model, self.layout.d_ffn, self.layout.page_bytes)
        self._kd = get_or_compile_moe_tlb_kernels(self.device)
        self._tlb_ptr.value = tlb_pages.data_ptr()
        self._exp_to_slot_ptr.value = expert_to_slot.data_ptr()
        self._partial_ptr.value = self._ws.out_f32.data_ptr()

        lay = self.layout
        refs: List[Any] = []
        self._args_zero = self._build_c_args([self._partial_ptr, self._zero_elems], refs)
        self._args_pack = self._build_c_args(
            [
                self._topk_idx_ptr,
                self._topk_w_ptr,
                self._exp_to_slot_ptr,
                self._ws.active_slots,
                self._ws.active_counts,
                self._ws.active_tok_ids,
                self._ws.active_pair_ids,
                self._ws.active_weights,
                self._ws.num_unique,
                self._partial_ptr,
                self._T,
                self._K,
                self.num_experts,
                lay.d_model,
            ],
            refs,
        )
        self._args_gu = self._build_c_args(
            [
                self._x_ptr,
                self._tlb_ptr,
                self._l2_ptr,
                self._ws.active_slots,
                self._ws.active_counts,
                self._ws.active_tok_ids,
                self._ws.num_unique,
                self._ws.inter_f32,
                lay.d_model,
                lay.d_ffn,
                lay.d_pad_half,
                lay.n_groups_d,
                lay.group_size,
                lay.page_bytes,
                lay.off_gu_q4,
                lay.off_gu_wmin,
                lay.off_gu_scale,
                self.act_kind,
            ],
            refs,
        )
        self._args_dn = self._build_c_args(
            [
                self._ws.inter_f32,
                self._tlb_ptr,
                self._l2_ptr,
                self._ws.active_slots,
                self._ws.active_counts,
                self._ws.active_pair_ids,
                self._ws.active_weights,
                self._ws.num_unique,
                self._partial_ptr,
                lay.d_model,
                lay.d_ffn,
                lay.i_pad_half,
                lay.n_groups_i,
                lay.group_size,
                lay.page_bytes,
                lay.off_dn_q4,
                lay.off_dn_wmin,
                lay.off_dn_scale,
            ],
            refs,
        )
        self._args_ep = self._build_c_args(
            [
                self._partial_ptr,
                self._out_bf16_ptr,
                self._total_elems,
                self._K,
                lay.d_model,
            ],
            refs,
        )
        if self.wina_layout is not None:
            wl = self.wina_layout
            self._args_pack_wina = self._build_c_args(
                [
                    self._topk_idx_ptr,
                    self._topk_w_ptr,
                    self._exp_to_slot_ptr,
                    self._ws.active_slots,
                    self._ws.active_counts,
                    self._ws.active_tok_ids,
                    self._ws.active_pair_ids,
                    self._ws.active_weights,
                    self._ws.num_unique,
                    self._ws.wina_l2_slots,
                    self._ws.wina_l2_tok_ids,
                    self._ws.wina_l2_pair_ids,
                    self._ws.wina_l2_weights,
                    self._ws.wina_num_l2,
                    self._partial_ptr,
                    self._T,
                    self._K,
                    self.num_experts,
                    lay.d_model,
                ],
                refs,
            )
            self._args_wina_gate = self._build_c_args(
                [
                    self._x_ptr,
                    self._l2_wina_ptr,
                    self._ws.wina_l2_slots,
                    self._ws.wina_l2_tok_ids,
                    self._ws.wina_num_l2,
                    self._ws.wina_gate_act,
                    self._ws.wina_scores,
                    wl.d_model,
                    wl.d_ffn,
                    wl.n_groups_d,
                    wl.row_bytes,
                    wl.page_bytes,
                    wl.off_gate,
                    wl.off_wina_norm,
                    self.act_kind,
                ],
                refs,
            )
            self._args_wina_sel = self._build_c_args(
                [
                    self._ws.wina_scores,
                    self._ws.wina_num_l2,
                    self._ws.wina_active_neurons,
                    wl.d_ffn,
                    self._wina_k_keep,
                ],
                refs,
            )
            self._args_wina_ud = self._build_c_args(
                [
                    self._x_ptr,
                    self._l2_wina_ptr,
                    self._ws.wina_l2_slots,
                    self._ws.wina_l2_tok_ids,
                    self._ws.wina_l2_pair_ids,
                    self._ws.wina_l2_weights,
                    self._ws.wina_num_l2,
                    self._ws.wina_gate_act,
                    self._ws.wina_active_neurons,
                    self._partial_ptr,
                    wl.d_model,
                    wl.d_ffn,
                    self._wina_k_keep,
                    wl.n_groups_d,
                    wl.row_bytes,
                    wl.slab_bytes,
                    wl.page_bytes,
                    wl.off_slabs,
                ],
                refs,
            )
        self._refs = refs
        self._bound_ws_version = self._ws.version
        self._bound_tlb_ptr = tlb_pages.data_ptr()

    def launch(
        self,
        x_bf16: torch.Tensor,
        top_k_idx_i64: torch.Tensor,
        top_k_w_f32: torch.Tensor,
        tlb_pages: torch.Tensor,
        expert_to_slot: torch.Tensor,
        out_bf16: torch.Tensor,
        l2_zero_copy_dev_ptr: int = 0,
        l2_wina_dev_ptr: int = 0,
        wina_keep_ratio: float = 1.0,
    ) -> None:
        T = int(x_bf16.shape[0])
        K = int(top_k_idx_i64.shape[1])
        if (
            self._kd is None
            or self._bound_ws_version != self._ws.version
            or self._bound_tlb_ptr != tlb_pages.data_ptr()
        ):
            self._rebind(tlb_pages, expert_to_slot)

        assert self._kd is not None
        libcuda = self._kd["libcuda"]
        stream_h = torch.cuda.current_stream(self.device)._as_parameter_

        self._x_ptr.value = x_bf16.data_ptr()
        self._topk_idx_ptr.value = top_k_idx_i64.data_ptr()
        self._topk_w_ptr.value = top_k_w_f32.data_ptr()
        self._exp_to_slot_ptr.value = expert_to_slot.data_ptr()
        self._tlb_ptr.value = tlb_pages.data_ptr()
        self._l2_ptr.value = int(l2_zero_copy_dev_ptr)
        self._l2_wina_ptr.value = int(l2_wina_dev_ptr)
        self._partial_ptr.value = self._ws.out_f32.data_ptr()
        self._out_bf16_ptr.value = out_bf16.data_ptr()
        self._T.value = T
        self._K.value = K
        tot_el = T * self.layout.d_model
        self._total_elems.value = tot_el

        max_u = min(self.num_experts, T * K, MAX_ACTIVE_UNIQUE)
        smem_bytes = min(T, MAX_FUSED_DECODE_TOKENS) * 1024 * 4
        use_wina = (
            l2_wina_dev_ptr != 0
            and self.wina_layout is not None
            and self._args_pack_wina is not None
            and (0.0 < float(wina_keep_ratio) < 1.0 or l2_zero_copy_dev_ptr == 0)
        )

        if use_wina:
            k_keep = max(1, min(self.layout.d_ffn, int(round(self.layout.d_ffn * float(wina_keep_ratio)))))
            self._wina_k_keep.value = k_keep
            max_l2_pairs = min(T * K, MAX_ACTIVE_UNIQUE)

            r0 = libcuda.cuLaunchKernel(
                self._kd["moe_pack_routes_wina_kernel"].func,
                1, 1, 1, 256, 1, 1, 0, stream_h, self._args_pack_wina, None,
            )
            # L1 VRAM hits (`slot >= 0`) run exact UINT4 kernels
            r1 = libcuda.cuLaunchKernel(
                self._kd["moe_uint4_gate_up_act_kernel"].func,
                self._grid_gu_x, max_u, 1, 128, 1, 1, smem_bytes, stream_h, self._args_gu, None,
            )
            r2 = libcuda.cuLaunchKernel(
                self._kd["moe_uint4_down_accum_kernel"].func,
                self._grid_dn_x, max_u, 1, 128, 1, 1, smem_bytes, stream_h, self._args_dn, None,
            )
            # L2 Zero-Copy hits (`slot <= -2`) run Gate-First `UINT4-WINA` conditional fetch
            rw0 = libcuda.cuLaunchKernel(
                self._kd["moe_wina_gate_q4_kernel"].func,
                self._grid_gu_x, max_l2_pairs, 1, 128, 1, 1, 0, stream_h, self._args_wina_gate, None,
            )
            rw1 = 0
            if k_keep < self.layout.d_ffn:
                rw1 = libcuda.cuLaunchKernel(
                    self._kd["moe_wina_select_top_kernel"].func,
                    max_l2_pairs, 1, 1, 256, 1, 1, 0, stream_h, self._args_wina_sel, None,
                )
            rw2 = libcuda.cuLaunchKernel(
                self._kd["moe_wina_up_down_q4_kernel"].func,
                (k_keep + 3) // 4, max_l2_pairs, 1, 128, 1, 1, 0, stream_h, self._args_wina_ud, None,
            )
            grid_ep = (tot_el + 255) // 256
            r3 = libcuda.cuLaunchKernel(
                self._kd["moe_f32_to_bf16_kernel"].func,
                grid_ep, 1, 1, 256, 1, 1, 0, stream_h, self._args_ep, None,
            )
            if (r0 | r1 | r2 | rw0 | rw1 | rw2 | r3) != 0:
                raise RuntimeError(f"MoE WINA kernel launch failed: r=({r0},{r1},{r2},{rw0},{rw1},{rw2},{r3})")
            return

        r0 = libcuda.cuLaunchKernel(
            self._kd["moe_pack_routes_kernel"].func,
            1, 1, 1, 256, 1, 1, 0, stream_h, self._args_pack, None,
        )
        r1 = libcuda.cuLaunchKernel(
            self._kd["moe_uint4_gate_up_act_kernel"].func,
            self._grid_gu_x, max_u, 1, 128, 1, 1, smem_bytes, stream_h, self._args_gu, None,
        )
        r2 = libcuda.cuLaunchKernel(
            self._kd["moe_uint4_down_accum_kernel"].func,
            self._grid_dn_x, max_u, 1, 128, 1, 1, smem_bytes, stream_h, self._args_dn, None,
        )
        grid_ep = (tot_el + 255) // 256
        r3 = libcuda.cuLaunchKernel(
            self._kd["moe_f32_to_bf16_kernel"].func,
            grid_ep, 1, 1, 256, 1, 1, 0, stream_h, self._args_ep, None,
        )
        if (r0 | r1 | r2 | r3) != 0:
            raise RuntimeError(f"MoE TLB kernel launch failed: r=({r0}, {r1}, {r2}, {r3})")

    def launch_wave_items(
        self,
        x_bf16: torch.Tensor,
        tlb_pages_ptr: int,
        l2_zero_copy_dev_ptr: int,
        partial_f32_ptr: int,
        num_items: int,
        max_m_in_wave: int,
    ) -> None:
        """Launches `moe_uint4_gate_up_act_kernel` + `moe_uint4_down_accum_kernel` for a pre-packed wave (`<= 128`)."""
        if num_items <= 0:
            return
        assert self._kd is not None
        libcuda = self._kd["libcuda"]
        stream_h = torch.cuda.current_stream(self.device)._as_parameter_

        self._x_ptr.value = x_bf16.data_ptr()
        self._tlb_ptr.value = int(tlb_pages_ptr)
        self._l2_ptr.value = int(l2_zero_copy_dev_ptr)
        self._partial_ptr.value = int(partial_f32_ptr)
        smem_bytes = max(1, min(int(max_m_in_wave), MAX_FUSED_DECODE_TOKENS)) * 1024 * 4

        r1 = libcuda.cuLaunchKernel(
            self._kd["moe_uint4_gate_up_act_kernel"].func,
            self._grid_gu_x, int(num_items), 1, 128, 1, 1, smem_bytes, stream_h, self._args_gu, None,
        )
        r2 = libcuda.cuLaunchKernel(
            self._kd["moe_uint4_down_accum_kernel"].func,
            self._grid_dn_x, int(num_items), 1, 128, 1, 1, smem_bytes, stream_h, self._args_dn, None,
        )
        if (r1 | r2) != 0:
            raise RuntimeError(f"MoE wave kernel launch failed: r=({r1}, {r2})")

    def launch_zero_and_epilogue(
        self,
        partial_f32_ptr: int,
        out_bf16: Optional[torch.Tensor],
        T_chunk: int,
        K: int,
        mode: str,
    ) -> None:
        """Executes `zero` (`partial_f32`) or `epilogue` (`partial_f32 -> out_bf16`) for grouped prefill."""
        assert self._kd is not None
        libcuda = self._kd["libcuda"]
        stream_h = torch.cuda.current_stream(self.device)._as_parameter_
        self._partial_ptr.value = int(partial_f32_ptr)
        if mode == "zero":
            n_zero = int(T_chunk) * int(K) * self.layout.d_model
            self._zero_elems.value = n_zero
            grid_z = (n_zero + 255) // 256
            rz = libcuda.cuLaunchKernel(
                self._kd["moe_zero_f32_kernel"].func,
                grid_z, 1, 1, 256, 1, 1, 0, stream_h, self._args_zero, None,
            )
            if rz != 0:
                raise RuntimeError(f"moe_zero_f32_kernel failed: {rz}")
        else:
            assert out_bf16 is not None
            tot_el = int(T_chunk) * self.layout.d_model
            self._out_bf16_ptr.value = out_bf16.data_ptr()
            self._total_elems.value = tot_el
            self._K.value = int(K)
            grid_ep = (tot_el + 255) // 256
            re = libcuda.cuLaunchKernel(
                self._kd["moe_f32_to_bf16_kernel"].func,
                grid_ep, 1, 1, 256, 1, 1, 0, stream_h, self._args_ep, None,
            )
            if re != 0:
                raise RuntimeError(f"moe_f32_to_bf16_kernel failed: {re}")
