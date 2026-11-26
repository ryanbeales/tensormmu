"""Single-Layer NVRTC CUDA Benchmark: `UINT4-WINA` & `WHT-INT3-WINA` vs. Current `UINT4` L2 Zero-Copy.

Tests whether Gate-First WINA conditional neuron-major PCIe fetch combined with:
  1. `UINT4-WINA` (4-bit group-64, zero WHT/trellis overhead)
  2. `WHT-INT3-WINA` (Block-64 Walsh-Hadamard + bit-plane 3-bit group-64, 0 bank conflicts)
hits BOTH:
  - Fidelity Gate: Cosine Similarity >= 0.975 vs. exact BF16 (coding/reasoning safe!)
  - Speed Gate:    >= 1.6x–2.2x faster L2 Zero-Copy PCIe decode inside CUDA Graphs!
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F

from runtime.moe_expert_tlb import _quantize_uint4_2d_with_salience, quantize_pack_moe_experts_uint4
from runtime.moe_tlb_kernels import (
    ContiguousExpertPageLayout,
    MoETLBLayerLauncher,
    get_shared_moe_workspace,
    get_zero_copy_device_ptr,
)

# =============================================================================
# 1. NVRTC CUDA Kernels: `UINT4-WINA` & Bit-Plane `WHT-INT3-WINA`
# =============================================================================
WINA_Q3_Q4_CUDA_SRC = r"""
typedef unsigned short bf16_t;
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;

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

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffU, v, offset);
    }
    return v;
}

__device__ __forceinline__ float apply_act(float x, int act_kind) {
    if (act_kind == 2) {
        return x / (1.0f + __expf(-x));
    }
    float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Block-64 Walsh-Hadamard Transform (`H_64 / 8`)
extern "C" __global__ void wht64_bf16_to_f32_kernel(
    const bf16_t* __restrict__ x_in,
    float* __restrict__ x_out,
    int d_model,
    int do_wht
) {
    __shared__ float s[64];
    const int base = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int idx = base + tid;
    float v = (idx < d_model) ? bf16_to_f32(x_in[idx]) : 0.0f;
    if (!do_wht) {
        if (idx < d_model) x_out[idx] = v;
        return;
    }
    s[tid] = v;
    __syncthreads();
    #pragma unroll
    for (int stride = 1; stride < 64; stride <<= 1) {
        int partner = tid ^ stride;
        float a = s[tid];
        float b = s[partner];
        __syncthreads();
        s[tid] = (tid < partner) ? (a + b) : (b - a);
        __syncthreads();
    }
    if (idx < d_model) x_out[idx] = s[tid] * 0.125f;
}

extern "C" __global__ void iwht64_f32_to_bf16_kernel(
    const float* __restrict__ y_in,
    bf16_t* __restrict__ y_out,
    int d_model,
    int do_wht
) {
    __shared__ float s[64];
    const int base = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int idx = base + tid;
    float v = (idx < d_model) ? y_in[idx] : 0.0f;
    if (!do_wht) {
        if (idx < d_model) y_out[idx] = f32_to_bf16(v);
        return;
    }
    s[tid] = v;
    __syncthreads();
    #pragma unroll
    for (int stride = 1; stride < 64; stride <<= 1) {
        int partner = tid ^ stride;
        float a = s[tid];
        float b = s[partner];
        __syncthreads();
        s[tid] = (tid < partner) ? (a + b) : (b - a);
        __syncthreads();
    }
    if (idx < d_model) y_out[idx] = f32_to_bf16(s[tid] * 0.125f);
}

// =============================================================================
// Shared WINA Top-K Selector (1 block per active expert, `< 3 us`, CUDA Graph safe)
// =============================================================================
extern "C" __global__ void wina_select_top_neurons_kernel(
    const float* __restrict__ wina_scores,
    int* __restrict__ active_neurons,
    int d_ffn,
    int k_keep
) {
    const int a = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float s_sc[1024];
    __shared__ float s_lo, s_hi, s_thr;
    __shared__ int s_cnt;

    const float* sc_row = wina_scores + a * d_ffn;
    float local_max = 0.0f;
    for (int j = tid; j < d_ffn; j += blockDim.x) {
        float v = sc_row[j];
        s_sc[j] = v;
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
            if (s_sc[j] >= thr) c++;
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
    int* out_row = active_neurons + a * k_keep;

    for (int j = tid; j < d_ffn; j += blockDim.x) {
        if (s_sc[j] >= final_thr) {
            int pos = atomicAdd(&s_cnt, 1);
            if (pos < k_keep) out_row[pos] = j;
        }
    }
    __syncthreads();
    for (int j = tid; j < d_ffn; j += blockDim.x) {
        if (s_sc[j] < final_thr && s_sc[j] >= s_lo) {
            int pos = atomicAdd(&s_cnt, 1);
            if (pos < k_keep) out_row[pos] = j;
        }
    }
    __syncthreads();
    for (int j = tid; j < k_keep; j += blockDim.x) {
        if (j >= s_cnt) out_row[j] = j;
    }
}

// =============================================================================
// Helper: Decode 1 group of 64 UINT4 weights (32 bytes = 2x uint4) + (wmin, scale)
// =============================================================================
__device__ __forceinline__ float dot_q4_grp64(
    const uint4* __restrict__ q4_u4,
    uint32_t mn_sc_packed,
    const float* __restrict__ xp
) {
    const float mn = bf16_to_f32((bf16_t)(mn_sc_packed & 0xFFFFU));
    const float sc = bf16_to_f32((bf16_t)(mn_sc_packed >> 16));
    uint4 v0 = __ldg(&q4_u4[0]);
    uint4 v1 = __ldg(&q4_u4[1]);
    uint32_t words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    float qdot = 0.0f;
    float xsum = 0.0f;
    #pragma unroll
    for (int wi = 0; wi < 8; ++wi) {
        uint32_t w = words[wi];
        const float* xw = xp + (wi << 3);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float xv = xw[k];
            xsum += xv;
            qdot += (float)((w >> (k << 2)) & 0x0FU) * xv;
        }
    }
    return qdot * sc + xsum * mn;
}

__device__ __forceinline__ void axpy_q4_grp64(
    const uint4* __restrict__ q4_u4,
    uint32_t mn_sc_packed,
    float coeff,
    float* __restrict__ yp
) {
    const float mn_c = bf16_to_f32((bf16_t)(mn_sc_packed & 0xFFFFU)) * coeff;
    const float sc_c = bf16_to_f32((bf16_t)(mn_sc_packed >> 16)) * coeff;
    uint4 v0 = __ldg(&q4_u4[0]);
    uint4 v1 = __ldg(&q4_u4[1]);
    uint32_t words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    #pragma unroll
    for (int wi = 0; wi < 8; ++wi) {
        uint32_t w = words[wi];
        float* yw = yp + (wi << 3);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = (float)((w >> (k << 2)) & 0x0FU) * sc_c + mn_c;
            atomicAdd(&yw[k], val);
        }
    }
}

// =============================================================================
// Helper: Decode 1 group of 64 Bit-Plane INT3 weights (24 bytes = 6x uint32)
// Layout per 64-weight group (28 bytes packed: [b0_lo, b1_lo, b2_lo, b0_hi, b1_hi, b2_hi, mn_sc]):
//   NO cross-word bit splits, 100% coalesced, 3.50 bits/weight total including scales!
// =============================================================================
__device__ __forceinline__ float dot_q3_grp64(
    const uint32_t* __restrict__ g_words,  // 7 uint32s = 28 bytes
    const float* __restrict__ xp
) {
    uint32_t b0_lo = __ldg(&g_words[0]);
    uint32_t b1_lo = __ldg(&g_words[1]);
    uint32_t b2_lo = __ldg(&g_words[2]);
    uint32_t b0_hi = __ldg(&g_words[3]);
    uint32_t b1_hi = __ldg(&g_words[4]);
    uint32_t b2_hi = __ldg(&g_words[5]);
    uint32_t mn_sc = __ldg(&g_words[6]);
    const float mn = bf16_to_f32((bf16_t)(mn_sc & 0xFFFFU));
    const float sc = bf16_to_f32((bf16_t)(mn_sc >> 16));

    float qdot = 0.0f;
    float xsum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        float x0 = xp[k];
        float x1 = xp[32 + k];
        xsum += x0 + x1;
        int q0 = ((b0_lo >> k) & 1U) | (((b1_lo >> k) & 1U) << 1) | (((b2_lo >> k) & 1U) << 2);
        int q1 = ((b0_hi >> k) & 1U) | (((b1_hi >> k) & 1U) << 1) | (((b2_hi >> k) & 1U) << 2);
        qdot += (float)q0 * x0 + (float)q1 * x1;
    }
    return qdot * sc + xsum * mn;
}

__device__ __forceinline__ void axpy_q3_grp64(
    const uint32_t* __restrict__ g_words,
    float coeff,
    float* __restrict__ yp
) {
    uint32_t b0_lo = __ldg(&g_words[0]);
    uint32_t b1_lo = __ldg(&g_words[1]);
    uint32_t b2_lo = __ldg(&g_words[2]);
    uint32_t b0_hi = __ldg(&g_words[3]);
    uint32_t b1_hi = __ldg(&g_words[4]);
    uint32_t b2_hi = __ldg(&g_words[5]);
    uint32_t mn_sc = __ldg(&g_words[6]);
    const float mn_c = bf16_to_f32((bf16_t)(mn_sc & 0xFFFFU)) * coeff;
    const float sc_c = bf16_to_f32((bf16_t)(mn_sc >> 16)) * coeff;

    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        int q0 = ((b0_lo >> k) & 1U) | (((b1_lo >> k) & 1U) << 1) | (((b2_lo >> k) & 1U) << 2);
        int q1 = ((b0_hi >> k) & 1U) | (((b1_hi >> k) & 1U) << 1) | (((b2_hi >> k) & 1U) << 2);
        atomicAdd(&yp[k], (float)q0 * sc_c + mn_c);
        atomicAdd(&yp[32 + k], (float)q1 * sc_c + mn_c);
    }
}

// =============================================================================
// Unified Gate-First Kernel (`qbits == 4` -> UINT4, `qbits == 3` -> Bit-Plane INT3)
// =============================================================================
extern "C" __global__ void wina_gate_q3q4_kernel(
    const float* __restrict__ x_in,
    const uint8_t* __restrict__ l2_pages,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_act_out,
    float* __restrict__ wina_score_out,
    float* __restrict__ y_accum_out,
    int d_model,
    int d_ffn,
    int n_groups,
    int row_bytes,
    int page_bytes,
    int off_gate,
    int off_wina_norm,
    int act_kind,
    int qbits
) {
    const int a = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 4 + warp;

    if (blockIdx.x == 0 && blockIdx.y == 0) {
        for (int i = tid; i < d_model; i += blockDim.x) y_accum_out[i] = 0.0f;
    }
    if (row >= d_ffn) return;

    const uint8_t* page = l2_pages + (size_t)expert_ids[a] * (size_t)page_bytes;
    const uint8_t* row_ptr = page + off_gate + (size_t)row * (size_t)row_bytes;
    const float dn_norm = bf16_to_f32(((const bf16_t*)(page + off_wina_norm))[row]);

    float acc = 0.0f;
    if (qbits == 4) {
        // Row layout: [n_groups * 32 B of q4] + [n_groups * 4 B of packed (wmin, scale)]
        const uint4* q4_base = (const uint4*)row_ptr;
        const uint32_t* ms_base = (const uint32_t*)(row_ptr + n_groups * 32);
        for (int g = lane; g < n_groups; g += 32) {
            acc += dot_q4_grp64(q4_base + g * 2, __ldg(&ms_base[g]), x_in + (g << 6));
        }
    } else {
        // Row layout: [n_groups * 28 B] (each group = 7 uint32s)
        const uint32_t* g_base = (const uint32_t*)row_ptr;
        for (int g = lane; g < n_groups; g += 32) {
            acc += dot_q3_grp64(g_base + g * 7, x_in + (g << 6));
        }
    }

    float g_val = warp_sum(acc);
    if (lane == 0) {
        float ga = apply_act(g_val, act_kind);
        gate_act_out[a * d_ffn + row] = ga;
        wina_score_out[a * d_ffn + row] = fabsf(ga) * dn_norm;
    }
}

// =============================================================================
// Unified Conditional Neuron-Major Up + Down Kernel (`qbits == 4` or `3`)
// =============================================================================
extern "C" __global__ void wina_up_down_q3q4_kernel(
    const float* __restrict__ x_in,
    const uint8_t* __restrict__ l2_pages,
    const int* __restrict__ expert_ids,
    const float* __restrict__ router_w,
    const float* __restrict__ gate_act,
    const int* __restrict__ active_neurons,
    float* __restrict__ y_accum_out,
    int d_model,
    int d_ffn,
    int k_keep,
    int n_groups,
    int row_bytes,
    int slab_bytes,
    int page_bytes,
    int off_slabs,
    int qbits
) {
    const int a = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int slot_k = blockIdx.x * 4 + warp;
    if (slot_k >= k_keep) return;

    const int j = (active_neurons != nullptr) ? active_neurons[a * k_keep + slot_k] : slot_k;
    const uint8_t* page = l2_pages + (size_t)expert_ids[a] * (size_t)page_bytes;
    const uint8_t* slab = page + off_slabs + (size_t)j * (size_t)slab_bytes;
    const uint8_t* up_ptr = slab;
    const uint8_t* dn_ptr = slab + row_bytes;

    float u_acc = 0.0f;
    if (qbits == 4) {
        const uint4* q4_base = (const uint4*)up_ptr;
        const uint32_t* ms_base = (const uint32_t*)(up_ptr + n_groups * 32);
        for (int g = lane; g < n_groups; g += 32) {
            u_acc += dot_q4_grp64(q4_base + g * 2, __ldg(&ms_base[g]), x_in + (g << 6));
        }
    } else {
        const uint32_t* g_base = (const uint32_t*)up_ptr;
        for (int g = lane; g < n_groups; g += 32) {
            u_acc += dot_q3_grp64(g_base + g * 7, x_in + (g << 6));
        }
    }

    float u_val = warp_sum(u_acc);
    u_val = __shfl_sync(0xffffffffU, u_val, 0);
    const float coeff = gate_act[a * d_ffn + j] * u_val * router_w[a];

    if (qbits == 4) {
        const uint4* q4_base = (const uint4*)dn_ptr;
        const uint32_t* ms_base = (const uint32_t*)(dn_ptr + n_groups * 32);
        for (int g = lane; g < n_groups; g += 32) {
            axpy_q4_grp64(q4_base + g * 2, __ldg(&ms_base[g]), coeff, y_accum_out + (g << 6));
        }
    } else {
        const uint32_t* g_base = (const uint32_t*)dn_ptr;
        for (int g = lane; g < n_groups; g += 32) {
            axpy_q3_grp64(g_base + g * 7, coeff, y_accum_out + (g << 6));
        }
    }
}
"""


def compile_kernels(device: torch.device) -> Dict[str, any]:
    from torch.cuda._utils import _cuda_load_module, _get_gpu_rtc_library, _get_gpu_runtime_library

    dev_idx = 0 if device.index is None else int(device.index)
    with torch.cuda.device(dev_idx):
        libnvrtc = _get_gpu_rtc_library()
        libcuda = _get_gpu_runtime_library()
        props = torch.cuda.get_device_properties(dev_idx)
        arch = f"--gpu-architecture=sm_{props.major}{props.minor}".encode("utf-8")
        opts = [arch, b"--use_fast_math", b"-default-device"]
        opts_arr = (ctypes.c_char_p * len(opts))(*opts)

        prog = ctypes.c_void_p()
        libnvrtc.nvrtcCreateProgram(ctypes.byref(prog), WINA_Q3_Q4_CUDA_SRC.encode("utf-8"), b"wina_q3q4.cu", 0, None, None)
        res = libnvrtc.nvrtcCompileProgram(prog, len(opts), opts_arr)
        if res != 0:
            log_sz = ctypes.c_size_t()
            libnvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
            log_buf = ctypes.create_string_buffer(log_sz.value)
            libnvrtc.nvrtcGetProgramLog(prog, log_buf)
            libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            raise RuntimeError(f"NVRTC failed:\n{log_buf.value.decode('utf-8', errors='replace')}")

        cubin_sz = ctypes.c_size_t()
        libnvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(cubin_sz))
        cubin_buf = ctypes.create_string_buffer(cubin_sz.value)
        libnvrtc.nvrtcGetCUBIN(prog, cubin_buf)
        libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

        names = [
            "wht64_bf16_to_f32_kernel",
            "iwht64_f32_to_bf16_kernel",
            "wina_select_top_neurons_kernel",
            "wina_gate_q3q4_kernel",
            "wina_up_down_q3q4_kernel",
        ]
        kd = _cuda_load_module(cubin_buf.raw, names)
        kd["libcuda"] = libcuda
        return kd


@dataclass(frozen=True)
class WINAQPageLayout:
    d_model: int
    d_ffn: int
    n_groups: int
    qbits: int
    row_bytes: int
    slab_bytes: int
    off_gate: int
    off_wina_norm: int
    off_slabs: int
    page_bytes: int

    @classmethod
    def compute(cls, d_model: int, d_ffn: int, qbits: int) -> "WINAQPageLayout":
        assert d_model % 64 == 0 and qbits in (3, 4)
        n_groups = d_model // 64
        raw_row = (n_groups * 36) if qbits == 4 else (n_groups * 28)
        row_b = ((raw_row + 15) // 16) * 16
        slab_b = 2 * row_b
        off_gate = 0
        off_wina = ((d_ffn * row_b + 15) // 16) * 16
        off_slabs = ((off_wina + d_ffn * 2 + 15) // 16) * 16
        page_b = ((off_slabs + d_ffn * slab_b + 4095) // 4096) * 4096
        return cls(d_model, d_ffn, n_groups, qbits, row_b, slab_b, off_gate, off_wina, off_slabs, page_b)


def _wht64_matrix(device: torch.device) -> torch.Tensor:
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    for _ in range(6):
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / 8.0


def apply_wht64(t: torch.Tensor, H64: torch.Tensor) -> torch.Tensor:
    return (t.float().reshape(-1, 64) @ H64).reshape(t.shape)


@torch.no_grad()
def pack_row_q4(w_f32: torch.Tensor, layout: WINAQPageLayout) -> torch.Tensor:
    """Packs `[rows, d_model]` into `[rows, layout.row_bytes]` (`q4` + packed `wmin|scale`)."""
    rows = w_f32.shape[0]
    packed_u8, wmin_bf, sc_bf = _quantize_uint4_2d_with_salience(w_f32, group_size=64)
    ms_u16 = torch.stack([wmin_bf.view(torch.uint16), sc_bf.view(torch.uint16)], dim=-1)  # [rows, n_groups, 2]
    out = torch.zeros((rows, layout.row_bytes), dtype=torch.uint8, device=w_f32.device)
    n_q4 = layout.n_groups * 32
    out[:, :n_q4].copy_(packed_u8.reshape(rows, n_q4))
    out[:, n_q4 : n_q4 + layout.n_groups * 4].copy_(ms_u16.view(torch.uint8).reshape(rows, -1))
    return out


@torch.no_grad()
def pack_row_q3_bitplane(w_f32: torch.Tensor, layout: WINAQPageLayout) -> torch.Tensor:
    """MSE-clipped group-64 3-bit quantization packed as `[rows, n_groups, 7 uint32]` bit-planes."""
    dev = w_f32.device
    rows, d_model = w_f32.shape
    ng = layout.n_groups
    wg = w_f32.view(rows, ng, 64)

    mn0 = wg.amin(dim=-1, keepdim=True)
    mx0 = wg.amax(dim=-1, keepdim=True)
    mid = 0.5 * (mn0 + mx0)
    half0 = 0.5 * (mx0 - mn0)

    best_mse = torch.full((rows, ng, 1), float("inf"), device=dev, dtype=torch.float32)
    wmin_bf = mn0.to(torch.bfloat16)
    sc_bf = ((mx0 - mn0) / 7.0).clamp_min(1e-8).to(torch.bfloat16)

    for alpha in (0.80, 0.86, 0.91, 0.95, 0.98, 1.00):
        cmn = (mid - half0 * alpha).to(torch.bfloat16)
        csc = ((2.0 * half0 * alpha) / 7.0).clamp_min(1e-8).to(torch.bfloat16)
        cq = torch.clamp(torch.round((wg - cmn.float()) / csc.float()), 0.0, 7.0)
        cdq = (cq * csc.float() + cmn.float())
        mse = (wg - cdq).square().mean(dim=-1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        wmin_bf = torch.where(better, cmn, wmin_bf)
        sc_bf = torch.where(better, csc, sc_bf)

    q = torch.clamp(torch.round((wg - wmin_bf.float()) / sc_bf.float()), 0.0, 7.0).to(torch.int64)  # [rows, ng, 64]
    shifts = torch.arange(32, device=dev, dtype=torch.int64).view(1, 1, 32)
    q_lo = q[:, :, :32]
    q_hi = q[:, :, 32:]

    def _plane(bits: torch.Tensor) -> torch.Tensor:
        return (bits << shifts).sum(dim=-1).to(torch.uint32)

    b0_lo = _plane(q_lo & 1)
    b1_lo = _plane((q_lo >> 1) & 1)
    b2_lo = _plane((q_lo >> 2) & 1)
    b0_hi = _plane(q_hi & 1)
    b1_hi = _plane((q_hi >> 1) & 1)
    b2_hi = _plane((q_hi >> 2) & 1)

    mn_u32 = wmin_bf.squeeze(-1).view(torch.uint16).to(torch.int64)
    sc_u32 = sc_bf.squeeze(-1).view(torch.uint16).to(torch.int64)
    ms_u32 = (mn_u32 | (sc_u32 << 16)).to(torch.uint32)

    words = torch.stack([b0_lo, b1_lo, b2_lo, b0_hi, b1_hi, b2_hi, ms_u32], dim=-1)  # [rows, ng, 7] uint32
    out = torch.zeros((rows, layout.row_bytes), dtype=torch.uint8, device=dev)
    out[:, : ng * 28].copy_(words.view(torch.uint8).reshape(rows, ng * 28))
    return out


def _build_c_args(items: List[any], refs: List[any]) -> any:
    arr = (ctypes.c_void_p * len(items))()
    for i, item in enumerate(items):
        if isinstance(item, torch.Tensor):
            p = ctypes.c_void_p(item.data_ptr())
            refs.append(p)
            arr[i] = ctypes.cast(ctypes.byref(p), ctypes.c_void_p)
        elif isinstance(item, (ctypes.c_void_p, ctypes.c_int)):
            refs.append(item)
            arr[i] = ctypes.cast(ctypes.byref(item), ctypes.c_void_p)
        elif isinstance(item, int):
            ci = ctypes.c_int(item)
            refs.append(ci)
            arr[i] = ctypes.cast(ctypes.byref(ci), ctypes.c_void_p)
        else:
            raise TypeError(type(item))
    return arr


def bench_geometry(
    kd: Dict[str, any],
    name: str,
    d_model: int,
    d_ffn: int,
    top_k: int,
    num_layers: int,
    act_kind: str,
    miss_counts: List[int],
) -> None:
    torch.manual_seed(42)
    dev = torch.device("cuda:0")
    H64 = _wht64_matrix(dev)
    act_int_gpu = 2 if act_kind == "silu" else 0
    libcuda = kd["libcuda"]

    lay_base = ContiguousExpertPageLayout.compute(d_model, d_ffn, 64)
    lay_q4w = WINAQPageLayout.compute(d_model, d_ffn, qbits=4)
    lay_q3w = WINAQPageLayout.compute(d_model, d_ffn, qbits=3)

    print(f"\n{'=' * 100}")
    print(f"  {name} | d_model={d_model}, d_ffn={d_ffn}, top_k={top_k}, layers={num_layers} ({act_kind})")
    print(
        f"  Page Sizes: Baseline UINT4={lay_base.page_bytes/(1024**2):.2f} MiB | "
        f"UINT4-WINA={lay_q4w.page_bytes/(1024**2):.2f} MiB | "
        f"WHT-INT3-WINA={lay_q3w.page_bytes/(1024**2):.2f} MiB "
        f"({lay_base.page_bytes/lay_q3w.page_bytes:.2f}x smaller!)"
    )
    print(f"{'=' * 100}")

    # Realistic heavy-tailed weights (2% 4x outliers) & heavy-tailed activations (8 outlier channels)
    E_acc = 4
    w_gate = torch.randn((E_acc, d_ffn, d_model), device=dev, dtype=torch.bfloat16) * 0.02
    w_up = torch.randn((E_acc, d_ffn, d_model), device=dev, dtype=torch.bfloat16) * 0.02
    w_dn = torch.randn((E_acc, d_model, d_ffn), device=dev, dtype=torch.bfloat16) * 0.02
    outlier_mask = torch.rand_like(w_gate.float()) < 0.02
    w_gate = (w_gate.float() * torch.where(outlier_mask, 4.0, 1.0)).to(torch.bfloat16)

    x_bf16 = torch.randn((1, d_model), device=dev, dtype=torch.bfloat16)
    x_bf16[0, :8] *= 5.0
    rw_acc = torch.full((E_acc,), 1.0 / E_acc, device=dev, dtype=torch.float32)

    # Exact BF16 reference
    y_bf16_ref = torch.zeros((1, d_model), device=dev, dtype=torch.float32)
    for e in range(E_acc):
        g = F.linear(x_bf16.float(), w_gate[e].float())
        u = F.linear(x_bf16.float(), w_up[e].float())
        h = (F.silu(g) if act_kind == "silu" else F.gelu(g, approximate="tanh")) * u
        y_bf16_ref += rw_acc[e] * F.linear(h, w_dn[e].float())

    # Pack Baseline UINT4
    q4_base_cpu, _ = quantize_pack_moe_experts_uint4(
        torch.cat([w_gate, w_up], dim=1).cpu(), w_dn.cpu(), group_size=64, compute_device="cuda:0"
    )

    def _pack_wina_pages(lay: WINAQPageLayout, use_wht: bool) -> torch.Tensor:
        pages_cpu = torch.zeros((E_acc, lay.page_bytes), dtype=torch.uint8, device="cpu")
        pack_fn = pack_row_q4 if lay.qbits == 4 else pack_row_q3_bitplane
        for e in range(E_acc):
            gw = apply_wht64(w_gate[e], H64) if use_wht else w_gate[e].float()
            uw = apply_wht64(w_up[e], H64) if use_wht else w_up[e].float()
            dw = apply_wht64(w_dn[e].T.contiguous(), H64) if use_wht else w_dn[e].T.float().contiguous()
            dn_norms = w_dn[e].float().norm(dim=0).to(torch.bfloat16)

            g_pk = pack_fn(gw, lay)
            u_pk = pack_fn(uw, lay)
            d_pk = pack_fn(dw, lay)

            pg = torch.zeros((lay.page_bytes,), dtype=torch.uint8, device=dev)
            pg[lay.off_gate : lay.off_gate + d_ffn * lay.row_bytes].copy_(g_pk.reshape(-1))
            pg[lay.off_wina_norm : lay.off_wina_norm + d_ffn * 2].copy_(dn_norms.view(torch.uint8).reshape(-1))
            slabs = torch.cat([u_pk, d_pk], dim=1)  # [d_ffn, 2 * row_bytes]
            pg[lay.off_slabs : lay.off_slabs + d_ffn * lay.slab_bytes].copy_(slabs.reshape(-1))
            pages_cpu[e].copy_(pg.cpu())
        return pages_cpu

    q4w_base_cpu = _pack_wina_pages(lay_q4w, use_wht=False)
    q3w_base_cpu = _pack_wina_pages(lay_q3w, use_wht=True)

    # Allocate >180 MiB pinned L2 pools for each format (exceeding CPU L3 cache)
    def _make_pool(base_cpu: torch.Tensor, pb: int) -> Tuple[torch.Tensor, int, int]:
        n_p = min(180, max(64, int((180 * 1024 * 1024) // pb)))
        pin = torch.empty((n_p, pb), dtype=torch.uint8, device="cpu").pin_memory()
        for i in range(n_p):
            pin[i].copy_(base_cpu[i % E_acc])
        return pin, get_zero_copy_device_ptr(pin, dev), n_p

    l2_base_pin, l2_base_zc, n_pool_base = _make_pool(q4_base_cpu, lay_base.page_bytes)
    l2_q4w_pin, l2_q4w_zc, n_pool_q4w = _make_pool(q4w_base_cpu, lay_q4w.page_bytes)
    l2_q3w_pin, l2_q3w_zc, n_pool_q3w = _make_pool(q3w_base_cpu, lay_q3w.page_bytes)

    # Scratch GPU buffers
    x_tmp = torch.empty((d_model,), device=dev, dtype=torch.float32)
    y_tmp = torch.empty((d_model,), device=dev, dtype=torch.float32)
    y_out = torch.empty((1, d_model), device=dev, dtype=torch.bfloat16)
    gate_act = torch.empty((top_k, d_ffn), device=dev, dtype=torch.float32)
    wina_sc = torch.empty((top_k, d_ffn), device=dev, dtype=torch.float32)
    act_neur = torch.empty((top_k, d_ffn), device=dev, dtype=torch.int32)

    def launch_wina(
        lay: WINAQPageLayout,
        l2_zc_ptr: int,
        exp_ids: torch.Tensor,
        rw: torch.Tensor,
        keep_ratio: float,
        do_wht: int,
    ) -> None:
        n_act = int(exp_ids.numel())
        k_keep = max(1, min(d_ffn, int(round(d_ffn * keep_ratio))))
        stream_h = torch.cuda.current_stream(dev)._as_parameter_
        refs: List[any] = []

        a0 = _build_c_args([x_bf16, x_tmp, d_model, do_wht], refs)
        libcuda.cuLaunchKernel(kd["wht64_bf16_to_f32_kernel"].func, d_model // 64, 1, 1, 64, 1, 1, 0, stream_h, a0, None)

        a1 = _build_c_args([
            x_tmp, ctypes.c_void_p(l2_zc_ptr), exp_ids, gate_act, wina_sc, y_tmp,
            d_model, d_ffn, lay.n_groups, lay.row_bytes, lay.page_bytes,
            lay.off_gate, lay.off_wina_norm, act_int_gpu, lay.qbits,
        ], refs)
        libcuda.cuLaunchKernel(kd["wina_gate_q3q4_kernel"].func, (d_ffn + 3) // 4, n_act, 1, 128, 1, 1, 0, stream_h, a1, None)

        use_mask = k_keep < d_ffn
        if use_mask:
            a2 = _build_c_args([wina_sc, act_neur, d_ffn, k_keep], refs)
            libcuda.cuLaunchKernel(kd["wina_select_top_neurons_kernel"].func, n_act, 1, 1, 256, 1, 1, 0, stream_h, a2, None)

        neur_arg = act_neur if use_mask else ctypes.c_void_p(0)
        a3 = _build_c_args([
            x_tmp, ctypes.c_void_p(l2_zc_ptr), exp_ids, rw, gate_act, neur_arg, y_tmp,
            d_model, d_ffn, k_keep, lay.n_groups, lay.row_bytes, lay.slab_bytes, lay.page_bytes,
            lay.off_slabs, lay.qbits,
        ], refs)
        libcuda.cuLaunchKernel(kd["wina_up_down_q3q4_kernel"].func, (k_keep + 3) // 4, n_act, 1, 128, 1, 1, 0, stream_h, a3, None)

        a4 = _build_c_args([y_tmp, y_out, d_model, do_wht], refs)
        libcuda.cuLaunchKernel(kd["iwht64_f32_to_bf16_kernel"].func, d_model // 64, 1, 1, 64, 1, 1, 0, stream_h, a4, None)

    def _fid(pred: torch.Tensor) -> Tuple[float, float]:
        p = pred.float().reshape(-1)
        r = y_bf16_ref.float().reshape(-1)
        return (
            float(F.cosine_similarity(p.unsqueeze(0), r.unsqueeze(0)).item()),
            float((p - r).norm().item() / r.norm().clamp_min(1e-8).item()),
        )

    # Accuracy Evaluation
    ws = get_shared_moe_workspace(dev)
    ws.ensure_capacity(d_model, d_ffn, lay_base.page_bytes)
    base_launcher = MoETLBLayerLauncher(dev, n_pool_base, lay_base, act_int_gpu)
    slot_map_base = torch.tensor([-2 - i for i in range(n_pool_base)], dtype=torch.int32, device=dev)
    dummy_l1 = torch.empty((1, lay_base.page_bytes), dtype=torch.uint8, device=dev)

    y_base_acc = torch.empty((1, d_model), device=dev, dtype=torch.bfloat16)
    base_launcher.launch(
        x_bf16, torch.arange(E_acc, device=dev, dtype=torch.int64).view(1, E_acc),
        rw_acc.view(1, E_acc), dummy_l1, slot_map_base, y_base_acc, l2_zero_copy_dev_ptr=l2_base_zc,
    )
    torch.cuda.synchronize()

    exp_acc_i32 = torch.arange(E_acc, device=dev, dtype=torch.int32)
    results_acc = [("Baseline UINT4 (100% neurons)         ", y_base_acc.clone())]
    for lbl, lay, zc, kr, wht in [
        ("UINT4-WINA (Top-50% neurons)          ", lay_q4w, l2_q4w_zc, 0.50, 0),
        ("UINT4-WINA (Top-40% neurons)          ", lay_q4w, l2_q4w_zc, 0.40, 0),
        ("WHT-INT3   (100% neurons)             ", lay_q3w, l2_q3w_zc, 1.00, 1),
        ("WHT-INT3-WINA (Top-50% neurons)       ", lay_q3w, l2_q3w_zc, 0.50, 1),
        ("WHT-INT3-WINA (Top-40% neurons)       ", lay_q3w, l2_q3w_zc, 0.40, 1),
    ]:
        launch_wina(lay, zc, exp_acc_i32, rw_acc, kr, wht)
        torch.cuda.synchronize()
        results_acc.append((lbl, y_out.clone()))

    print("\n  [Accuracy vs. Exact BF16 (Heavy-Tailed Activations + 2% Weight Outliers)]")
    for lbl, t in results_acc:
        cos, rl2 = _fid(t)
        print(f"    {lbl} | Cosine Sim = {cos:.5f} | Rel L2 Err = {rl2:.4f}")

    # CUDA-Graphed Wall-Clock PCIe Zero-Copy Speed Benchmark (`ms/layer`)
    print(
        f"\n  [CUDA-Graphed L2 Zero-Copy Latency (ms/layer, zero Python launch overhead)]"
    )
    print(
        f"  {'Misses':>6} | {'UINT4 Base':>10} | {'UINT4-WINA 50%':>14} | {'INT3-WINA 50%':>13} | "
        f"{'INT3-WINA 40%':>13} | {'Speedup (Q4-50 / Q3-50 / Q3-40)':>31}"
    )
    print("  " + "-" * 100)

    iters = 50
    for m_cnt in miss_counts:
        rw_m = torch.full((m_cnt,), 1.0 / m_cnt, device=dev, dtype=torch.float32)
        rw_m_2d = rw_m.view(1, m_cnt)

        # Static input buffers updated before each graph replay so every iter hits cold host RAM!
        stat_idx_i64 = torch.arange(m_cnt, device=dev, dtype=torch.int64).view(1, m_cnt)
        stat_idx_i32 = torch.arange(m_cnt, device=dev, dtype=torch.int32)

        def _capture_graph(fn) -> torch.cuda.CUDAGraph:
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            return g

        g_base = _capture_graph(
            lambda: base_launcher.launch(x_bf16, stat_idx_i64, rw_m_2d, dummy_l1, slot_map_base, y_base_acc, l2_zero_copy_dev_ptr=l2_base_zc)
        )
        g_q4w50 = _capture_graph(lambda: launch_wina(lay_q4w, l2_q4w_zc, stat_idx_i32, rw_m, 0.50, 0))
        g_q3w50 = _capture_graph(lambda: launch_wina(lay_q3w, l2_q3w_zc, stat_idx_i32, rw_m, 0.50, 1))
        g_q3w40 = _capture_graph(lambda: launch_wina(lay_q3w, l2_q3w_zc, stat_idx_i32, rw_m, 0.40, 1))

        def _time_graph(g: torch.cuda.CUDAGraph, pool_n: int, is_i64: bool) -> float:
            # Prebuild index tensors on GPU to avoid CPU->GPU sync inside timing loop
            seq = [
                torch.arange((it * m_cnt) % (pool_n - m_cnt), (it * m_cnt) % (pool_n - m_cnt) + m_cnt,
                             device=dev, dtype=(torch.int64 if is_i64 else torch.int32))
                for it in range(iters)
            ]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for it in range(iters):
                if is_i64:
                    stat_idx_i64[0].copy_(seq[it])
                else:
                    stat_idx_i32.copy_(seq[it])
                g.replay()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0 / iters

        ms_base = _time_graph(g_base, n_pool_base, True)
        ms_q4_50 = _time_graph(g_q4w50, n_pool_q4w, False)
        ms_q3_50 = _time_graph(g_q3w50, n_pool_q3w, False)
        ms_q3_40 = _time_graph(g_q3w40, n_pool_q3w, False)

        print(
            f"  {m_cnt:>6} | {ms_base:>8.3f}ms | {ms_q4_50:>12.3f}ms | {ms_q3_50:>11.3f}ms | "
            f"{ms_q3_40:>11.3f}ms | {ms_base/ms_q4_50:>8.2f}x / {ms_base/ms_q3_50:>6.2f}x / {ms_base/ms_q3_40:>6.2f}x"
        )


def main() -> None:
    dev = torch.device("cuda:0")
    print(f"Compiling UINT4-WINA & WHT-INT3-WINA NVRTC kernels on {torch.cuda.get_device_name(dev)}...", flush=True)
    kd = compile_kernels(dev)
    print("Compiled successfully!", flush=True)

    bench_geometry(
        kd, "Qwen3-Next-80B-A3B", d_model=2048, d_ffn=512, top_k=10, num_layers=48, act_kind="silu",
        miss_counts=[1, 2, 4, 6, 8, 10],
    )
    bench_geometry(
        kd, "Gemma-4-26B-A4B", d_model=2816, d_ffn=704, top_k=8, num_layers=30, act_kind="gelu_tanh",
        miss_counts=[1, 2, 4, 6, 8],
    )


if __name__ == "__main__":
    main()
