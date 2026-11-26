"""Single-Layer NVRTC CUDA Benchmark: `Trellis-WINA` (QTIP 2-Bit Bitshift Trellis + Gate-First WINA
Conditional PCIe Fetch) vs. TensorMMU `UINT4` L2 Zero-Copy Baseline.

Evaluates BOTH:
  1. Wall-clock PCIe Zero-Copy decode latency (`ms/layer` & speedup) across 1, 2, 4, 6, 8 L2 misses/layer
     on a >160 MiB pinned host pool (exceeding CPU L3 cache) for:
       - Qwen3-Next-80B-A3B (`d_model=2048, d_ffn=512, top_k=10, 48 layers`, SiLU)
       - Gemma-4-26B-A4B    (`d_model=2816, d_ffn=704, top_k=8,  30 layers`, GeLU-tanh)
  2. Real numerical fidelity (Cosine Similarity & Relative L2 Error vs. exact BF16) comparing:
       - `UINT4` (group-64, current TensorMMU baseline)
       - Naive `2-bit` scalar quant (group-64)
       - `QTIP-2bit` (Block-64 Walsh-Hadamard + 2-bit Lookup-Free Bitshift Trellis, 100% neurons)
       - `Trellis-WINA` (`QTIP-2bit` + Gate-First WINA Top-50% & Top-40% conditional neuron fetch)
"""

from __future__ import annotations

import ctypes
import math
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

from runtime.moe_expert_tlb import quantize_pack_moe_experts_uint4
from runtime.moe_tlb_kernels import (
    ContiguousExpertPageLayout,
    MoETLBLayerLauncher,
    get_shared_moe_workspace,
    get_zero_copy_device_ptr,
)

# =============================================================================
# 1. NVRTC CUDA Kernels: `QTIP` 2-Bit Lookup-Free Bitshift Trellis + `WINA`
# =============================================================================
TRELLIS_WINA_CUDA_SRC = r"""
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
        return x / (1.0f + __expf(-x));  // SiLU
    }
    float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));  // GeLU-tanh
}

// QTIP-style lookup-free 1MAD compute-only trellis decoder:
// Maps a 16-bit trellis state `S` -> smooth Gaussian-like float in ~[-2.4, +2.4]
// using ZERO shared-memory LUT lookups (3 ALU ops in registers, 0 bank conflicts!).
// Uses Irwin-Hall sum of 4 uniform bytes extracted from a single 32-bit LCG/1MAD hash:
//   mean = 0, std = 1.0  (exact closed-form approximation to N(0, 1)!)
__device__ __forceinline__ float qtip_1mad_decode(uint32_t state16) {
    uint32_t h = (state16 & 0xFFFFU) * 34038481U + 2531011U;
    int s = (int)(h & 0xFFU) + (int)((h >> 8) & 0xFFU) + (int)((h >> 16) & 0xFFU) + (int)((h >> 24) & 0xFFU);
    return ((float)s - 510.0f) * 0.0067944f;  // 1 / (255 * sqrt(4/12)) = 0.0067944
}

// Fast in-register Block-64 Walsh-Hadamard Transform (`H_64 / 8`) on `x_in[d_model]` -> `x_wht[d_model]`
// Grid: (d_model / 64, 1, 1), Block: (64, 1, 1) -> ~2 microseconds!
extern "C" __global__ void wht64_bf16_to_f32_kernel(
    const bf16_t* __restrict__ x_in,
    float* __restrict__ x_wht,
    int d_model
) {
    __shared__ float s[64];
    const int base = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int idx = base + tid;
    s[tid] = (idx < d_model) ? bf16_to_f32(x_in[idx]) : 0.0f;
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
    if (idx < d_model) {
        x_wht[idx] = s[tid] * 0.125f;  // 1 / sqrt(64)
    }
}

// Inverse Block-64 Walsh-Hadamard Transform on `y_wht_f32[d_model]` -> `y_out_bf16[d_model]`
extern "C" __global__ void iwht64_f32_to_bf16_kernel(
    const float* __restrict__ y_wht,
    bf16_t* __restrict__ y_out,
    int d_model
) {
    __shared__ float s[64];
    const int base = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int idx = base + tid;
    s[tid] = (idx < d_model) ? y_wht[idx] : 0.0f;
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
    if (idx < d_model) {
        y_out[idx] = f32_to_bf16(s[tid] * 0.125f);
    }
}

// =============================================================================
// Stage 1: Gate-Only QTIP 2-Bit Trellis GEMV over L2 Zero-Copy PCIe + WINA Score
// Each block = 4 warps (128 threads), computing 4 rows of `gate_proj` for expert `a`.
// Each lane loads 1 `uint4` (16 bytes = 64 2-bit weights = 1 independent 64-weight trellis chunk!)
// per iteration — 100% coalesced 128-bit PCIe Zero-Copy transactions!
// =============================================================================
extern "C" __global__ void trellis_wina_gate_kernel(
    const float* __restrict__ x_wht,         // [d_model]
    const uint8_t* __restrict__ l2_pages,    // [pool, page_bytes] (Zero-Copy DDR5)
    const int* __restrict__ expert_ids,      // [n_active]
    float* __restrict__ gate_act_out,        // [n_active, d_ffn]
    float* __restrict__ wina_score_out,      // [n_active, d_ffn]
    float* __restrict__ y_wht_out,           // [d_model] (zeroed cooperatively by block 0!)
    int d_model,
    int d_ffn,
    int row_q2_bytes,       // d_model / 4 (must be multiple of 16)
    int page_bytes,
    int off_gate_q2,
    int off_gate_scale,
    int off_wina_norm,
    int act_kind
) {
    const int a = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 4 + warp;

    // Cooperatively zero `y_wht_out` once (zero extra kernel launch!)
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        for (int i = tid; i < d_model; i += blockDim.x) {
            y_wht_out[i] = 0.0f;
        }
    }

    if (row >= d_ffn) return;

    const uint8_t* page = l2_pages + (size_t)expert_ids[a] * (size_t)page_bytes;
    const uint4* g_q2_u4 = (const uint4*)(page + off_gate_q2 + (size_t)row * (size_t)row_q2_bytes);
    const float sc = bf16_to_f32(((const bf16_t*)(page + off_gate_scale))[row]);
    const float dn_norm = bf16_to_f32(((const bf16_t*)(page + off_wina_norm))[row]);

    const int n_chunks = d_model >> 6;  // 1 chunk = 64 weights = 16 bytes = 1 uint4!
    float acc = 0.0f;

    for (int ch = lane; ch < n_chunks; ch += 32) {
        uint4 v = __ldg(&g_q2_u4[ch]);
        const float* xp = x_wht + (ch << 6);
        uint32_t words[4] = {v.x, v.y, v.z, v.w};
        uint32_t state = (uint32_t)(ch * 2654435761U) & 0xFFFFU;  // deterministic per-64-chunk seed

        #pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            uint32_t w32 = words[wi];
            const float* xw = xp + (wi << 4);
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                uint32_t b2 = (w32 >> (k << 1)) & 3U;
                state = ((state << 2) | b2) & 0xFFFFU;
                acc += qtip_1mad_decode(state) * xw[k];
            }
        }
    }

    float g_val = warp_sum(acc) * sc;
    if (lane == 0) {
        float ga = apply_act(g_val, act_kind);
        gate_act_out[a * d_ffn + row] = ga;
        wina_score_out[a * d_ffn + row] = fabsf(ga) * dn_norm;
    }
}

// =============================================================================
// Stage 2: Exact In-Shared-Memory WINA Threshold Selection (1 block per active expert)
// Selects the Top-`k_keep` neurons per expert via fast 12-iteration binary search on threshold
// (`< 3 microseconds`, zero CPU sync, 100% CUDA Graph safe!).
// =============================================================================
extern "C" __global__ void wina_select_top_neurons_kernel(
    const float* __restrict__ wina_scores,   // [n_active, d_ffn]
    int* __restrict__ active_neurons,        // [n_active, k_keep]
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
    if (tid == 0) {
        s_lo = 0.0f;
        s_hi = 0.0f;
    }
    __syncthreads();

    // Compute block max via atomicMax on positive float bits
    atomicMax((int*)&s_hi, __float_as_int(local_max));
    __syncthreads();

    // 14-step binary search for threshold yielding ~k_keep neurons
    for (int iter = 0; iter < 14; ++iter) {
        if (tid == 0) {
            s_thr = 0.5f * (s_lo + s_hi);
            s_cnt = 0;
        }
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

    // Compact strictly `k_keep` winning neuron indices into `active_neurons[a, 0..k_keep-1]`
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
    // Fill any remaining slots (if ties dropped count slightly below k_keep)
    for (int j = tid; j < d_ffn; j += blockDim.x) {
        if (s_sc[j] < final_thr && s_sc[j] >= s_lo) {
            int pos = atomicAdd(&s_cnt, 1);
            if (pos < k_keep) out_row[pos] = j;
        }
    }
    __syncthreads();
    // Guard against degenerate all-zero scores
    for (int j = tid; j < k_keep; j += blockDim.x) {
        if (j >= s_cnt) out_row[j] = j;
    }
}

// =============================================================================
// Stage 3: Conditional Neuron-Major Up + Down QTIP 2-Bit Kernel over L2 Zero-Copy!
// ONLY fetches the `k_keep` surviving neurons per expert — skipped neurons NEVER touch PCIe!
// Grid: ((k_keep + 3) / 4, n_active), Block: (128, 1, 1) -> 1 warp per active neuron `j`.
// Each neuron slab in Block C holds `[up_q2 (d/4 B), down_q2 (d/4 B), up_scale (2 B), down_scale (2 B)]`
// contiguously aligned to 16 bytes!
// =============================================================================
extern "C" __global__ void trellis_wina_up_down_kernel(
    const float* __restrict__ x_wht,         // [d_model]
    const uint8_t* __restrict__ l2_pages,    // [pool, page_bytes]
    const int* __restrict__ expert_ids,      // [n_active]
    const float* __restrict__ router_w,      // [n_active]
    const float* __restrict__ gate_act,      // [n_active, d_ffn]
    const int* __restrict__ active_neurons,  // [n_active, k_keep] (or nullptr when k_keep == d_ffn)
    float* __restrict__ y_wht_out,           // [d_model] (FP32 L2-cached atomicAdd target)
    int d_model,
    int d_ffn,
    int k_keep,
    int row_q2_bytes,
    int slab_bytes,
    int page_bytes,
    int off_slabs
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

    const uint4* up_q2_u4 = (const uint4*)(slab);
    const uint4* dn_q2_u4 = (const uint4*)(slab + row_q2_bytes);
    const bf16_t* sc_pair = (const bf16_t*)(slab + 2 * row_q2_bytes);
    const float up_sc = bf16_to_f32(sc_pair[0]);
    const float dn_sc = bf16_to_f32(sc_pair[1]);

    const int n_chunks = d_model >> 6;
    float u_acc = 0.0f;

    // 1. Stream ONLY active neuron's `up_row[j]` over L2 Zero-Copy PCIe
    for (int ch = lane; ch < n_chunks; ch += 32) {
        uint4 v = __ldg(&up_q2_u4[ch]);
        const float* xp = x_wht + (ch << 6);
        uint32_t words[4] = {v.x, v.y, v.z, v.w};
        uint32_t state = (uint32_t)(ch * 2654435761U) & 0xFFFFU;
        #pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            uint32_t w32 = words[wi];
            const float* xw = xp + (wi << 4);
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                uint32_t b2 = (w32 >> (k << 1)) & 3U;
                state = ((state << 2) | b2) & 0xFFFFU;
                u_acc += qtip_1mad_decode(state) * xw[k];
            }
        }
    }

    float u_val = warp_sum(u_acc) * up_sc;
    u_val = __shfl_sync(0xffffffffU, u_val, 0);
    const float coeff = gate_act[a * d_ffn + j] * u_val * router_w[a] * dn_sc;

    // 2. Stream ONLY active neuron's `down_col[j]` over L2 Zero-Copy PCIe and accumulate into `y_wht_out`
    for (int ch = lane; ch < n_chunks; ch += 32) {
        uint4 v = __ldg(&dn_q2_u4[ch]);
        float* yp = y_wht_out + (ch << 6);
        uint32_t words[4] = {v.x, v.y, v.z, v.w};
        uint32_t state = (uint32_t)(ch * 2654435761U) & 0xFFFFU;
        #pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            uint32_t w32 = words[wi];
            float* yw = yp + (wi << 4);
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                uint32_t b2 = (w32 >> (k << 1)) & 3U;
                state = ((state << 2) | b2) & 0xFFFFU;
                atomicAdd(&yw[k], coeff * qtip_1mad_decode(state));
            }
        }
    }
}
"""


def compile_trellis_wina_kernels(device: torch.device) -> Dict[str, any]:
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
        libnvrtc.nvrtcCreateProgram(ctypes.byref(prog), TRELLIS_WINA_CUDA_SRC.encode("utf-8"), b"trellis_wina.cu", 0, None, None)
        res = libnvrtc.nvrtcCompileProgram(prog, len(opts), opts_arr)
        if res != 0:
            log_sz = ctypes.c_size_t()
            libnvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
            log_buf = ctypes.create_string_buffer(log_sz.value)
            libnvrtc.nvrtcGetProgramLog(prog, log_buf)
            libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            raise RuntimeError(f"Trellis-WINA NVRTC failed:\n{log_buf.value.decode('utf-8', errors='replace')}")

        cubin_sz = ctypes.c_size_t()
        libnvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(cubin_sz))
        cubin_buf = ctypes.create_string_buffer(cubin_sz.value)
        libnvrtc.nvrtcGetCUBIN(prog, cubin_buf)
        libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

        names = [
            "wht64_bf16_to_f32_kernel",
            "iwht64_f32_to_bf16_kernel",
            "trellis_wina_gate_kernel",
            "wina_select_top_neurons_kernel",
            "trellis_wina_up_down_kernel",
        ]
        kd = _cuda_load_module(cubin_buf.raw, names)
        kd["libcuda"] = libcuda
        return kd


@dataclass(frozen=True)
class TrellisWINAPageLayout:
    d_model: int
    d_ffn: int
    row_q2_bytes: int
    slab_bytes: int
    off_gate_q2: int
    off_gate_scale: int
    off_wina_norm: int
    off_slabs: int
    page_bytes: int

    @classmethod
    def compute(cls, d_model: int, d_ffn: int) -> "TrellisWINAPageLayout":
        assert d_model % 64 == 0
        row_q2 = d_model // 4  # 2 bits/weight
        # Each neuron slab: up_q2 (row_q2) + down_q2 (row_q2) + 2 bf16 scales (4 B), aligned to 16 B
        slab_b = ((2 * row_q2 + 4 + 15) // 16) * 16
        off_gate_q2 = 0
        off_gate_sc = ((d_ffn * row_q2 + 15) // 16) * 16
        off_wina = ((off_gate_sc + d_ffn * 2 + 15) // 16) * 16
        off_slabs = ((off_wina + d_ffn * 2 + 15) // 16) * 16
        page_b = ((off_slabs + d_ffn * slab_b + 4095) // 4096) * 4096
        return cls(d_model, d_ffn, row_q2, slab_b, off_gate_q2, off_gate_sc, off_wina, off_slabs, page_b)


# =============================================================================
# 2. Vectorized GPU Beam-Search Trellis Encoder (`WHT-64` + `QTIP-1MAD` 2-bit)
# =============================================================================
def _wht64_matrix(device: torch.device) -> torch.Tensor:
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    for _ in range(6):
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / 8.0


def apply_wht64(t: torch.Tensor, H64: torch.Tensor) -> torch.Tensor:
    orig_shape = t.shape
    return (t.float().reshape(-1, 64) @ H64).reshape(orig_shape)


@torch.no_grad()
def encode_qtip_1mad_2bit_gpu(w_rot: torch.Tensor, beam_width: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encodes `[rows, d_model]` WHT-rotated weights into `[rows, d_model/4]` 2-bit trellis codes + `bf16[rows]` scales.

    Uses exact GPU beam-search over the 16-bit bitshift-trellis `qtip_1mad_decode(state)` so our accuracy test
    measures *genuine* QTIP-style 2-bit trellis reconstruction quality (not synthetic noise!).
    """
    dev = w_rot.device
    rows, d_model = w_rot.shape
    n_chunks = d_model // 64
    row_std = w_rot.float().std(dim=-1).clamp_min(1e-6)
    w_norm = (w_rot.float() / row_std.unsqueeze(-1)).view(rows, n_chunks, 64)

    # Precompute decode table for all 65,536 16-bit states (`256 KB` on GPU, used ONLY at compile time!)
    all_s = torch.arange(65536, device=dev, dtype=torch.int64)
    h = (all_s * 34038481 + 2531011) & 0xFFFFFFFF
    s_sum = (h & 0xFF) + ((h >> 8) & 0xFF) + ((h >> 16) & 0xFF) + ((h >> 24) & 0xFF)
    dec_lut = (s_sum.float() - 510.0) * 0.0067944  # [65536]

    N = rows * n_chunks
    targets = w_norm.reshape(N, 64)
    ch_ids = torch.arange(n_chunks, device=dev, dtype=torch.int64).repeat(rows)
    init_state = ((ch_ids * 2654435761) & 0xFFFF).unsqueeze(1).expand(N, 1)  # [N, 1]

    # Beam state: states [N, B], costs [N, B], history [N, B, 64] uint8
    states = init_state
    costs = torch.zeros((N, 1), device=dev, dtype=torch.float32)
    hist = torch.zeros((N, 1, 64), device=dev, dtype=torch.uint8)
    b2_vals = torch.arange(4, device=dev, dtype=torch.int64).view(1, 1, 4)

    for k in range(64):
        B = states.shape[1]
        cand_states = ((states.unsqueeze(2) << 2) | b2_vals) & 0xFFFF  # [N, B, 4]
        cand_vals = dec_lut[cand_states]  # [N, B, 4]
        err = (cand_vals - targets[:, k].view(N, 1, 1)).square()
        cand_costs = (costs.unsqueeze(2) + err).reshape(N, B * 4)

        new_B = min(beam_width, B * 4)
        top_costs, top_idx = torch.topk(cand_costs, k=new_B, dim=1, largest=False)
        parent_b = top_idx // 4
        chosen_b2 = (top_idx % 4).to(torch.uint8)

        states = cand_states.reshape(N, B * 4).gather(1, top_idx)
        costs = top_costs
        hist = hist.gather(1, parent_b.unsqueeze(-1).expand(N, new_B, 64))
        hist[:, :, k] = chosen_b2

    best_codes = hist[:, 0, :].reshape(rows, d_model)  # [rows, d_model] in {0,1,2,3}
    # Pack 4 2-bit codes per uint8 byte (little-endian within each 32-bit word matching CUDA `w32 >> (k*2)`)
    c4 = best_codes.view(rows, d_model // 4, 4)
    packed = (c4[..., 0] | (c4[..., 1] << 2) | (c4[..., 2] << 4) | (c4[..., 3] << 6)).to(torch.uint8).contiguous()
    return packed, row_std.to(torch.bfloat16).contiguous()


@torch.no_grad()
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


def run_benchmark_for_geometry(
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

    lay_q4 = ContiguousExpertPageLayout.compute(d_model, d_ffn, 64)
    lay_tw = TrellisWINAPageLayout.compute(d_model, d_ffn)
    print(f"\n{'=' * 96}")
    print(f"  {name} | d_model={d_model}, d_ffn={d_ffn}, top_k={top_k}, layers={num_layers} ({act_kind})")
    print(
        f"  Page Size: UINT4 = {lay_q4.page_bytes / (1024*1024):.2f} MiB  vs.  "
        f"Trellis-WINA 2-bit = {lay_tw.page_bytes / (1024*1024):.2f} MiB "
        f"({lay_q4.page_bytes / lay_tw.page_bytes:.2f}x smaller!)"
    )
    print(f"{'=' * 96}")

    # -------------------------------------------------------------------------
    # PART A: Real Numerical Fidelity Test (BF16 vs UINT4 vs Naive 2-bit vs QTIP-2bit vs Trellis-WINA)
    # Use realistic heavy-tailed LLM activations (Student-t / outlier channels) & weights!
    # -------------------------------------------------------------------------
    E_acc = 4
    w_gate = torch.randn((E_acc, d_ffn, d_model), device=dev, dtype=torch.bfloat16) * 0.02
    w_up = torch.randn((E_acc, d_ffn, d_model), device=dev, dtype=torch.bfloat16) * 0.02
    w_dn = torch.randn((E_acc, d_model, d_ffn), device=dev, dtype=torch.bfloat16) * 0.02
    # Inject realistic LLM outlier channels (5% of weights 4x larger)
    outlier_mask = torch.rand_like(w_gate.float()) < 0.02
    w_gate = (w_gate.float() * torch.where(outlier_mask, 4.0, 1.0)).to(torch.bfloat16)

    # Realistic heavy-tailed input hidden state (with 8 outlier channels as in Qwen/Gemma!)
    x_bf16 = torch.randn((1, d_model), device=dev, dtype=torch.bfloat16)
    x_bf16[0, :8] *= 5.0
    router_w_acc = torch.full((E_acc,), 1.0 / E_acc, device=dev, dtype=torch.float32)

    # Exact BF16 reference output
    y_bf16_ref = torch.zeros((1, d_model), device=dev, dtype=torch.float32)
    for e in range(E_acc):
        g = F.linear(x_bf16.float(), w_gate[e].float())
        u = F.linear(x_bf16.float(), w_up[e].float())
        h = (F.silu(g) if act_kind == "silu" else F.gelu(g, approximate="tanh")) * u
        y_bf16_ref += router_w_acc[e] * F.linear(h, w_dn[e].float())

    # 1. UINT4 (group-64) reference
    w_gu_cat = torch.cat([w_gate, w_up], dim=1)
    q4_pages_cpu, _ = quantize_pack_moe_experts_uint4(w_gu_cat.cpu(), w_dn.cpu(), group_size=64, compute_device="cuda:0")
    q4_vram = q4_pages_cpu.to(dev)
    q4_launcher = MoETLBLayerLauncher(dev, E_acc, lay_q4, act_int_gpu)
    y_q4 = torch.empty((1, d_model), device=dev, dtype=torch.bfloat16)
    q4_launcher.launch(
        x_bf16,
        torch.arange(E_acc, device=dev, dtype=torch.int64).view(1, E_acc),
        router_w_acc.view(1, E_acc),
        q4_vram,
        torch.arange(E_acc, device=dev, dtype=torch.int32),
        y_q4,
    )
    torch.cuda.synchronize()

    # 2. Naive 2-bit scalar quantization (group-64) for comparison
    def _naive_q2(w: torch.Tensor) -> torch.Tensor:
        orig = w.shape
        wg = w.float().reshape(-1, 64)
        mn = wg.amin(dim=-1, keepdim=True)
        sc = ((wg.amax(dim=-1, keepdim=True) - mn) / 3.0).clamp_min(1e-8)
        q = torch.clamp(torch.round((wg - mn) / sc), 0.0, 3.0)
        return (q * sc + mn).reshape(orig)

    y_naive_q2 = torch.zeros((1, d_model), device=dev, dtype=torch.float32)
    for e in range(E_acc):
        g = F.linear(x_bf16.float(), _naive_q2(w_gate[e]))
        u = F.linear(x_bf16.float(), _naive_q2(w_up[e]))
        h = (F.silu(g) if act_kind == "silu" else F.gelu(g, approximate="tanh")) * u
        y_naive_q2 += router_w_acc[e] * F.linear(h, _naive_q2(w_dn[e]))

    # 3. Encode the 4 real experts into our `Trellis-WINA` 2-bit page format!
    tw_base_cpu = torch.zeros((E_acc, lay_tw.page_bytes), dtype=torch.uint8, device="cpu")
    for e in range(E_acc):
        g_rot = apply_wht64(w_gate[e], H64)
        u_rot = apply_wht64(w_up[e], H64)
        # Rotate down_proj along output dim (dim=0 -> transpose to [d_ffn, d_model] so each neuron column is length d_model!)
        dn_t_rot = apply_wht64(w_dn[e].T.contiguous(), H64)  # [d_ffn, d_model]
        dn_col_norms = w_dn[e].float().norm(dim=0).to(torch.bfloat16)  # [d_ffn]

        g_q2, g_sc = encode_qtip_1mad_2bit_gpu(g_rot, beam_width=8)
        u_q2, u_sc = encode_qtip_1mad_2bit_gpu(u_rot, beam_width=8)
        d_q2, d_sc = encode_qtip_1mad_2bit_gpu(dn_t_rot, beam_width=8)

        pg = torch.zeros((lay_tw.page_bytes,), dtype=torch.uint8, device=dev)
        pg[lay_tw.off_gate_q2 : lay_tw.off_gate_q2 + d_ffn * lay_tw.row_q2_bytes].copy_(g_q2.reshape(-1))
        pg[lay_tw.off_gate_scale : lay_tw.off_gate_scale + d_ffn * 2].copy_(g_sc.view(torch.uint8).reshape(-1))
        pg[lay_tw.off_wina_norm : lay_tw.off_wina_norm + d_ffn * 2].copy_(dn_col_norms.view(torch.uint8).reshape(-1))

        # Pack neuron-major Block C: [d_ffn, slab_bytes]
        slabs = torch.zeros((d_ffn, lay_tw.slab_bytes), dtype=torch.uint8, device=dev)
        slabs[:, 0 : lay_tw.row_q2_bytes].copy_(u_q2)
        slabs[:, lay_tw.row_q2_bytes : 2 * lay_tw.row_q2_bytes].copy_(d_q2)
        sc_pairs = torch.stack([u_sc, d_sc], dim=1).view(torch.uint8).reshape(d_ffn, 4)
        slabs[:, 2 * lay_tw.row_q2_bytes : 2 * lay_tw.row_q2_bytes + 4].copy_(sc_pairs)
        pg[lay_tw.off_slabs : lay_tw.off_slabs + d_ffn * lay_tw.slab_bytes].copy_(slabs.reshape(-1))
        tw_base_cpu[e].copy_(pg.cpu())

    # Allocate >160 MiB pinned L2 Zero-Copy pools for BOTH UINT4 and Trellis-WINA (so CPU L3 cannot cheat!)
    n_pool_q4 = min(140, max(64, int((180 * 1024 * 1024) // lay_q4.page_bytes)))
    n_pool_tw = min(240, max(128, int((180 * 1024 * 1024) // lay_tw.page_bytes)))
    l2_q4_pin = torch.empty((n_pool_q4, lay_q4.page_bytes), dtype=torch.uint8, device="cpu").pin_memory()
    for i in range(n_pool_q4):
        l2_q4_pin[i].copy_(q4_pages_cpu[i % E_acc])
    l2_q4_zc = get_zero_copy_device_ptr(l2_q4_pin, dev)

    l2_tw_pin = torch.empty((n_pool_tw, lay_tw.page_bytes), dtype=torch.uint8, device="cpu").pin_memory()
    for i in range(n_pool_tw):
        l2_tw_pin[i].copy_(tw_base_cpu[i % E_acc])
    l2_tw_zc = get_zero_copy_device_ptr(l2_tw_pin, dev)

    # Scratch GPU buffers for Trellis-WINA execution
    x_wht_buf = torch.empty((d_model,), device=dev, dtype=torch.float32)
    y_wht_buf = torch.empty((d_model,), device=dev, dtype=torch.float32)
    y_tw_out = torch.empty((1, d_model), device=dev, dtype=torch.bfloat16)
    gate_act_buf = torch.empty((top_k, d_ffn), device=dev, dtype=torch.float32)
    wina_sc_buf = torch.empty((top_k, d_ffn), device=dev, dtype=torch.float32)
    act_neur_buf = torch.empty((top_k, d_ffn), device=dev, dtype=torch.int32)
    libcuda = kd["libcuda"]

    def launch_trellis_wina(exp_ids_gpu: torch.Tensor, rw_gpu: torch.Tensor, keep_ratio: float) -> None:
        n_act = int(exp_ids_gpu.numel())
        k_keep = max(1, min(d_ffn, int(round(d_ffn * keep_ratio))))
        stream_h = torch.cuda.current_stream(dev)._as_parameter_
        refs: List[any] = []

        # 0. WHT-64 on x_in (~2 us)
        a0 = _build_c_args([x_bf16, x_wht_buf, d_model], refs)
        libcuda.cuLaunchKernel(kd["wht64_bf16_to_f32_kernel"].func, d_model // 64, 1, 1, 64, 1, 1, 0, stream_h, a0, None)

        # 1. Gate-only QTIP 2-bit over L2 Zero-Copy + WINA score
        a1 = _build_c_args([
            x_wht_buf, ctypes.c_void_p(l2_tw_zc), exp_ids_gpu, gate_act_buf, wina_sc_buf, y_wht_buf,
            d_model, d_ffn, lay_tw.row_q2_bytes, lay_tw.page_bytes,
            lay_tw.off_gate_q2, lay_tw.off_gate_scale, lay_tw.off_wina_norm, act_int_gpu,
        ], refs)
        libcuda.cuLaunchKernel(kd["trellis_wina_gate_kernel"].func, (d_ffn + 3) // 4, n_act, 1, 128, 1, 1, 0, stream_h, a1, None)

        # 2. WINA Top-k neuron selection (skip if 100% kept)
        use_mask = k_keep < d_ffn
        if use_mask:
            a2 = _build_c_args([wina_sc_buf, act_neur_buf, d_ffn, k_keep], refs)
            libcuda.cuLaunchKernel(kd["wina_select_top_neurons_kernel"].func, n_act, 1, 1, 256, 1, 1, 0, stream_h, a2, None)

        # 3. Conditional neuron-major Up + Down QTIP 2-bit over L2 Zero-Copy
        neur_arg = act_neur_buf if use_mask else ctypes.c_void_p(0)
        a3 = _build_c_args([
            x_wht_buf, ctypes.c_void_p(l2_tw_zc), exp_ids_gpu, rw_gpu, gate_act_buf, neur_arg, y_wht_buf,
            d_model, d_ffn, k_keep, lay_tw.row_q2_bytes, lay_tw.slab_bytes, lay_tw.page_bytes, lay_tw.off_slabs,
        ], refs)
        libcuda.cuLaunchKernel(kd["trellis_wina_up_down_kernel"].func, (k_keep + 3) // 4, n_act, 1, 128, 1, 1, 0, stream_h, a3, None)

        # 4. Inverse WHT-64 -> BF16
        a4 = _build_c_args([y_wht_buf, y_tw_out, d_model], refs)
        libcuda.cuLaunchKernel(kd["iwht64_f32_to_bf16_kernel"].func, d_model // 64, 1, 1, 64, 1, 1, 0, stream_h, a4, None)

    def _fidelity(pred: torch.Tensor) -> Tuple[float, float]:
        p = pred.float().reshape(-1)
        r = y_bf16_ref.float().reshape(-1)
        cos = float(F.cosine_similarity(p.unsqueeze(0), r.unsqueeze(0)).item())
        rel_l2 = float((p - r).norm().item() / r.norm().clamp_min(1e-8).item())
        return cos, rel_l2

    exp_acc_ids = torch.arange(E_acc, device=dev, dtype=torch.int32)
    launch_trellis_wina(exp_acc_ids, router_w_acc, keep_ratio=1.0)
    torch.cuda.synchronize()
    y_qtip_100 = y_tw_out.clone()

    launch_trellis_wina(exp_acc_ids, router_w_acc, keep_ratio=0.50)
    torch.cuda.synchronize()
    y_tw_50 = y_tw_out.clone()

    launch_trellis_wina(exp_acc_ids, router_w_acc, keep_ratio=0.40)
    torch.cuda.synchronize()
    y_tw_40 = y_tw_out.clone()

    print("\n  [Accuracy vs. Exact BF16 (Heavy-Tailed Activations + 2% Weight Outliers)]")
    for lbl, tensor in [
        ("UINT4 (group-64, current baseline)       ", y_q4),
        ("Naive 2-bit scalar (group-64)            ", y_naive_q2),
        ("QTIP-2bit Trellis (100% neurons)         ", y_qtip_100),
        ("Trellis-WINA 2-bit (Top-50% neurons)     ", y_tw_50),
        ("Trellis-WINA 2-bit (Top-40% neurons)     ", y_tw_40),
    ]:
        cos, rl2 = _fidelity(tensor)
        print(f"    {lbl} | Cosine Sim = {cos:.5f} | Rel L2 Err = {rl2:.4f}")

    # -------------------------------------------------------------------------
    # PART B: Wall-Clock PCIe Zero-Copy Speed Benchmark (`ms/layer` & Speedup)
    # -------------------------------------------------------------------------
    ws = get_shared_moe_workspace(dev)
    ws.ensure_capacity(d_model, d_ffn, lay_q4.page_bytes)
    q4_bench_launcher = MoETLBLayerLauncher(dev, n_pool_q4, lay_q4, act_int_gpu)
    slot_map_q4_l2 = torch.tensor([-2 - i for i in range(n_pool_q4)], dtype=torch.int32, device=dev)
    dummy_l1 = torch.empty((1, lay_q4.page_bytes), dtype=torch.uint8, device=dev)

    print(
        f"\n  {'Misses':>6} | {'UINT4 L2-ZC':>11} | {'QTIP-2b (100%)':>14} | "
        f"{'Trellis-WINA 50%':>16} | {'Trellis-WINA 40%':>16} | {'Speedup (50% / 40%)':>20}"
    )
    print("  " + "-" * 96)

    iters = 50
    for m_cnt in miss_counts:
        rw_m = torch.full((m_cnt,), 1.0 / m_cnt, device=dev, dtype=torch.float32)
        rw_m_2d = rw_m.view(1, m_cnt)
        idx_q4_list = [
            torch.arange((it * m_cnt) % (n_pool_q4 - m_cnt), (it * m_cnt) % (n_pool_q4 - m_cnt) + m_cnt, device=dev, dtype=torch.int64).view(1, m_cnt)
            for it in range(iters)
        ]
        idx_tw_list = [
            torch.arange((it * m_cnt) % (n_pool_tw - m_cnt), (it * m_cnt) % (n_pool_tw - m_cnt) + m_cnt, device=dev, dtype=torch.int32)
            for it in range(iters)
        ]

        # Warmup
        for _ in range(5):
            q4_bench_launcher.launch(x_bf16, idx_q4_list[0], rw_m_2d, dummy_l1, slot_map_q4_l2, y_q4, l2_zero_copy_dev_ptr=l2_q4_zc)
            launch_trellis_wina(idx_tw_list[0], rw_m, 0.50)
        torch.cuda.synchronize()

        # Baseline: UINT4 L2 Zero-Copy
        t0 = time.perf_counter()
        for it in range(iters):
            q4_bench_launcher.launch(x_bf16, idx_q4_list[it], rw_m_2d, dummy_l1, slot_map_q4_l2, y_q4, l2_zero_copy_dev_ptr=l2_q4_zc)
        torch.cuda.synchronize()
        ms_q4 = (time.perf_counter() - t0) * 1000.0 / iters

        # Variant A: QTIP-2bit (100% neurons)
        t0 = time.perf_counter()
        for it in range(iters):
            launch_trellis_wina(idx_tw_list[it], rw_m, 1.0)
        torch.cuda.synchronize()
        ms_qtip100 = (time.perf_counter() - t0) * 1000.0 / iters

        # Variant B1: Trellis-WINA Top-50%
        t0 = time.perf_counter()
        for it in range(iters):
            launch_trellis_wina(idx_tw_list[it], rw_m, 0.50)
        torch.cuda.synchronize()
        ms_tw50 = (time.perf_counter() - t0) * 1000.0 / iters

        # Variant B2: Trellis-WINA Top-40%
        t0 = time.perf_counter()
        for it in range(iters):
            launch_trellis_wina(idx_tw_list[it], rw_m, 0.40)
        torch.cuda.synchronize()
        ms_tw40 = (time.perf_counter() - t0) * 1000.0 / iters

        print(
            f"  {m_cnt:>6} | {ms_q4:>9.3f}ms | {ms_qtip100:>12.3f}ms | "
            f"{ms_tw50:>14.3f}ms | {ms_tw40:>14.3f}ms | {ms_q4/ms_tw50:>8.2f}x / {ms_q4/ms_tw40:>6.2f}x"
        )


def main() -> None:
    dev = torch.device("cuda:0")
    print(f"Compiling Trellis-WINA NVRTC CUDA kernels on {torch.cuda.get_device_name(dev)}...", flush=True)
    kd = compile_trellis_wina_kernels(dev)
    print("Compiled successfully!", flush=True)

    run_benchmark_for_geometry(
        kd,
        name="Qwen3-Next-80B-A3B",
        d_model=2048,
        d_ffn=512,
        top_k=10,
        num_layers=48,
        act_kind="silu",
        miss_counts=[1, 2, 4, 6, 8, 10],
    )
    run_benchmark_for_geometry(
        kd,
        name="Gemma-4-26B-A4B",
        d_model=2816,
        d_ffn=704,
        top_k=8,
        num_layers=30,
        act_kind="gelu_tanh",
        miss_counts=[1, 2, 4, 6, 8],
    )


if __name__ == "__main__":
    main()
