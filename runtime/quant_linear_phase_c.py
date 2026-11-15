"""Phase C2 Register-Fused Quantized Linear (`INT8` Row-Wise & `INT4` Group-Wise) for Attention & Boundary Projections.

Provides a drop-in `nn.Module` replacement for `nn.Linear` (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.)
backed by NVRTC-compiled CUDA C++ kernels for `1 <= M <= 8` decode (zero unpacked BF16 HBM writes) and
Tensor Core GEMM for `M > 8` prefill. Crucial for keeping 32B-70B Self-Attention inside a 24 GB GPU budget.
"""

import ctypes
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from runtime.gpu_page_tlb import (
    SharedQuantLUTPool,
    quantize_and_pack_uint4_group,
    quantize_int8_rowwise,
)
from runtime.phase_c_kernels import MAX_FUSED_DECODE_TOKENS

QUANT_LINEAR_CUDA_SOURCE = r"""
typedef unsigned short bf16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;

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

#define MAX_M 8
#define TILE_D 1280

extern "C" __global__ void phase_c_linear_int8_kernel(
    const bf16_t* __restrict__ x_in,      // [M, d_in]
    const int8_t* __restrict__ q_weight,  // [d_out, d_in]
    const bf16_t* __restrict__ row_scale, // [d_out]
    const bf16_t* __restrict__ bias,      // [d_out] or nullptr (has_bias == 0)
    bf16_t* __restrict__ out_bf16,        // [M, d_out]
    int M,
    int d_in,
    int d_out,
    int has_bias
) {
    extern __shared__ float s_x[]; // [M * TILE_D]
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp_id;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    const int aligned4 = ((d_in & 3) == 0);
    const int n_tiles = (d_in + TILE_D - 1) / TILE_D;
    for (int tile = 0; tile < n_tiles; ++tile) {
        const int base_k = tile * TILE_D;
        const int k_lim = (base_k + TILE_D < d_in) ? TILE_D : (d_in - base_k);

        for (int idx = tid; idx < k_lim; idx += blockDim.x) {
            const int col = base_k + idx;
            for (int m = 0; m < M; ++m) {
                s_x[m * TILE_D + idx] = bf16_to_f32(__ldg(&x_in[m * d_in + col]));
            }
        }
        __syncthreads();

        if (row < d_out) {
            const int8_t* rw = &q_weight[(size_t)row * d_in + base_k];
            int idx = lane * 4;
            if (aligned4) {
                for (; idx + 3 < k_lim; idx += 128) {
                    const int p4 = __ldg((const int*)&rw[idx]);
                    const float w0 = (float)((int8_t)(p4 & 0xFF));
                    const float w1 = (float)((int8_t)((p4 >> 8) & 0xFF));
                    const float w2 = (float)((int8_t)((p4 >> 16) & 0xFF));
                    const float w3 = (float)((int8_t)((p4 >> 24) & 0xFF));
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            const float* sx = &s_x[m * TILE_D + idx];
                            acc[m] += w0 * sx[0] + w1 * sx[1] + w2 * sx[2] + w3 * sx[3];
                        }
                    }
                }
            } else {
                for (; idx + 3 < k_lim; idx += 128) {
                    const float w0 = (float)__ldg(&rw[idx + 0]);
                    const float w1 = (float)__ldg(&rw[idx + 1]);
                    const float w2 = (float)__ldg(&rw[idx + 2]);
                    const float w3 = (float)__ldg(&rw[idx + 3]);
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            const float* sx = &s_x[m * TILE_D + idx];
                            acc[m] += w0 * sx[0] + w1 * sx[1] + w2 * sx[2] + w3 * sx[3];
                        }
                    }
                }
            }
            for (int k = (k_lim & ~3) + lane; k < k_lim; k += 32) {
                float fw = (float)__ldg(&rw[k]);
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        acc[m] += fw * s_x[m * TILE_D + k];
                    }
                }
            }
        }
        __syncthreads();
    }

    if (row < d_out) {
        const float sc = bf16_to_f32(__ldg(&row_scale[row]));
        const float b_val = has_bias ? bf16_to_f32(__ldg(&bias[row])) : 0.0f;
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                float s = warp_sum(acc[m]);
                if (lane == 0) {
                    out_bf16[m * d_out + row] = f32_to_bf16(s * sc + b_val);
                }
            }
        }
    }
}

extern "C" __global__ void phase_c_linear_uint4_kernel(
    const bf16_t* __restrict__ x_in,       // [M, d_in]
    const uint8_t* __restrict__ p_u8,      // [d_out, d_pad_half]
    const bf16_t* __restrict__ wmin,       // [d_out, n_groups]
    const bf16_t* __restrict__ scale,      // [d_out, n_groups]
    const bf16_t* __restrict__ bias,       // [d_out] or nullptr
    bf16_t* __restrict__ out_bf16,         // [M, d_out]
    int M,
    int d_in,
    int d_out,
    int d_pad_half,
    int n_groups,
    int group_size,
    int has_bias
) {
    extern __shared__ float s_x[]; // [M * TILE_D]
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp_id;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    const int d_pad = d_pad_half * 2;
    const int g_half = group_size >> 1;
    const int vec4_ok = (((d_pad_half & 3) == 0) && ((g_half & 3) == 0) && (g_half >= 4));
    const int n_tiles = (d_pad + TILE_D - 1) / TILE_D;

    for (int tile = 0; tile < n_tiles; ++tile) {
        const int base_k = tile * TILE_D;
        for (int idx = tid; idx < TILE_D; idx += blockDim.x) {
            const int col = base_k + idx;
            for (int m = 0; m < M; ++m) {
                s_x[m * TILE_D + idx] = (col < d_in) ? bf16_to_f32(__ldg(&x_in[m * d_in + col])) : 0.0f;
            }
        }
        __syncthreads();

        if (row < d_out) {
            const int base_b = base_k >> 1;
            const int b_lim = ((base_b + (TILE_D >> 1)) < d_pad_half) ? (TILE_D >> 1) : (d_pad_half - base_b);
            const uint8_t* rp = &p_u8[(size_t)row * d_pad_half + base_b];
            const bf16_t* rmn = &wmin[(size_t)row * n_groups];
            const bf16_t* rsc = &scale[(size_t)row * n_groups];

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
                    const int x_idx = b_l << 1;

                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            const float* sx = &s_x[m * TILE_D + x_idx];
                            acc[m] += w0 * sx[0] + w1 * sx[1] + w2 * sx[2] + w3 * sx[3]
                                    + w4 * sx[4] + w5 * sx[5] + w6 * sx[6] + w7 * sx[7];
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
                    const int x_idx = b_rem << 1;
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            acc[m] += w0 * s_x[m * TILE_D + x_idx + 0] + w1 * s_x[m * TILE_D + x_idx + 1];
                        }
                    }
                }
            } else {
                for (int b_l = lane; b_l < b_lim; b_l += 32) {
                    const int b_global = base_b + b_l;
                    const int grp = b_global / g_half;
                    const uint8_t packed = __ldg(&rp[b_l]);
                    const float sc = bf16_to_f32(__ldg(&rsc[grp]));
                    const float mn = bf16_to_f32(__ldg(&rmn[grp]));

                    const float w0 = (float)(packed & 0x0FU) * sc + mn;
                    const float w1 = (float)((packed >> 4) & 0x0FU) * sc + mn;
                    const int x_idx = b_l << 1;
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            acc[m] += w0 * s_x[m * TILE_D + x_idx + 0] + w1 * s_x[m * TILE_D + x_idx + 1];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (row < d_out) {
        const float b_val = has_bias ? bf16_to_f32(__ldg(&bias[row])) : 0.0f;
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                float s = warp_sum(acc[m]);
                if (lane == 0) {
                    out_bf16[m * d_out + row] = f32_to_bf16(s + b_val);
                }
            }
        }
    }
}
"""

_QLIN_KERNELS_BY_DEVICE: Dict[int, Dict[str, Any]] = {}


def get_or_compile_quant_linear_kernels(device: torch.device) -> Dict[str, Any]:
    dev_idx = 0 if device.index is None else int(device.index)
    if dev_idx in _QLIN_KERNELS_BY_DEVICE:
        return _QLIN_KERNELS_BY_DEVICE[dev_idx]

    from torch.cuda._utils import _cuda_load_module, _get_gpu_rtc_library, _get_gpu_runtime_library

    with torch.cuda.device(dev_idx):
        libnvrtc = _get_gpu_rtc_library()
        libcuda = _get_gpu_runtime_library()
        props = torch.cuda.get_device_properties(dev_idx)
        arch = f"--gpu-architecture=sm_{props.major}{props.minor}".encode("utf-8")
        opts = [arch, b"--use_fast_math", b"-default-device"]
        opts_arr = (ctypes.c_char_p * len(opts))(*opts)

        src = QUANT_LINEAR_CUDA_SOURCE.encode("utf-8")
        prog = ctypes.c_void_p()
        libnvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"quant_linear.cu", 0, None, None)
        res = libnvrtc.nvrtcCompileProgram(prog, len(opts), opts_arr)
        if res != 0:
            log_sz = ctypes.c_size_t()
            libnvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
            log_buf = ctypes.create_string_buffer(log_sz.value)
            libnvrtc.nvrtcGetProgramLog(prog, log_buf)
            libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            raise RuntimeError(f"QuantLinear NVRTC failed:\n{log_buf.value.decode('utf-8', errors='replace')}")

        cubin_sz = ctypes.c_size_t()
        libnvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(cubin_sz))
        cubin_buf = ctypes.create_string_buffer(cubin_sz.value)
        libnvrtc.nvrtcGetCUBIN(prog, cubin_buf)
        libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

        names = ["phase_c_linear_int8_kernel", "phase_c_linear_uint4_kernel"]
        kd = _cuda_load_module(cubin_buf.raw, names)
        kd["phase_c_linear_int8_kernel"].set_shared_memory_config(65536)
        kd["phase_c_linear_uint4_kernel"].set_shared_memory_config(65536)
        kd["libcuda"] = libcuda

    _QLIN_KERNELS_BY_DEVICE[dev_idx] = kd
    return kd


class PhaseCQuantLinear(nn.Module):
    """Hardware-packed `INT8` or `UINT4 (g=64)` Linear layer with register-fused NVRTC decode."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_mode: str,
        q_weight: torch.Tensor,
        scale: torch.Tensor,
        wmin: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        group_size: int = 64,
        lut_pool: Optional[SharedQuantLUTPool] = None,
    ) -> None:
        super().__init__()
        if quant_mode not in ("int8", "int4"):
            raise ValueError(f"Unsupported quant_mode '{quant_mode}'.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.quant_mode = quant_mode
        self.group_size = int(group_size)
        self.lut_pool = lut_pool

        self.register_buffer("q_weight", q_weight.contiguous(), persistent=True)
        self.register_buffer("scale", scale.contiguous(), persistent=True)
        if wmin is not None:
            self.register_buffer("wmin", wmin.contiguous(), persistent=True)
        else:
            self.wmin = None
        if bias is not None:
            self.register_buffer("bias", bias.contiguous(), persistent=True)
        else:
            self.bias = None

        self.d_pad = int(q_weight.shape[1] * 2) if quant_mode == "int4" else self.in_features
        self.pad_len = self.d_pad - self.in_features

        self._bound_ready: bool = False
        self._x_ptr = ctypes.c_void_p(0)
        self._out_ptr = ctypes.c_void_p(0)
        self._M = ctypes.c_int(1)
        self._refs: List[Any] = []
        self._c_args: Any = None
        self._kernel_func: Any = None
        self._libcuda: Any = None
        self._grid_x = (self.out_features + 3) // 4

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        quant_mode: str = "int8",
        group_size: int = 64,
        target_device: str = "cuda:0",
        scale_dtype: torch.dtype = torch.bfloat16,
        lut_pool: Optional[SharedQuantLUTPool] = None,
    ) -> "PhaseCQuantLinear":
        w_fp32 = linear.weight.detach().to(device=target_device, dtype=torch.float32)
        bias_bf = (
            linear.bias.detach().to(device=target_device, dtype=scale_dtype).contiguous()
            if linear.bias is not None
            else None
        )
        if quant_mode == "int8":
            q_i8, sc_bf = quantize_int8_rowwise(w_fp32, s_in=None, target_device=target_device, scale_dtype=scale_dtype)
            del w_fp32
            return cls(
                in_features=linear.in_features,
                out_features=linear.out_features,
                quant_mode="int8",
                q_weight=q_i8,
                scale=sc_bf,
                wmin=None,
                bias=bias_bf,
                group_size=group_size,
                lut_pool=lut_pool,
            )
        else:
            p_u8, mn_bf, sc_bf, _ = quantize_and_pack_uint4_group(
                w_fp32, s_in=None, group_size=group_size, target_device=target_device, scale_dtype=scale_dtype
            )
            del w_fp32
            return cls(
                in_features=linear.in_features,
                out_features=linear.out_features,
                quant_mode="int4",
                q_weight=p_u8,
                scale=sc_bf,
                wmin=mn_bf,
                bias=bias_bf,
                group_size=group_size,
                lut_pool=lut_pool,
            )

    def vram_bytes(self) -> int:
        total = int(self.q_weight.numel() * self.q_weight.element_size() + self.scale.numel() * self.scale.element_size())
        if self.wmin is not None:
            total += int(self.wmin.numel() * self.wmin.element_size())
        if self.bias is not None:
            total += int(self.bias.numel() * self.bias.element_size())
        return total

    def _init_cuda_launcher(self) -> None:
        kd = get_or_compile_quant_linear_kernels(self.q_weight.device)
        self._libcuda = kd["libcuda"]
        has_bias = 1 if self.bias is not None else 0
        bias_ptr_val = self.bias.data_ptr() if self.bias is not None else 0

        if self.quant_mode == "int8":
            self._kernel_func = kd["phase_c_linear_int8_kernel"].func
            spec = [
                self._x_ptr,
                self.q_weight,
                self.scale,
                ctypes.c_void_p(bias_ptr_val),
                self._out_ptr,
                self._M,
                self.in_features,
                self.out_features,
                has_bias,
            ]
        else:
            assert self.wmin is not None
            self._kernel_func = kd["phase_c_linear_uint4_kernel"].func
            spec = [
                self._x_ptr,
                self.q_weight,
                self.wmin,
                self.scale,
                ctypes.c_void_p(bias_ptr_val),
                self._out_ptr,
                self._M,
                self.in_features,
                self.out_features,
                int(self.q_weight.shape[1]),
                int(self.wmin.shape[1]),
                self.group_size,
                has_bias,
            ]

        refs: List[Any] = []
        arr = (ctypes.c_void_p * len(spec))()
        for i, item in enumerate(spec):
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
        self._refs = refs
        self._c_args = arr
        self._bound_ready = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        flat_x = x.reshape(-1, self.in_features)
        M = int(flat_x.shape[0])

        if 1 <= M <= MAX_FUSED_DECODE_TOKENS and flat_x.is_cuda and self.scale.dtype == torch.bfloat16:
            if not self._bound_ready:
                self._init_cuda_launcher()
            x_bf = flat_x.contiguous() if flat_x.dtype == torch.bfloat16 else flat_x.to(torch.bfloat16).contiguous()
            out_bf = torch.empty((M, self.out_features), dtype=torch.bfloat16, device=x_bf.device)
            self._x_ptr.value = x_bf.data_ptr()
            self._out_ptr.value = out_bf.data_ptr()
            self._M.value = M
            stream_h = torch.cuda.current_stream(x_bf.device)._as_parameter_
            res = self._libcuda.cuLaunchKernel(
                self._kernel_func, self._grid_x, 1, 1, 128, 1, 1, M * 1280 * 4, stream_h, self._c_args, None
            )
            if res != 0:
                raise RuntimeError(f"PhaseCQuantLinear kernel launch failed with CUDA error {res}")
            out = out_bf.view(*orig_shape[:-1], self.out_features)
            return out if out.dtype == orig_dtype else out.to(orig_dtype)

        # Batched Tensor Core path for prompt prefill (`M > 8`)
        dt = self.scale.dtype
        x_dt = flat_x.to(dt)
        if self.quant_mode == "int8":
            w = self.q_weight.to(dt) * self.scale
            y = x_dt @ w.T
        else:
            assert self.wmin is not None and self.lut_pool is not None
            n_grp = self.wmin.shape[1]
            q_bf = F.embedding(self.q_weight.to(torch.int32), self.lut_pool.lut_uint4).view(
                self.out_features, n_grp, self.group_size
            )
            w_pad = (q_bf * self.scale + self.wmin).view(self.out_features, self.d_pad)
            x_p = F.pad(x_dt, (0, self.pad_len), value=0.0) if self.pad_len > 0 else x_dt
            y = x_p @ w_pad.T
        if self.bias is not None:
            y = y + self.bias
        return y.view(*orig_shape[:-1], self.out_features).to(orig_dtype)
