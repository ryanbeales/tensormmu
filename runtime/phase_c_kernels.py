"""Phase C Runtime: Register-Fused NVRTC CUDA C++ Kernels for Bit-Tiered Neural VM & GPU TLB.

Eliminates all temporary HBM BF16 weight unpacking and collapses ~48 per-layer PyTorch kernel
launches down to 2 launches (`core_only`) or 3-4 launches (`vm_paged` with `speculative_tlb`)
during autoregressive decoding (`1 <= M <= 8`), using PyTorch's bundled NVRTC runtime compiler
with zero external `nvcc` or MSVC build dependencies.
"""

import ctypes
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

MAX_FUSED_DECODE_TOKENS = 8

PHASE_C_CUDA_SOURCE = r"""
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

__device__ __forceinline__ float round_bf16(float f) {
    return bf16_to_f32(f32_to_bf16(f));
}

__device__ __forceinline__ float apply_act(float x, int act_kind) {
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
#define TILE_PK 256
#define TILE_D 1280

extern "C" __global__ void phase_c_stage1_kernel(
    const bf16_t* __restrict__ x_in,
    const bf16_t* __restrict__ inv_s_in,
    const int8_t* __restrict__ q_gu_hot,
    const bf16_t* __restrict__ sc_gu_hot,
    const bf16_t* __restrict__ inv_smid_hot,
    const unsigned short* __restrict__ p_gu_cold,
    const bf16_t* __restrict__ mn_gu_cold,
    const bf16_t* __restrict__ sc_gu_cold,
    const bf16_t* __restrict__ inv_smid_cold,
    const bf16_t* __restrict__ col_err_w,
    const bf16_t* __restrict__ U_bridge,
    float* __restrict__ x_scaled_ws,
    float* __restrict__ h_hot_ws,
    float* __restrict__ g_cold_ws,
    float* __restrict__ u_cold_ws,
    float* __restrict__ h_cold_scaled_ws,
    float* __restrict__ z_bridge_ws,
    float* __restrict__ page_scores_ws,
    int M,
    int d_model,
    int n_hot,
    int n_cold,
    int n_pk_gu,
    int n_grp_gu,
    int g5_gu,
    int bridge_rank,
    int page_size,
    int num_pages,
    int act_kind,
    int vm_paged
) {
    extern __shared__ float smem[];
    float* s_x = smem;
    float* s_sum5 = smem + M * TILE_D;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int task_id = blockIdx.x * warps_per_block + warp_id;

    const int is_hot = (task_id < n_hot);
    const int is_cold = (!is_hot) && (task_id < n_hot + n_cold);
    const int is_bridge = (!is_hot) && (!is_cold) && (task_id < n_hot + n_cold + bridge_rank);

    float acc_g[MAX_M];
    float acc_u[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc_g[m] = 0.0f;
        acc_u[m] = 0.0f;
    }

    const int d_total = n_pk_gu * 5;
    const int total_tiles = (d_total + TILE_D - 1) / TILE_D;

    const int hot_r = task_id;
    const int cold_c = task_id - n_hot;
    const int bridge_b = task_id - (n_hot + n_cold);

    for (int tile = 0; tile < total_tiles; ++tile) {
        const int base_d = tile * TILE_D;
        const int base_pk = tile * TILE_PK;

        for (int idx = tid; idx < TILE_D; idx += blockDim.x) {
            const int col = base_d + idx;
            if (col < d_model) {
                float inv_s = bf16_to_f32(__ldg(&inv_s_in[col]));
                for (int m = 0; m < M; ++m) {
                    float xv = bf16_to_f32(__ldg(&x_in[m * d_model + col]));
                    float xs = round_bf16(xv * inv_s);
                    s_x[m * TILE_D + idx] = xs;
                    if (vm_paged && blockIdx.x == 0) {
                        x_scaled_ws[m * d_model + col] = xs;
                    }
                }
            } else {
                for (int m = 0; m < M; ++m) {
                    s_x[m * TILE_D + idx] = 0.0f;
                }
            }
        }
        __syncthreads();

        for (int pk_local = tid; pk_local < TILE_PK; pk_local += blockDim.x) {
            const int i5 = pk_local * 5;
            for (int m = 0; m < M; ++m) {
                const float* sx_m = &s_x[m * TILE_D + i5];
                s_sum5[m * TILE_PK + pk_local] = sx_m[0] + sx_m[1] + sx_m[2] + sx_m[3] + sx_m[4];
            }
        }
        __syncthreads();

        if (is_cold) {
            const int pk_end = (base_pk + TILE_PK < n_pk_gu) ? TILE_PK : (n_pk_gu - base_pk);
            const unsigned short* row_g = &p_gu_cold[(size_t)cold_c * n_pk_gu + base_pk];
            const unsigned short* row_u = &p_gu_cold[(size_t)(n_cold + cold_c) * n_pk_gu + base_pk];
            const bf16_t* mng_row = &mn_gu_cold[(size_t)cold_c * n_grp_gu];
            const bf16_t* scg_row = &sc_gu_cold[(size_t)cold_c * n_grp_gu];
            const bf16_t* mnu_row = &mn_gu_cold[(size_t)(n_cold + cold_c) * n_grp_gu];
            const bf16_t* scu_row = &sc_gu_cold[(size_t)(n_cold + cold_c) * n_grp_gu];

            int pk_done = 0;
            if ((n_pk_gu & 1) == 0) {
                pk_done = pk_end & ~1;
                for (int pk_l = lane * 2; pk_l + 1 < pk_end; pk_l += 64) {
                    const unsigned int ug2 = __ldg((const unsigned int*)&row_g[pk_l]);
                    const unsigned int uu2 = __ldg((const unsigned int*)&row_u[pk_l]);

                    const int grp0 = (base_pk + pk_l) / g5_gu;
                    const int grp1 = (base_pk + pk_l + 1) / g5_gu;
                    const float sc_g0 = bf16_to_f32(__ldg(&scg_row[grp0]));
                    const float mn_g0 = bf16_to_f32(__ldg(&mng_row[grp0]));
                    const float sc_u0 = bf16_to_f32(__ldg(&scu_row[grp0]));
                    const float mn_u0 = bf16_to_f32(__ldg(&mnu_row[grp0]));

                    const float sc_g1 = (grp1 == grp0) ? sc_g0 : bf16_to_f32(__ldg(&scg_row[grp1]));
                    const float mn_g1 = (grp1 == grp0) ? mn_g0 : bf16_to_f32(__ldg(&mng_row[grp1]));
                    const float sc_u1 = (grp1 == grp0) ? sc_u0 : bf16_to_f32(__ldg(&scu_row[grp1]));
                    const float mn_u1 = (grp1 == grp0) ? mn_u0 : bf16_to_f32(__ldg(&mnu_row[grp1]));

                    const unsigned int ug_a = ug2 & 0xFFFFU;
                    const unsigned int ug_b = ug2 >> 16;
                    const unsigned int uu_a = uu2 & 0xFFFFU;
                    const unsigned int uu_b = uu2 >> 16;

                    const float qg0 = (float)(ug_a & 7U), qg1 = (float)((ug_a >> 3) & 7U), qg2 = (float)((ug_a >> 6) & 7U), qg3 = (float)((ug_a >> 9) & 7U), qg4 = (float)((ug_a >> 12) & 7U);
                    const float qg5 = (float)(ug_b & 7U), qg6 = (float)((ug_b >> 3) & 7U), qg7 = (float)((ug_b >> 6) & 7U), qg8 = (float)((ug_b >> 9) & 7U), qg9 = (float)((ug_b >> 12) & 7U);

                    const float qu0 = (float)(uu_a & 7U), qu1 = (float)((uu_a >> 3) & 7U), qu2 = (float)((uu_a >> 6) & 7U), qu3 = (float)((uu_a >> 9) & 7U), qu4 = (float)((uu_a >> 12) & 7U);
                    const float qu5 = (float)(uu_b & 7U), qu6 = (float)((uu_b >> 3) & 7U), qu7 = (float)((uu_b >> 6) & 7U), qu8 = (float)((uu_b >> 9) & 7U), qu9 = (float)((uu_b >> 12) & 7U);

                    const int i5 = pk_l * 5;
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            const float* sx = &s_x[m * TILE_D + i5];
                            const float sum5_0 = s_sum5[m * TILE_PK + pk_l];
                            const float sum5_1 = s_sum5[m * TILE_PK + pk_l + 1];
                            const float dot_qg0 = qg0 * sx[0] + qg1 * sx[1] + qg2 * sx[2] + qg3 * sx[3] + qg4 * sx[4];
                            const float dot_qu0 = qu0 * sx[0] + qu1 * sx[1] + qu2 * sx[2] + qu3 * sx[3] + qu4 * sx[4];
                            const float dot_qg1 = qg5 * sx[5] + qg6 * sx[6] + qg7 * sx[7] + qg8 * sx[8] + qg9 * sx[9];
                            const float dot_qu1 = qu5 * sx[5] + qu6 * sx[6] + qu7 * sx[7] + qu8 * sx[8] + qu9 * sx[9];
                            acc_g[m] += (sc_g0 * dot_qg0 + mn_g0 * sum5_0) + (sc_g1 * dot_qg1 + mn_g1 * sum5_1);
                            acc_u[m] += (sc_u0 * dot_qu0 + mn_u0 * sum5_0) + (sc_u1 * dot_qu1 + mn_u1 * sum5_1);
                        }
                    }
                }
            }

            for (int pk_l = pk_done + lane; pk_l < pk_end; pk_l += 32) {
                const int pk_global = base_pk + pk_l;
                const int grp = pk_global / g5_gu;
                const unsigned int ug = (unsigned int)__ldg(&row_g[pk_l]);
                const unsigned int uu = (unsigned int)__ldg(&row_u[pk_l]);

                const float sc_g = bf16_to_f32(__ldg(&scg_row[grp]));
                const float mn_g = bf16_to_f32(__ldg(&mng_row[grp]));
                const float sc_u = bf16_to_f32(__ldg(&scu_row[grp]));
                const float mn_u = bf16_to_f32(__ldg(&mnu_row[grp]));

                const float qg0 = (float)(ug & 7U);
                const float qg1 = (float)((ug >> 3) & 7U);
                const float qg2 = (float)((ug >> 6) & 7U);
                const float qg3 = (float)((ug >> 9) & 7U);
                const float qg4 = (float)((ug >> 12) & 7U);

                const float qu0 = (float)(uu & 7U);
                const float qu1 = (float)((uu >> 3) & 7U);
                const float qu2 = (float)((uu >> 6) & 7U);
                const float qu3 = (float)((uu >> 9) & 7U);
                const float qu4 = (float)((uu >> 12) & 7U);

                const int i5 = pk_l * 5;
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const float* sx = &s_x[m * TILE_D + i5];
                        const float sum5 = s_sum5[m * TILE_PK + pk_l];
                        const float dot_qg = qg0 * sx[0] + qg1 * sx[1] + qg2 * sx[2] + qg3 * sx[3] + qg4 * sx[4];
                        const float dot_qu = qu0 * sx[0] + qu1 * sx[1] + qu2 * sx[2] + qu3 * sx[3] + qu4 * sx[4];
                        acc_g[m] += sc_g * dot_qg + mn_g * sum5;
                        acc_u[m] += sc_u * dot_qu + mn_u * sum5;
                    }
                }
            }
        } else if (is_hot && base_d < d_model) {
            const int d_lim = (base_d + TILE_D < d_model) ? TILE_D : (d_model - base_d);
            const int8_t* rg = &q_gu_hot[(size_t)hot_r * d_model + base_d];
            const int8_t* ru = &q_gu_hot[(size_t)(n_hot + hot_r) * d_model + base_d];

            int idx = lane * 4;
            if ((d_model & 3) == 0) {
                for (; idx + 3 < d_lim; idx += 128) {
                    const int pg = __ldg((const int*)&rg[idx]);
                    const int pu = __ldg((const int*)&ru[idx]);
                    const float g0 = (float)((int8_t)(pg & 0xFF));
                    const float g1 = (float)((int8_t)((pg >> 8) & 0xFF));
                    const float g2 = (float)((int8_t)((pg >> 16) & 0xFF));
                    const float g3 = (float)((int8_t)((pg >> 24) & 0xFF));
                    const float u0 = (float)((int8_t)(pu & 0xFF));
                    const float u1 = (float)((int8_t)((pu >> 8) & 0xFF));
                    const float u2 = (float)((int8_t)((pu >> 16) & 0xFF));
                    const float u3 = (float)((int8_t)((pu >> 24) & 0xFF));
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            const float* sx = &s_x[m * TILE_D + idx];
                            acc_g[m] += g0 * sx[0] + g1 * sx[1] + g2 * sx[2] + g3 * sx[3];
                            acc_u[m] += u0 * sx[0] + u1 * sx[1] + u2 * sx[2] + u3 * sx[3];
                        }
                    }
                }
            } else {
                for (; idx + 3 < d_lim; idx += 128) {
                    const float g0 = (float)__ldg(&rg[idx + 0]);
                    const float g1 = (float)__ldg(&rg[idx + 1]);
                    const float g2 = (float)__ldg(&rg[idx + 2]);
                    const float g3 = (float)__ldg(&rg[idx + 3]);
                    const float u0 = (float)__ldg(&ru[idx + 0]);
                    const float u1 = (float)__ldg(&ru[idx + 1]);
                    const float u2 = (float)__ldg(&ru[idx + 2]);
                    const float u3 = (float)__ldg(&ru[idx + 3]);
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            const float* sx = &s_x[m * TILE_D + idx];
                            acc_g[m] += g0 * sx[0] + g1 * sx[1] + g2 * sx[2] + g3 * sx[3];
                            acc_u[m] += u0 * sx[0] + u1 * sx[1] + u2 * sx[2] + u3 * sx[3];
                        }
                    }
                }
            }
            for (int k = (d_lim & ~3) + lane; k < d_lim; k += 32) {
                float fg = (float)__ldg(&rg[k]);
                float fu = (float)__ldg(&ru[k]);
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        float sxv = s_x[m * TILE_D + k];
                        acc_g[m] += fg * sxv;
                        acc_u[m] += fu * sxv;
                    }
                }
            }
        } else if (is_bridge && base_d < d_model) {
            const int d_lim = (base_d + TILE_D < d_model) ? TILE_D : (d_model - base_d);
            for (int k_l = lane; k_l < d_lim; k_l += 32) {
                const int col = base_d + k_l;
                const float u_val = bf16_to_f32(__ldg(&U_bridge[(size_t)col * bridge_rank + bridge_b]));
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const float xv = bf16_to_f32(__ldg(&x_in[m * d_model + col]));
                        acc_g[m] += xv * u_val;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (is_hot) {
        const float sc_g = bf16_to_f32(__ldg(&sc_gu_hot[hot_r]));
        const float sc_u = bf16_to_f32(__ldg(&sc_gu_hot[n_hot + hot_r]));
        const float inv_sm = bf16_to_f32(__ldg(&inv_smid_hot[hot_r]));
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                float sg = warp_sum(acc_g[m]);
                float su = warp_sum(acc_u[m]);
                if (lane == 0) {
                    float g = round_bf16(sg * sc_g);
                    float u = round_bf16(su * sc_u);
                    float h = round_bf16(apply_act(g, act_kind)) * u;
                    h_hot_ws[m * n_hot + hot_r] = round_bf16(round_bf16(h) * inv_sm);
                }
            }
        }
    } else if (is_cold) {
        const float inv_sm = bf16_to_f32(__ldg(&inv_smid_cold[cold_c]));
        const float cw = vm_paged ? bf16_to_f32(__ldg(&col_err_w[cold_c])) : 0.0f;
        const int page_id = cold_c / page_size;
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                float sg = warp_sum(acc_g[m]);
                float su = warp_sum(acc_u[m]);
                if (lane == 0) {
                    float g = round_bf16(sg);
                    float u = round_bf16(su);
                    float h = round_bf16(round_bf16(apply_act(g, act_kind)) * u);
                    h_cold_scaled_ws[m * n_cold + cold_c] = round_bf16(h * inv_sm);
                    if (vm_paged) {
                        g_cold_ws[m * n_cold + cold_c] = g;
                        u_cold_ws[m * n_cold + cold_c] = u;
                        if (page_id < num_pages) {
                            atomicAdd(&page_scores_ws[m * num_pages + page_id], fabsf(h) * cw);
                        }
                    }
                }
            }
        }
    } else if (is_bridge && bridge_rank > 0) {
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                float sb = warp_sum(acc_g[m]);
                if (lane == 0) {
                    z_bridge_ws[m * bridge_rank + bridge_b] = round_bf16(sb);
                }
            }
        }
    }
}

extern "C" __global__ void phase_c_router_topk_kernel(
    float* __restrict__ page_scores_ws,
    int* __restrict__ req_out,
    int M,
    int num_pages,
    int k_req
) {
    __shared__ float s_agg[256];
    const int tid = threadIdx.x;
    for (int p = tid; p < num_pages; p += blockDim.x) {
        float agg = 0.0f;
        if (M == 1) {
            agg = page_scores_ws[p];
            page_scores_ws[p] = 0.0f;
        } else {
            float sum_v = 0.0f;
            float max_v = 0.0f;
            for (int m = 0; m < M; ++m) {
                float v = page_scores_ws[m * num_pages + p];
                page_scores_ws[m * num_pages + p] = 0.0f;
                sum_v += v;
                if (v > max_v) max_v = v;
            }
            agg = (sum_v / (float)M) + 0.35f * max_v;
        }
        s_agg[p] = agg;
    }
    __syncthreads();

    if (tid == 0) {
        for (int sel = 0; sel < k_req; ++sel) {
            int best_p = 0;
            float best_val = -1e30f;
            for (int p = 0; p < num_pages; ++p) {
                if (s_agg[p] > best_val) {
                    best_val = s_agg[p];
                    best_p = p;
                }
            }
            req_out[sel] = best_p;
            s_agg[best_p] = -2e30f;
        }
    }
}

extern "C" __global__ void phase_c_tlb_gu_upgrade_kernel(
    const float* __restrict__ x_scaled_ws,
    const uint8_t* __restrict__ tlb_packed_q4,
    const bf16_t* __restrict__ tlb_scales,
    const long long* __restrict__ tlb_cols,
    const float* __restrict__ g_cold_ws,
    const float* __restrict__ u_cold_ws,
    const bf16_t* __restrict__ inv_smid_cold,
    float* __restrict__ h_cold_scaled_ws,
    float* __restrict__ h_tlb_down_ws,
    float* __restrict__ page_scores_ws,
    int* __restrict__ req_out,
    int M,
    int d_model,
    int d_even_half,
    int page_size,
    int n_cold,
    int n_act,
    int num_pages,
    int k_req,
    int act_kind,
    int do_router
) {
    if (do_router && blockIdx.x == 0) {
        __shared__ float s_agg[256];
        const int tid = threadIdx.x;
        for (int p = tid; p < num_pages; p += blockDim.x) {
            float agg = 0.0f;
            if (M == 1) {
                agg = page_scores_ws[p];
                page_scores_ws[p] = 0.0f;
            } else {
                float sum_v = 0.0f;
                float max_v = 0.0f;
                for (int m = 0; m < M; ++m) {
                    float v = page_scores_ws[m * num_pages + p];
                    page_scores_ws[m * num_pages + p] = 0.0f;
                    sum_v += v;
                    if (v > max_v) max_v = v;
                }
                agg = (sum_v / (float)M) + 0.35f * max_v;
            }
            s_agg[p] = agg;
        }
        __syncthreads();
        if (tid == 0) {
            for (int sel = 0; sel < k_req; ++sel) {
                int best_p = 0;
                float best_val = -1e30f;
                for (int p = 0; p < num_pages; ++p) {
                    if (s_agg[p] > best_val) {
                        best_val = s_agg[p];
                        best_p = p;
                    }
                }
                req_out[sel] = best_p;
                s_agg[best_p] = -2e30f;
            }
        }
        return;
    }

    extern __shared__ float s_x_tlb[];
    const int eff_block = do_router ? (blockIdx.x - 1) : blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int r_tlb = eff_block * warps_per_block + warp_id;
    const int c_act = n_act * page_size;

    float acc_g[MAX_M];
    float acc_u[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc_g[m] = 0.0f;
        acc_u[m] = 0.0f;
    }

    const int slot = r_tlb / page_size;
    const int p_off = r_tlb % page_size;
    const size_t slot_stride = (size_t)3 * page_size * d_even_half;
    const size_t plane_stride = (size_t)page_size * d_even_half;
    const uint8_t* row_g = (r_tlb < c_act) ? (&tlb_packed_q4[slot * slot_stride + 0 * plane_stride + (size_t)p_off * d_even_half]) : nullptr;
    const uint8_t* row_u = (r_tlb < c_act) ? (&tlb_packed_q4[slot * slot_stride + 1 * plane_stride + (size_t)p_off * d_even_half]) : nullptr;

    const int total_tiles = (d_model + TILE_D - 1) / TILE_D;
    for (int tile = 0; tile < total_tiles; ++tile) {
        const int base_d = tile * TILE_D;
        for (int idx = tid; idx < TILE_D; idx += blockDim.x) {
            const int col = base_d + idx;
            for (int m = 0; m < M; ++m) {
                s_x_tlb[m * TILE_D + idx] = (col < d_model) ? x_scaled_ws[m * d_model + col] : 0.0f;
            }
        }
        __syncthreads();

        if (r_tlb < c_act) {
            const int base_b = base_d >> 1;
            const int b_lim = ((base_b + (TILE_D >> 1)) < d_even_half) ? (TILE_D >> 1) : (d_even_half - base_b);
            int b_done = 0;
            if ((d_even_half & 3) == 0) {
                b_done = b_lim & ~3;
                for (int b_l = lane * 4; b_l + 3 < b_lim; b_l += 128) {
                    const unsigned int bg4 = __ldg((const unsigned int*)&row_g[base_b + b_l]);
                    const unsigned int bu4 = __ldg((const unsigned int*)&row_u[base_b + b_l]);
                    const float qg0 = (float)((int)((bg4 >>  0) & 0x0FU) - 8);
                    const float qg1 = (float)((int)((bg4 >>  4) & 0x0FU) - 8);
                    const float qg2 = (float)((int)((bg4 >>  8) & 0x0FU) - 8);
                    const float qg3 = (float)((int)((bg4 >> 12) & 0x0FU) - 8);
                    const float qg4 = (float)((int)((bg4 >> 16) & 0x0FU) - 8);
                    const float qg5 = (float)((int)((bg4 >> 20) & 0x0FU) - 8);
                    const float qg6 = (float)((int)((bg4 >> 24) & 0x0FU) - 8);
                    const float qg7 = (float)((int)((bg4 >> 28) & 0x0FU) - 8);

                    const float qu0 = (float)((int)((bu4 >>  0) & 0x0FU) - 8);
                    const float qu1 = (float)((int)((bu4 >>  4) & 0x0FU) - 8);
                    const float qu2 = (float)((int)((bu4 >>  8) & 0x0FU) - 8);
                    const float qu3 = (float)((int)((bu4 >> 12) & 0x0FU) - 8);
                    const float qu4 = (float)((int)((bu4 >> 16) & 0x0FU) - 8);
                    const float qu5 = (float)((int)((bu4 >> 20) & 0x0FU) - 8);
                    const float qu6 = (float)((int)((bu4 >> 24) & 0x0FU) - 8);
                    const float qu7 = (float)((int)((bu4 >> 28) & 0x0FU) - 8);

                    const int x_idx = b_l << 1;
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            const float* sx = &s_x_tlb[m * TILE_D + x_idx];
                            acc_g[m] += qg0 * sx[0] + qg1 * sx[1] + qg2 * sx[2] + qg3 * sx[3]
                                      + qg4 * sx[4] + qg5 * sx[5] + qg6 * sx[6] + qg7 * sx[7];
                            acc_u[m] += qu0 * sx[0] + qu1 * sx[1] + qu2 * sx[2] + qu3 * sx[3]
                                      + qu4 * sx[4] + qu5 * sx[5] + qu6 * sx[6] + qu7 * sx[7];
                        }
                    }
                }
            }
            for (int b_l = b_done + lane; b_l < b_lim; b_l += 32) {
                const uint8_t bg = __ldg(&row_g[base_b + b_l]);
                const uint8_t bu = __ldg(&row_u[base_b + b_l]);
                const float qg0 = (float)((int)(bg & 0x0FU) - 8);
                const float qg1 = (float)((int)((bg >> 4) & 0x0FU) - 8);
                const float qu0 = (float)((int)(bu & 0x0FU) - 8);
                const float qu1 = (float)((int)((bu >> 4) & 0x0FU) - 8);
                const int x_idx = b_l << 1;
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const float x0 = s_x_tlb[m * TILE_D + x_idx + 0];
                        const float x1 = s_x_tlb[m * TILE_D + x_idx + 1];
                        acc_g[m] += qg0 * x0 + qg1 * x1;
                        acc_u[m] += qu0 * x0 + qu1 * x1;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (r_tlb < c_act) {
        const size_t sc_base = (size_t)slot * 3 * page_size + p_off;
        const float sc_g = bf16_to_f32(__ldg(&tlb_scales[sc_base]));
        const float sc_u = bf16_to_f32(__ldg(&tlb_scales[(size_t)slot * 3 * page_size + page_size + p_off]));
        const float sc_d = bf16_to_f32(__ldg(&tlb_scales[(size_t)slot * 3 * page_size + 2 * page_size + p_off]));
        const long long col = __ldg(&tlb_cols[r_tlb]);
        const float inv_sm = bf16_to_f32(__ldg(&inv_smid_cold[col]));

        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                float dg = warp_sum(acc_g[m]) * sc_g;
                float du = warp_sum(acc_u[m]) * sc_u;
                if (lane == 0) {
                    float g_new = round_bf16(g_cold_ws[m * n_cold + col] + round_bf16(dg));
                    float u_new = round_bf16(u_cold_ws[m * n_cold + col] + round_bf16(du));
                    float h_new = round_bf16(round_bf16(apply_act(g_new, act_kind)) * u_new);
                    h_cold_scaled_ws[m * n_cold + col] = round_bf16(h_new * inv_sm);
                    h_tlb_down_ws[m * c_act + r_tlb] = h_new * sc_d;
                }
            }
        }
    }
}

extern "C" __global__ void phase_c_stage2_down_kernel(
    const float* __restrict__ h_hot_ws,
    const int8_t* __restrict__ q_down_hot,
    const bf16_t* __restrict__ sc_down_hot,
    const float* __restrict__ h_cold_scaled_ws,
    const unsigned short* __restrict__ p_down_cold,
    const bf16_t* __restrict__ mn_down_cold,
    const bf16_t* __restrict__ sc_down_cold,
    const float* __restrict__ z_bridge_ws,
    const bf16_t* __restrict__ V_bridge,
    bf16_t* __restrict__ out_bf16,
    int M,
    int d_model,
    int n_hot,
    int n_cold,
    int n_pk_down,
    int n_grp_down,
    int g5_down,
    int bridge_rank
) {
    extern __shared__ float smem[];
    float* s_h = smem;
    float* s_sum5 = smem + M * TILE_D;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int out_o = blockIdx.x * warps_per_block + warp_id;

    float acc_hot[MAX_M];
    float acc_cold[MAX_M];
    float acc_br[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc_hot[m] = 0.0f;
        acc_cold[m] = 0.0f;
        acc_br[m] = 0.0f;
    }

    if (out_o < d_model && bridge_rank > 0) {
        for (int r = lane; r < bridge_rank; r += 32) {
            float v_val = bf16_to_f32(__ldg(&V_bridge[(size_t)r * d_model + out_o]));
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < M) {
                    acc_br[m] += z_bridge_ws[m * bridge_rank + r] * v_val;
                }
            }
        }
    }

    const int hot_tiles = (n_hot + TILE_D - 1) / TILE_D;
    for (int tile = 0; tile < hot_tiles; ++tile) {
        const int base_j = tile * TILE_D;
        const int j_lim = (base_j + TILE_D < n_hot) ? TILE_D : (n_hot - base_j);
        for (int idx = tid; idx < j_lim; idx += blockDim.x) {
            for (int m = 0; m < M; ++m) {
                s_h[m * TILE_D + idx] = h_hot_ws[m * n_hot + base_j + idx];
            }
        }
        __syncthreads();

        if (out_o < d_model) {
            const int8_t* rd = &q_down_hot[(size_t)out_o * n_hot + base_j];
            int idx = lane * 4;
            if ((n_hot & 3) == 0) {
                for (; idx + 3 < j_lim; idx += 128) {
                    const int p4 = __ldg((const int*)&rd[idx]);
                    const float w0 = (float)((int8_t)(p4 & 0xFF));
                    const float w1 = (float)((int8_t)((p4 >> 8) & 0xFF));
                    const float w2 = (float)((int8_t)((p4 >> 16) & 0xFF));
                    const float w3 = (float)((int8_t)((p4 >> 24) & 0xFF));
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            const float* sh = &s_h[m * TILE_D + idx];
                            acc_hot[m] += w0 * sh[0] + w1 * sh[1] + w2 * sh[2] + w3 * sh[3];
                        }
                    }
                }
            } else {
                for (; idx + 3 < j_lim; idx += 128) {
                    const float w0 = (float)__ldg(&rd[idx + 0]);
                    const float w1 = (float)__ldg(&rd[idx + 1]);
                    const float w2 = (float)__ldg(&rd[idx + 2]);
                    const float w3 = (float)__ldg(&rd[idx + 3]);
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            const float* sh = &s_h[m * TILE_D + idx];
                            acc_hot[m] += w0 * sh[0] + w1 * sh[1] + w2 * sh[2] + w3 * sh[3];
                        }
                    }
                }
            }
            for (int k = (j_lim & ~3) + lane; k < j_lim; k += 32) {
                float fw = (float)__ldg(&rd[k]);
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        acc_hot[m] += fw * s_h[m * TILE_D + k];
                    }
                }
            }
        }
        __syncthreads();
    }

    const int d_total_cold = n_pk_down * 5;
    const int cold_tiles = (d_total_cold + TILE_D - 1) / TILE_D;
    for (int tile = 0; tile < cold_tiles; ++tile) {
        const int base_c = tile * TILE_D;
        const int base_pk = tile * TILE_PK;
        for (int idx = tid; idx < TILE_D; idx += blockDim.x) {
            const int c_col = base_c + idx;
            for (int m = 0; m < M; ++m) {
                s_h[m * TILE_D + idx] = (c_col < n_cold) ? h_cold_scaled_ws[m * n_cold + c_col] : 0.0f;
            }
        }
        __syncthreads();

        for (int pk_l = tid; pk_l < TILE_PK; pk_l += blockDim.x) {
            const int i5 = pk_l * 5;
            for (int m = 0; m < M; ++m) {
                const float* sh = &s_h[m * TILE_D + i5];
                s_sum5[m * TILE_PK + pk_l] = sh[0] + sh[1] + sh[2] + sh[3] + sh[4];
            }
        }
        __syncthreads();

        if (out_o < d_model) {
            const int pk_end = (base_pk + TILE_PK < n_pk_down) ? TILE_PK : (n_pk_down - base_pk);
            const unsigned short* row_p = &p_down_cold[(size_t)out_o * n_pk_down + base_pk];
            const bf16_t* mn_row = &mn_down_cold[(size_t)out_o * n_grp_down];
            const bf16_t* sc_row = &sc_down_cold[(size_t)out_o * n_grp_down];

            int pk_done = 0;
            if ((n_pk_down & 1) == 0) {
                pk_done = pk_end & ~1;
                for (int pk_l = lane * 2; pk_l + 1 < pk_end; pk_l += 64) {
                    const unsigned int upk2 = __ldg((const unsigned int*)&row_p[pk_l]);
                    const int grp0 = (base_pk + pk_l) / g5_down;
                    const int grp1 = (base_pk + pk_l + 1) / g5_down;
                    const float sc0 = bf16_to_f32(__ldg(&sc_row[grp0]));
                    const float mn0 = bf16_to_f32(__ldg(&mn_row[grp0]));
                    const float sc1 = (grp1 == grp0) ? sc0 : bf16_to_f32(__ldg(&sc_row[grp1]));
                    const float mn1 = (grp1 == grp0) ? mn0 : bf16_to_f32(__ldg(&mn_row[grp1]));

                    const unsigned int ua = upk2 & 0xFFFFU;
                    const unsigned int ub = upk2 >> 16;
                    const float q0 = (float)(ua & 7U), q1 = (float)((ua >> 3) & 7U), q2 = (float)((ua >> 6) & 7U), q3 = (float)((ua >> 9) & 7U), q4 = (float)((ua >> 12) & 7U);
                    const float q5 = (float)(ub & 7U), q6 = (float)((ub >> 3) & 7U), q7 = (float)((ub >> 6) & 7U), q8 = (float)((ub >> 9) & 7U), q9 = (float)((ub >> 12) & 7U);

                    const int i5 = pk_l * 5;
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            const float* sh = &s_h[m * TILE_D + i5];
                            const float sum5_0 = s_sum5[m * TILE_PK + pk_l];
                            const float sum5_1 = s_sum5[m * TILE_PK + pk_l + 1];
                            const float dot0 = q0 * sh[0] + q1 * sh[1] + q2 * sh[2] + q3 * sh[3] + q4 * sh[4];
                            const float dot1 = q5 * sh[5] + q6 * sh[6] + q7 * sh[7] + q8 * sh[8] + q9 * sh[9];
                            acc_cold[m] += (sc0 * dot0 + mn0 * sum5_0) + (sc1 * dot1 + mn1 * sum5_1);
                        }
                    }
                }
            }

            for (int pk_l = pk_done + lane; pk_l < pk_end; pk_l += 32) {
                const int pk_global = base_pk + pk_l;
                const int grp = pk_global / g5_down;
                const unsigned int upk = (unsigned int)__ldg(&row_p[pk_l]);
                const float sc = bf16_to_f32(__ldg(&sc_row[grp]));
                const float mn = bf16_to_f32(__ldg(&mn_row[grp]));

                const float q0 = (float)(upk & 7U);
                const float q1 = (float)((upk >> 3) & 7U);
                const float q2 = (float)((upk >> 6) & 7U);
                const float q3 = (float)((upk >> 9) & 7U);
                const float q4 = (float)((upk >> 12) & 7U);

                const int i5 = pk_l * 5;
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const float* sh = &s_h[m * TILE_D + i5];
                        const float sum5 = s_sum5[m * TILE_PK + pk_l];
                        const float dot_q = q0 * sh[0] + q1 * sh[1] + q2 * sh[2] + q3 * sh[3] + q4 * sh[4];
                        acc_cold[m] += sc * dot_q + mn * sum5;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (out_o < d_model) {
        const float sc_h = bf16_to_f32(__ldg(&sc_down_hot[out_o]));
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                float yh = round_bf16(warp_sum(acc_hot[m]) * sc_h);
                float yc = round_bf16(warp_sum(acc_cold[m]));
                float yb = (bridge_rank > 0) ? round_bf16(warp_sum(acc_br[m])) : 0.0f;
                if (lane == 0) {
                    out_bf16[m * d_model + out_o] = f32_to_bf16(yh + yc + yb);
                }
            }
        }
    }
}

extern "C" __global__ void phase_c_tlb_down_add_kernel(
    const float* __restrict__ h_tlb_down_ws,
    const uint8_t* __restrict__ tlb_packed_q4,
    bf16_t* __restrict__ out_bf16,
    int M,
    int d_model,
    int d_even_half,
    int page_size,
    int n_act
) {
    __shared__ float s_red[4][64 * MAX_M];
    const int tx = threadIdx.x & 31;
    const int ty = threadIdx.x >> 5;
    const int byte_col = blockIdx.x * 32 + tx;
    const int c_act = n_act * page_size;

    float acc0[MAX_M];
    float acc1[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc0[m] = 0.0f;
        acc1[m] = 0.0f;
    }

    if (byte_col < d_even_half) {
        const size_t slot_stride = (size_t)3 * page_size * d_even_half;
        const size_t plane2_offset = (size_t)2 * page_size * d_even_half;
        for (int r = ty; r < c_act; r += 4) {
            const int slot = r / page_size;
            const int p_off = r % page_size;
            const uint8_t b = __ldg(&tlb_packed_q4[slot * slot_stride + plane2_offset + (size_t)p_off * d_even_half + byte_col]);
            const float w0 = (float)((int)(b & 0x0FU) - 8);
            const float w1 = (float)((int)((b >> 4) & 0x0FU) - 8);
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < M) {
                    const float hs = h_tlb_down_ws[m * c_act + r];
                    acc0[m] += hs * w0;
                    acc1[m] += hs * w1;
                }
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            s_red[ty][(tx * 2 + 0) * MAX_M + m] = acc0[m];
            s_red[ty][(tx * 2 + 1) * MAX_M + m] = acc1[m];
        }
    }
    __syncthreads();

    if (ty == 0 && byte_col < d_even_half) {
        const int o0 = byte_col * 2 + 0;
        const int o1 = byte_col * 2 + 1;
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                #pragma unroll
                for (int s = 0; s < 4; ++s) {
                    sum0 += s_red[s][(tx * 2 + 0) * MAX_M + m];
                    sum1 += s_red[s][(tx * 2 + 1) * MAX_M + m];
                }
                if (o0 < d_model) {
                    float cur0 = bf16_to_f32(out_bf16[m * d_model + o0]);
                    out_bf16[m * d_model + o0] = f32_to_bf16(cur0 + round_bf16(sum0));
                }
                if (o1 < d_model) {
                    float cur1 = bf16_to_f32(out_bf16[m * d_model + o1]);
                    out_bf16[m * d_model + o1] = f32_to_bf16(cur1 + round_bf16(sum1));
                }
            }
        }
    }
}
"""

_COMPILED_MODULES_BY_DEVICE: Dict[int, Dict[str, Any]] = {}
_SHARED_WORKSPACES_BY_DEVICE: Dict[int, "PhaseCWorkspace"] = {}


def detect_act_kind(act_fn: nn.Module) -> int:
    """Maps standard HuggingFace/PyTorch gated-MLP activations to Phase C device `act_kind`."""
    cls_name = type(act_fn).__name__.lower()
    if "silu" in cls_name or "swish" in cls_name:
        return 2
    if "gelutanh" in cls_name or "newgelu" in cls_name or "fastgelu" in cls_name:
        return 0
    approx = str(getattr(act_fn, "approximate", "none")).lower()
    if approx == "tanh":
        return 0
    if "gelu" in cls_name:
        return 1
    return -1


def get_or_compile_phase_c_kernels(device: torch.device) -> Dict[str, Any]:
    """Compiles all 5 Phase C CUDA kernels once per CUDA device directly to native sm_XX CUBIN."""
    dev_idx = 0 if device.index is None else int(device.index)
    if dev_idx in _COMPILED_MODULES_BY_DEVICE:
        return _COMPILED_MODULES_BY_DEVICE[dev_idx]

    from torch.cuda._utils import _cuda_load_module, _get_gpu_rtc_library, _get_gpu_runtime_library

    with torch.cuda.device(dev_idx):
        libnvrtc = _get_gpu_rtc_library()
        libcuda = _get_gpu_runtime_library()
        props = torch.cuda.get_device_properties(dev_idx)
        arch = f"--gpu-architecture=sm_{props.major}{props.minor}".encode("utf-8")
        opts = [arch, b"--use_fast_math", b"-default-device"]
        opts_arr = (ctypes.c_char_p * len(opts))(*opts)

        src_bytes = PHASE_C_CUDA_SOURCE.encode("utf-8")
        prog = ctypes.c_void_p()
        res = libnvrtc.nvrtcCreateProgram(ctypes.byref(prog), src_bytes, b"phase_c_kernels.cu", 0, None, None)
        if res != 0:
            raise RuntimeError(f"nvrtcCreateProgram failed with code {res}")

        res = libnvrtc.nvrtcCompileProgram(prog, len(opts), opts_arr)
        if res != 0:
            log_sz = ctypes.c_size_t()
            libnvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
            log_buf = ctypes.create_string_buffer(log_sz.value)
            libnvrtc.nvrtcGetProgramLog(prog, log_buf)
            libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            raise RuntimeError(f"Phase C NVRTC compilation failed:\n{log_buf.value.decode('utf-8', errors='replace')}")

        cubin_sz = ctypes.c_size_t()
        libnvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(cubin_sz))
        cubin_buf = ctypes.create_string_buffer(cubin_sz.value)
        libnvrtc.nvrtcGetCUBIN(prog, cubin_buf)
        libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

        kernel_names = [
            "phase_c_stage1_kernel",
            "phase_c_router_topk_kernel",
            "phase_c_tlb_gu_upgrade_kernel",
            "phase_c_stage2_down_kernel",
            "phase_c_tlb_down_add_kernel",
        ]
        kdict = _cuda_load_module(cubin_buf.raw, kernel_names)
        kdict["phase_c_stage1_kernel"].set_shared_memory_config(65536)
        kdict["phase_c_tlb_gu_upgrade_kernel"].set_shared_memory_config(65536)
        kdict["phase_c_stage2_down_kernel"].set_shared_memory_config(65536)
        kdict["libcuda"] = libcuda

    _COMPILED_MODULES_BY_DEVICE[dev_idx] = kdict
    return kdict


class PhaseCWorkspace:
    """Shared L2-resident scratch workspace across all layers on a single CUDA device (~1 MB total)."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.version: int = 0
        self.max_d_model: int = 0
        self.max_n_hot: int = 0
        self.max_n_cold: int = 0
        self.max_c_act: int = 0
        self.max_bridge_rank: int = 0
        self.max_pages: int = 0

        self.x_scaled = torch.empty((0,), dtype=torch.float32, device=device)
        self.h_hot = torch.empty((0,), dtype=torch.float32, device=device)
        self.g_cold = torch.empty((0,), dtype=torch.float32, device=device)
        self.u_cold = torch.empty((0,), dtype=torch.float32, device=device)
        self.h_cold_scaled = torch.empty((0,), dtype=torch.float32, device=device)
        self.h_tlb_down = torch.empty((0,), dtype=torch.float32, device=device)
        self.z_bridge = torch.empty((0,), dtype=torch.float32, device=device)
        self.page_scores = torch.empty((0,), dtype=torch.float32, device=device)
        self.req_tmp = torch.empty((0,), dtype=torch.int32, device=device)

    def ensure_capacity(
        self,
        d_model: int,
        n_hot: int,
        n_cold: int,
        c_act: int,
        bridge_rank: int,
        num_pages: int,
    ) -> None:
        need_realloc = (
            d_model > self.max_d_model
            or n_hot > self.max_n_hot
            or n_cold > self.max_n_cold
            or c_act > self.max_c_act
            or bridge_rank > self.max_bridge_rank
            or num_pages > self.max_pages
        )
        if not need_realloc:
            return

        self.max_d_model = max(self.max_d_model, int(d_model), 64)
        self.max_n_hot = max(self.max_n_hot, int(n_hot), 64)
        self.max_n_cold = max(self.max_n_cold, int(n_cold), 64)
        self.max_c_act = max(self.max_c_act, int(c_act), 64)
        self.max_bridge_rank = max(self.max_bridge_rank, int(bridge_rank), 16)
        self.max_pages = max(self.max_pages, int(num_pages), 64)

        m = MAX_FUSED_DECODE_TOKENS
        self.x_scaled = torch.zeros((m, self.max_d_model), dtype=torch.float32, device=self.device)
        self.h_hot = torch.zeros((m, self.max_n_hot), dtype=torch.float32, device=self.device)
        self.g_cold = torch.zeros((m, self.max_n_cold), dtype=torch.float32, device=self.device)
        self.u_cold = torch.zeros((m, self.max_n_cold), dtype=torch.float32, device=self.device)
        self.h_cold_scaled = torch.zeros((m, self.max_n_cold), dtype=torch.float32, device=self.device)
        self.h_tlb_down = torch.zeros((m, self.max_c_act), dtype=torch.float32, device=self.device)
        self.z_bridge = torch.zeros((m, self.max_bridge_rank), dtype=torch.float32, device=self.device)
        self.page_scores = torch.zeros((m, self.max_pages), dtype=torch.float32, device=self.device)
        self.req_tmp = torch.zeros((self.max_pages,), dtype=torch.int32, device=self.device)
        self.version += 1

    def zero_page_scores(self) -> None:
        if self.page_scores.numel() > 0:
            self.page_scores.zero_()

    @property
    def total_vram_bytes(self) -> int:
        return sum(
            int(t.numel() * t.element_size())
            for t in (
                self.x_scaled,
                self.h_hot,
                self.g_cold,
                self.u_cold,
                self.h_cold_scaled,
                self.h_tlb_down,
                self.z_bridge,
                self.page_scores,
                self.req_tmp,
            )
        )


def get_shared_phase_c_workspace(device: torch.device) -> PhaseCWorkspace:
    dev_idx = 0 if device.index is None else int(device.index)
    if dev_idx not in _SHARED_WORKSPACES_BY_DEVICE:
        _SHARED_WORKSPACES_BY_DEVICE[dev_idx] = PhaseCWorkspace(torch.device(f"cuda:{dev_idx}"))
    return _SHARED_WORKSPACES_BY_DEVICE[dev_idx]


class PhaseCLayerLauncher:
    """Pre-bound `cuLaunchKernel` dispatcher for a single `BitTieredTLBMLP` layer (~3 us CPU overhead)."""

    def __init__(self, mlp: Any, act_kind: int) -> None:
        self.mlp = mlp
        self.act_kind = int(act_kind)
        self.device = mlp.q_gu_hot_i8.device
        self.kdict = get_or_compile_phase_c_kernels(self.device)
        self.libcuda = self.kdict["libcuda"]
        self.k1 = self.kdict["phase_c_stage1_kernel"]
        self.kr = self.kdict["phase_c_router_topk_kernel"]
        self.ktlb = self.kdict["phase_c_tlb_gu_upgrade_kernel"]
        self.k2 = self.kdict["phase_c_stage2_down_kernel"]
        self.ktd = self.kdict["phase_c_tlb_down_add_kernel"]

        self.ws = get_shared_phase_c_workspace(self.device)
        max_c_act = mlp.tlb_cache.tlb_slots * mlp.page_size
        br = int(mlp.U_bridge.shape[1]) if mlp.U_bridge.ndim == 2 else 0
        self.ws.ensure_capacity(
            d_model=mlp.d_model,
            n_hot=mlp.n_hot,
            n_cold=mlp.n_cold,
            c_act=max_c_act,
            bridge_rank=br,
            num_pages=mlp.num_pages,
        )

        self.n_pk_gu = int(mlp.p_gu_cold_i16.shape[1])
        self.n_grp_gu = int(mlp.mn_gu_cold.shape[1])
        self.g5_gu = int(mlp.gu_group_size // 5)
        self.n_pk_down = int(mlp.p_down_cold_i16.shape[1])
        self.n_grp_down = int(mlp.mn_down_cold.shape[1])
        self.g5_down = int(mlp.down_group_size // 5)
        self.br = br
        self.d_even_half = int(mlp.tlb_cache.d_even_half)

        total_s1_tasks = mlp.n_hot + mlp.n_cold + self.br
        self.grid_s1_x = (total_s1_tasks + 3) // 4
        self.grid_s2_x = (mlp.d_model + 3) // 4
        self.grid_td_x = (self.d_even_half + 31) // 32

        self._x_ptr = ctypes.c_void_p(0)
        self._out_ptr = ctypes.c_void_p(0)
        self._tlb_q4_ptr = ctypes.c_void_p(0)
        self._tlb_sc_ptr = ctypes.c_void_p(0)
        self._tlb_col_ptr = ctypes.c_void_p(0)
        self._req_ptr = ctypes.c_void_p(0)

        self._ws_x_ptr = ctypes.c_void_p(0)
        self._ws_h_hot_ptr = ctypes.c_void_p(0)
        self._ws_g_cold_ptr = ctypes.c_void_p(0)
        self._ws_u_cold_ptr = ctypes.c_void_p(0)
        self._ws_hc_sc_ptr = ctypes.c_void_p(0)
        self._ws_htlb_ptr = ctypes.c_void_p(0)
        self._ws_zbr_ptr = ctypes.c_void_p(0)
        self._ws_ps_ptr = ctypes.c_void_p(0)

        self._M = ctypes.c_int(1)
        self._vm_paged = ctypes.c_int(1)
        self._n_act = ctypes.c_int(0)
        self._k_req = ctypes.c_int(mlp.decode_req_pages)
        self._do_router = ctypes.c_int(1)
        self._bound_ws_version: int = -1

        self._rebind_workspace_ptrs()
        self._build_arg_arrays()

    def _rebind_workspace_ptrs(self) -> None:
        self._ws_x_ptr.value = self.ws.x_scaled.data_ptr()
        self._ws_h_hot_ptr.value = self.ws.h_hot.data_ptr()
        self._ws_g_cold_ptr.value = self.ws.g_cold.data_ptr()
        self._ws_u_cold_ptr.value = self.ws.u_cold.data_ptr()
        self._ws_hc_sc_ptr.value = self.ws.h_cold_scaled.data_ptr()
        self._ws_htlb_ptr.value = self.ws.h_tlb_down.data_ptr()
        self._ws_zbr_ptr.value = self.ws.z_bridge.data_ptr()
        self._ws_ps_ptr.value = self.ws.page_scores.data_ptr()
        self._bound_ws_version = self.ws.version

    def _build_arg_arrays(self) -> None:
        mlp = self.mlp

        def _pack(spec: List[Any]) -> Tuple[List[Any], Any]:
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
                else:
                    raise TypeError(f"Unsupported kernel arg type: {type(item)}")
            return refs, arr

        self._s1_refs, self._s1_args = _pack([
            self._x_ptr,
            mlp.inv_s_in,
            mlp.q_gu_hot_i8,
            mlp.sc_gu_hot,
            mlp.inv_s_mid_hot,
            mlp.p_gu_cold_i16,
            mlp.mn_gu_cold,
            mlp.sc_gu_cold,
            mlp.inv_s_mid_cold,
            mlp.col_err_weight,
            mlp.U_bridge,
            self._ws_x_ptr,
            self._ws_h_hot_ptr,
            self._ws_g_cold_ptr,
            self._ws_u_cold_ptr,
            self._ws_hc_sc_ptr,
            self._ws_zbr_ptr,
            self._ws_ps_ptr,
            self._M,
            mlp.d_model,
            mlp.n_hot,
            mlp.n_cold,
            self.n_pk_gu,
            self.n_grp_gu,
            self.g5_gu,
            self.br,
            mlp.page_size,
            mlp.num_pages,
            self.act_kind,
            self._vm_paged,
        ])

        self._kr_refs, self._kr_args = _pack([
            self._ws_ps_ptr,
            self._req_ptr,
            self._M,
            mlp.num_pages,
            self._k_req,
        ])

        self._ktlb_refs, self._ktlb_args = _pack([
            self._ws_x_ptr,
            self._tlb_q4_ptr,
            self._tlb_sc_ptr,
            self._tlb_col_ptr,
            self._ws_g_cold_ptr,
            self._ws_u_cold_ptr,
            mlp.inv_s_mid_cold,
            self._ws_hc_sc_ptr,
            self._ws_htlb_ptr,
            self._ws_ps_ptr,
            self._req_ptr,
            self._M,
            mlp.d_model,
            self.d_even_half,
            mlp.page_size,
            mlp.n_cold,
            self._n_act,
            mlp.num_pages,
            self._k_req,
            self.act_kind,
            self._do_router,
        ])

        self._s2_refs, self._s2_args = _pack([
            self._ws_h_hot_ptr,
            mlp.q_down_hot_i8,
            mlp.sc_down_hot,
            self._ws_hc_sc_ptr,
            mlp.p_down_cold_i16,
            mlp.mn_down_cold,
            mlp.sc_down_cold,
            self._ws_zbr_ptr,
            mlp.V_bridge,
            self._out_ptr,
            self._M,
            mlp.d_model,
            mlp.n_hot,
            mlp.n_cold,
            self.n_pk_down,
            self.n_grp_down,
            self.g5_down,
            self.br,
        ])

        self._ktd_refs, self._ktd_args = _pack([
            self._ws_htlb_ptr,
            self._tlb_q4_ptr,
            self._out_ptr,
            self._M,
            mlp.d_model,
            self.d_even_half,
            mlp.page_size,
            self._n_act,
        ])

    def forward_decode(self, flat_x_bf16: torch.Tensor, orig_shape: torch.Size, orig_dtype: torch.dtype) -> torch.Tensor:
        if self._bound_ws_version != self.ws.version:
            self._rebind_workspace_ptrs()

        mlp = self.mlp
        if not flat_x_bf16.is_contiguous():
            flat_x_bf16 = flat_x_bf16.contiguous()
        M = int(flat_x_bf16.shape[0])
        out_bf16 = torch.empty_like(flat_x_bf16)

        stream_h = torch.cuda.current_stream(self.device)._as_parameter_
        launch = self.libcuda.cuLaunchKernel

        vm_paged = 1 if (mlp.mode == "vm_paged" and mlp.tlb_cache.tlb_slots > 0) else 0
        # Re-bind bridge pointers dynamically in case U_bridge / V_bridge tensors were updated after init
        self._x_ptr.value = flat_x_bf16.data_ptr()
        self._out_ptr.value = out_bf16.data_ptr()
        self._M.value = M
        self._vm_paged.value = vm_paged

        smem_bytes = M * (1280 + 256) * 4
        res = launch(self.k1.func, self.grid_s1_x, 1, 1, 128, 1, 1, smem_bytes, stream_h, self._s1_args, None)
        if res != 0:
            raise RuntimeError(f"phase_c_stage1_kernel launch failed with CUDA error {res}")

        n_act = 0
        if vm_paged:
            is_prefill_step = (orig_shape[1] > 1) if len(orig_shape) == 3 else (M > 1)
            tlb = mlp.tlb_cache
            use_spec_batch = (
                (mlp.tlb_strategy == "speculative_tlb" and not is_prefill_step)
                or (mlp.tlb_strategy == "amortized_spec_tlb")
            )
            if (
                use_spec_batch
                and mlp.batch_controller is not None
                and tlb._num_valid > 0
            ):
                ctrl = mlp.batch_controller
                if mlp.controller_slot_idx == 0:
                    ctrl.drain_and_prefetch_all()
                n_act = tlb._num_valid
                c_act = n_act * mlp.page_size
                self._tlb_q4_ptr.value = tlb.tlb_packed_q4.data_ptr()
                self._tlb_sc_ptr.value = tlb.tlb_scales.data_ptr()
                self._tlb_col_ptr.value = tlb.tlb_col_indices.data_ptr()
                self._req_ptr.value = ctrl.gpu_req_matrix[mlp.controller_slot_idx].data_ptr()
                self._n_act.value = n_act
                self._k_req.value = mlp.decode_req_pages
                self._do_router.value = 1

                grid_tlb_x = 1 + ((c_act + 3) // 4)
                res = launch(self.ktlb.func, grid_tlb_x, 1, 1, 128, 1, 1, M * 1280 * 4, stream_h, self._ktlb_args, None)
                if res != 0:
                    raise RuntimeError(f"phase_c_tlb_gu_upgrade_kernel launch failed with CUDA error {res}")

                if mlp.controller_slot_idx == ctrl.num_layers - 1:
                    if ctrl.dma_stream is not None:
                        cur = torch.cuda.current_stream(self.device)
                        ctrl.dma_stream.wait_stream(cur)
                        with torch.cuda.stream(ctrl.dma_stream):
                            ctrl.cpu_pinned_req_matrix.copy_(ctrl.gpu_req_matrix, non_blocking=True)
                            if ctrl.d2h_event is not None:
                                ctrl.d2h_event.record(ctrl.dma_stream)
                    else:
                        ctrl.cpu_pinned_req_matrix.copy_(ctrl.gpu_req_matrix)
                    ctrl.has_pending_batch = True
            else:
                k_req = tlb.tlb_slots if (is_prefill_step or tlb._num_valid < tlb.tlb_slots) else mlp.decode_req_pages
                if mlp.tlb_strategy == "stateless_dma":
                    k_req = mlp.decode_req_pages
                req_buf = self.ws.req_tmp[:k_req]
                self._req_ptr.value = req_buf.data_ptr()
                self._k_req.value = k_req
                res = launch(self.kr.func, 1, 1, 1, 64, 1, 1, 0, stream_h, self._kr_args, None)
                if res != 0:
                    raise RuntimeError(f"phase_c_router_topk_kernel launch failed with CUDA error {res}")

                if mlp.tlb_strategy == "stateless_dma":
                    _, _, _, n_act = tlb.request_pages_stateless_dma(req_buf)
                else:
                    _, _, _, n_act = tlb.request_pages_sync_tlb(req_buf)

                if n_act > 0:
                    c_act = n_act * mlp.page_size
                    self._tlb_q4_ptr.value = tlb.tlb_packed_q4.data_ptr()
                    self._tlb_sc_ptr.value = tlb.tlb_scales.data_ptr()
                    self._tlb_col_ptr.value = tlb.tlb_col_indices.data_ptr()
                    self._n_act.value = n_act
                    self._do_router.value = 0
                    res = launch(self.ktlb.func, (c_act + 3) // 4, 1, 1, 128, 1, 1, M * 1280 * 4, stream_h, self._ktlb_args, None)
                    if res != 0:
                        raise RuntimeError(f"phase_c_tlb_gu_upgrade_kernel (sync) failed with CUDA error {res}")

        res = launch(self.k2.func, self.grid_s2_x, 1, 1, 128, 1, 1, smem_bytes, stream_h, self._s2_args, None)
        if res != 0:
            raise RuntimeError(f"phase_c_stage2_down_kernel launch failed with CUDA error {res}")

        if vm_paged and n_act > 0:
            res = launch(self.ktd.func, self.grid_td_x, 1, 1, 128, 1, 1, 0, stream_h, self._ktd_args, None)
            if res != 0:
                raise RuntimeError(f"phase_c_tlb_down_add_kernel launch failed with CUDA error {res}")

        out = out_bf16.view(*orig_shape)
        return out if out.dtype == orig_dtype else out.to(orig_dtype)
