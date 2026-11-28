"""Fast NVRTC CUDA Core Kernels & Speculative State Rollback for Sparse MoE (`Qwen3-Next` & `Gemma-4`).

Why this module exists:
1. On Windows (`transformers` without Linux-only `fla` / `causal_conv1d` Triton wheels), `Qwen3NextGatedDeltaNet`
   falls back to Python per-token/chunked loops, un-fused `causal_conv1d`, `fix_query_key_value_ordering`,
   `Qwen3NextRMSNormGated`, `Qwen3NextRMSNorm`, and `Qwen3NextTopKRouter` (~1,500 tiny CUDA launches/token,
   adding ~70+ ms of pure CPU launch overhead per pass on `Qwen3-Next-80B-A3B`). Similarly, `Gemma-4-26B-A4B`
   executes ~330 Python `Gemma4RMSNorm` + 30 `Gemma4TextRouter` softmax/topk dispatches per token.
2. Upstream `LinearAttentionLayer.crop()` in `transformers.cache_utils` only rolls back `conv_states` and leaves
   `recurrent_states` corrupted if speculative decoding rejects draft tokens (`n_reject > 0`).

This module provides:
- `qwen3_next_gated_delta_rule_cuda`: Register-resident NVRTC `sm_89` CUDA C++ kernel with native GQA `H_k`
  indexing (zero `repeat_interleave`) and optional fused `Qwen3NextRMSNormGated` epilogue.
- `qwen3_next_gdn_prep_conv_cuda`: Single-launch NVRTC kernel fusing `fix_query_key_value_ordering` + in-place
  or speculative-recorded `conv_state` update + 4-tap depthwise `conv1d` + `silu` + `beta`/`g` discretization.
- `nvm_rmsnorm_bf16_cuda` & `qwen3_next_rmsnorm_gated_cuda`: Single-launch FP32-accumulated NVRTC RMSNorm kernels
  for `Qwen3NextRMSNorm`, `Qwen3NextRMSNormGated`, and `Gemma4RMSNorm`.
- `moe_router_topk_softmax_cuda`: Single-launch FP32 `softmax` + deterministic `topk` + probability normalization
  for `Qwen3NextTopKRouter`, `Gemma4TextRouter`, and Phase G2 `LookaheadRouterBank`.
- Exact `O(1)` speculative state rollback (`step_recurrent_states` + zero-alloc `step_conv_states` crop) for
  `0.0-ulp` lossless parity between `M=1` and `M>1`.
"""

from __future__ import annotations

import ctypes
import types
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

_QWEN3_NEXT_CUDA_SRC = r"""
typedef unsigned short bf16_t;

__device__ __forceinline__ float bf16_to_f32(bf16_t u) {
    return __uint_as_float(((unsigned int)u) << 16);
}

__device__ __forceinline__ bf16_t f32_to_bf16(float f) {
    unsigned int x = __float_as_uint(f);
    unsigned int lsb = (x >> 16) & 1U;
    x += 0x7FFFU + lsb;
    return (bf16_t)(x >> 16);
}

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus_f32(float x) {
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

extern "C" __global__ void qwen3_next_gated_delta_rule_kernel(
    const bf16_t* __restrict__ q_bf16,             // [B, S, H_k, K]
    const bf16_t* __restrict__ k_bf16,             // [B, S, H_k, K]
    const bf16_t* __restrict__ v_bf16,             // [B, S, H, V]
    const float* __restrict__ g_f32,               // [B, S, H]
    const bf16_t* __restrict__ beta_bf16,          // [B, S, H]
    const float* init_state_f32,                   // [B, H, K, V] or nullptr (may alias final_state_f32!)
    const bf16_t* __restrict__ z_bf16,             // [B, S, H, V] or nullptr (optional fused RMSNormGated)
    const bf16_t* __restrict__ norm_w_bf16,        // [V] or nullptr (optional fused RMSNormGated)
    bf16_t* __restrict__ out_bf16,                 // [B, S, H, V]
    float* final_state_f32,                        // [B, H, K, V] or nullptr (may alias init_state_f32!)
    float* __restrict__ step_states_f32,           // [S, B, H, K, V] or nullptr (for M>1 speculative rollback)
    int B,
    int S,
    int H,
    int H_k,
    int K,
    int V,
    int has_init_state,
    int use_qk_l2norm,
    float inv_sqrt_k,
    float norm_eps
) {
    const int bh = blockIdx.x;
    if (bh >= B * H) return;
    const int b = bh / H;
    const int h = bh % H;
    const int n_rep = (H_k > 0) ? (H / H_k) : 1;
    const int kh = (n_rep > 1) ? (h / n_rep) : h;
    const int tid = threadIdx.x;

    __shared__ float sq[128];
    __shared__ float sk[128];
    __shared__ float s_red_q[8];
    __shared__ float s_red_k[8];
    __shared__ float s_inv_q;
    __shared__ float s_inv_k;
    __shared__ float s_inv_v;

    // Each thread tid < V holds column `tid` of `state[0..K-1, tid]` (`K <= 128`) in registers!
    float s_col[128];
    const long long state_bh_base = (long long)bh * (long long)(K * V);
    if (tid < V) {
        if (has_init_state && init_state_f32 != nullptr) {
            for (int d = 0; d < K; ++d) {
                s_col[d] = init_state_f32[state_bh_base + (long long)d * V + tid];
            }
        } else {
            for (int d = 0; d < K; ++d) {
                s_col[d] = 0.0f;
            }
        }
    }

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;
    const int fuse_gated_norm = (z_bf16 != nullptr && norm_w_bf16 != nullptr) ? 1 : 0;
    const float nw_val = (fuse_gated_norm && tid < V) ? bf16_to_f32(norm_w_bf16[tid]) : 1.0f;

    for (int t = 0; t < S; ++t) {
        const long long bth = ((long long)b * S + t) * H + h;
        const long long btkh = ((long long)b * S + t) * H_k + kh;
        const long long qk_base = btkh * K;
        const long long v_base = bth * V;

        float q_val = 0.0f;
        float k_val = 0.0f;
        if (tid < K) {
            q_val = bf16_to_f32(q_bf16[qk_base + tid]);
            k_val = bf16_to_f32(k_bf16[qk_base + tid]);
            sq[tid] = q_val;
            sk[tid] = k_val;
        }

        if (use_qk_l2norm) {
            float q2 = q_val * q_val;
            float k2 = k_val * k_val;
            for (int mask = 16; mask > 0; mask >>= 1) {
                q2 += __shfl_xor_sync(0xffffffffu, q2, mask);
                k2 += __shfl_xor_sync(0xffffffffu, k2, mask);
            }
            if (lane == 0) {
                s_red_q[warp_id] = q2;
                s_red_k[warp_id] = k2;
            }
            __syncthreads();
            if (tid == 0) {
                float tot_q2 = 0.0f;
                float tot_k2 = 0.0f;
                for (int w = 0; w < num_warps; ++w) {
                    tot_q2 += s_red_q[w];
                    tot_k2 += s_red_k[w];
                }
                s_inv_q = rsqrtf(tot_q2 + 1e-6f) * inv_sqrt_k;
                s_inv_k = rsqrtf(tot_k2 + 1e-6f);
            }
            __syncthreads();
            if (tid < K) {
                sq[tid] *= s_inv_q;
                sk[tid] *= s_inv_k;
            }
        } else {
            __syncthreads();
            if (tid < K) {
                sq[tid] *= inv_sqrt_k;
            }
        }
        __syncthreads();

        float raw_out_f = 0.0f;
        bf16_t raw_out_bf = 0;
        if (tid < V) {
            const float exp_g = expf(g_f32[bth]);
            const float beta_t = bf16_to_f32(beta_bf16[bth]);
            const float v_t = bf16_to_f32(v_bf16[v_base + tid]);

            float kv_mem = 0.0f;
            for (int d = 0; d < K; ++d) {
                float sd = s_col[d] * exp_g;
                s_col[d] = sd;
                kv_mem += sd * sk[d];
            }
            const float delta = (v_t - kv_mem) * beta_t;
            float out_acc = 0.0f;
            for (int d = 0; d < K; ++d) {
                float sd = s_col[d] + sk[d] * delta;
                s_col[d] = sd;
                out_acc += sd * sq[d];
            }
            raw_out_bf = f32_to_bf16(out_acc);
            raw_out_f = bf16_to_f32(raw_out_bf);

            if (step_states_f32 != nullptr) {
                const long long step_base = ((long long)t * (B * H) + bh) * (long long)(K * V);
                for (int d = 0; d < K; ++d) {
                    step_states_f32[step_base + (long long)d * V + tid] = s_col[d];
                }
            }
        }

        if (fuse_gated_norm) {
            float v2 = raw_out_f * raw_out_f;
            for (int mask = 16; mask > 0; mask >>= 1) {
                v2 += __shfl_xor_sync(0xffffffffu, v2, mask);
            }
            if (lane == 0) {
                s_red_q[warp_id] = v2;
            }
            __syncthreads();
            if (tid == 0) {
                float tot_v2 = 0.0f;
                for (int w = 0; w < num_warps; ++w) {
                    tot_v2 += s_red_q[w];
                }
                s_inv_v = rsqrtf(tot_v2 / (float)V + norm_eps);
            }
            __syncthreads();
            if (tid < V) {
                float normed_bf = bf16_to_f32(f32_to_bf16(raw_out_f * s_inv_v));
                float w_bf = bf16_to_f32(f32_to_bf16(nw_val * normed_bf));
                float z_f = bf16_to_f32(z_bf16[v_base + tid]);
                out_bf16[v_base + tid] = f32_to_bf16(w_bf * silu_f32(z_f));
            }
        } else {
            if (tid < V) {
                out_bf16[v_base + tid] = raw_out_bf;
            }
        }
        __syncthreads();
    }

    if (final_state_f32 != nullptr && tid < V) {
        for (int d = 0; d < K; ++d) {
            final_state_f32[state_bh_base + (long long)d * V + tid] = s_col[d];
        }
    }
}

// Fuses fix_query_key_value_ordering + conv_state update/record + 4-tap depthwise conv1d + silu + beta/g!
extern "C" __global__ void qwen3_next_gdn_prep_conv_kernel(
    const bf16_t* __restrict__ proj_qkvz,          // [B, S, P_qkvz]
    const bf16_t* __restrict__ proj_ba,            // [B, S, 2 * H_v]
    bf16_t* __restrict__ conv_state,               // [B, conv_dim, 4] (updated in-place when !is_rec_past)
    bf16_t* __restrict__ step_conv_buf,            // [B, conv_dim, max_step_w] (written when is_rec_past)
    const bf16_t* __restrict__ conv_w,             // [conv_dim, 4]
    const bf16_t* __restrict__ conv_bias,          // [conv_dim] or nullptr
    const float* __restrict__ A_log_f32,           // [H_v]
    const bf16_t* __restrict__ dt_bias_bf16,       // [H_v]
    bf16_t* __restrict__ q_out,                    // [B, S, H_k, K]
    bf16_t* __restrict__ k_out,                    // [B, S, H_k, K]
    bf16_t* __restrict__ v_out,                    // [B, S, H_v, V]
    bf16_t* __restrict__ z_out,                    // [B, S, H_v, V]
    float* __restrict__ g_out,                     // [B, S, H_v]
    bf16_t* __restrict__ beta_out,                 // [B, S, H_v]
    int B,
    int S,
    int H_k,
    int H_v,
    int K,
    int V,
    int is_rec_past,
    int step_conv_stride
) {
    const int key_dim = H_k * K;
    const int value_dim = H_v * V;
    const int conv_dim = 2 * key_dim + value_dim;
    const int total_ch = B * conv_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_ch) return;

    const int b = idx / conv_dim;
    const int c = idx % conv_dim;
    const int n_rep = H_v / H_k;
    const int group_qkvz = 2 * K + 2 * n_rep * V;
    const int P_qkvz = H_k * group_qkvz;

    // Decode channel mapping inside proj_qkvz[b, t, :]
    int qkvz_col = 0;
    int target_kind = 0; // 0=q, 1=k, 2=v
    int target_offset = 0;
    if (c < key_dim) {
        const int kh = c / K;
        const int kd = c % K;
        qkvz_col = kh * group_qkvz + kd;
        target_kind = 0;
        target_offset = c;
    } else if (c < 2 * key_dim) {
        const int c1 = c - key_dim;
        const int kh = c1 / K;
        const int kd = c1 % K;
        qkvz_col = kh * group_qkvz + K + kd;
        target_kind = 1;
        target_offset = c1;
    } else {
        const int c2 = c - 2 * key_dim;
        const int vh = c2 / V;
        const int vd = c2 % V;
        const int kh = vh / n_rep;
        const int vr = vh % n_rep;
        qkvz_col = kh * group_qkvz + 2 * K + vr * V + vd;
        target_kind = 2;
        target_offset = c2;
    }

    const long long cs_base = (long long)idx * 4LL;
    bf16_t x_win[20]; // 4 state taps + up to 16 decode/speculative tokens
    x_win[0] = conv_state[cs_base + 0];
    x_win[1] = conv_state[cs_base + 1];
    x_win[2] = conv_state[cs_base + 2];
    x_win[3] = conv_state[cs_base + 3];

    for (int t = 0; t < S; ++t) {
        const long long qkvz_base = ((long long)b * S + t) * (long long)P_qkvz;
        x_win[4 + t] = proj_qkvz[qkvz_base + qkvz_col];
    }

    if (is_rec_past && step_conv_buf != nullptr) {
        const long long sb_base = (long long)idx * (long long)step_conv_stride;
        for (int j = 0; j < 4 + S; ++j) {
            step_conv_buf[sb_base + j] = x_win[j];
        }
    } else {
        conv_state[cs_base + 0] = x_win[S + 0];
        conv_state[cs_base + 1] = x_win[S + 1];
        conv_state[cs_base + 2] = x_win[S + 2];
        conv_state[cs_base + 3] = x_win[S + 3];
    }

    const long long cw_base = (long long)c * 4LL;
    const float w0 = bf16_to_f32(conv_w[cw_base + 0]);
    const float w1 = bf16_to_f32(conv_w[cw_base + 1]);
    const float w2 = bf16_to_f32(conv_w[cw_base + 2]);
    const float w3 = bf16_to_f32(conv_w[cw_base + 3]);
    const float bias_f = (conv_bias != nullptr) ? bf16_to_f32(conv_bias[c]) : 0.0f;

    for (int t = 0; t < S; ++t) {
        float acc = bias_f
            + bf16_to_f32(x_win[t + 1]) * w0
            + bf16_to_f32(x_win[t + 2]) * w1
            + bf16_to_f32(x_win[t + 3]) * w2
            + bf16_to_f32(x_win[t + 4]) * w3;
        float conv_rounded = bf16_to_f32(f32_to_bf16(acc));
        bf16_t act_bf = f32_to_bf16(silu_f32(conv_rounded));
        const long long bt = (long long)b * S + t;
        if (target_kind == 0) {
            q_out[bt * key_dim + target_offset] = act_bf;
        } else if (target_kind == 1) {
            k_out[bt * key_dim + target_offset] = act_bf;
        } else {
            v_out[bt * value_dim + target_offset] = act_bf;
        }
    }

    // Also extract z [B, S, H_v, V] on threads c < value_dim
    if (c < value_dim) {
        const int vh = c / V;
        const int vd = c % V;
        const int kh = vh / n_rep;
        const int vr = vh % n_rep;
        const int z_col = kh * group_qkvz + 2 * K + n_rep * V + vr * V + vd;
        for (int t = 0; t < S; ++t) {
            const long long bt = (long long)b * S + t;
            z_out[bt * value_dim + c] = proj_qkvz[bt * (long long)P_qkvz + z_col];
        }
    }

    // Also compute beta & g [B, S, H_v] on threads c < H_v
    if (c < H_v) {
        const int vh = c;
        const int kh = vh / n_rep;
        const int vr = vh % n_rep;
        const int b_col = kh * (2 * n_rep) + vr;
        const int a_col = kh * (2 * n_rep) + n_rep + vr;
        const float neg_exp_A = -expf(A_log_f32[vh]);
        const float dt_b = bf16_to_f32(dt_bias_bf16[vh]);
        for (int t = 0; t < S; ++t) {
            const long long bt = (long long)b * S + t;
            const long long ba_base = bt * (long long)(2 * H_v);
            const float b_val = bf16_to_f32(proj_ba[ba_base + b_col]);
            const float a_val = bf16_to_f32(proj_ba[ba_base + a_col]);
            beta_out[bt * H_v + vh] = f32_to_bf16(1.0f / (1.0f + expf(-b_val)));
            g_out[bt * H_v + vh] = neg_exp_A * softplus_f32(a_val + dt_b);
        }
    }
}

// General single-launch BF16 RMSNorm (supports Qwen3Next 1.0+w, Gemma4 w, and unscaled RMSNorm)
extern "C" __global__ void nvm_rmsnorm_bf16_kernel(
    const bf16_t* __restrict__ x_bf16,
    const bf16_t* __restrict__ w_bf16,
    bf16_t* __restrict__ out_bf16,
    int N_rows,
    int D,
    float eps,
    float w_offset,
    int use_rsqrt
) {
    const int row = blockIdx.x;
    if (row >= N_rows) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    __shared__ float s_red[8];
    __shared__ float s_inv;

    const long long base = (long long)row * (long long)D;
    float sum_sq = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float xv = bf16_to_f32(x_bf16[base + d]);
        sum_sq += xv * xv;
    }
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, mask);
    }
    if (lane == 0) {
        s_red[warp_id] = sum_sq;
    }
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            tot += s_red[w];
        }
        float ms = tot / (float)D + eps;
        s_inv = use_rsqrt ? rsqrtf(ms) : (1.0f / sqrtf(ms));
    }
    __syncthreads();

    const float inv = s_inv;
    if (w_bf16 != nullptr) {
        for (int d = tid; d < D; d += blockDim.x) {
            float xv = bf16_to_f32(x_bf16[base + d]);
            float wv = w_offset + bf16_to_f32(w_bf16[d]);
            out_bf16[base + d] = f32_to_bf16((xv * inv) * wv);
        }
    } else {
        for (int d = tid; d < D; d += blockDim.x) {
            float xv = bf16_to_f32(x_bf16[base + d]);
            out_bf16[base + d] = f32_to_bf16(xv * inv);
        }
    }
}

// Standalone Qwen3NextRMSNormGated matching exact BF16 intermediate rounding
extern "C" __global__ void qwen3_next_rmsnorm_gated_kernel(
    const bf16_t* __restrict__ x_bf16,
    const bf16_t* __restrict__ gate_bf16,
    const bf16_t* __restrict__ w_bf16,
    bf16_t* __restrict__ out_bf16,
    int N_rows,
    int D,
    float eps
) {
    const int row = blockIdx.x;
    if (row >= N_rows) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    __shared__ float s_red[8];
    __shared__ float s_inv;

    const long long base = (long long)row * (long long)D;
    float sum_sq = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float xv = bf16_to_f32(x_bf16[base + d]);
        sum_sq += xv * xv;
    }
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, mask);
    }
    if (lane == 0) {
        s_red[warp_id] = sum_sq;
    }
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            tot += s_red[w];
        }
        s_inv = rsqrtf(tot / (float)D + eps);
    }
    __syncthreads();

    const float inv = s_inv;
    for (int d = tid; d < D; d += blockDim.x) {
        float xv = bf16_to_f32(x_bf16[base + d]);
        float normed_bf = bf16_to_f32(f32_to_bf16(xv * inv));
        float w_bf = bf16_to_f32(f32_to_bf16(bf16_to_f32(w_bf16[d]) * normed_bf));
        float gf = bf16_to_f32(gate_bf16[base + d]);
        out_bf16[base + d] = f32_to_bf16(w_bf * silu_f32(gf));
    }
}

// Single-launch FP32 Softmax + Deterministic Top-K + Norm + Optional Per-Expert Scale / Prob Threshold
extern "C" __global__ void moe_router_topk_softmax_kernel(
    const bf16_t* __restrict__ logits_bf16,        // [T, E]
    const bf16_t* __restrict__ expert_scale_bf16,  // [E] or nullptr (Gemma4 per_expert_scale)
    float* __restrict__ topk_weights_f32,          // [T, K]
    long long* __restrict__ topk_indices_i64,      // [T, K]
    int T,
    int E,
    int K,
    int norm_topk,
    int round_bf16_before_scale,
    float min_prob_thresh
) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const int tid = threadIdx.x; // 256 threads for E <= 512
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    __shared__ float s_probs[512];
    __shared__ float s_red[8];
    __shared__ int s_idx[8];
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_chosen_p[16];
    __shared__ int s_chosen_i[16];
    __shared__ float s_topk_sum;

    const long long base = (long long)t * (long long)E;
    float local_max = -1e30f;
    for (int e = tid; e < E; e += 256) {
        float lv = bf16_to_f32(logits_bf16[base + e]);
        s_probs[e] = lv;
        if (lv > local_max) local_max = lv;
    }
    for (int mask = 16; mask > 0; mask >>= 1) {
        float oth = __shfl_xor_sync(0xffffffffu, local_max, mask);
        if (oth > local_max) local_max = oth;
    }
    if (lane == 0) s_red[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float mx = s_red[0];
        for (int w = 1; w < 8; ++w) {
            if (s_red[w] > mx) mx = s_red[w];
        }
        s_max = mx;
    }
    __syncthreads();

    const float mx = s_max;
    float local_sum = 0.0f;
    for (int e = tid; e < E; e += 256) {
        float ev = expf(s_probs[e] - mx);
        s_probs[e] = ev;
        local_sum += ev;
    }
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffffu, local_sum, mask);
    }
    if (lane == 0) s_red[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float sm = 0.0f;
        for (int w = 0; w < 8; ++w) sm += s_red[w];
        s_sum = sm;
    }
    __syncthreads();

    const float inv_sum = 1.0f / fmaxf(s_sum, 1e-30f);
    for (int e = tid; e < E; e += 256) {
        s_probs[e] *= inv_sum;
    }
    __syncthreads();

    // Deterministic Top-K selection (tie-breaks to smaller expert index, matching torch.topk)
    for (int k = 0; k < K; ++k) {
        float best_v = -1.0f;
        int best_i = 2147483647;
        for (int e = tid; e < E; e += 256) {
            float pv = s_probs[e];
            if (pv > best_v || (pv == best_v && e < best_i)) {
                best_v = pv;
                best_i = e;
            }
        }
        for (int mask = 16; mask > 0; mask >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, best_v, mask);
            int oi = __shfl_xor_sync(0xffffffffu, best_i, mask);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0) {
            s_red[warp_id] = best_v;
            s_idx[warp_id] = best_i;
        }
        __syncthreads();
        if (tid == 0) {
            float fv = s_red[0];
            int fi = s_idx[0];
            for (int w = 1; w < 8; ++w) {
                float wv = s_red[w];
                int wi = s_idx[w];
                if (wv > fv || (wv == fv && wi < fi)) {
                    fv = wv;
                    fi = wi;
                }
            }
            s_chosen_p[k] = fv;
            s_chosen_i[k] = fi;
            if (fi >= 0 && fi < E) {
                s_probs[fi] = -2.0f;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float tsum = 0.0f;
        for (int k = 0; k < K; ++k) tsum += s_chosen_p[k];
        s_topk_sum = fmaxf(tsum, 1e-30f);
    }
    __syncthreads();

    if (tid < K) {
        const float raw_p = s_chosen_p[tid];
        const int exp_i = s_chosen_i[tid];
        float w = norm_topk ? (raw_p / s_topk_sum) : raw_p;
        if (round_bf16_before_scale) {
            w = bf16_to_f32(f32_to_bf16(w));
        }
        if (expert_scale_bf16 != nullptr && exp_i >= 0 && exp_i < E) {
            // Match PyTorch `float32 * bfloat16 -> float32`
            w = w * bf16_to_f32(expert_scale_bf16[exp_i]);
        }
        topk_weights_f32[(long long)t * K + tid] = w;
        topk_indices_i64[(long long)t * K + tid] = (min_prob_thresh > 0.0f && raw_p < min_prob_thresh)
            ? -1LL
            : (long long)exp_i;
    }
}

// Arbitrary-length (S >= 1) streaming 4-tap causal conv1d + QKVZ/BA ordering + state update for prefill!
extern "C" __global__ void qwen3_next_gdn_prefill_prep_conv_kernel(
    const bf16_t* __restrict__ proj_qkvz,          // [B, S, P_qkvz]
    const bf16_t* __restrict__ proj_ba,            // [B, S, 2 * H_v]
    bf16_t* __restrict__ conv_state,               // [B, conv_dim, 4] or nullptr
    const bf16_t* __restrict__ conv_w,             // [conv_dim, 4]
    const bf16_t* __restrict__ conv_bias,          // [conv_dim] or nullptr
    const float* __restrict__ A_log_f32,           // [H_v]
    const bf16_t* __restrict__ dt_bias_bf16,       // [H_v]
    bf16_t* __restrict__ q_out,                    // [B, S, H_k, K]
    bf16_t* __restrict__ k_out,                    // [B, S, H_k, K]
    bf16_t* __restrict__ v_out,                    // [B, S, H_v, V]
    bf16_t* __restrict__ z_out,                    // [B, S, H_v, V]
    float* __restrict__ g_out,                     // [B, S, H_v]
    bf16_t* __restrict__ beta_out,                 // [B, S, H_v]
    int B,
    int S,
    int H_k,
    int H_v,
    int K,
    int V,
    int has_prev_conv
) {
    const int key_dim = H_k * K;
    const int value_dim = H_v * V;
    const int conv_dim = 2 * key_dim + value_dim;
    const int total_ch = B * conv_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_ch) return;

    const int b = idx / conv_dim;
    const int c = idx % conv_dim;
    const int n_rep = H_v / H_k;
    const int group_qkvz = 2 * K + 2 * n_rep * V;
    const int P_qkvz = H_k * group_qkvz;

    int qkvz_col = 0;
    int target_kind = 0;
    int target_offset = 0;
    if (c < key_dim) {
        const int kh = c / K;
        const int kd = c % K;
        qkvz_col = kh * group_qkvz + kd;
        target_kind = 0;
        target_offset = c;
    } else if (c < 2 * key_dim) {
        const int c1 = c - key_dim;
        const int kh = c1 / K;
        const int kd = c1 % K;
        qkvz_col = kh * group_qkvz + K + kd;
        target_kind = 1;
        target_offset = c1;
    } else {
        const int c2 = c - 2 * key_dim;
        const int vh = c2 / V;
        const int vd = c2 % V;
        const int kh = vh / n_rep;
        const int vr = vh % n_rep;
        qkvz_col = kh * group_qkvz + 2 * K + vr * V + vd;
        target_kind = 2;
        target_offset = c2;
    }

    const long long cs_base = (long long)idx * 4LL;
    bf16_t s0 = (has_prev_conv && conv_state != nullptr) ? conv_state[cs_base + 0] : (bf16_t)0;
    bf16_t s1 = (has_prev_conv && conv_state != nullptr) ? conv_state[cs_base + 1] : (bf16_t)0;
    bf16_t s2 = (has_prev_conv && conv_state != nullptr) ? conv_state[cs_base + 2] : (bf16_t)0;
    bf16_t s3 = (has_prev_conv && conv_state != nullptr) ? conv_state[cs_base + 3] : (bf16_t)0;

    const long long cw_base = (long long)c * 4LL;
    const float w0 = bf16_to_f32(conv_w[cw_base + 0]);
    const float w1 = bf16_to_f32(conv_w[cw_base + 1]);
    const float w2 = bf16_to_f32(conv_w[cw_base + 2]);
    const float w3 = bf16_to_f32(conv_w[cw_base + 3]);
    const float bias_f = (conv_bias != nullptr) ? bf16_to_f32(conv_bias[c]) : 0.0f;

    for (int t = 0; t < S; ++t) {
        const long long bt = (long long)b * S + t;
        const bf16_t x_t = proj_qkvz[bt * (long long)P_qkvz + qkvz_col];
        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = x_t;
        float acc = bias_f
            + bf16_to_f32(s0) * w0
            + bf16_to_f32(s1) * w1
            + bf16_to_f32(s2) * w2
            + bf16_to_f32(s3) * w3;
        float conv_rounded = bf16_to_f32(f32_to_bf16(acc));
        bf16_t act_bf = f32_to_bf16(silu_f32(conv_rounded));
        if (target_kind == 0) {
            q_out[bt * key_dim + target_offset] = act_bf;
        } else if (target_kind == 1) {
            k_out[bt * key_dim + target_offset] = act_bf;
        } else {
            v_out[bt * value_dim + target_offset] = act_bf;
        }
    }

    if (conv_state != nullptr) {
        conv_state[cs_base + 0] = s0;
        conv_state[cs_base + 1] = s1;
        conv_state[cs_base + 2] = s2;
        conv_state[cs_base + 3] = s3;
    }

    if (c < value_dim) {
        const int vh = c / V;
        const int vd = c % V;
        const int kh = vh / n_rep;
        const int vr = vh % n_rep;
        const int z_col = kh * group_qkvz + 2 * K + n_rep * V + vr * V + vd;
        for (int t = 0; t < S; ++t) {
            const long long bt = (long long)b * S + t;
            z_out[bt * value_dim + c] = proj_qkvz[bt * (long long)P_qkvz + z_col];
        }
    }

    if (c < H_v) {
        const int vh = c;
        const int kh = vh / n_rep;
        const int vr = vh % n_rep;
        const int b_col = kh * (2 * n_rep) + vr;
        const int a_col = kh * (2 * n_rep) + n_rep + vr;
        const float neg_exp_A = -expf(A_log_f32[vh]);
        const float dt_b = bf16_to_f32(dt_bias_bf16[vh]);
        for (int t = 0; t < S; ++t) {
            const long long bt = (long long)b * S + t;
            const long long ba_base = bt * (long long)(2 * H_v);
            const float b_val = bf16_to_f32(proj_ba[ba_base + b_col]);
            const float a_val = bf16_to_f32(proj_ba[ba_base + a_col]);
            beta_out[bt * H_v + vh] = f32_to_bf16(1.0f / (1.0f + expf(-b_val)));
            g_out[bt * H_v + vh] = neg_exp_A * softplus_f32(a_val + dt_b);
        }
    }
}
"""

_COMPILED_GDN_MODULE: Dict[int, Dict[str, Any]] = {}
_GDN_WORKSPACES: Dict[int, Dict[str, torch.Tensor]] = {}


def _get_gdn_kernel(device: torch.device) -> Dict[str, Any]:
    dev_idx = 0 if device.index is None else int(device.index)
    if dev_idx in _COMPILED_GDN_MODULE:
        return _COMPILED_GDN_MODULE[dev_idx]

    from torch.cuda._utils import _cuda_load_module, _get_gpu_rtc_library, _get_gpu_runtime_library

    with torch.cuda.device(dev_idx):
        libnvrtc = _get_gpu_rtc_library()
        libcuda = _get_gpu_runtime_library()
        props = torch.cuda.get_device_properties(dev_idx)
        arch = f"--gpu-architecture=sm_{props.major}{props.minor}".encode("utf-8")
        opts = [arch, b"-default-device"]
        opts_arr = (ctypes.c_char_p * len(opts))(*opts)

        src = _QWEN3_NEXT_CUDA_SRC.encode("utf-8")
        prog = ctypes.c_void_p()
        libnvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"qwen3_next_core.cu", 0, None, None)
        res = libnvrtc.nvrtcCompileProgram(prog, len(opts), opts_arr)
        if res != 0:
            log_sz = ctypes.c_size_t()
            libnvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
            log_buf = ctypes.create_string_buffer(log_sz.value)
            libnvrtc.nvrtcGetProgramLog(prog, log_buf)
            libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            raise RuntimeError(f"Qwen3Next Core NVRTC failed:\n{log_buf.value.decode('utf-8', errors='replace')}")

        cubin_sz = ctypes.c_size_t()
        libnvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(cubin_sz))
        cubin_buf = ctypes.create_string_buffer(cubin_sz.value)
        libnvrtc.nvrtcGetCUBIN(prog, cubin_buf)
        libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

        kd = _cuda_load_module(
            cubin_buf.raw,
            [
                "qwen3_next_gated_delta_rule_kernel",
                "qwen3_next_gdn_prep_conv_kernel",
                "qwen3_next_gdn_prefill_prep_conv_kernel",
                "nvm_rmsnorm_bf16_kernel",
                "qwen3_next_rmsnorm_gated_kernel",
                "moe_router_topk_softmax_kernel",
            ],
        )
        kd["libcuda"] = libcuda

    _COMPILED_GDN_MODULE[dev_idx] = kd
    return kd


def _get_gdn_decode_workspace(
    device: torch.device,
    B: int,
    max_s: int,
    H_k: int,
    H_v: int,
    K: int,
    V: int,
) -> Dict[str, torch.Tensor]:
    dev_idx = 0 if device.index is None else int(device.index)
    ws = _GDN_WORKSPACES.get(dev_idx)
    cap_s = max(16, int(max_s))
    if (
        ws is not None
        and int(ws["q"].shape[0]) >= B
        and int(ws["q"].shape[1]) >= cap_s
        and int(ws["q"].shape[2]) == H_k
        and int(ws["v"].shape[2]) == H_v
        and int(ws["q"].shape[3]) == K
        and int(ws["v"].shape[3]) == V
    ):
        return ws

    ws = {
        "q": torch.empty((B, cap_s, H_k, K), dtype=torch.bfloat16, device=device),
        "k": torch.empty((B, cap_s, H_k, K), dtype=torch.bfloat16, device=device),
        "v": torch.empty((B, cap_s, H_v, V), dtype=torch.bfloat16, device=device),
        "z": torch.empty((B, cap_s, H_v, V), dtype=torch.bfloat16, device=device),
        "g": torch.empty((B, cap_s, H_v), dtype=torch.float32, device=device),
        "beta": torch.empty((B, cap_s, H_v), dtype=torch.bfloat16, device=device),
    }
    _GDN_WORKSPACES[dev_idx] = ws
    return ws


def nvm_rmsnorm_bf16_cuda(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float = 1e-6,
    w_offset: float = 0.0,
    use_rsqrt: bool = True,
) -> torch.Tensor:
    """Single-launch NVRTC BF16 RMSNorm supporting `Qwen3NextRMSNorm` and `Gemma4RMSNorm`."""
    orig_shape = x.shape
    D = int(orig_shape[-1])
    x_2d = x.contiguous().view(-1, D)
    N_rows = int(x_2d.shape[0])
    if N_rows == 0:
        return x.clone()

    dev = x.device
    out_bf = torch.empty_like(x_2d)
    kd = _get_gdn_kernel(dev)
    libcuda = kd["libcuda"]
    stream_h = torch.cuda.current_stream(dev)._as_parameter_

    p_x = ctypes.c_void_p(x_2d.data_ptr())
    p_w = ctypes.c_void_p(weight.data_ptr() if weight is not None else 0)
    p_out = ctypes.c_void_p(out_bf.data_ptr())
    c_rows = ctypes.c_int(N_rows)
    c_D = ctypes.c_int(D)
    c_eps = ctypes.c_float(float(eps))
    c_woff = ctypes.c_float(float(w_offset))
    c_rsq = ctypes.c_int(1 if use_rsqrt else 0)

    args = (ctypes.c_void_p * 8)(
        ctypes.cast(ctypes.byref(p_x), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_w), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_rows), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_D), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_eps), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_woff), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_rsq), ctypes.c_void_p),
    )

    threads = 256 if D >= 256 else max(32, ((D + 31) // 32) * 32)
    res = libcuda.cuLaunchKernel(
        kd["nvm_rmsnorm_bf16_kernel"].func,
        N_rows, 1, 1,
        threads, 1, 1,
        0,
        stream_h,
        args,
        None,
    )
    if res != 0:
        raise RuntimeError(f"nvm_rmsnorm_bf16_kernel failed with CUDA error {res}")
    return out_bf.view(orig_shape)


def qwen3_next_rmsnorm_gated_cuda(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Single-launch NVRTC `Qwen3NextRMSNormGated` matching exact BF16 intermediate rounding."""
    orig_shape = hidden_states.shape
    D = int(orig_shape[-1])
    x_2d = hidden_states.contiguous().view(-1, D)
    g_2d = gate.contiguous().view(-1, D)
    N_rows = int(x_2d.shape[0])
    if N_rows == 0:
        return hidden_states.clone()

    dev = hidden_states.device
    out_bf = torch.empty_like(x_2d)
    kd = _get_gdn_kernel(dev)
    libcuda = kd["libcuda"]
    stream_h = torch.cuda.current_stream(dev)._as_parameter_

    p_x = ctypes.c_void_p(x_2d.data_ptr())
    p_g = ctypes.c_void_p(g_2d.data_ptr())
    p_w = ctypes.c_void_p(weight.data_ptr())
    p_out = ctypes.c_void_p(out_bf.data_ptr())
    c_rows = ctypes.c_int(N_rows)
    c_D = ctypes.c_int(D)
    c_eps = ctypes.c_float(float(eps))

    args = (ctypes.c_void_p * 7)(
        ctypes.cast(ctypes.byref(p_x), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_g), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_w), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_rows), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_D), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_eps), ctypes.c_void_p),
    )

    threads = min(256, max(32, ((D + 31) // 32) * 32))
    res = libcuda.cuLaunchKernel(
        kd["qwen3_next_rmsnorm_gated_kernel"].func,
        N_rows, 1, 1,
        threads, 1, 1,
        0,
        stream_h,
        args,
        None,
    )
    if res != 0:
        raise RuntimeError(f"qwen3_next_rmsnorm_gated_kernel failed with CUDA error {res}")
    return out_bf.view(orig_shape)


def moe_router_topk_softmax_cuda(
    logits_bf16: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool = True,
    round_bf16_before_scale: bool = True,
    per_expert_scale_bf16: Optional[torch.Tensor] = None,
    min_prob_thresh: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-launch FP32 Softmax + deterministic Top-K (`E <= 512, K <= 16`).

    Returns `(topk_weights_f32 [T, K], topk_indices_i64 [T, K])`.
    """
    logits_2d = logits_bf16.contiguous().view(-1, logits_bf16.shape[-1])
    T, E = int(logits_2d.shape[0]), int(logits_2d.shape[1])
    K = int(top_k)
    assert E <= 512 and 1 <= K <= 16, f"Unsupported router dims E={E}, K={K}"

    dev = logits_2d.device
    out_w = torch.empty((T, K), dtype=torch.float32, device=dev)
    out_i = torch.empty((T, K), dtype=torch.int64, device=dev)
    if T == 0:
        return out_w, out_i

    kd = _get_gdn_kernel(dev)
    libcuda = kd["libcuda"]
    stream_h = torch.cuda.current_stream(dev)._as_parameter_

    p_l = ctypes.c_void_p(logits_2d.data_ptr())
    p_es = ctypes.c_void_p(per_expert_scale_bf16.data_ptr() if per_expert_scale_bf16 is not None else 0)
    p_w = ctypes.c_void_p(out_w.data_ptr())
    p_i = ctypes.c_void_p(out_i.data_ptr())
    c_T = ctypes.c_int(T)
    c_E = ctypes.c_int(E)
    c_K = ctypes.c_int(K)
    c_norm = ctypes.c_int(1 if norm_topk_prob else 0)
    c_rnd = ctypes.c_int(1 if round_bf16_before_scale else 0)
    c_minp = ctypes.c_float(float(min_prob_thresh))

    args = (ctypes.c_void_p * 10)(
        ctypes.cast(ctypes.byref(p_l), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_es), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_w), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_i), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_T), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_E), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_K), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_norm), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_rnd), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_minp), ctypes.c_void_p),
    )

    res = libcuda.cuLaunchKernel(
        kd["moe_router_topk_softmax_kernel"].func,
        T, 1, 1,
        256, 1, 1,
        0,
        stream_h,
        args,
        None,
    )
    if res != 0:
        raise RuntimeError(f"moe_router_topk_softmax_kernel failed with CUDA error {res}")
    return out_w, out_i


def qwen3_next_gated_delta_rule_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    step_states_out: Optional[torch.Tensor] = None,
    z_gate: Optional[torch.Tensor] = None,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
    inplace_initial_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Executes `Qwen3NextGatedDeltaNet` recurrence on CUDA in FP32 registers (`K <= 128, V <= 256`).

    Supports native GQA `H_k <= H` without `repeat_interleave`, optional fused `Qwen3NextRMSNormGated`,
    and zero-allocation `inplace_initial_state` updates for CUDA Graph decode.
    """
    orig_dtype = query.dtype
    dev = query.device
    B, S, H_k, K = key.shape
    H = int(value.shape[2])
    V = int(value.shape[-1])
    assert K <= 128 and V <= 256, f"Unsupported GDN head dims K={K}, V={V}"
    assert H % H_k == 0, f"Expected H ({H}) divisible by H_k ({H_k})"

    q_bf = query if (query.dtype == torch.bfloat16 and query.is_contiguous()) else query.to(torch.bfloat16).contiguous()
    k_bf = key if (key.dtype == torch.bfloat16 and key.is_contiguous()) else key.to(torch.bfloat16).contiguous()
    v_bf = value if (value.dtype == torch.bfloat16 and value.is_contiguous()) else value.to(torch.bfloat16).contiguous()
    g_f32 = g if (g.dtype == torch.float32 and g.is_contiguous()) else g.to(torch.float32).contiguous()
    beta_bf = beta if (beta.dtype == torch.bfloat16 and beta.is_contiguous()) else beta.to(torch.bfloat16).contiguous()

    has_init = 1 if initial_state is not None else 0
    init_f32 = (
        (initial_state if (initial_state.dtype == torch.float32 and initial_state.is_contiguous()) else initial_state.to(torch.float32).contiguous())
        if initial_state is not None
        else None
    )
    z_bf = (
        (z_gate if (z_gate.dtype == torch.bfloat16 and z_gate.is_contiguous()) else z_gate.to(torch.bfloat16).contiguous())
        if (z_gate is not None and norm_weight is not None)
        else None
    )
    nw_bf = (
        (norm_weight if (norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()) else norm_weight.to(torch.bfloat16).contiguous())
        if (z_bf is not None and norm_weight is not None)
        else None
    )

    out_bf = torch.empty((B, S, H, V), dtype=torch.bfloat16, device=dev)
    if output_final_state:
        if inplace_initial_state and init_f32 is not None and init_f32 is initial_state:
            final_f32 = init_f32
        else:
            final_f32 = torch.empty((B, H, K, V), dtype=torch.float32, device=dev)
    else:
        final_f32 = None

    kd = _get_gdn_kernel(dev)
    libcuda = kd["libcuda"]
    stream_h = torch.cuda.current_stream(dev)._as_parameter_

    p_q = ctypes.c_void_p(q_bf.data_ptr())
    p_k = ctypes.c_void_p(k_bf.data_ptr())
    p_v = ctypes.c_void_p(v_bf.data_ptr())
    p_g = ctypes.c_void_p(g_f32.data_ptr())
    p_beta = ctypes.c_void_p(beta_bf.data_ptr())
    p_init = ctypes.c_void_p(init_f32.data_ptr() if init_f32 is not None else 0)
    p_z = ctypes.c_void_p(z_bf.data_ptr() if z_bf is not None else 0)
    p_nw = ctypes.c_void_p(nw_bf.data_ptr() if nw_bf is not None else 0)
    p_out = ctypes.c_void_p(out_bf.data_ptr())
    p_fin = ctypes.c_void_p(final_f32.data_ptr() if final_f32 is not None else 0)
    p_step = ctypes.c_void_p(step_states_out.data_ptr() if step_states_out is not None else 0)

    c_B = ctypes.c_int(B)
    c_S = ctypes.c_int(S)
    c_H = ctypes.c_int(H)
    c_Hk = ctypes.c_int(H_k)
    c_K = ctypes.c_int(K)
    c_V = ctypes.c_int(V)
    c_has_init = ctypes.c_int(has_init)
    c_l2 = ctypes.c_int(1 if use_qk_l2norm_in_kernel else 0)
    c_scale = ctypes.c_float(float(K) ** -0.5)
    c_neps = ctypes.c_float(float(norm_eps))

    args = (ctypes.c_void_p * 21)(
        ctypes.cast(ctypes.byref(p_q), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_k), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_v), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_g), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_beta), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_init), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_z), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_nw), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_fin), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_step), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_B), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_S), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_H), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_Hk), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_K), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_V), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_has_init), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_l2), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_scale), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_neps), ctypes.c_void_p),
    )

    threads = ((max(K, V) + 31) // 32) * 32
    res = libcuda.cuLaunchKernel(
        kd["qwen3_next_gated_delta_rule_kernel"].func,
        B * H, 1, 1,
        threads, 1, 1,
        0,
        stream_h,
        args,
        None,
    )
    if res != 0:
        raise RuntimeError(f"qwen3_next_gated_delta_rule_kernel launch failed with CUDA error {res}")

    out = out_bf if orig_dtype == torch.bfloat16 else out_bf.to(orig_dtype)
    return out, final_f32


def _launch_gdn_prep_conv_cuda(
    proj_qkvz: torch.Tensor,
    proj_ba: torch.Tensor,
    conv_state: torch.Tensor,
    step_conv_buf: Optional[torch.Tensor],
    conv_w_2d: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    a_log_f32: torch.Tensor,
    dt_bias_bf16: torch.Tensor,
    H_k: int,
    H_v: int,
    K: int,
    V: int,
    is_rec_past: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = proj_qkvz.device
    B, S, _ = proj_qkvz.shape
    ws = _get_gdn_decode_workspace(dev, B=B, max_s=S, H_k=H_k, H_v=H_v, K=K, V=V)
    q_out = ws["q"][:B, :S]
    k_out = ws["k"][:B, :S]
    v_out = ws["v"][:B, :S]
    z_out = ws["z"][:B, :S]
    g_out = ws["g"][:B, :S]
    beta_out = ws["beta"][:B, :S]

    kd = _get_gdn_kernel(dev)
    libcuda = kd["libcuda"]
    stream_h = torch.cuda.current_stream(dev)._as_parameter_

    qkvz_c = proj_qkvz if proj_qkvz.is_contiguous() else proj_qkvz.contiguous()
    ba_c = proj_ba if proj_ba.is_contiguous() else proj_ba.contiguous()

    p_qkvz = ctypes.c_void_p(qkvz_c.data_ptr())
    p_ba = ctypes.c_void_p(ba_c.data_ptr())
    p_cs = ctypes.c_void_p(conv_state.data_ptr())
    p_sb = ctypes.c_void_p(step_conv_buf.data_ptr() if step_conv_buf is not None else 0)
    p_cw = ctypes.c_void_p(conv_w_2d.data_ptr())
    p_cb = ctypes.c_void_p(conv_bias.data_ptr() if conv_bias is not None else 0)
    p_alog = ctypes.c_void_p(a_log_f32.data_ptr())
    p_dtb = ctypes.c_void_p(dt_bias_bf16.data_ptr())
    p_qo = ctypes.c_void_p(q_out.data_ptr())
    p_ko = ctypes.c_void_p(k_out.data_ptr())
    p_vo = ctypes.c_void_p(v_out.data_ptr())
    p_zo = ctypes.c_void_p(z_out.data_ptr())
    p_go = ctypes.c_void_p(g_out.data_ptr())
    p_bo = ctypes.c_void_p(beta_out.data_ptr())

    c_B = ctypes.c_int(B)
    c_S = ctypes.c_int(S)
    c_Hk = ctypes.c_int(H_k)
    c_Hv = ctypes.c_int(H_v)
    c_K = ctypes.c_int(K)
    c_V = ctypes.c_int(V)
    c_rec = ctypes.c_int(1 if is_rec_past else 0)
    c_stride = ctypes.c_int(int(step_conv_buf.shape[-1]) if step_conv_buf is not None else 0)

    args = (ctypes.c_void_p * 22)(
        ctypes.cast(ctypes.byref(p_qkvz), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_ba), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_cs), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_sb), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_cw), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_cb), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_alog), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_dtb), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_qo), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_ko), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_vo), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_zo), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_go), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_bo), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_B), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_S), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_Hk), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_Hv), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_K), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_V), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_rec), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_stride), ctypes.c_void_p),
    )

    conv_dim = 2 * H_k * K + H_v * V
    total_ch = B * conv_dim
    threads = 256
    blocks = (total_ch + threads - 1) // threads
    res = libcuda.cuLaunchKernel(
        kd["qwen3_next_gdn_prep_conv_kernel"].func,
        blocks, 1, 1,
        threads, 1, 1,
        0,
        stream_h,
        args,
        None,
    )
    if res != 0:
        raise RuntimeError(f"qwen3_next_gdn_prep_conv_kernel failed with CUDA error {res}")
    return q_out, k_out, v_out, z_out, g_out, beta_out


def _launch_gdn_prefill_prep_conv_cuda(
    proj_qkvz: torch.Tensor,
    proj_ba: torch.Tensor,
    conv_state: Optional[torch.Tensor],
    conv_w_2d: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    a_log_f32: torch.Tensor,
    dt_bias_bf16: torch.Tensor,
    H_k: int,
    H_v: int,
    K: int,
    V: int,
    has_prev_conv: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = proj_qkvz.device
    B, S, _ = proj_qkvz.shape
    q_out = torch.empty((B, S, H_k, K), dtype=torch.bfloat16, device=dev)
    k_out = torch.empty((B, S, H_k, K), dtype=torch.bfloat16, device=dev)
    v_out = torch.empty((B, S, H_v, V), dtype=torch.bfloat16, device=dev)
    z_out = torch.empty((B, S, H_v, V), dtype=torch.bfloat16, device=dev)
    g_out = torch.empty((B, S, H_v), dtype=torch.float32, device=dev)
    beta_out = torch.empty((B, S, H_v), dtype=torch.bfloat16, device=dev)

    kd = _get_gdn_kernel(dev)
    libcuda = kd["libcuda"]
    stream_h = torch.cuda.current_stream(dev)._as_parameter_

    qkvz_c = proj_qkvz if proj_qkvz.is_contiguous() else proj_qkvz.contiguous()
    ba_c = proj_ba if proj_ba.is_contiguous() else proj_ba.contiguous()

    p_qkvz = ctypes.c_void_p(qkvz_c.data_ptr())
    p_ba = ctypes.c_void_p(ba_c.data_ptr())
    p_cs = ctypes.c_void_p(conv_state.data_ptr() if conv_state is not None else 0)
    p_cw = ctypes.c_void_p(conv_w_2d.data_ptr())
    p_cb = ctypes.c_void_p(conv_bias.data_ptr() if conv_bias is not None else 0)
    p_alog = ctypes.c_void_p(a_log_f32.data_ptr())
    p_dtb = ctypes.c_void_p(dt_bias_bf16.data_ptr())
    p_qo = ctypes.c_void_p(q_out.data_ptr())
    p_ko = ctypes.c_void_p(k_out.data_ptr())
    p_vo = ctypes.c_void_p(v_out.data_ptr())
    p_zo = ctypes.c_void_p(z_out.data_ptr())
    p_go = ctypes.c_void_p(g_out.data_ptr())
    p_bo = ctypes.c_void_p(beta_out.data_ptr())

    c_B = ctypes.c_int(B)
    c_S = ctypes.c_int(S)
    c_Hk = ctypes.c_int(H_k)
    c_Hv = ctypes.c_int(H_v)
    c_K = ctypes.c_int(K)
    c_V = ctypes.c_int(V)
    c_prev = ctypes.c_int(1 if has_prev_conv else 0)

    args = (ctypes.c_void_p * 20)(
        ctypes.cast(ctypes.byref(p_qkvz), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_ba), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_cs), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_cw), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_cb), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_alog), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_dtb), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_qo), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_ko), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_vo), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_zo), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_go), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(p_bo), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_B), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_S), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_Hk), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_Hv), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_K), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_V), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_prev), ctypes.c_void_p),
    )

    conv_dim = 2 * H_k * K + H_v * V
    total_ch = B * conv_dim
    threads = 256
    blocks = (total_ch + threads - 1) // threads
    res = libcuda.cuLaunchKernel(
        kd["qwen3_next_gdn_prefill_prep_conv_kernel"].func,
        blocks, 1, 1,
        threads, 1, 1,
        0,
        stream_h,
        args,
        None,
    )
    if res != 0:
        raise RuntimeError(f"qwen3_next_gdn_prefill_prep_conv_kernel failed with CUDA error {res}")
    return q_out, k_out, v_out, z_out, g_out, beta_out


def prepare_hybrid_cache_for_speculative_step(past_key_values: Any, num_draft_tokens: int) -> None:
    """Enables past-state recording on `LinearAttentionLayer` entries before an `M = K + 1` speculative step."""
    if past_key_values is None or num_draft_tokens <= 0:
        return
    layers = getattr(past_key_values, "layers", None)
    if not layers:
        return
    for layer in layers:
        if hasattr(layer, "activate_past_recording") and hasattr(layer, "recurrent_states"):
            layer.activate_past_recording()


def finalize_or_rollback_hybrid_cache(past_key_values: Any, total_step_tokens: int, valid_kv_added: int) -> None:
    """Rolls back (or trims) both full-attention KV caches and `LinearAttentionLayer` conv + recurrent states."""
    if past_key_values is None:
        return
    n_reject = int(total_step_tokens) - int(valid_kv_added)
    layers = getattr(past_key_values, "layers", None)
    if not layers:
        if n_reject > 0:
            past_key_values.crop(-n_reject)
        return

    for layer in layers:
        if hasattr(layer, "recurrent_states") and hasattr(layer, "record_past"):
            if getattr(layer, "record_past", False):
                # 1. Crop conv_states window (either zero-alloc `_nvm_step_conv_buf` or fallback HF `conv_states`)
                step_conv_len = int(getattr(layer, "_nvm_step_conv_len", 0))
                step_conv_buf = getattr(layer, "_nvm_step_conv_states", None)
                if step_conv_len > 0 and step_conv_buf is not None and layer.is_conv_states_initialized.get(0, False):
                    ck = int(layer.conv_kernel_size[0])
                    layer.conv_states[0].copy_(step_conv_buf[..., valid_kv_added : valid_kv_added + ck])
                    layer._nvm_step_conv_len = 0
                elif any(layer.is_conv_states_initialized.values()):
                    for st_idx in range(layer.number_of_states):
                        if layer.is_conv_states_initialized[st_idx]:
                            ck = layer.conv_kernel_size[st_idx]
                            if n_reject == 0:
                                layer.conv_states[st_idx] = (
                                    layer.conv_states[st_idx][..., -ck:].contiguous()
                                )
                            else:
                                layer.conv_states[st_idx] = (
                                    layer.conv_states[st_idx][..., -n_reject - ck : -n_reject].contiguous()
                                )
                # 2. Roll back recurrent_states to exact verified position `valid_kv_added - 1` when `n_reject > 0`
                step_buf = getattr(layer, "_nvm_step_recurrent_states", None)
                step_rec_len = int(getattr(layer, "_nvm_step_recurrent_len", 0))
                if n_reject > 0 and step_rec_len > 0 and step_buf is not None and valid_kv_added >= 1:
                    if layer.is_recurrent_states_initialized.get(0, False):
                        layer.recurrent_states[0].copy_(step_buf[valid_kv_added - 1])
                layer._nvm_step_recurrent_len = 0
                layer.record_past = False
            # Also crop full-attention part if this is a combined Hybrid layer
            if hasattr(layer, "keys") and getattr(layer, "keys", None) is not None and n_reject > 0:
                if layer.keys.numel() > 0:
                    layer.keys = layer.keys[..., :-n_reject, :]
                    layer.values = layer.values[..., :-n_reject, :]
        else:
            if n_reject > 0 and hasattr(layer, "crop"):
                layer.crop(-n_reject)


def install_fast_moe_core_kernels(model: nn.Module) -> Dict[str, int]:
    """Installs fused NVRTC RMSNorm and Top-K Router kernels across `Qwen3-Next` and `Gemma-4` models."""
    import torch.nn.functional as F

    counts = {"rmsnorm": 0, "rmsnorm_gated": 0, "router": 0}

    def _qwen_rmsnorm_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.bfloat16 and self.weight.dtype == torch.bfloat16:
            return nvm_rmsnorm_bf16_cuda(x, self.weight, eps=self.eps, w_offset=1.0, use_rsqrt=True)
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)

    def _gemma4_rmsnorm_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16:
            w = self.weight if getattr(self, "with_scale", True) else None
            if w is None or w.dtype == torch.bfloat16:
                return nvm_rmsnorm_bf16_cuda(
                    hidden_states, w, eps=self.eps, w_offset=0.0, use_rsqrt=False
                )
        xf = hidden_states.float()
        normed = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if getattr(self, "with_scale", True):
            normed = normed * self.weight.float()
        return normed.type_as(hidden_states)

    def _qwen_rmsnorm_gated_forward(self: nn.Module, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if (
            hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and gate.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
        ):
            return qwen3_next_rmsnorm_gated_cuda(hidden_states, gate, self.weight, eps=self.variance_epsilon)
        orig_dt = hidden_states.dtype
        xf = hidden_states.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        xf = self.weight * xf.to(orig_dt)
        return (xf * F.silu(gate.float())).to(orig_dt)

    def _qwen_router_forward(self: nn.Module, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        if (
            router_logits.is_cuda
            and router_logits.dtype == torch.bfloat16
            and self.num_experts <= 512
            and self.top_k <= 16
        ):
            scores_f32, indices_i64 = moe_router_topk_softmax_cuda(
                router_logits,
                top_k=self.top_k,
                norm_topk_prob=bool(self.norm_topk_prob),
                round_bf16_before_scale=True,
                per_expert_scale_bf16=None,
            )
            return router_logits, scores_f32.to(router_logits.dtype), indices_i64
        router_probs = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_scores = router_top_value.to(router_logits.dtype)
        return router_logits, router_scores, router_indices

    def _gemma4_router_forward(self: nn.Module, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        expert_scores = self.proj(hidden_states)
        top_k = int(self.config.top_k_experts)
        if (
            expert_scores.is_cuda
            and expert_scores.dtype == torch.bfloat16
            and len(self._forward_hooks) == 0
            and int(expert_scores.shape[-1]) <= 512
            and top_k <= 16
            and self.per_expert_scale.dtype == torch.bfloat16
        ):
            top_k_weights, top_k_index = moe_router_topk_softmax_cuda(
                expert_scores,
                top_k=top_k,
                norm_topk_prob=True,
                round_bf16_before_scale=False,
                per_expert_scale_bf16=self.per_expert_scale,
            )
            return expert_scores, top_k_weights, top_k_index
        router_probabilities = F.softmax(expert_scores, dim=-1, dtype=torch.float32)
        top_k_weights, top_k_index = torch.topk(router_probabilities, k=top_k, dim=-1)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return router_probabilities, top_k_weights, top_k_index

    for mod in model.modules():
        cls_name = mod.__class__.__name__
        if cls_name == "Qwen3NextRMSNorm":
            mod.forward = types.MethodType(_qwen_rmsnorm_forward, mod)
            counts["rmsnorm"] += 1
        elif cls_name == "Gemma4RMSNorm":
            mod.forward = types.MethodType(_gemma4_rmsnorm_forward, mod)
            counts["rmsnorm"] += 1
        elif cls_name == "Qwen3NextRMSNormGated":
            mod.forward = types.MethodType(_qwen_rmsnorm_gated_forward, mod)
            counts["rmsnorm_gated"] += 1
        elif cls_name == "Qwen3NextTopKRouter":
            mod.forward = types.MethodType(_qwen_router_forward, mod)
            counts["router"] += 1
        elif cls_name == "Gemma4TextRouter":
            mod.forward = types.MethodType(_gemma4_router_forward, mod)
            counts["router"] += 1

    return counts


def install_qwen3_next_fast_gdn(model: nn.Module) -> int:
    """Installs fused NVRTC GDN + Conv1d + RMSNorm + Router kernels on `Qwen3-Next` models."""
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextGatedDeltaNet,
        apply_mask_to_padding_states,
        causal_conv1d_fn,
    )
    import torch.nn.functional as F

    install_fast_moe_core_kernels(model)
    patched = 0

    def _fast_gdn_forward(
        self: Qwen3NextGatedDeltaNet,
        hidden_states: torch.Tensor,
        cache_params: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if cache_params is None:
            cache_params = cache
        seq_len = int(hidden_states.shape[1])
        # Note: during cached decode (`seq_len <= 16`, batch=1), attention_mask is all-1s (`ones_mask_buf`),
        # and even if passed, its tail `seq_len` entries are 1; only apply mask when lengths match (`prefill`)!
        if attention_mask is not None and attention_mask.shape[1] == seq_len:
            hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        if cache_params is not None:
            conv_state = cache_params.get_conv_state(self.layer_idx, state_idx=0)
            recurrent_state = cache_params.get_recurrent_state(self.layer_idx, state_idx=0)
            is_rec_past = bool(cache_params.is_recording_past(self.layer_idx))
            cache_layer = cache_params.layers[self.layer_idx]
            has_prev_conv = bool(cache_layer.has_previous_state.get(0, False))
        else:
            conv_state, recurrent_state = None, None
            is_rec_past = False
            cache_layer = None
            has_prev_conv = False

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)

        # Cached decode & speculative multi-token fastpath (`1 <= seq_len <= 16`):
        # Collapses ordering + conv1d + silu + beta/g + GDN recurrence + gated RMSNorm into 2 NVRTC launches!
        if (
            hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and conv_state is not None
            and has_prev_conv
            and self.conv_kernel_size == 4
            and self.activation == "silu"
            and 1 <= seq_len <= 16
            and self.head_k_dim <= 128
            and self.head_v_dim <= 256
        ):
            cw_2d = getattr(self, "_nvm_conv_w_2d", None)
            if cw_2d is None or cw_2d.device != hidden_states.device:
                cw_2d = self.conv1d.weight.squeeze(1).to(torch.bfloat16).contiguous()
                self._nvm_conv_w_2d = cw_2d
            alog_f32 = getattr(self, "_nvm_alog_f32", None)
            if alog_f32 is None or alog_f32.device != hidden_states.device:
                alog_f32 = self.A_log.float().contiguous()
                self._nvm_alog_f32 = alog_f32
            dtb_bf16 = getattr(self, "_nvm_dt_bias_bf16", None)
            if dtb_bf16 is None or dtb_bf16.device != hidden_states.device:
                dtb_bf16 = self.dt_bias.to(torch.bfloat16).contiguous()
                self._nvm_dt_bias_bf16 = dtb_bf16

            B_cur = int(hidden_states.shape[0])
            step_conv_buf: Optional[torch.Tensor] = None
            step_rec_buf: Optional[torch.Tensor] = None
            if is_rec_past and cache_layer is not None:
                existing_cb = getattr(cache_layer, "_nvm_step_conv_states", None)
                need_w = 4 + max(8, seq_len)
                if (
                    existing_cb is None
                    or existing_cb.shape[0] != B_cur
                    or existing_cb.shape[1] != self.conv_dim
                    or existing_cb.shape[2] < 4 + seq_len
                ):
                    existing_cb = torch.empty(
                        (B_cur, self.conv_dim, need_w),
                        dtype=torch.bfloat16,
                        device=hidden_states.device,
                    )
                    cache_layer._nvm_step_conv_states = existing_cb
                step_conv_buf = existing_cb
                cache_layer._nvm_step_conv_len = seq_len

                if recurrent_state is not None and seq_len > 1:
                    existing_rb = getattr(cache_layer, "_nvm_step_recurrent_states", None)
                    req_shape = (seq_len, B_cur, self.num_v_heads, self.head_k_dim, self.head_v_dim)
                    if existing_rb is None or existing_rb.shape[0] < seq_len or existing_rb.shape[1:] != req_shape[1:]:
                        existing_rb = torch.empty(
                            (max(8, seq_len), B_cur, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                            dtype=torch.float32,
                            device=hidden_states.device,
                        )
                        cache_layer._nvm_step_recurrent_states = existing_rb
                    step_rec_buf = existing_rb[:seq_len]
                    cache_layer._nvm_step_recurrent_len = seq_len

            q_c, k_c, v_c, z_c, g_c, beta_c = _launch_gdn_prep_conv_cuda(
                proj_qkvz=projected_states_qkvz,
                proj_ba=projected_states_ba,
                conv_state=conv_state,
                step_conv_buf=step_conv_buf,
                conv_w_2d=cw_2d,
                conv_bias=self.conv1d.bias,
                a_log_f32=alog_f32,
                dt_bias_bf16=dtb_bf16,
                H_k=self.num_k_heads,
                H_v=self.num_v_heads,
                K=self.head_k_dim,
                V=self.head_v_dim,
                is_rec_past=is_rec_past,
            )

            normed_attn_out, last_recurrent_state = qwen3_next_gated_delta_rule_cuda(
                query=q_c,
                key=k_c,
                value=v_c,
                g=g_c,
                beta=beta_c,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                step_states_out=step_rec_buf,
                z_gate=z_c,
                norm_weight=self.norm.weight,
                norm_eps=self.norm.variance_epsilon,
                inplace_initial_state=True,
            )
            if (
                cache_params is not None
                and last_recurrent_state is not None
                and last_recurrent_state is not recurrent_state
            ):
                cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

            return self.out_proj(normed_attn_out.view(B_cur, seq_len, -1))

        # Fused NVRTC Prefill Fastpath (`S > 16` or initial cache creation, when not recording past):
        # Eliminates slow PyTorch `causal_conv1d_fn` grouped-conv fallback + HF warning!
        if (
            hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and not is_rec_past
            and self.conv_kernel_size == 4
            and self.activation == "silu"
            and self.head_k_dim <= 128
            and self.head_v_dim <= 256
        ):
            cw_2d = getattr(self, "_nvm_conv_w_2d", None)
            if cw_2d is None or cw_2d.device != hidden_states.device:
                cw_2d = self.conv1d.weight.squeeze(1).to(torch.bfloat16).contiguous()
                self._nvm_conv_w_2d = cw_2d
            alog_f32 = getattr(self, "_nvm_alog_f32", None)
            if alog_f32 is None or alog_f32.device != hidden_states.device:
                alog_f32 = self.A_log.float().contiguous()
                self._nvm_alog_f32 = alog_f32
            dtb_bf16 = getattr(self, "_nvm_dt_bias_bf16", None)
            if dtb_bf16 is None or dtb_bf16.device != hidden_states.device:
                dtb_bf16 = self.dt_bias.to(torch.bfloat16).contiguous()
                self._nvm_dt_bias_bf16 = dtb_bf16

            B_cur = int(hidden_states.shape[0])
            if cache_layer is not None:
                if not cache_layer.is_conv_states_initialized.get(0, False):
                    dummy_cs = torch.empty(
                        (B_cur, self.conv_dim, self.conv_kernel_size),
                        dtype=torch.bfloat16,
                        device=hidden_states.device,
                    )
                    cache_layer.lazy_initialization(
                        conv_states=dummy_cs,
                        state_idx=0,
                        conv_kernel_size=self.conv_kernel_size,
                    )
                conv_state = cache_layer.conv_states[0]
                cache_layer.has_previous_state[0] = True

            q_p, k_p, v_p, z_p, g_p, beta_p = _launch_gdn_prefill_prep_conv_cuda(
                proj_qkvz=projected_states_qkvz,
                proj_ba=projected_states_ba,
                conv_state=conv_state,
                conv_w_2d=cw_2d,
                conv_bias=self.conv1d.bias,
                a_log_f32=alog_f32,
                dt_bias_bf16=dtb_bf16,
                H_k=self.num_k_heads,
                H_v=self.num_v_heads,
                K=self.head_k_dim,
                V=self.head_v_dim,
                has_prev_conv=has_prev_conv,
            )

            normed_attn_out, last_recurrent_state = qwen3_next_gated_delta_rule_cuda(
                query=q_p,
                key=k_p,
                value=v_p,
                g=g_p,
                beta=beta_p,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                step_states_out=None,
                z_gate=z_p,
                norm_weight=self.norm.weight,
                norm_eps=self.norm.variance_epsilon,
                inplace_initial_state=True,
            )
            if (
                cache_params is not None
                and last_recurrent_state is not None
                and last_recurrent_state is not recurrent_state
            ):
                cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
            return self.out_proj(normed_attn_out.view(B_cur, seq_len, -1))

        # General fallback (CPU / non-BF16 / custom conv_kernel_size)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))
        mixed_qkv = torch.cat((query, key, value), dim=-1).transpose(1, 2)

        if cache_params is not None:
            mixed_qkv = cache_params.update_conv_state(
                mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
            )
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            activation=self.activation,
        )
        if cache_params is not None:
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        if hidden_states.is_cuda and self.head_k_dim <= 128 and self.head_v_dim <= 256:
            core_attn_out, last_recurrent_state = qwen3_next_gated_delta_rule_cuda(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                step_states_out=None,
                z_gate=z,
                norm_weight=self.norm.weight,
                norm_eps=self.norm.variance_epsilon,
            )
            if cache_params is not None and last_recurrent_state is not None:
                cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
            return self.out_proj(core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1))

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)
        from transformers.models.qwen3_next.modeling_qwen3_next import torch_recurrent_gated_delta_rule

        core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )
        if cache_params is not None and last_recurrent_state is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
        return self.out_proj(core_attn_out)

    for mod in model.modules():
        if isinstance(mod, Qwen3NextGatedDeltaNet):
            mod.forward = _fast_gdn_forward.__get__(mod, mod.__class__)
            patched += 1
    return patched
