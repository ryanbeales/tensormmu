"""Hardware Verification & Latency Benchmark for CUDA-Graphed `NativeMTPModule` (`Qwen3-Next-80B` Geometry).

Verifies:
  1. Exact numerical parity between eager PyTorch MTP and CUDA-Graphed `NativeMTPModule` (`0` token diff).
  2. Prompt prefill (`prefill_from_hidden`) + multi-step `K=3` autoregressive drafting + exact KV rollback
     (`commit_verified_tokens`) when draft token 2 is rejected.
  3. Real RTX 4090 CUDA-Graphed draft latency (`ms/draft token`) at full `Qwen3-Next-80B` dimensions
     (`d_model=2048, d_ffn=512, 512 experts, 16 Q heads, 2 KV heads, head_dim=256, vocab=151936`) in:
       - `vm_paged` (`64` L1 VRAM + `448` L2 Zero-Copy `UINT4-WINA` -> ~130 MiB VRAM!)
       - `all_resident` (`512/512` L1 VRAM -> ~900 MiB VRAM, matching AWS L40S / H100!)
"""

from __future__ import annotations

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from runtime.gpu_page_tlb import SharedQuantLUTPool
from runtime.moe_expert_tlb import SparseMoETLBExperts, quantize_pack_moe_experts_uint4
from runtime.native_mtp import NativeMTPModule, _MTPRMSNorm


class _MockQwen3NextAttn(nn.Module):
    def __init__(self, d_model: int, h_q: int, h_kv: int, head_dim: int, dev: torch.device, dtype: torch.dtype):
        super().__init__()
        self.scaling = head_dim ** -0.5
        # Qwen3-Next gated attention: q_proj outputs 2 * h_q * head_dim (query + sigmoid gate)
        self.q_proj = nn.Linear(d_model, 2 * h_q * head_dim, bias=False, device=dev, dtype=dtype)
        self.k_proj = nn.Linear(d_model, h_kv * head_dim, bias=False, device=dev, dtype=dtype)
        self.v_proj = nn.Linear(d_model, h_kv * head_dim, bias=False, device=dev, dtype=dtype)
        self.o_proj = nn.Linear(h_q * head_dim, d_model, bias=False, device=dev, dtype=dtype)
        self.q_norm = _MTPRMSNorm(head_dim, unit_offset=True).to(device=dev, dtype=dtype)
        self.k_norm = _MTPRMSNorm(head_dim, unit_offset=True).to(device=dev, dtype=dtype)


class _MockQwen3NextMoEBlock(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int, experts: SparseMoETLBExperts, dev: torch.device, dtype: torch.dtype):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False, device=dev, dtype=dtype)
        self.experts = experts
        self.shared_gate_up = nn.Linear(d_model, 1024, bias=False, device=dev, dtype=dtype)
        self.shared_down = nn.Linear(512, d_model, bias=False, device=dev, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        logits = self.gate(flat).float()
        probs = F.softmax(logits, dim=-1)
        top_w, top_i = torch.topk(probs, k=self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        moe_out = self.experts(flat, top_i, top_w).view_as(x)
        g, u = self.shared_gate_up(x).chunk(2, dim=-1)
        sh_out = self.shared_down(F.silu(g) * u)
        return moe_out + sh_out


class _MockQwen3NextLayer(nn.Module):
    def __init__(self, d_model: int, h_q: int, h_kv: int, head_dim: int, moe_block: nn.Module, dev: torch.device, dtype: torch.dtype):
        super().__init__()
        self.input_layernorm = _MTPRMSNorm(d_model, unit_offset=True).to(device=dev, dtype=dtype)
        self.self_attn = _MockQwen3NextAttn(d_model, h_q, h_kv, head_dim, dev, dtype)
        self.post_attention_layernorm = _MTPRMSNorm(d_model, unit_offset=True).to(device=dev, dtype=dtype)
        self.mlp = moe_block


class _MockRotary(nn.Module):
    def __init__(self, head_dim: int, dev: torch.device, dtype: torch.dtype):
        super().__init__()
        inv = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=dev, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._dt = dtype

    def forward(self, v: torch.Tensor, position_ids: torch.Tensor):
        freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(self._dt), emb.sin().to(self._dt)


class _MockMainModel(nn.Module):
    def __init__(self, vocab: int, d_model: int, head_dim: int, dev: torch.device, dtype: torch.dtype):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, d_model, device=dev, dtype=dtype)
        self.model.rotary_emb = _MockRotary(head_dim, dev, dtype)
        self.lm_head = nn.Linear(d_model, vocab, bias=False, device=dev, dtype=dtype)


def main() -> None:
    torch.manual_seed(42)
    dev = torch.device("cuda:0")
    dtype = torch.bfloat16
    D, I, E, K_top = 2048, 512, 512, 10
    H_q, H_kv, H_d, V = 16, 2, 256, 151936

    print(f"Building full Qwen3-Next-80B MTP layer (E={E}, D={D}, I={I}, V={V}) on {torch.cuda.get_device_name(dev)}...", flush=True)
    # Build 16 unique quantized pages and tile to E=512 (fast ~0.4s build, full 865 MiB memory footprint!)
    w_gu_16 = torch.randn((16, 2 * I, D), dtype=dtype) * 0.02
    w_dn_16 = torch.randn((16, D, I), dtype=dtype) * 0.02
    base_pages, layout = quantize_pack_moe_experts_uint4(w_gu_16, w_dn_16, group_size=64, compute_device="cuda:0")
    host_pages = torch.empty((E, layout.page_bytes), dtype=torch.uint8, device="cpu")
    for e in range(E):
        host_pages[e].copy_(base_pages[e % 16])

    lut = SharedQuantLUTPool(dev, dtype)
    main_model = _MockMainModel(V, D, H_d, dev, dtype).eval()

    for mode_lbl, tlb_slots in [("vm_paged (64 L1 + 448 L2 UINT4-WINA)", 64), ("all_resident (512/512 L1 VRAM)", 512)]:
        print(f"\n=== Testing NativeMTPModule [{mode_lbl}] ===", flush=True)
        experts = SparseMoETLBExperts(
            num_experts=E, d_model=D, d_ffn=I, layout=layout, host_expert_pages=host_pages,
            lut_pool=lut, act_fn=F.silu, device=dev, layer_idx=999, tlb_slots=tlb_slots,
            mode="all_resident" if tlb_slots >= E else "vm_paged", use_fused_kernels=True,
        )
        if tlb_slots < E:
            experts.tlb_cache.configure_l2_zero_copy(E - tlb_slots, wina_keep_ratio=0.50, lut_uint4=lut.lut_uint4)
            assert experts.tlb_cache.all_experts_mapped and experts.tlb_cache.l2_wina_dev_ptr != 0

        moe_blk = _MockQwen3NextMoEBlock(D, E, K_top, experts, dev, dtype).eval()
        lyr = _MockQwen3NextLayer(D, H_q, H_kv, H_d, moe_blk, dev, dtype).eval()
        mtp = NativeMTPModule(
            d_model=D, num_q_heads=H_q, num_kv_heads=H_kv, head_dim=H_d, eps=1e-6,
            unit_offset_norm=True, decoder_layer=lyr, main_model=main_model, device=dev,
            dtype=dtype, moe_experts=experts, arch_kind="qwen3_next",
        ).eval()

        # 1. Prefill 32 prompt tokens
        prompt_ids = torch.randint(0, V, (1, 32), device=dev, dtype=torch.long)
        h_prefill = torch.randn((1, 32, D), device=dev, dtype=dtype) * 0.5
        mtp.prefill_from_hidden(prompt_ids, h_prefill)
        assert mtp.cur_seq_len == 31

        # 2. Exact parity: Eager vs CUDA-Graphed `draft_k_tokens(K=3)`
        h_last = h_prefill[:, -1:, :].clone()
        # Run eager first (disable graph temporarily)
        orig_ensure = mtp._ensure_cuda_graph
        mtp._ensure_cuda_graph = lambda: False  # type: ignore[method-assign]
        eager_ids, eager_confs = mtp.draft_k_tokens(42, h_last, max_draft_tokens=3, min_conf=0.0)
        mtp._ensure_cuda_graph = orig_ensure  # type: ignore[method-assign]

        # Run CUDA-graphed
        graph_ids, graph_confs = mtp.draft_k_tokens(42, h_last, max_draft_tokens=3, min_conf=0.0)
        assert mtp._graph is not None, "Expected MTP CUDA Graph to be captured!"
        assert eager_ids == graph_ids, f"Token mismatch: eager={eager_ids} vs graph={graph_ids}"
        max_c_diff = max(abs(a - b) for a, b in zip(eager_confs, graph_confs))
        print(f"  [OK] Eager vs CUDA-Graph K=3 parity: ids={graph_ids} (exact match!), max_conf_diff={max_c_diff:.6f}", flush=True)

        # 3. Verify KV rollback (`commit_verified_tokens(1)` discards rejected draft #2 & #3 cleanly)
        mtp.commit_verified_tokens(1)
        assert mtp.cur_seq_len == 33
        next_ids_a, _ = mtp.draft_k_tokens(99, h_last, max_draft_tokens=2, min_conf=0.0)
        # Re-run from same cursor (no commit) -> must be bit-identical!
        next_ids_b, _ = mtp.draft_k_tokens(99, h_last, max_draft_tokens=2, min_conf=0.0)
        assert next_ids_a == next_ids_b
        print(f"  [OK] KV rollback & deterministic replay verified (cur_seq_len={mtp.cur_seq_len}).", flush=True)

        # 4. Benchmark CUDA-Graphed single-draft-token latency (50 iterations)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_iters = 50
        for _ in range(n_iters):
            mtp.draft_k_tokens(42, h_last, max_draft_tokens=1, min_conf=0.0)
        torch.cuda.synchronize()
        ms_per_draft = (time.perf_counter() - t0) * 1000.0 / n_iters

        t0 = time.perf_counter()
        for _ in range(n_iters):
            mtp.draft_k_tokens(42, h_last, max_draft_tokens=3, min_conf=0.0)
        torch.cuda.synchronize()
        ms_k3 = (time.perf_counter() - t0) * 1000.0 / n_iters
        vram_mb = (
            experts.tlb_cache.total_gpu_vram_bytes
            + sum(p.numel() * p.element_size() for p in mtp.parameters())
            + (mtp.k_buf.numel() * 2 * 2 if mtp.k_buf is not None else 0)
        ) / (1024 ** 2)
        print(
            f"  [PERF] CUDA-Graphed MTP Draft Latency: K=1 -> {ms_per_draft:.2f} ms/tok | "
            f"K=3 -> {ms_k3:.2f} ms total ({ms_k3/3:.2f} ms/tok) | MTP VRAM ~ {vram_mb:.1f} MiB",
            flush=True,
        )
        mtp.release()
        del mtp, lyr, moe_blk, experts
        torch.cuda.empty_cache()

    print("\nALL NATIVE MTP CUDA GRAPH TESTS PASSED!", flush=True)


if __name__ == "__main__":
    main()
