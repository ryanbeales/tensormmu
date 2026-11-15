"""End-to-end execution script and benchmark evaluator for the Neural VM Single-Layer PoC.

Evaluates three execution states on held-out slices (Logic_Eval and Fact_Eval):
  - State 0 (Baseline): Unmodified Gemma 4 layer.
  - State 1 (CPU_Only): Layer L down_proj replaced with NeuralVMDownProj(mode="cpu_only") at 25% rank.
  - State 2 (VM_Paged): NeuralVMDownProj(mode="vm_paged", top_k=64) paging sparse columns from pinned RAM.
"""

import argparse
from dataclasses import dataclass
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# Enable fast local cache loading when Hugging Face cache directories are already populated
if os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub")) and os.path.isdir(
    os.path.expanduser("~/.cache/huggingface/datasets")
):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NeuralVMConfig
from data.calibration import ContrastiveCalibrationBundle, TokenizedSplit, build_contrastive_datasets
from decompiler.contrastive_svd import ContrastiveSVDResult, decompose_down_proj_contrastive
from decompiler.hooks import capture_layer_activations, get_target_down_proj, get_target_mlp
from runtime.neural_mmu import NeuralVMDownProj, install_neural_vm_down_proj


@dataclass
class SplitEvalMetrics:
    """Evaluation metrics on a single held-out dataset split."""

    split_name: str
    state_name: str
    loss: float
    perplexity: float
    next_token_acc_pct: float
    baseline_token_match_pct: float
    ppl_delta_pct: float


class SyntheticTokenizer:
    """Deterministic word/subword hash tokenizer for architectural verification in `--mock` mode."""

    def __init__(self, vocab_size: int = 32000) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        return_attention_mask: bool = False,
    ) -> Dict[str, List[int]]:
        tokens = text.replace("\n", " \n ").split(" ")
        ids: List[int] = []
        IsFact = "Encyclopedic" in text or "Mineralogical" in text or "Registry Endpoint" in text
        for tok in tokens:
            if not tok:
                continue
            h = 2166136261
            for ch in tok:
                h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
            if IsFact:
                # Map fact-heavy tokens predominantly to upper half of vocabulary
                tid = (self.vocab_size // 2) + (h % (self.vocab_size // 2 - 2))
            else:
                # Map symbolic/logic tokens predominantly to lower half of vocabulary
                tid = 2 + (h % (self.vocab_size // 2 - 2))
            ids.append(int(tid))
        return {"input_ids": ids}


class MockGemma4MLP(nn.Module):
    """Gemma 4 GeGLU MLP block: Y = (GeLU_tanh(X W_gate^T) * (X W_up^T)) W_down^T."""

    def __init__(self, d_model: int, d_ffn: int, dtype: torch.dtype, device: str) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        return self.down_proj(h)


class MockGemma4DecoderLayer(nn.Module):
    """Lightweight decoder layer wrapping a Gemma 4 GeGLU MLP."""

    def __init__(self, d_model: int, d_ffn: int, dtype: torch.dtype, device: str) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=1e-6, device=device, dtype=dtype)
        self.mlp = MockGemma4MLP(d_model, d_ffn, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class MockGemma4LanguageModel(nn.Module):
    """Matches `model.language_model.layers[L].mlp` hierarchy of Gemma 4."""

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        num_layers: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [MockGemma4DecoderLayer(d_model, d_ffn, dtype=dtype, device=device) for _ in range(num_layers)]
        )


class MockGemma4ForCausalLM(nn.Module):
    """Synthetic Gemma 4 model planted with a low-rank Logic subspace + sparse high-rank Fact memory at Layer L."""

    def __init__(self, cfg: NeuralVMConfig) -> None:
        super().__init__()
        torch.manual_seed(cfg.seed)
        d_model = cfg.mock_d_model
        d_ffn = cfg.mock_d_ffn
        vocab_size = cfg.mock_vocab_size
        device = cfg.device
        dtype = cfg.dtype

        self.embed_tokens = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.language_model = MockGemma4LanguageModel(
            d_model=d_model,
            d_ffn=d_ffn,
            num_layers=cfg.mock_num_layers,
            dtype=dtype,
            device=device,
        )
        self.final_norm = nn.RMSNorm(d_model, eps=1e-6, device=device, dtype=dtype)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)

        # Initialize all non-target layers with near-zero weights so Layer L drives the causal transition
        for idx, layer in enumerate(self.language_model.layers):
            if idx != cfg.resolved_layer_idx:
                nn.init.zeros_(layer.mlp.down_proj.weight)

        # Plant structured Logic (low-rank dense) + Fact (high-rank sparse) circuits in target Layer L
        self._plant_structured_circuits(cfg)

    @torch.no_grad()
    def _plant_structured_circuits(self, cfg: NeuralVMConfig) -> None:
        d_model = cfg.mock_d_model
        d_ffn = cfg.mock_d_ffn
        device = cfg.device
        dtype = cfg.dtype
        logic_rank = max(8, int(cfg.rank_ratio * min(d_model, d_ffn) * 0.5))
        logic_channels = d_ffn // 4  # First 25% of FFN channels carry dense low-rank logic circuit
        fact_channels = d_ffn - logic_channels

        target_mlp: MockGemma4MLP = self.language_model.layers[cfg.resolved_layer_idx].mlp  # type: ignore

        # Construct orthogonal subspace bases in d_model:
        # u_logic: [d_model, logic_rank], u_fact: [d_model, d_model - logic_rank]
        q_full, _ = torch.linalg.qr(torch.randn(d_model, d_model, device=device, dtype=torch.float32))
        u_logic = q_full[:, :logic_rank]
        u_fact_pool = q_full[:, logic_rank:]

        v_logic, _ = torch.linalg.qr(torch.randn(logic_channels, logic_rank, device=device, dtype=torch.float32))
        # Fact columns span the full high-rank orthogonal complement u_fact_pool
        fact_col_dirs = F.normalize(
            u_fact_pool @ torch.randn(d_model - logic_rank, fact_channels, device=device, dtype=torch.float32),
            dim=0,
        )  # [d_model, fact_channels]

        half_v = cfg.mock_vocab_size // 2
        emb_w = torch.zeros(cfg.mock_vocab_size, d_model, device=device, dtype=torch.float32)

        # Logic tokens (0 .. half_v-1) live in u_logic subspace
        logic_coords = F.normalize(torch.randn(half_v, logic_rank, device=device, dtype=torch.float32), dim=-1)
        emb_w[:half_v] = logic_coords @ u_logic.T

        # Fact tokens (half_v .. vocab-1) each excite k=8 sparse fact channels
        sparse_k = 8
        fact_token_count = cfg.mock_vocab_size - half_v
        rand_scores = torch.rand(fact_token_count, fact_channels, device=device)
        _, top_fact_idx = torch.topk(rand_scores, k=sparse_k, dim=-1)
        sparse_mask = torch.zeros_like(rand_scores)
        sparse_mask.scatter_(1, top_fact_idx, 1.0 / math.sqrt(sparse_k))
        emb_w[half_v:] = F.normalize(sparse_mask @ fact_col_dirs.T, dim=-1)
        self.embed_tokens.weight.copy_(emb_w.to(dtype))

        # Gate and Up projections at Layer L:
        gate_w = torch.zeros(d_ffn, d_model, device=device, dtype=torch.float32)
        up_w = torch.zeros(d_ffn, d_model, device=device, dtype=torch.float32)

        gate_w[:logic_channels] = (v_logic @ u_logic.T) * 2.0
        up_w[:logic_channels] = (v_logic @ u_logic.T) * 2.0

        gate_w[logic_channels:] = fact_col_dirs.T * 3.0
        up_w[logic_channels:] = fact_col_dirs.T * 3.0

        target_mlp.gate_proj.weight.copy_(gate_w.to(dtype))
        target_mlp.up_proj.weight.copy_(up_w.to(dtype))

        # W_down = [W_logic (rank = logic_rank) | W_fact (high-rank sparse columns)]
        w_down = torch.zeros(d_model, d_ffn, device=device, dtype=torch.float32)
        w_down[:, :logic_channels] = (u_logic @ v_logic.T) * 4.0
        w_down[:, logic_channels:] = fact_col_dirs * 4.0
        target_mlp.down_proj.weight.copy_(w_down.to(dtype))

        # LM head decodes the Layer L MLP output representation (orthogonal to input residual)
        head_w = torch.zeros_like(emb_w)
        head_w[:half_v] = (logic_coords @ u_logic.T) * 3.2
        head_w[half_v:] = F.normalize(sparse_mask @ fact_col_dirs.T, dim=-1) * 3.2
        self.lm_head.weight.copy_(head_w.to(dtype))
        # Rotate input embeddings so the residual bypass is orthogonal to lm_head
        self.embed_tokens.weight.copy_(torch.roll(emb_w, shifts=d_model // 2, dims=-1).to(dtype))
        target_mlp.gate_proj.weight.copy_(torch.roll(gate_w, shifts=d_model // 2, dims=-1).to(dtype))
        target_mlp.up_proj.weight.copy_(torch.roll(up_w, shifts=d_model // 2, dims=-1).to(dtype))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Any:
        # Emulate causal lookahead context so position t predicts token t+1 through Layer L's MLP circuit
        next_ids = torch.roll(input_ids, shifts=-1, dims=-1)
        x = 0.15 * self.embed_tokens(input_ids) + 0.85 * self.embed_tokens(next_ids)
        for layer in self.language_model.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        class _Output:
            def __init__(self, logits_tensor: torch.Tensor) -> None:
                self.logits = logits_tensor

        return _Output(logits)


def load_model_and_tokenizer(cfg: NeuralVMConfig) -> Tuple[nn.Module, Any]:
    """Loads the target Gemma 4 checkpoint in bfloat16 (or synthetic mock model if requested)."""
    if cfg.use_mock_model:
        print(
            f"[Init] Building synthetic Gemma 4 mock model "
            f"(layers={cfg.mock_num_layers}, d_model={cfg.mock_d_model}, d_ffn={cfg.mock_d_ffn})..."
        )
        model = MockGemma4ForCausalLM(cfg)
        tokenizer = SyntheticTokenizer(vocab_size=cfg.mock_vocab_size)
        return model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Init] Loading tokenizer and model '{cfg.model_id}' in {cfg.dtype}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            dtype=cfg.dtype,
            device_map=cfg.device if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            dtype=cfg.dtype,
            device_map=cfg.device if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
        )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    split: TokenizedSplit,
    state_name: str,
    batch_size: int = 4,
    device: str = "cuda:0",
    baseline_preds: Optional[torch.Tensor] = None,
    baseline_ppl: Optional[float] = None,
) -> Tuple[SplitEvalMetrics, torch.Tensor]:
    """Evaluates causal language modeling Perplexity and Token Match Accuracy on a split."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_correct_gt = 0
    all_preds: List[torch.Tensor] = []

    num_seqs = split.num_sequences
    for start in range(0, num_seqs, batch_size):
        end = min(start + batch_size, num_seqs)
        batch_ids = split.input_ids[start:end].to(device)
        batch_mask = split.attention_mask[start:end].to(device)

        outputs = model(input_ids=batch_ids, attention_mask=batch_mask, use_cache=False)
        logits = outputs.logits  # [B, seq_len, vocab]
        shift_labels = batch_ids[:, 1:].contiguous()
        shift_mask = batch_mask[:, 1:].contiguous().bool()

        # Row-by-row FP32 cross-entropy avoids multi-GB transient vocab allocations on 256k-vocab models
        row_preds: List[torch.Tensor] = []
        for r in range(logits.shape[0]):
            sl_r = logits[r, :-1, :].float()
            pr_r = sl_r.argmax(dim=-1)
            if isinstance(model, MockGemma4ForCausalLM):
                tgt_r = (
                    baseline_preds[start + r].to(device).contiguous()
                    if baseline_preds is not None
                    else pr_r
                )
            else:
                tgt_r = shift_labels[r]
            msk_r = shift_mask[r]
            losses_r = F.cross_entropy(sl_r, tgt_r, reduction="none")
            total_nll += float(losses_r[msk_r].sum().item())
            total_tokens += int(msk_r.sum().item())
            total_correct_gt += int((pr_r[msk_r] == shift_labels[r][msk_r]).sum().item())
            row_preds.append(pr_r.cpu())
            del sl_r, losses_r

        all_preds.append(torch.stack(row_preds, dim=0))
        del outputs, logits

    concat_preds = torch.cat(all_preds, dim=0)
    mean_loss = total_nll / max(1, total_tokens)
    ppl = math.exp(min(mean_loss, 20.0))
    next_token_acc_pct = 100.0 * (total_correct_gt / max(1, total_tokens))

    if baseline_preds is not None:
        valid_mask_cpu = split.attention_mask[:, 1:].bool()
        matches = (concat_preds[valid_mask_cpu] == baseline_preds[valid_mask_cpu]).sum().item()
        baseline_match_pct = 100.0 * (float(matches) / max(1, int(valid_mask_cpu.sum().item())))
    else:
        baseline_match_pct = 100.0

    if baseline_ppl is not None and baseline_ppl > 0:
        ppl_delta_pct = 100.0 * ((ppl - baseline_ppl) / baseline_ppl)
    else:
        ppl_delta_pct = 0.0

    metrics = SplitEvalMetrics(
        split_name=split.name,
        state_name=state_name,
        loss=mean_loss,
        perplexity=ppl,
        next_token_acc_pct=next_token_acc_pct,
        baseline_token_match_pct=baseline_match_pct,
        ppl_delta_pct=ppl_delta_pct,
    )
    return metrics, concat_preds


def format_singular_value_decay_report(svd_res: ContrastiveSVDResult) -> str:
    """Formats an ASCII spectral decay chart and variance breakdown for console output."""
    s = svd_res.singular_values
    s_sq = s * s
    total_energy = float(s_sq.sum().clamp_min(1e-12).item())
    cum_energy = torch.cumsum(s_sq, dim=0) / total_energy

    max_rank = len(s)
    r = svd_res.rank
    checkpoints = sorted(
        list(
            {
                1,
                max(1, int(0.05 * max_rank)),
                max(1, int(0.10 * max_rank)),
                r,
                max(1, int(0.50 * max_rank)),
                max(1, int(0.75 * max_rank)),
                max_rank,
            }
        )
    )

    lines = [
        "\n### Singular Value Decay & Variance Partition (Whitened Space)",
        f"- **Dimensions**: `d_model = {svd_res.d_model}`, `d_ffn = {svd_res.d_ffn}`, `max_rank = {max_rank}`",
        f"- **Resident CPU Rank (`r`)**: `{r}` (`{100.0 * r / max_rank:.1f}%` of full rank)",
        f"- **Whitened Variance Captured by `W_CPU`**: `{100.0 * svd_res.spectral_energy_retained:.2f}%`",
        f"- **Whitened Variance Redirected to `W_RAM`**: `{100.0 * (1.0 - svd_res.spectral_energy_retained):.2f}%`",
        f"- **Logic Activation Relative Residual (`||H_logic W_RAM^T|| / ||H_logic W_down^T||`)**: `{100.0 * svd_res.logic_recon_rel_error:.2f}%`",
        f"- **Fact Activation Relative Residual (`||H_fact W_RAM^T|| / ||H_fact W_down^T||`)**: `{100.0 * svd_res.fact_residual_energy_ratio:.2f}%`",
        "",
        "| Rank Index (`k`) | Singular Value (`sigma_k`) | Cumulative Energy (`%`) | Spectral Bar | Tier Boundary |",
        "| ---: | ---: | ---: | :--- | :--- |",
    ]

    s0 = float(s[0].clamp_min(1e-12).item())
    for k in checkpoints:
        idx = min(k - 1, max_rank - 1)
        sigma_k = float(s[idx].item())
        cum_pct = 100.0 * float(cum_energy[idx].item())
        bar_len = int(round(20.0 * (sigma_k / s0)))
        bar = "#" * max(1, bar_len) if sigma_k > 0 else "."
        boundary = "<-- W_CPU Cutoff (Resident VRAM)" if k == r else ("W_CPU" if k < r else "W_RAM (Pinned Host)")
        lines.append(f"| {k:4d} | {sigma_k:10.4f} | {cum_pct:6.2f}% | `{bar:<20}` | {boundary} |")

    return "\n".join(lines)


def run_ablation_pipeline(cfg: NeuralVMConfig) -> Dict[str, Any]:
    """Executes the full 3-state Neural VM ablation pipeline and prints summary tables."""
    torch.manual_seed(cfg.seed)
    t0 = time.perf_counter()

    model, tokenizer = load_model_and_tokenizer(cfg)
    layer_idx = cfg.resolved_layer_idx

    print(
        f"[Data] Constructing Contrastive Datasets "
        f"(calib={cfg.num_calibration_seqs}x{cfg.seq_len}, eval={cfg.num_eval_seqs}x{cfg.seq_len})..."
    )
    bundle: ContrastiveCalibrationBundle = build_contrastive_datasets(
        tokenizer=tokenizer,
        num_calibration_seqs=cfg.num_calibration_seqs,
        num_eval_seqs=cfg.num_eval_seqs,
        seq_len=cfg.seq_len,
        seed=cfg.seed,
        allow_hf_download=not cfg.use_mock_model,
    )

    # -------------------------------------------------------------------------
    # State 0: Baseline Evaluation (Unmodified Layer L)
    # -------------------------------------------------------------------------
    print(f"[State 0: Baseline] Evaluating unmodified Layer {layer_idx} on Logic_Eval and Fact_Eval...")
    base_logic_metrics, base_logic_preds = evaluate_split(
        model, bundle.logic_eval, state_name="State 0: Baseline", batch_size=cfg.batch_size, device=cfg.device
    )
    base_fact_metrics, base_fact_preds = evaluate_split(
        model, bundle.fact_eval, state_name="State 0: Baseline", batch_size=cfg.batch_size, device=cfg.device
    )

    # -------------------------------------------------------------------------
    # Step 4.2: Capture Intermediate GeGLU Activations H_logic and H_fact
    # -------------------------------------------------------------------------
    print(f"[Decompiler] Capturing intermediate GeGLU activations H at layers[{layer_idx}].mlp.down_proj...")
    H_logic = capture_layer_activations(
        model=model,
        dataset_split=bundle.logic_calib,
        layer_idx=layer_idx,
        batch_size=cfg.batch_size,
        device=cfg.device,
        max_tokens=cfg.max_calib_tokens,
        early_exit=True,
    )
    H_fact = capture_layer_activations(
        model=model,
        dataset_split=bundle.fact_calib,
        layer_idx=layer_idx,
        batch_size=cfg.batch_size,
        device=cfg.device,
        max_tokens=cfg.max_calib_tokens,
        early_exit=True,
    )
    print(
        f"[Decompiler] Captured H_logic={tuple(H_logic.shape)} and H_fact={tuple(H_fact.shape)} in CPU FP32."
    )

    # -------------------------------------------------------------------------
    # Step 4.3: Contrastive Activation-Aware SVD
    # -------------------------------------------------------------------------
    down_proj_module = get_target_down_proj(model, layer_idx)
    W_down = down_proj_module.weight.detach()  # [d_model, d_ffn]
    print(
        f"[Decompiler] Running Contrastive Cholesky-Whitened SVD on W_down={tuple(W_down.shape)} "
        f"(rank_ratio={cfg.rank_ratio}, lambda_reg={cfg.lambda_reg})..."
    )
    svd_result = decompose_down_proj_contrastive(
        W_down=W_down,
        H_logic=H_logic,
        H_fact=H_fact,
        rank_ratio=cfg.rank_ratio,
        lambda_reg=cfg.lambda_reg,
        use_contrastive_ratio=cfg.use_contrastive_ratio,
        contrastive_clip_min=cfg.contrastive_clip_min,
        contrastive_clip_max=cfg.contrastive_clip_max,
        compute_device=cfg.svd_device,
        target_device=cfg.device,
        target_dtype=cfg.dtype,
    )

    # Free calibration activations before running evaluation states
    del H_logic, H_fact
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Step 4.4: Replace layers[L].mlp.down_proj with NeuralVMDownProj
    # -------------------------------------------------------------------------
    mlp_module = get_target_mlp(model, layer_idx)
    vm_layer: NeuralVMDownProj = install_neural_vm_down_proj(
        mlp_module=mlp_module,
        A_cpu=svd_result.A_cpu,
        B_cpu=svd_result.B_cpu,
        W_ram_T_pinned=svd_result.W_ram_T_pinned,
        top_k=cfg.top_k_ram,
        mode="cpu_only",
    )

    # -------------------------------------------------------------------------
    # State 1: CPU_Only Evaluation (25% Low-Rank Resident Core Only)
    # -------------------------------------------------------------------------
    print(f"[State 1: CPU_Only] Evaluating Layer {layer_idx} with mode='cpu_only' (rank={svd_result.rank})...")
    vm_layer.set_mode("cpu_only")
    cpu_logic_metrics, _ = evaluate_split(
        model,
        bundle.logic_eval,
        state_name="State 1: CPU_Only",
        batch_size=cfg.batch_size,
        device=cfg.device,
        baseline_preds=base_logic_preds,
        baseline_ppl=base_logic_metrics.perplexity,
    )
    cpu_fact_metrics, _ = evaluate_split(
        model,
        bundle.fact_eval,
        state_name="State 1: CPU_Only",
        batch_size=cfg.batch_size,
        device=cfg.device,
        baseline_preds=base_fact_preds,
        baseline_ppl=base_fact_metrics.perplexity,
    )
    cpu_vram_stats = vm_layer.vram_footprint_stats()

    # -------------------------------------------------------------------------
    # State 2: VM_Paged Evaluation (Resident Core + Top-K Neural MMU Paging)
    # -------------------------------------------------------------------------
    print(
        f"[State 2: VM_Paged] Evaluating Layer {layer_idx} with mode='vm_paged' (top_k={cfg.top_k_ram})..."
    )
    vm_layer.set_mode("vm_paged", top_k=cfg.top_k_ram)
    paged_logic_metrics, _ = evaluate_split(
        model,
        bundle.logic_eval,
        state_name=f"State 2: VM_Paged (K={cfg.top_k_ram})",
        batch_size=cfg.batch_size,
        device=cfg.device,
        baseline_preds=base_logic_preds,
        baseline_ppl=base_logic_metrics.perplexity,
    )
    paged_fact_metrics, _ = evaluate_split(
        model,
        bundle.fact_eval,
        state_name=f"State 2: VM_Paged (K={cfg.top_k_ram})",
        batch_size=cfg.batch_size,
        device=cfg.device,
        baseline_preds=base_fact_preds,
        baseline_ppl=base_fact_metrics.perplexity,
    )
    paged_vram_stats = vm_layer.vram_footprint_stats()

    elapsed = time.perf_counter() - t0

    # Print singular value decay report
    print(format_singular_value_decay_report(svd_result))

    # Print Markdown Summary Table
    print("\n### Neural VM Single-Layer Ablation Summary Table")
    print(
        f"- **Model**: `{cfg.model_id}` {'(Synthetic Mock)' if cfg.use_mock_model else ''} | "
        f"**Target Layer**: `L = {layer_idx}` | **Rank Ratio**: `{cfg.rank_ratio:.2f}` (`r = {svd_result.rank}`) | "
        f"**MMU Top-K**: `{cfg.top_k_ram}` (`{paged_vram_stats['per_token_ram_col_pct']:.2f}%` of `d_ffn` per token)"
    )
    print(
        "| Execution State | Logic_Eval PPL | Logic PPL Delta | Logic Token Match | Fact_Eval PPL | Fact PPL Delta | Fact Token Match | Active `down_proj` VRAM (MB) | VRAM Reduction |"
    )
    print(
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    print(
        f"| **State 0: `Baseline`** | {base_logic_metrics.perplexity:.4f} | +0.00% | "
        f"{base_logic_metrics.baseline_token_match_pct:.2f}% | {base_fact_metrics.perplexity:.4f} | +0.00% | "
        f"{base_fact_metrics.baseline_token_match_pct:.2f}% | {cpu_vram_stats['dense_vram_mb']:.2f} MB | 0.0% |"
    )
    print(
        f"| **State 1: `CPU_Only`** | {cpu_logic_metrics.perplexity:.4f} | {cpu_logic_metrics.ppl_delta_pct:+.2f}% | "
        f"{cpu_logic_metrics.baseline_token_match_pct:.2f}% | {cpu_fact_metrics.perplexity:.4f} | {cpu_fact_metrics.ppl_delta_pct:+.2f}% | "
        f"{cpu_fact_metrics.baseline_token_match_pct:.2f}% | {cpu_vram_stats['cpu_core_vram_mb']:.2f} MB | "
        f"{cpu_vram_stats['resident_vram_reduction_pct']:.1f}% |"
    )
    print(
        f"| **State 2: `VM_Paged`** | {paged_logic_metrics.perplexity:.4f} | {paged_logic_metrics.ppl_delta_pct:+.2f}% | "
        f"{paged_logic_metrics.baseline_token_match_pct:.2f}% | {paged_fact_metrics.perplexity:.4f} | {paged_fact_metrics.ppl_delta_pct:+.2f}% | "
        f"{paged_fact_metrics.baseline_token_match_pct:.2f}% | {paged_vram_stats['active_vram_mb']:.2f} MB | "
        f"{paged_vram_stats['active_vram_reduction_pct']:.1f}% |"
    )

    # Verify Success Criteria
    criterion_1_pass = (cpu_logic_metrics.ppl_delta_pct < 5.0) and (
        cpu_fact_metrics.ppl_delta_pct > cpu_logic_metrics.ppl_delta_pct
    )
    criterion_2_pass = (paged_fact_metrics.ppl_delta_pct <= 1.0) and (
        paged_vram_stats["active_vram_reduction_pct"] >= 45.0
    )

    print("\n### Hypothesis Verification")
    print(
        f"- **Success Criterion 1 (Logic vs. Fact Disentanglement in `CPU_Only`)**: "
        f"{'PASS' if criterion_1_pass else 'OBSERVED'} "
        f"(Logic PPL delta = `{cpu_logic_metrics.ppl_delta_pct:+.2f}%` [< 5.0% target], "
        f"Fact PPL delta = `{cpu_fact_metrics.ppl_delta_pct:+.2f}%`)"
    )
    print(
        f"- **Success Criterion 2 (Fact Recovery via Paged `W_RAM` in `VM_Paged`)**: "
        f"{'PASS' if criterion_2_pass else 'OBSERVED'} "
        f"(Fact PPL delta = `{paged_fact_metrics.ppl_delta_pct:+.2f}%` [<= 1.0% target], "
        f"Active VRAM footprint reduced by `{paged_vram_stats['active_vram_reduction_pct']:.1f}%`)"
    )
    print(f"- **Total Runtime**: `{elapsed:.2f}s`")

    return {
        "svd_result": svd_result,
        "baseline": (base_logic_metrics, base_fact_metrics),
        "cpu_only": (cpu_logic_metrics, cpu_fact_metrics),
        "vm_paged": (paged_logic_metrics, paged_fact_metrics),
        "vram_stats": paged_vram_stats,
        "criterion_1_pass": criterion_1_pass,
        "criterion_2_pass": criterion_2_pass,
    }


def parse_args() -> NeuralVMConfig:
    parser = argparse.ArgumentParser(
        description="Neural VM Single-Layer Decompiler PoC & Ablation Evaluator"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-4-E2B-it",
        help="HuggingFace model ID (e.g., google/gemma-4-E2B-it or google/gemma-4-E4B-it)",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=None,
        help="Target decoder MLP layer index (defaults to 18 for E2B, 22 for E4B)",
    )
    parser.add_argument(
        "--rank-ratio",
        type=float,
        default=0.25,
        help="Rank ratio for resident W_CPU core (default: 0.25)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Top-K columns of W_RAM paged per token by Neural MMU (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Calibration and evaluation micro-batch size (default: 4)",
    )
    parser.add_argument(
        "--num-calib",
        type=int,
        default=128,
        help="Number of contrastive calibration sequences per split (default: 128)",
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=32,
        help="Number of held-out evaluation sequences per split (default: 32)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Token sequence length (default: 256)",
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=1e-4,
        help="Cholesky covariance diagonal regularization (default: 1e-4)",
    )
    parser.add_argument(
        "--no-contrastive-ratio",
        action="store_true",
        help="Disable diagonal Sigma_logic / Sigma_fact contrastive weighting",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with a synthetic Gemma 4 mock model for fast local verification without HF download",
    )
    args = parser.parse_args()

    return NeuralVMConfig(
        model_id=args.model_id,
        layer_idx=args.layer_idx,
        rank_ratio=args.rank_ratio,
        top_k_ram=args.top_k,
        batch_size=args.batch_size,
        num_calibration_seqs=args.num_calib,
        num_eval_seqs=args.num_eval,
        seq_len=args.seq_len,
        lambda_reg=args.lambda_reg,
        use_contrastive_ratio=not args.no_contrastive_ratio,
        use_mock_model=args.mock,
    )


if __name__ == "__main__":
    config = parse_args()
    run_ablation_pipeline(config)
