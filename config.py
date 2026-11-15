"""Hyperparameter and runtime configuration for the Neural VM Single-Layer Decompiler PoC."""

from dataclasses import dataclass
from typing import Optional
import torch


DEFAULT_LAYER_BY_MODEL = {
    "google/gemma-4-E2B-it": 18,
    "google/gemma-4-E2B": 18,
    "google/gemma-4-E4B-it": 22,
    "google/gemma-4-E4B": 22,
}


@dataclass
class NeuralVMConfig:
    """Configuration for single-layer Neural VM decompilation and evaluation."""

    # Model & target layer settings
    model_id: str = "google/gemma-4-E2B-it"
    layer_idx: Optional[int] = None  # Defaults to 18 for E2B or 22 for E4B

    # Contrastive SVD & Neural MMU hyperparameters
    rank_ratio: float = 0.25
    top_k_ram: int = 64
    lambda_reg: float = 1e-4
    use_contrastive_ratio: bool = True
    contrastive_clip_min: float = 0.55
    contrastive_clip_max: float = 1.80

    # Calibration & evaluation dataset parameters
    num_calibration_seqs: int = 128
    num_eval_seqs: int = 32
    seq_len: int = 256
    batch_size: int = 4
    max_calib_tokens: Optional[int] = 32768  # 128 * 256 tokens

    # Execution & hardware parameters
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    svd_device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    seed: int = 42

    # Synthetic/mock mode for local architectural validation without full HF checkpoint
    use_mock_model: bool = False
    mock_d_model: int = 2048
    mock_d_ffn: int = 8192
    mock_num_layers: int = 26
    mock_vocab_size: int = 32000

    # Phase 2 Multi-Layer Cascade & Async Lookahead MMU parameters
    start_layer: int = 10
    end_layer: int = 25  # Inclusive: 16 layers [10..25]
    router_rank: int = 64
    closed_loop_ridge: float = 1e-2
    phase2_calib_seqs: int = 32  # 32 * 256 = 8192 tokens per domain for fast 16-layer sequential SVD

    # Phase 3 Full-MLP Tri-Matrix Decompilation & Clustered Memory Pages parameters
    num_pages: int = 64
    max_active_pages: int = 4  # Strict cap: 4 / 64 = 6.25% of W_RAM (768 / 12288 cols)
    num_fact_pages: int = 4  # Fact-selective Knowledge Pages displaced in CPU_Only mode
    decode_tokens: int = 128  # Single-token autoregressive decode benchmark length

    # Phase 4 Quantized DMA Pages, Full-Stack (28-Layer) + PLE Offloading, and GSM8K Zero-Shot parameters
    phase4_start_layer: int = 4
    phase4_end_layer: int = 31  # Inclusive: 28 interior Full-MLP blocks [4..31]
    use_int8_pages: bool = True
    offload_ple: bool = True
    gsm8k_eval_problems: int = 50

    # Phase 5 Cross-Model Subspace Alignment & Teacher-to-Student Knowledge Cartridges parameters
    teacher_model_id: str = "google/gemma-4-E4B-it"
    teacher_d_model: int = 2560
    teacher_num_layers: int = 42
    student_num_layers: int = 35
    cartridge_blend_alpha: float = 0.30
    cartridge_ridge_lambda: float = 5e-2

    def __post_init__(self) -> None:
        if self.layer_idx is None:
            self.layer_idx = DEFAULT_LAYER_BY_MODEL.get(self.model_id, 18)

    @property
    def resolved_layer_idx(self) -> int:
        return self.layer_idx if self.layer_idx is not None else 18
