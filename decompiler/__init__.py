"""Decompiler package for activation capture, contrastive SVD, and sequential multi-layer cascade."""

from decompiler.hooks import (
    DownProjActivationHook,
    capture_layer_activations,
    get_target_down_proj,
    get_target_mlp,
    resolve_decoder_layers,
)
from decompiler.contrastive_svd import (
    ContrastiveSVDResult,
    compute_covariance_matrix,
    decompose_down_proj_contrastive,
)
from decompiler.sequential_cascade import (
    LayerCascadeStats,
    MultiLayerActivationCollector,
    capture_multi_layer_teacher_activations,
    decompile_naive_parallel_stack,
    decompile_sequential_cascade_stack,
    decompose_layer_with_drift_compensation,
    restore_baseline_down_proj_stack,
    snapshot_original_mlp_weights,
)

__all__ = [
    "DownProjActivationHook",
    "capture_layer_activations",
    "get_target_down_proj",
    "get_target_mlp",
    "resolve_decoder_layers",
    "ContrastiveSVDResult",
    "compute_covariance_matrix",
    "decompose_down_proj_contrastive",
    "LayerCascadeStats",
    "MultiLayerActivationCollector",
    "capture_multi_layer_teacher_activations",
    "decompile_naive_parallel_stack",
    "decompile_sequential_cascade_stack",
    "decompose_layer_with_drift_compensation",
    "restore_baseline_down_proj_stack",
    "snapshot_original_mlp_weights",
]
