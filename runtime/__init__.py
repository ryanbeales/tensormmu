"""Runtime package for Neural MMU paged inference and dual-stream async lookahead prefetching."""

from runtime.neural_mmu import NeuralVMDownProj, install_neural_vm_down_proj
from runtime.lookahead_mmu import (
    LookaheadNeuralVMDownProj,
    SubspaceLookaheadRouter,
    VALID_PAGING_STRATEGIES,
)

__all__ = [
    "NeuralVMDownProj",
    "install_neural_vm_down_proj",
    "LookaheadNeuralVMDownProj",
    "SubspaceLookaheadRouter",
    "VALID_PAGING_STRATEGIES",
]
