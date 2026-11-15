"""Forward hooks for capturing intermediate GeGLU activations H at layers[L].mlp.down_proj."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from data.calibration import TokenizedSplit


class _EarlyExitAfterHook(Exception):
    """Internal control-flow signal to halt forward pass once Layer L activations are captured."""


def resolve_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locates the decoder layer ModuleList across Gemma 4 / CausalLM wrapper variants.

    Checks `model.language_model.layers` first per spec, followed by common HF paths.
    """
    candidate_paths = [
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "layers"),
        ("model", "layers"),
        ("layers",),
    ]
    for path in candidate_paths:
        curr = model
        found = True
        for attr in path:
            if hasattr(curr, attr):
                curr = getattr(curr, attr)
            else:
                found = False
                break
        if found and isinstance(curr, (nn.ModuleList, list)):
            return curr

    raise AttributeError(
        "Could not locate decoder layers on model. Tried `model.language_model.layers`, "
        "`model.model.layers`, and `model.layers`."
    )


def get_target_mlp(model: nn.Module, layer_idx: int) -> nn.Module:
    """Returns `layers[layer_idx].mlp` from the target Gemma 4 model."""
    layers = resolve_decoder_layers(model)
    if not (0 <= layer_idx < len(layers)):
        raise IndexError(f"Requested layer_idx={layer_idx}, but model has {len(layers)} layers.")
    layer = layers[layer_idx]
    if not hasattr(layer, "mlp"):
        raise AttributeError(f"Decoder layer {layer_idx} ({type(layer).__name__}) has no `.mlp` attribute.")
    return layer.mlp


def get_target_down_proj(model: nn.Module, layer_idx: int) -> nn.Module:
    """Returns `layers[layer_idx].mlp.down_proj` from the target Gemma 4 model."""
    mlp = get_target_mlp(model, layer_idx)
    if not hasattr(mlp, "down_proj"):
        raise AttributeError(f"MLP block at layer {layer_idx} ({type(mlp).__name__}) has no `.down_proj` attribute.")
    return mlp.down_proj


class DownProjActivationHook:
    """Registers a PyTorch forward hook on `layers[L].mlp.down_proj` to capture input H.

    In Gemma 4:
        H = GeLU_tanh(X @ W_gate^T) * (X @ W_up^T)
        Y = H @ W_down^T
    Capturing the input `inputs[0]` to `down_proj` yields H of shape [batch, seq_len, d_ffn].
    All captured token activations are flattened to [N, d_ffn] and immediately offloaded to
    CPU in FP32 without storing gradients.
    """

    def __init__(
        self,
        down_proj_module: nn.Module,
        max_tokens: Optional[int] = None,
        early_exit: bool = True,
    ) -> None:
        self.down_proj_module = down_proj_module
        self.max_tokens = max_tokens
        self.early_exit = early_exit
        self.captured_batches: List[torch.Tensor] = []
        self.total_tokens: int = 0
        self._active_mask: Optional[torch.Tensor] = None
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def set_batch_mask(self, attention_mask: Optional[torch.Tensor]) -> None:
        """Sets the attention mask [batch, seq_len] for filtering padding tokens in the current batch."""
        self._active_mask = attention_mask

    def _hook_fn(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not inputs:
            return
        h = inputs[0].detach()  # [batch, seq_len, d_ffn] or [N, d_ffn]
        if h.dim() == 3:
            if self._active_mask is not None:
                mask = self._active_mask.to(device=h.device, dtype=torch.bool)
                flat_h = h[mask]  # [valid_tokens, d_ffn]
            else:
                flat_h = h.reshape(-1, h.shape[-1])
        elif h.dim() == 2:
            flat_h = h
        else:
            flat_h = h.view(-1, h.shape[-1])

        # Offload immediately to CPU FP32
        flat_h_cpu = flat_h.to(device="cpu", dtype=torch.float32, non_blocking=False)

        if self.max_tokens is not None:
            remaining = self.max_tokens - self.total_tokens
            if remaining <= 0:
                if self.early_exit:
                    raise _EarlyExitAfterHook()
                return
            if flat_h_cpu.shape[0] > remaining:
                flat_h_cpu = flat_h_cpu[:remaining]

        self.captured_batches.append(flat_h_cpu)
        self.total_tokens += int(flat_h_cpu.shape[0])

        if self.early_exit:
            raise _EarlyExitAfterHook()

    def __enter__(self) -> "DownProjActivationHook":
        self._handle = self.down_proj_module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def get_concatenated_activations(self) -> torch.Tensor:
        """Returns the full captured activation matrix H in R^{N x d_ffn} (CPU, float32)."""
        if not self.captured_batches:
            raise RuntimeError("No activations were captured by DownProjActivationHook.")
        return torch.cat(self.captured_batches, dim=0)


@torch.no_grad()
def capture_layer_activations(
    model: nn.Module,
    dataset_split: TokenizedSplit,
    layer_idx: int,
    batch_size: int = 4,
    device: str = "cuda:0",
    max_tokens: Optional[int] = None,
    early_exit: bool = True,
) -> torch.Tensor:
    """Runs calibration sequences through `model` and captures H at `layers[layer_idx].mlp.down_proj`.

    Returns:
        H: Tensor of shape [N, d_ffn] in CPU FP32.
    """
    model.eval()
    down_proj = get_target_down_proj(model, layer_idx)
    num_seqs = dataset_split.num_sequences

    with DownProjActivationHook(
        down_proj_module=down_proj,
        max_tokens=max_tokens,
        early_exit=early_exit,
    ) as collector:
        for start in range(0, num_seqs, batch_size):
            end = min(start + batch_size, num_seqs)
            batch_ids = dataset_split.input_ids[start:end].to(device)
            batch_mask = dataset_split.attention_mask[start:end].to(device)
            collector.set_batch_mask(batch_mask)

            try:
                model(input_ids=batch_ids, attention_mask=batch_mask, use_cache=False)
            except _EarlyExitAfterHook:
                pass

            if max_tokens is not None and collector.total_tokens >= max_tokens:
                break

    return collector.get_concatenated_activations()
