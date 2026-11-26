"""Phase C2: Streaming Layer-by-Layer `.safetensors` Compiler & `.nvm` Disk Cartridge Serializer/Loader.

Eliminates the monolithic BF16 `AutoModelForCausalLM.from_pretrained()` RAM/VRAM bottleneck:
  1. Streams a HuggingFace `.safetensors` checkpoint (single-file or multi-shard) ONE decoder layer at a time
     (`O(1)` peak RAM & VRAM in model depth `L`), enabling 27B-70B compilation on a single 24 GB GPU + 34 GB RAM PC.
  2. Optionally compresses `self_attn` projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) into register-fused
     `PhaseCQuantLinear` (`INT8` row-wise or `INT4` group-wise) alongside `BitTieredTLBMLP`.
  3. Serializes compiled models into standalone `.nvm` Neural VM disk cartridges (`base_resident.safetensors`,
     `ple_host.safetensors`, per-layer `layer_XXX_core.safetensors` + `layer_XXX_pages.safetensors`, `manifest.json`).
  4. Cold-boots `.nvm` cartridges directly onto GPU in seconds (`< 5s` on `E4B`) with zero dense BF16 allocation,
     supporting both `pinned_ram` (`~24 GB/s` PCIe DMA) and zero-RAM `nvme_mmap` OS-paged cold specialist storage.
"""

from collections import UserDict
from dataclasses import asdict
import gc
import json
import math
import os
import shutil
import struct
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.calibration import ContrastiveCalibrationBundle, TokenizedSplit
from decompiler.full_model_vm import DeterministicPLEOffloader
from decompiler.hooks import get_target_mlp, resolve_decoder_layers
from decompiler.phase_b_compiler import PhaseBLayerStats, compile_phase_b_layer
from runtime.gpu_page_tlb import (
    BitTieredTLBMLP,
    SharedBatchTLBController,
    SharedQuantLUTPool,
)
from runtime.moe_expert_tlb import (
    SparseMoETLBExperts,
    auto_configure_moe_l2_zero_copy,
    install_moe_intra_layer_overlap,
    quantize_pack_moe_experts_uint4,
)
from runtime.moe_tlb_kernels import ContiguousExpertPageLayout, get_shared_moe_workspace
from runtime.phase_c_kernels import get_shared_phase_c_workspace
from runtime.quant_linear_phase_c import PhaseCQuantLinear

VALID_ATTN_QUANT_MODES = ("none", "int8", "int4", "hybrid_int8_int4")
VALID_BOUNDARY_MLP_QUANT_MODES = ("none", "int8", "int4")
VALID_LM_HEAD_QUANT_MODES = ("none", "int8")
VALID_PAGE_STORAGE_MODES = ("pinned_ram", "nvme_mmap")

_SAFETENSOR_NP_DTYPES = {
    "U8": (np.uint8, torch.uint8),
    "I8": (np.int8, torch.int8),
    "I16": (np.int16, torch.int16),
    "U16": (np.uint16, None),
    "I32": (np.int32, torch.int32),
    "I64": (np.int64, torch.int64),
    "F16": (np.float16, torch.float16),
    "BF16": (np.uint16, torch.bfloat16),
    "F32": (np.float32, torch.float32),
    "BOOL": (np.bool_, torch.bool),
}


def _is_complete_hf_snapshot(cand: str) -> bool:
    if not os.path.isfile(os.path.join(cand, "config.json")):
        return False
    return any(fn.endswith(".safetensors") for fn in os.listdir(cand))


def resolve_hf_snapshot_dir(model_id_or_path: str) -> str:
    """Resolves a local directory or cached HuggingFace Hub repo ID to its local snapshot path."""
    if os.path.isdir(model_id_or_path):
        return os.path.abspath(model_id_or_path)

    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    repo_folder = os.path.join(cache_root, "models--" + model_id_or_path.replace("/", "--"))
    snaps_dir = os.path.join(repo_folder, "snapshots")
    if os.path.isdir(snaps_dir):
        ref_main = os.path.join(repo_folder, "refs", "main")
        if os.path.isfile(ref_main):
            with open(ref_main, "r", encoding="utf-8") as f:
                rev = f.read().strip()
            cand = os.path.join(snaps_dir, rev)
            if _is_complete_hf_snapshot(cand):
                return os.path.abspath(cand)
        for sub in sorted(os.listdir(snaps_dir), reverse=True):
            cand = os.path.join(snaps_dir, sub)
            if _is_complete_hf_snapshot(cand):
                return os.path.abspath(cand)

    from huggingface_hub import snapshot_download

    ig = ["*.bin", "*.pth", "original/*", ".eval_results/*"]
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    try:
        return snapshot_download(repo_id=model_id_or_path, local_files_only=True, ignore_patterns=ig)
    except Exception:
        if offline:
            raise
        return snapshot_download(repo_id=model_id_or_path, local_files_only=False, ignore_patterns=ig)


def _parse_safetensors_header(filepath: str) -> Tuple[int, Dict[str, Any]]:
    """Reads the 8-byte little-endian header length and JSON tensor metadata of a `.safetensors` file."""
    with open(filepath, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"Corrupt safetensors header in '{filepath}'.")
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        header = json.loads(f.read(header_len).decode("utf-8"))
    header.pop("__metadata__", None)
    return 8 + int(header_len), header


def _read_safetensor_slice_exact(
    fp: Any,
    data_base_offset: int,
    info: Dict[str, Any],
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Reads a single tensor via 64-bit `seek + readinto` without mapping multi-40GB shards into virtual memory."""
    dtype_str = info["dtype"]
    if dtype_str not in _SAFETENSOR_NP_DTYPES:
        raise ValueError(f"Unsupported safetensors dtype '{dtype_str}'.")
    _, torch_dt = _SAFETENSOR_NP_DTYPES[dtype_str]
    shape = tuple(int(d) for d in info["shape"])
    start_off, end_off = int(info["data_offsets"][0]), int(info["data_offsets"][1])
    nbytes = end_off - start_off

    buf_u8 = torch.empty((nbytes,), dtype=torch.uint8, device="cpu")
    if nbytes > 0:
        fp.seek(data_base_offset + start_off)
        mv = memoryview(buf_u8.numpy())
        read_so_far = 0
        while read_so_far < nbytes:
            n_got = fp.readinto(mv[read_so_far:])
            if not n_got:
                raise EOFError(f"Unexpected EOF reading {nbytes} bytes at offset {data_base_offset + start_off}.")
            read_so_far += n_got

    if dtype_str == "BF16":
        t = buf_u8.view(torch.bfloat16).reshape(shape)
    elif torch_dt is not None:
        t = buf_u8.view(torch_dt).reshape(shape)
    else:
        raise ValueError(f"Cannot view safetensors dtype '{dtype_str}' as torch.Tensor.")

    if dtype is not None and t.is_floating_point() and t.dtype != dtype:
        return t.to(device=device, dtype=dtype).contiguous()
    if str(t.device) != str(device):
        return t.to(device=device).contiguous()
    return t


def mmap_safetensor_tensor(filepath: str, key: str) -> torch.Tensor:
    """Memory-maps a single tensor inside a `.safetensors` file read-only (`0` anonymous RAM commit charge)."""
    data_base, header = _parse_safetensors_header(filepath)
    if key not in header:
        raise KeyError(f"Key '{key}' not found in '{filepath}'.")

    info = header[key]
    dtype_str = info["dtype"]
    if dtype_str not in _SAFETENSOR_NP_DTYPES:
        raise ValueError(f"Unsupported safetensors dtype '{dtype_str}' for mmap.")

    np_dt, torch_dt = _SAFETENSOR_NP_DTYPES[dtype_str]
    shape = tuple(int(d) for d in info["shape"])
    start_off = int(info["data_offsets"][0])
    byte_offset = data_base + start_off

    mm = np.memmap(filepath, dtype=np_dt, mode="r", offset=byte_offset, shape=shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        t = torch.from_numpy(mm)
    if dtype_str == "BF16":
        t = t.view(torch.bfloat16)
    elif torch_dt is not None and t.dtype != torch_dt:
        t = t.view(torch_dt)
    return t


class SafetensorShardReader:
    """On-demand 64-bit `seek + readinto` reader across single-file or multi-shard `.safetensors` checkpoints."""

    def __init__(self, snapshot_dir: str) -> None:
        self.snapshot_dir = os.path.abspath(snapshot_dir)
        index_path = os.path.join(self.snapshot_dir, "model.safetensors.index.json")
        self.key_to_file: Dict[str, str] = {}
        self._file_headers: Dict[str, Tuple[int, Dict[str, Any]]] = {}

        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            for k, rel_file in idx_data.get("weight_map", {}).items():
                fpath = os.path.join(self.snapshot_dir, rel_file)
                self.key_to_file[k] = fpath
                if fpath not in self._file_headers:
                    self._file_headers[fpath] = _parse_safetensors_header(fpath)
        else:
            single_path = os.path.join(self.snapshot_dir, "model.safetensors")
            st_files = (
                [single_path]
                if os.path.isfile(single_path)
                else sorted(
                    os.path.join(self.snapshot_dir, fn)
                    for fn in os.listdir(self.snapshot_dir)
                    if fn.endswith(".safetensors")
                )
            )
            if not st_files:
                raise FileNotFoundError(f"No `.safetensors` files found in '{self.snapshot_dir}'.")
            for fpath in st_files:
                data_base, hdr = _parse_safetensors_header(fpath)
                self._file_headers[fpath] = (data_base, hdr)
                for k in hdr.keys():
                    self.key_to_file[k] = fpath

    def keys(self) -> List[str]:
        return list(self.key_to_file.keys())

    def has_key(self, key: str) -> bool:
        return key in self.key_to_file

    def get_file_for_key(self, key: str) -> str:
        return self.key_to_file[key]

    def get_tensor(
        self,
        key: str,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        fpath = self.key_to_file[key]
        data_base, hdr = self._file_headers[fpath]
        with open(fpath, "rb") as fp:
            return _read_safetensor_slice_exact(fp, data_base, hdr[key], device=device, dtype=dtype)

    def get_tensor_shape(self, key: str) -> Tuple[int, ...]:
        fpath = self.key_to_file[key]
        _, hdr = self._file_headers[fpath]
        return tuple(int(d) for d in hdr[key]["shape"])

    def get_tensor_dim0_slice(
        self,
        key: str,
        start_idx: int,
        end_idx: int,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Streams rows `[start_idx:end_idx]` along `dim=0` of a large 3D/2D `.safetensors` tensor (`O(chunk)` RAM)."""
        fpath = self.key_to_file[key]
        data_base, hdr = self._file_headers[fpath]
        info = hdr[key]
        dtype_str = info["dtype"]
        if dtype_str not in _SAFETENSOR_NP_DTYPES:
            raise ValueError(f"Unsupported safetensors dtype '{dtype_str}'.")
        _, torch_dt = _SAFETENSOR_NP_DTYPES[dtype_str]
        shape = tuple(int(d) for d in info["shape"])
        if not shape:
            raise ValueError(f"Cannot slice scalar tensor '{key}'.")
        d0 = shape[0]
        s0 = max(0, min(int(start_idx), d0))
        e0 = max(s0, min(int(end_idx), d0))
        start_off, end_off = int(info["data_offsets"][0]), int(info["data_offsets"][1])
        total_bytes = end_off - start_off
        row_bytes = total_bytes // max(1, d0)
        slice_off = start_off + s0 * row_bytes
        slice_end = start_off + e0 * row_bytes
        slice_info = {
            "dtype": dtype_str,
            "shape": [e0 - s0, *shape[1:]],
            "data_offsets": [slice_off, slice_end],
        }
        with open(fpath, "rb") as fp:
            return _read_safetensor_slice_exact(fp, data_base, slice_info, device=device, dtype=dtype)

    def get_prefix_tensors(
        self,
        prefix: str,
        strip_prefix: bool = True,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        allowed_rel_keys: Optional[set[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        by_file: Dict[str, List[str]] = {}
        plen = len(prefix) if strip_prefix else 0
        for k, fpath in self.key_to_file.items():
            if k.startswith(prefix):
                rel = k[plen:]
                if allowed_rel_keys is not None and rel not in allowed_rel_keys:
                    continue
                by_file.setdefault(fpath, []).append(k)

        out: Dict[str, torch.Tensor] = {}
        for fpath, k_list in by_file.items():
            data_base, hdr = self._file_headers[fpath]
            # Sort keys by file offset for sequential NVMe streaming!
            k_sorted = sorted(k_list, key=lambda kk: int(hdr[kk]["data_offsets"][0]))
            with open(fpath, "rb") as fp:
                for k in k_sorted:
                    out[k[plen:]] = _read_safetensor_slice_exact(
                        fp, data_base, hdr[k], device=device, dtype=dtype
                    )
        return out


def _find_module_qualname(root: nn.Module, target: nn.Module) -> str:
    for name, mod in root.named_modules():
        if mod is target:
            return name
    raise RuntimeError("Target module not found inside root model.")


def _get_parent_and_attr(root: nn.Module, qualname: str) -> Tuple[nn.Module, str]:
    parts = qualname.split(".")
    cur = root
    for p in parts[:-1]:
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    return cur, parts[-1]


def _set_tensor_by_qualname(root: nn.Module, qualname: str, tensor: torch.Tensor) -> None:
    parent, attr = _get_parent_and_attr(root, qualname)
    if attr in parent._parameters:
        parent._parameters[attr] = nn.Parameter(tensor, requires_grad=False)
    elif attr in parent._buffers:
        parent._buffers[attr] = tensor
    else:
        setattr(parent, attr, nn.Parameter(tensor, requires_grad=False))


def _move_cpu_buffers_to_device(
    module: nn.Module,
    device: torch.device,
    exclude_prefixes: Tuple[str, ...] = (),
) -> None:
    for mod_name, submod in module.named_modules():
        if isinstance(submod, DeterministicPLEOffloader):
            continue
        if any(mod_name == p or mod_name.startswith(p + ".") for p in exclude_prefixes):
            continue
        for buf_name, buf in list(submod._buffers.items()):
            if buf is not None and buf.device.type == "cpu":
                submod._buffers[buf_name] = buf.to(device=device)


def _strip_unused_multimodal_towers(model: nn.Module) -> List[str]:
    """Removes unused vision/audio encoder towers on multimodal CausalLM wrappers (saves ~912 MB VRAM on E4B)."""
    stripped: List[str] = []
    containers = [model]
    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        containers.append(model.model)

    multimodal_attrs = (
        "vision_tower",
        "audio_tower",
        "embed_vision",
        "embed_audio",
        "multi_modal_projector",
    )
    for container in containers:
        cname = _find_module_qualname(model, container)
        for attr in multimodal_attrs:
            if hasattr(container, attr) and getattr(container, attr) is not None:
                qual = f"{cname}.{attr}" if cname else attr
                setattr(container, attr, None)
                stripped.append(qual)
    return stripped


def _resolve_ple_module_path(model: nn.Module) -> Optional[str]:
    lm = model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
    elif hasattr(model, "language_model"):
        lm = model.language_model
    if hasattr(lm, "embed_tokens_per_layer") and lm.embed_tokens_per_layer is not None:
        return _find_module_qualname(model, lm.embed_tokens_per_layer)
    return None


def _create_empty_skeleton(
    config_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    skip_multimodal: bool = True,
) -> Tuple[nn.Module, List[str]]:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
    except Exception:
        config = AutoConfig.from_pretrained(config_dir)

    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)

    stripped = _strip_unused_multimodal_towers(model) if skip_multimodal else []
    return model, stripped


def _evict_module_to_meta(module: nn.Module) -> None:
    """Replaces all parameters and buffers inside `module` with 0-byte `meta` placeholders."""
    for sub in module.modules():
        if isinstance(sub, (SharedQuantLUTPool, DeterministicPLEOffloader)):
            continue
        for k, p in list(sub._parameters.items()):
            if p is not None and p.device.type != "meta":
                sub._parameters[k] = nn.Parameter(
                    torch.empty(0, dtype=p.dtype, device="meta"), requires_grad=False
                )
        for k, b in list(sub._buffers.items()):
            if b is not None and b.device.type != "meta":
                sub._buffers[k] = torch.empty(0, dtype=b.dtype, device="meta")


def _tree_to_cpu_dedup(obj: Any, memo: Dict[Tuple[Any, ...], torch.Tensor]) -> Any:
    if isinstance(obj, torch.Tensor):
        tkey = (obj.data_ptr(), tuple(obj.shape), tuple(obj.stride()), obj.dtype)
        if tkey not in memo:
            memo[tkey] = obj.detach().to("cpu").contiguous()
        return memo[tkey]
    if isinstance(obj, tuple):
        return tuple(_tree_to_cpu_dedup(v, memo) for v in obj)
    if isinstance(obj, list):
        return [_tree_to_cpu_dedup(v, memo) for v in obj]
    if isinstance(obj, (dict, UserDict)):
        return {k: _tree_to_cpu_dedup(v, memo) for k, v in obj.items()}
    return obj


def _tree_to_device_dedup(
    obj: Any, device: torch.device, memo: Dict[Tuple[Any, ...], torch.Tensor]
) -> Any:
    if isinstance(obj, torch.Tensor):
        tkey = (obj.data_ptr(), tuple(obj.shape), tuple(obj.stride()), obj.dtype)
        if tkey not in memo:
            memo[tkey] = obj.to(device=device, non_blocking=True)
        return memo[tkey]
    if isinstance(obj, tuple):
        return tuple(_tree_to_device_dedup(v, device, memo) for v in obj)
    if isinstance(obj, list):
        return [_tree_to_device_dedup(v, device, memo) for v in obj]
    if isinstance(obj, dict):
        return {k: _tree_to_device_dedup(v, device, memo) for k, v in obj.items()}
    return obj


class _BatchCallEnvelope:
    """Stores initial `H_0`, `attention_mask_2d`, and deduplicated per-layer `(args, kwargs)` on CPU."""

    def __init__(self, split_kind: str, attn_mask_2d_cpu: torch.Tensor) -> None:
        self.split_kind = split_kind  # "logic" or "fact"
        self.attn_mask_2d_cpu = attn_mask_2d_cpu
        self.h0_cpu: Optional[torch.Tensor] = None
        self.layer_args_cpu: Dict[int, Tuple[Any, ...]] = {}
        self.layer_kwargs_cpu: Dict[int, Dict[str, Any]] = {}
        self._memo: Dict[Tuple[Any, ...], torch.Tensor] = {}


class _PreambleSpyLayer(nn.Module):
    """Zero-FLOP pass-through decoder layer used for 20 ms preamble & RoPE/mask/PLE envelope capture."""

    def __init__(self, orig_layer: nn.Module, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.gradient_checkpointing = False
        for attr in ("layer_type", "attention_type", "is_sliding", "block_type"):
            if hasattr(orig_layer, attr):
                setattr(self, attr, getattr(orig_layer, attr))
        self.active_envelope: Optional[_BatchCallEnvelope] = None

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        env = self.active_envelope
        if env is not None:
            if self.layer_idx == 0:
                env.h0_cpu = hidden_states.detach().to("cpu")
            clean_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ("shared_kv_states", "past_key_values")
            }
            env.layer_args_cpu[self.layer_idx] = _tree_to_cpu_dedup(args, env._memo)
            env.layer_kwargs_cpu[self.layer_idx] = _tree_to_cpu_dedup(clean_kwargs, env._memo)
        return hidden_states


class _StopAtMLPPreHook(Exception):
    """Halts a single-layer forward pass right after capturing `X_act` entering `layer.mlp`."""


def _unwrap_layer_output(out: Any) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out


def _resolve_eff_attn_quant(attn: nn.Module, layer: nn.Module, quant_mode: str) -> str:
    if quant_mode != "hybrid_int8_int4":
        return quant_mode
    ltype = str(
        getattr(attn, "layer_type", None)
        or getattr(layer, "layer_type", None)
        or getattr(attn, "attention_type", None)
        or getattr(layer, "attention_type", None)
        or ""
    ).lower()
    is_sliding = getattr(attn, "is_sliding", getattr(layer, "is_sliding", None))
    if ltype == "full_attention" or is_sliding is False:
        return "int8"
    return "int4"


def _quantize_layer_self_attn(
    layer: nn.Module,
    quant_mode: str,
    group_size: int,
    target_device: str,
    scale_dtype: torch.dtype,
    lut_pool: SharedQuantLUTPool,
) -> Dict[str, Dict[str, Any]]:
    """Compresses `layer.self_attn` (and optional `layer.linear_attn`) `nn.Linear` projections in-place."""
    if quant_mode == "none":
        return {}

    quant_meta: Dict[str, Dict[str, Any]] = {}
    if hasattr(layer, "self_attn") and layer.self_attn is not None:
        attn = layer.self_attn
        eff_mode = _resolve_eff_attn_quant(attn, layer, quant_mode)
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(attn, proj_name, None)
            if isinstance(lin, nn.Linear):
                qlin = PhaseCQuantLinear.from_linear(
                    lin,
                    quant_mode=eff_mode,
                    group_size=group_size,
                    target_device=target_device,
                    scale_dtype=scale_dtype,
                    lut_pool=lut_pool,
                )
                setattr(attn, proj_name, qlin)
                del lin
                quant_meta[f"self_attn.{proj_name}"] = {
                    "in_features": qlin.in_features,
                    "out_features": qlin.out_features,
                    "quant_mode": qlin.quant_mode,
                    "group_size": qlin.group_size,
                    "has_bias": qlin.bias is not None,
                    "has_wmin": qlin.wmin is not None,
                }

    if hasattr(layer, "linear_attn") and layer.linear_attn is not None:
        l_attn = layer.linear_attn
        eff_lin_mode = "int8" if quant_mode in ("int8", "hybrid_int8_int4") else quant_mode
        for proj_name in ("in_proj_qkvz", "out_proj"):
            lin = getattr(l_attn, proj_name, None)
            if isinstance(lin, nn.Linear):
                qlin = PhaseCQuantLinear.from_linear(
                    lin,
                    quant_mode=eff_lin_mode,
                    group_size=group_size,
                    target_device=target_device,
                    scale_dtype=scale_dtype,
                    lut_pool=lut_pool,
                )
                setattr(l_attn, proj_name, qlin)
                del lin
                quant_meta[f"linear_attn.{proj_name}"] = {
                    "in_features": qlin.in_features,
                    "out_features": qlin.out_features,
                    "quant_mode": qlin.quant_mode,
                    "group_size": qlin.group_size,
                    "has_bias": qlin.bias is not None,
                    "has_wmin": qlin.wmin is not None,
                }
    return quant_meta


def _quantize_layer_mlp_linear(
    layer: nn.Module,
    mlp: nn.Module,
    mlp_rel_name: str,
    quant_mode: str,
    group_size: int,
    target_device: str,
    scale_dtype: torch.dtype,
    lut_pool: SharedQuantLUTPool,
) -> Dict[str, Dict[str, Any]]:
    """Compresses boundary/uncompiled `mlp` (`gate_proj`, `up_proj`, `down_proj`) into `PhaseCQuantLinear`."""
    if quant_mode == "none":
        return {}
    quant_meta: Dict[str, Dict[str, Any]] = {}
    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        lin = getattr(mlp, proj_name, None)
        if isinstance(lin, nn.Linear):
            qlin = PhaseCQuantLinear.from_linear(
                lin,
                quant_mode=quant_mode,
                group_size=group_size,
                target_device=target_device,
                scale_dtype=scale_dtype,
                lut_pool=lut_pool,
            )
            setattr(mlp, proj_name, qlin)
            del lin
            quant_meta[f"{mlp_rel_name}.{proj_name}"] = {
                "in_features": qlin.in_features,
                "out_features": qlin.out_features,
                "quant_mode": qlin.quant_mode,
                "group_size": qlin.group_size,
                "has_bias": qlin.bias is not None,
                "has_wmin": qlin.wmin is not None,
            }
    return quant_meta


def _export_module_cpu_tensors(
    module: nn.Module,
    skip_submodules: Tuple[str, ...] = (),
) -> Dict[str, torch.Tensor]:
    """Exports all concrete parameters and persistent buffers of `module` to CPU dictionary."""
    out: Dict[str, torch.Tensor] = {}
    for mod_name, sub in module.named_modules():
        if isinstance(sub, SharedQuantLUTPool):
            continue
        if any(mod_name == s or mod_name.startswith(s + ".") for s in skip_submodules):
            continue
        prefix = f"{mod_name}." if mod_name else ""
        for p_name, p in sub.named_parameters(recurse=False):
            if p is not None and p.device.type != "meta":
                out[f"{prefix}{p_name}"] = p.detach().cpu().contiguous()
        non_pers = getattr(sub, "_non_persistent_buffers_set", set())
        for b_name, b in sub.named_buffers(recurse=False):
            if b_name in non_pers:
                continue
            if b is not None and b.device.type != "meta":
                out[f"{prefix}{b_name}"] = b.detach().cpu().contiguous()
    return out


def _copy_tokenizer_and_configs(src_dir: str, dst_dir: str) -> None:
    copy_Candidates = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
        "preprocessor_config.json",
    )
    for fn in copy_Candidates:
        sp = os.path.join(src_dir, fn)
        if os.path.isfile(sp):
            shutil.copy2(sp, os.path.join(dst_dir, fn))


@torch.no_grad()
def compile_streaming_to_nvm(
    model_id_or_path: str,
    nvm_dir: str,
    bundle: ContrastiveCalibrationBundle,
    layer_indices: Optional[List[int]] = None,
    boundary_front: int = 4,
    boundary_back: int = 3,
    hot_fraction: float = 0.11,
    num_pages: int = 64,
    tlb_slots: int = 12,
    decode_req_pages: int = 4,
    bridge_rank: int = 64,
    closed_loop_ridge: float = 0.08,
    attn_quant: str = "none",
    attn_group_size: int = 64,
    boundary_mlp_quant: str = "none",
    calib_batch_size: int = 4,
    max_calib_tokens: int = 8192,
    offload_ple: bool = True,
    save_standalone_ple: bool = True,
    skip_multimodal: bool = True,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Streams a `.safetensors` checkpoint one layer at a time and writes a standalone `.nvm` cartridge."""
    if attn_quant not in VALID_ATTN_QUANT_MODES:
        raise ValueError(f"Invalid attn_quant '{attn_quant}'. Expected {VALID_ATTN_QUANT_MODES}.")
    if boundary_mlp_quant not in VALID_BOUNDARY_MLP_QUANT_MODES:
        raise ValueError(
            f"Invalid boundary_mlp_quant '{boundary_mlp_quant}'. Expected {VALID_BOUNDARY_MLP_QUANT_MODES}."
        )

    t_start = time.perf_counter()
    dev = torch.device(device)
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    src_dir = resolve_hf_snapshot_dir(model_id_or_path)
    out_dir = os.path.abspath(nvm_dir)
    layers_out_dir = os.path.join(out_dir, "layers")
    os.makedirs(layers_out_dir, exist_ok=True)
    _copy_tokenizer_and_configs(src_dir, out_dir)

    reader = SafetensorShardReader(src_dir)
    model, stripped_towers = _create_empty_skeleton(src_dir, dtype=dtype, skip_multimodal=skip_multimodal)
    if torch.cuda.is_available() and dev.type == "cuda":
        if any(m.__class__.__name__ == "Qwen3NextGatedDeltaNet" for m in model.modules()):
            from runtime.qwen3_next_kernels import install_qwen3_next_fast_gdn

            install_qwen3_next_fast_gdn(model)
    decoder_layers = resolve_decoder_layers(model)
    num_total_layers = len(decoder_layers)
    layers_qualname = _find_module_qualname(model, decoder_layers)
    layers_prefix = f"{layers_qualname}."

    if layer_indices is None:
        if num_total_layers > (boundary_front + boundary_back):
            compiled_indices = list(range(boundary_front, num_total_layers - boundary_back))
        else:
            compiled_indices = list(range(num_total_layers))
    else:
        compiled_indices = sorted(layer_indices)
    compiled_set = set(compiled_indices)
    max_compiled_idx = max(compiled_indices) if compiled_indices else -1
    first_divergent_idx = (
        0
        if (attn_quant != "none" or boundary_mlp_quant != "none")
        else (min(compiled_indices) if compiled_indices else num_total_layers)
    )

    # 1. Materialize PLE on CPU/mmap (never touches GPU VRAM!) + Preamble modules on GPU
    ple_qualname = _resolve_ple_module_path(model)
    ple_prefix = f"{ple_qualname}." if ple_qualname else None
    ple_saved_mb = 0.0
    ple_meta: Optional[Dict[str, Any]] = None

    if offload_ple and ple_qualname is not None and reader.has_key(f"{ple_qualname}.weight"):
        ple_parent, ple_attr = _get_parent_and_attr(model, ple_qualname)
        orig_ple = getattr(ple_parent, ple_attr)
        raw_scale = getattr(orig_ple, "embed_scale", 1.0)
        ple_scale_float = float(raw_scale.item()) if isinstance(raw_scale, torch.Tensor) else float(raw_scale)
        ple_pad_idx = getattr(orig_ple, "padding_idx", None)

        ple_src_file = reader.get_file_for_key(f"{ple_qualname}.weight")
        ple_dst_path = os.path.join(out_dir, "ple_host.safetensors")
        if save_standalone_ple:
            w_ple_tmp = reader.get_tensor(f"{ple_qualname}.weight", device="cpu", dtype=dtype)
            save_file({"weight": w_ple_tmp}, ple_dst_path)
            del w_ple_tmp
            ple_mmap_w = mmap_safetensor_tensor(ple_dst_path, "weight")
            ple_source_mode = "standalone"
        else:
            ple_mmap_w = mmap_safetensor_tensor(ple_src_file, f"{ple_qualname}.weight")
            ple_source_mode = "source_ref"

        ple_offloader = DeterministicPLEOffloader.from_cpu_tensor(
            ple_mmap_w,
            embed_scale=ple_scale_float,
            padding_idx=ple_pad_idx,
        )
        ple_saved_mb = ple_offloader.offloaded_vram_mb
        setattr(ple_parent, ple_attr, ple_offloader)
        del orig_ple
        ple_meta = {
            "qualname": ple_qualname,
            "embed_scale": ple_scale_float,
            "padding_idx": ple_pad_idx,
            "storage": ple_source_mode,
            "source_file": ple_src_file if not save_standalone_ple else "ple_host.safetensors",
            "source_key": f"{ple_qualname}.weight" if not save_standalone_ple else "weight",
            "offloaded_vram_mb": ple_saved_mb,
        }

    # Materialize remaining non-layer preamble/head tensors onto GPU
    skipped_prefixes = tuple(stripped_towers)
    valid_model_keys = {k for k, _ in model.named_parameters()} | {k for k, _ in model.named_buffers()}
    base_cpu_tensors: Dict[str, torch.Tensor] = {}
    for key in reader.keys():
        if key.startswith(layers_prefix):
            continue
        if ple_meta is not None and ple_prefix is not None and key.startswith(ple_prefix):
            continue
        if any(key == sp or key.startswith(sp + ".") for sp in skipped_prefixes):
            continue
        if key not in valid_model_keys:
            continue
        t_gpu = reader.get_tensor(key, device=device, dtype=dtype)
        _set_tensor_by_qualname(model, key, t_gpu)
        base_cpu_tensors[key] = t_gpu.detach().cpu().contiguous()

    _move_cpu_buffers_to_device(model, dev, exclude_prefixes=(layers_qualname,))
    tie_word_embeddings = False
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight") and model.lm_head.weight.device.type == "meta":
        model.tie_weights()
        tie_word_embeddings = True

    save_file(base_cpu_tensors, os.path.join(out_dir, "base_resident.safetensors"))
    del base_cpu_tensors

    # 2. Capture exact Preamble envelopes (`H_0`, `per_layer_inputs`, RoPE, causal masks) in 20 ms!
    envelopes: List[_BatchCallEnvelope] = []
    if compiled_set:
        orig_meta_layers = [decoder_layers[i] for i in range(num_total_layers)]
        spy_layers = [_PreambleSpyLayer(orig_meta_layers[i], i) for i in range(num_total_layers)]
        for i in range(num_total_layers):
            decoder_layers[i] = spy_layers[i]
        model.eval()

        def _record_split_envelopes(split: TokenizedSplit, kind: str) -> None:
            tok_seen = 0
            n_rows = split.input_ids.shape[0]
            for s in range(0, n_rows, calib_batch_size):
                if tok_seen >= max_calib_tokens:
                    break
                e = min(n_rows, s + calib_batch_size)
                b_ids = split.input_ids[s:e].to(dev)
                b_mask = split.attention_mask[s:e].to(dev)
                env = _BatchCallEnvelope(split_kind=kind, attn_mask_2d_cpu=split.attention_mask[s:e].cpu())
                for spy in spy_layers:
                    spy.active_envelope = env
                try:
                    _ = model(input_ids=b_ids, attention_mask=b_mask, use_cache=False, logits_to_keep=1)
                except TypeError:
                    _ = model(input_ids=b_ids, attention_mask=b_mask, use_cache=False)
                for spy in spy_layers:
                    spy.active_envelope = None
                assert env.h0_cpu is not None
                envelopes.append(env)
                tok_seen += int(b_mask.sum().item())

        _record_split_envelopes(bundle.logic_calib, "logic")
        _record_split_envelopes(bundle.fact_calib, "fact")

        for i in range(num_total_layers):
            decoder_layers[i] = orig_meta_layers[i]
        del spy_layers, orig_meta_layers

    peak_preamble_vram_mb = (
        torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        if (torch.cuda.is_available() and dev.type == "cuda")
        else 0.0
    )

    # Evict Preamble (`embed_tokens`, `lm_head`, `per_layer_model_projection`, `norm`) from GPU VRAM
    # before starting the layer loop so peak compile VRAM is strictly O(1) single-layer!
    if tie_word_embeddings and hasattr(model, "lm_head"):
        model.lm_head.weight = nn.Parameter(torch.empty(0, dtype=dtype, device="meta"), requires_grad=False)
    for top_name, top_mod in model.named_children():
        if top_name == layers_qualname.split(".")[0]:
            continue
        _evict_module_to_meta(top_mod)
    # Also evict `embed_tokens`, `per_layer_model_projection`, `norm` inside `language_model`
    lm_container, _ = _get_parent_and_attr(model, layers_qualname)
    for child_name, child_mod in lm_container.named_children():
        if child_mod is decoder_layers or isinstance(child_mod, DeterministicPLEOffloader):
            continue
        _evict_module_to_meta(child_mod)
    gc.collect()
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    # 3. Lockstep Streaming Layer-by-Layer Compilation (`idx = 0 .. num_total_layers - 1`)
    lut_pool = SharedQuantLUTPool(device=dev, dtype=dtype)
    H_teacher_cpu: List[torch.Tensor] = [env.h0_cpu for env in envelopes]  # type: ignore[misc]
    H_student_cpu: List[torch.Tensor] = [env.h0_cpu for env in envelopes]  # type: ignore[misc]
    kv_teacher_cpu: List[UserDict] = [UserDict() for _ in envelopes]
    kv_student_cpu: List[UserDict] = [UserDict() for _ in envelopes]

    layer_manifest: Dict[str, Any] = {}
    compiled_stats: List[Dict[str, Any]] = []
    peak_layer_vram_mb = 0.0

    def _call_single_layer(
        layer_mod: nn.Module,
        idx: int,
        env: _BatchCallEnvelope,
        h_in_cpu: torch.Tensor,
        kv_cpu: UserDict,
    ) -> torch.Tensor:
        memo_gpu: Dict[int, torch.Tensor] = {}
        args_gpu = _tree_to_device_dedup(env.layer_args_cpu[idx], dev, memo_gpu)
        kwargs_gpu = _tree_to_device_dedup(env.layer_kwargs_cpu[idx], dev, memo_gpu)
        kv_work = UserDict(kv_cpu)
        h_gpu = h_in_cpu.to(device=dev, dtype=dtype, non_blocking=True)
        out_gpu = _unwrap_layer_output(
            layer_mod(h_gpu, *args_gpu, shared_kv_states=kv_work, **kwargs_gpu)
        )
        for k_type, pair in kv_work.items():
            if isinstance(pair, tuple) and len(pair) == 2:
                kv_cpu[k_type] = (pair[0].detach().cpu(), pair[1].detach().cpu())
        return out_gpu.detach().cpu()

    for idx in range(num_total_layers):
        t_layer = time.perf_counter()
        layer_mod = decoder_layers[idx]
        l_prefix = f"{layers_prefix}{idx}."
        valid_layer_keys = {k for k, _ in layer_mod.named_parameters()} | {
            k for k, _ in layer_mod.named_buffers()
        }
        raw_tensors = reader.get_prefix_tensors(
            l_prefix,
            strip_prefix=True,
            device=device,
            dtype=dtype,
            allowed_rel_keys=valid_layer_keys,
        )
        for rel_k, t_gpu in raw_tensors.items():
            _set_tensor_by_qualname(layer_mod, rel_k, t_gpu)
        del raw_tensors
        _move_cpu_buffers_to_device(layer_mod, dev)
        layer_mod.eval()

        if idx not in compiled_set:
            if not compiled_set or idx > max_compiled_idx:
                # Static boundary/post-compiled layer: no downstream BitTieredTLBMLP needs H_teacher/H_student
                q_meta = _quantize_layer_self_attn(
                    layer_mod, attn_quant, attn_group_size, device, dtype, lut_pool
                )
                if boundary_mlp_quant != "none":
                    bnd_mlp = get_target_mlp(model, idx)
                    bnd_mlp_rel = _find_module_qualname(layer_mod, bnd_mlp)
                    q_meta.update(
                        _quantize_layer_mlp_linear(
                            layer_mod, bnd_mlp, bnd_mlp_rel, boundary_mlp_quant, attn_group_size, device, dtype, lut_pool
                        )
                    )
            elif idx < first_divergent_idx and attn_quant == "none" and boundary_mlp_quant == "none":
                # Teacher and student trajectories are bit-identical before first modified layer
                for b_i, env in enumerate(envelopes):
                    h_next = _call_single_layer(layer_mod, idx, env, H_teacher_cpu[b_i], kv_teacher_cpu[b_i])
                    H_teacher_cpu[b_i] = h_next
                    H_student_cpu[b_i] = h_next
                    kv_student_cpu[b_i] = UserDict(kv_teacher_cpu[b_i])
                q_meta = {}
            else:
                # Advance teacher on unmodified dense layer first
                for b_i, env in enumerate(envelopes):
                    H_teacher_cpu[b_i] = _call_single_layer(
                        layer_mod, idx, env, H_teacher_cpu[b_i], kv_teacher_cpu[b_i]
                    )
                q_meta = _quantize_layer_self_attn(
                    layer_mod, attn_quant, attn_group_size, device, dtype, lut_pool
                )
                if boundary_mlp_quant != "none":
                    bnd_mlp = get_target_mlp(model, idx)
                    bnd_mlp_rel = _find_module_qualname(layer_mod, bnd_mlp)
                    q_meta.update(
                        _quantize_layer_mlp_linear(
                            layer_mod, bnd_mlp, bnd_mlp_rel, boundary_mlp_quant, attn_group_size, device, dtype, lut_pool
                        )
                    )
                for b_i, env in enumerate(envelopes):
                    H_student_cpu[b_i] = _call_single_layer(
                        layer_mod, idx, env, H_student_cpu[b_i], kv_student_cpu[b_i]
                    )

            bnd_tensors = _export_module_cpu_tensors(layer_mod)
            bnd_file = f"layer_{idx:03d}_boundary.safetensors"
            save_file(bnd_tensors, os.path.join(layers_out_dir, bnd_file))
            del bnd_tensors
            layer_manifest[str(idx)] = {
                "compiled_mlp": False,
                "boundary_file": f"layers/{bnd_file}",
                "quant_linears": q_meta,
            }
        else:
            orig_mlp = get_target_mlp(model, idx)
            mlp_rel_name = _find_module_qualname(layer_mod, orig_mlp)

            # A. Run Teacher through unmodified dense layer[idx] and capture `X_orig_logic` & `X_orig_fact`
            x_orig_logic_buf: List[torch.Tensor] = []
            x_orig_fact_buf: List[torch.Tensor] = []
            cnt_orig = {"logic": 0, "fact": 0}
            cur_env_ref: List[Optional[_BatchCallEnvelope]] = [None]

            def _teacher_mlp_pre_hook(_mod: nn.Module, args: Tuple[torch.Tensor, ...]) -> None:
                env_cur = cur_env_ref[0]
                if env_cur is None or not args:
                    return
                x_t = args[0].detach()
                mask_b = env_cur.attn_mask_2d_cpu.to(device=x_t.device, dtype=torch.bool)
                flat = x_t[mask_b] if x_t.dim() == 3 else x_t.reshape(-1, x_t.shape[-1])
                rem = max_calib_tokens - cnt_orig[env_cur.split_kind]
                if rem > 0:
                    sl = flat[:rem].to(device="cpu", dtype=torch.bfloat16)
                    if env_cur.split_kind == "logic":
                        x_orig_logic_buf.append(sl)
                    else:
                        x_orig_fact_buf.append(sl)
                    cnt_orig[env_cur.split_kind] += int(sl.shape[0])

            h_hook = orig_mlp.register_forward_pre_hook(_teacher_mlp_pre_hook)
            try:
                for b_i, env in enumerate(envelopes):
                    cur_env_ref[0] = env
                    H_teacher_cpu[b_i] = _call_single_layer(
                        layer_mod, idx, env, H_teacher_cpu[b_i], kv_teacher_cpu[b_i]
                    )
            finally:
                h_hook.remove()
                cur_env_ref[0] = None

            X_orig_logic = torch.cat(x_orig_logic_buf, dim=0)
            X_orig_fact = torch.cat(x_orig_fact_buf, dim=0)
            del x_orig_logic_buf, x_orig_fact_buf

            # B. Optionally quantize `self_attn` BEFORE capturing `X_act` so closed-loop bridge compensates
            # for BOTH upstream layer drift AND current-layer `self_attn` quantization drift!
            q_meta = _quantize_layer_self_attn(
                layer_mod, attn_quant, attn_group_size, device, dtype, lut_pool
            )

            # C. Capture exact Student `X_act_logic` & `X_act_fact` (`refresh_every = 1` on EVERY layer!)
            if idx == first_divergent_idx and attn_quant == "none":
                X_act_logic = X_orig_logic
                X_act_fact = X_orig_fact
            else:
                x_act_logic_buf: List[torch.Tensor] = []
                x_act_fact_buf: List[torch.Tensor] = []
                cnt_act = {"logic": 0, "fact": 0}

                def _student_stop_pre_hook(_mod: nn.Module, args: Tuple[torch.Tensor, ...]) -> None:
                    env_cur = cur_env_ref[0]
                    if env_cur is not None and args:
                        x_t = args[0].detach()
                        mask_b = env_cur.attn_mask_2d_cpu.to(device=x_t.device, dtype=torch.bool)
                        flat = x_t[mask_b] if x_t.dim() == 3 else x_t.reshape(-1, x_t.shape[-1])
                        rem = max_calib_tokens - cnt_act[env_cur.split_kind]
                        if rem > 0:
                            sl = flat[:rem].to(device="cpu", dtype=torch.bfloat16)
                            if env_cur.split_kind == "logic":
                                x_act_logic_buf.append(sl)
                            else:
                                x_act_fact_buf.append(sl)
                            cnt_act[env_cur.split_kind] += int(sl.shape[0])
                    raise _StopAtMLPPreHook()

                s_hook = orig_mlp.register_forward_pre_hook(_student_stop_pre_hook)
                try:
                    for b_i, env in enumerate(envelopes):
                        cur_env_ref[0] = env
                        memo_gpu: Dict[int, torch.Tensor] = {}
                        args_gpu = _tree_to_device_dedup(env.layer_args_cpu[idx], dev, memo_gpu)
                        kwargs_gpu = _tree_to_device_dedup(env.layer_kwargs_cpu[idx], dev, memo_gpu)
                        kv_tmp = UserDict(kv_student_cpu[b_i])
                        h_gpu = H_student_cpu[b_i].to(device=dev, dtype=dtype, non_blocking=True)
                        try:
                            _ = layer_mod(h_gpu, *args_gpu, shared_kv_states=kv_tmp, **kwargs_gpu)
                        except _StopAtMLPPreHook:
                            pass
                finally:
                    s_hook.remove()
                    cur_env_ref[0] = None

                X_act_logic = torch.cat(x_act_logic_buf, dim=0)
                X_act_fact = torch.cat(x_act_fact_buf, dim=0)
                del x_act_logic_buf, x_act_fact_buf

            # D. Extract dense MLP weights, free dense `orig_mlp`, and compile `BitTieredTLBMLP`
            act_fn = getattr(orig_mlp, "act_fn", nn.GELU(approximate="tanh"))
            Wg = orig_mlp.gate_proj.weight.detach()
            Wu = orig_mlp.up_proj.weight.detach()
            Wd = orig_mlp.down_proj.weight.detach()
            _evict_module_to_meta(orig_mlp)

            b_mlp, st = compile_phase_b_layer(
                W_gate=Wg,
                W_up=Wu,
                W_down=Wd,
                act_fn=act_fn,
                X_orig_logic=X_orig_logic,
                X_orig_fact=X_orig_fact,
                H_orig_logic=None,
                H_orig_fact=None,
                lut_pool=lut_pool,
                X_actual_logic=X_act_logic,
                X_actual_fact=X_act_fact,
                hot_fraction=hot_fraction,
                num_pages=num_pages,
                tlb_slots=tlb_slots,
                decode_req_pages=decode_req_pages,
                bridge_rank=bridge_rank,
                closed_loop_ridge=closed_loop_ridge,
                compute_device=device,
                target_device=device,
                target_dtype=dtype,
                layer_idx=idx,
                shared_dma_stream=None,
                batch_controller=None,
                controller_slot_idx=0,
            )
            del Wg, Wu, Wd, X_orig_logic, X_orig_fact, X_act_logic, X_act_fact

            # E. Install `b_mlp` and advance Student hidden state `H_student` + `kv_student` (if downstream compiled layers remain)
            mlp_parent, mlp_attr = _get_parent_and_attr(layer_mod, mlp_rel_name)
            setattr(mlp_parent, mlp_attr, b_mlp)
            if idx < max_compiled_idx:
                b_mlp.set_mode("vm_paged")
                b_mlp.set_tlb_strategy("sync_tlb")
                for b_i, env in enumerate(envelopes):
                    b_mlp.reset_telemetry(clear_tlb=True)
                    H_student_cpu[b_i] = _call_single_layer(
                        layer_mod, idx, env, H_student_cpu[b_i], kv_student_cpu[b_i]
                    )

            # F. Serialize `layer_{idx:03d}_core.safetensors` & `layer_{idx:03d}_pages.safetensors`
            non_mlp_tensors = _export_module_cpu_tensors(layer_mod, skip_submodules=(mlp_rel_name,))
            mlp_core_tensors, mlp_page_tensors, mlp_meta = b_mlp.export_serialized_core_and_pages()
            combined_core: Dict[str, torch.Tensor] = {}
            for k_nm, t_nm in non_mlp_tensors.items():
                combined_core[f"layer.{k_nm}"] = t_nm
            for k_mc, t_mc in mlp_core_tensors.items():
                combined_core[f"mlp.{k_mc}"] = t_mc

            core_file = f"layer_{idx:03d}_core.safetensors"
            pages_file = f"layer_{idx:03d}_pages.safetensors"
            save_file(combined_core, os.path.join(layers_out_dir, core_file))
            save_file(mlp_page_tensors, os.path.join(layers_out_dir, pages_file))
            del non_mlp_tensors, mlp_core_tensors, mlp_page_tensors, combined_core

            layer_manifest[str(idx)] = {
                "compiled_mlp": True,
                "mlp_rel_name": mlp_rel_name,
                "core_file": f"layers/{core_file}",
                "pages_file": f"layers/{pages_file}",
                "mlp_meta": mlp_meta,
                "quant_linears": q_meta,
                "stats": asdict(st),
            }
            compiled_stats.append(asdict(st))
            del b_mlp

        if torch.cuda.is_available() and dev.type == "cuda":
            cur_peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
            peak_layer_vram_mb = max(peak_layer_vram_mb, cur_peak)

        # Evict finished layer `idx` back to 0-byte `meta` immediately!
        for env in envelopes:
            env.layer_args_cpu.pop(idx, None)
            env.layer_kwargs_cpu.pop(idx, None)
        _evict_module_to_meta(layer_mod)
        decoder_layers[idx] = nn.Identity()  # type: ignore[assignment]
        del layer_mod
        gc.collect()
        if os.name != "nt":
            try:
                import ctypes

                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
        if torch.cuda.is_available() and dev.type == "cuda":
            torch.cuda.empty_cache()

        dt_l = time.perf_counter() - t_layer
        tag = "COMPILED" if idx in compiled_set else "BOUNDARY"
        print(
            f"[StreamCompile] Layer {idx:02d}/{num_total_layers - 1:02d} ({tag}, attn={attn_quant}) "
            f"flushed in {dt_l:.2f}s | Peak Compile VRAM={peak_layer_vram_mb:.1f} MB"
        )

    # Calculate total `.nvm` cartridge size on disk
    total_disk_bytes = 0
    pages_disk_bytes = 0
    ple_disk_bytes = 0
    for root_d, _, files in os.walk(out_dir):
        for fn in files:
            fp = os.path.join(root_d, fn)
            sz = os.path.getsize(fp)
            total_disk_bytes += sz
            if fn.endswith("_pages.safetensors"):
                pages_disk_bytes += sz
            elif fn == "ple_host.safetensors":
                ple_disk_bytes += sz

    compile_wall_s = time.perf_counter() - t_start
    manifest = {
        "format_version": "nvm_v1",
        "source_model": model_id_or_path,
        "num_total_layers": num_total_layers,
        "compiled_layer_indices": compiled_indices,
        "layers_qualname": layers_qualname,
        "stripped_multimodal_towers": stripped_towers,
        "tie_word_embeddings": tie_word_embeddings,
        "attn_quant": attn_quant,
        "attn_group_size": attn_group_size,
        "boundary_mlp_quant": boundary_mlp_quant,
        "hot_fraction": hot_fraction,
        "num_pages": num_pages,
        "tlb_slots": tlb_slots,
        "decode_req_pages": decode_req_pages,
        "bridge_rank": bridge_rank,
        "ple_meta": ple_meta,
        "compile_telemetry": {
            "compile_wall_s": compile_wall_s,
            "peak_preamble_vram_mb": peak_preamble_vram_mb,
            "peak_single_layer_vram_mb": peak_layer_vram_mb,
            "peak_compile_vram_mb": max(peak_preamble_vram_mb, peak_layer_vram_mb),
            "total_disk_mb": total_disk_bytes / (1024 ** 2),
            "core_resident_disk_mb": (total_disk_bytes - pages_disk_bytes - ple_disk_bytes) / (1024 ** 2),
            "pages_disk_mb": pages_disk_bytes / (1024 ** 2),
            "ple_disk_mb": ple_disk_bytes / (1024 ** 2),
        },
        "layers": layer_manifest,
        "compiled_layer_stats": compiled_stats,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    del model, envelopes, H_teacher_cpu, H_student_cpu, kv_teacher_cpu, kv_student_cpu, lut_pool
    gc.collect()
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.empty_cache()

    return manifest


def _locate_layer_moe_components(layer_mod: nn.Module) -> Optional[Dict[str, Any]]:
    """Detects MoE submodules inside a decoder layer across Gemma 4, Qwen3-Next/Qwen3-MoE, Mixtral, and DeepSeek."""
    experts_mod: Optional[nn.Module] = None
    experts_rel = ""
    router_rel = ""
    shared_mlp_mod: Optional[nn.Module] = None
    shared_mlp_rel = ""

    if getattr(layer_mod, "enable_moe_block", False) and getattr(layer_mod, "experts", None) is not None:
        experts_mod = layer_mod.experts
        experts_rel = "experts"
        router_rel = "router" if getattr(layer_mod, "router", None) is not None else ""
        if getattr(layer_mod, "mlp", None) is not None:
            shared_mlp_mod = layer_mod.mlp
            shared_mlp_rel = "mlp"
    elif hasattr(layer_mod, "mlp") and layer_mod.mlp is not None and getattr(layer_mod.mlp, "experts", None) is not None:
        experts_mod = layer_mod.mlp.experts
        experts_rel = "mlp.experts"
        router_rel = "mlp.gate" if getattr(layer_mod.mlp, "gate", None) is not None else ""
        if getattr(layer_mod.mlp, "shared_expert", None) is not None:
            shared_mlp_mod = layer_mod.mlp.shared_expert
            shared_mlp_rel = "mlp.shared_expert"
        elif getattr(layer_mod.mlp, "shared_experts", None) is not None:
            shared_mlp_mod = layer_mod.mlp.shared_experts
            shared_mlp_rel = "mlp.shared_experts"

    if experts_mod is None:
        return None

    gu_param = getattr(experts_mod, "gate_up_proj", None)
    dn_param = getattr(experts_mod, "down_proj", None)
    if isinstance(gu_param, torch.Tensor) and gu_param.ndim == 3:
        num_experts = int(gu_param.shape[0])
        d_ffn = int(gu_param.shape[1] // 2)
        d_model = int(gu_param.shape[2])
    elif isinstance(experts_mod, nn.ModuleList) and len(experts_mod) > 0:
        num_experts = len(experts_mod)
        e0 = experts_mod[0]
        g_lin = getattr(e0, "gate_proj", getattr(e0, "w1", None))
        if g_lin is None:
            return None
        d_ffn = int(g_lin.weight.shape[0])
        d_model = int(g_lin.weight.shape[1])
    else:
        num_experts = int(getattr(experts_mod, "num_experts", 0))
        d_model = int(getattr(experts_mod, "hidden_dim", getattr(experts_mod, "hidden_size", 0)))
        d_ffn = int(getattr(experts_mod, "intermediate_dim", getattr(experts_mod, "moe_intermediate_size", 0)))
        if num_experts <= 0 or d_model <= 0 or d_ffn <= 0:
            return None

    act_fn = getattr(experts_mod, "act_fn", None)
    if act_fn is None and isinstance(experts_mod, nn.ModuleList) and len(experts_mod) > 0:
        act_fn = getattr(experts_mod[0], "act_fn", None)
    if act_fn is None:
        act_fn = nn.GELU(approximate="tanh")

    return {
        "experts_mod": experts_mod,
        "experts_rel": experts_rel,
        "router_rel": router_rel,
        "shared_mlp_mod": shared_mlp_mod,
        "shared_mlp_rel": shared_mlp_rel,
        "num_experts": num_experts,
        "d_model": d_model,
        "d_ffn": d_ffn,
        "act_fn": act_fn,
        "has_stacked_3d_params": isinstance(gu_param, torch.Tensor) and isinstance(dn_param, torch.Tensor),
    }


def _read_moe_expert_chunk(
    reader: SafetensorShardReader,
    l_prefix: str,
    experts_rel: str,
    start_e: int,
    end_e: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reads experts `[start_e:end_e]` as `(w_gu [C, 2*I, D], w_dn [C, D, I])` across 3D or per-expert 2D shards."""
    gu_3d_key = f"{l_prefix}{experts_rel}.gate_up_proj"
    dn_3d_key = f"{l_prefix}{experts_rel}.down_proj"
    if reader.has_key(gu_3d_key) and reader.has_key(dn_3d_key):
        w_gu = reader.get_tensor_dim0_slice(gu_3d_key, start_e, end_e, device=device, dtype=dtype)
        w_dn = reader.get_tensor_dim0_slice(dn_3d_key, start_e, end_e, device=device, dtype=dtype)
        return w_gu, w_dn

    # Per-expert 2D keys (`gate_proj/up_proj/down_proj` or `w1/w3/w2`)
    sample_gate = f"{l_prefix}{experts_rel}.{start_e}.gate_proj.weight"
    if reader.has_key(sample_gate):
        g_suf, u_suf, d_suf = "gate_proj.weight", "up_proj.weight", "down_proj.weight"
    else:
        g_suf, u_suf, d_suf = "w1.weight", "w3.weight", "w2.weight"

    wanted_keys: List[str] = []
    for e in range(start_e, end_e):
        wanted_keys.append(f"{l_prefix}{experts_rel}.{e}.{g_suf}")
        wanted_keys.append(f"{l_prefix}{experts_rel}.{e}.{u_suf}")
        wanted_keys.append(f"{l_prefix}{experts_rel}.{e}.{d_suf}")

    by_file: Dict[str, List[str]] = {}
    for k in wanted_keys:
        fpath = reader.get_file_for_key(k)
        by_file.setdefault(fpath, []).append(k)

    loaded: Dict[str, torch.Tensor] = {}
    for fpath, k_list in by_file.items():
        data_base, hdr = reader._file_headers[fpath]
        k_sorted = sorted(k_list, key=lambda kk: int(hdr[kk]["data_offsets"][0]))
        with open(fpath, "rb") as fp:
            for k in k_sorted:
                loaded[k] = _read_safetensor_slice_exact(fp, data_base, hdr[k], device="cpu", dtype=dtype)

    gu_list: List[torch.Tensor] = []
    dn_list: List[torch.Tensor] = []
    for e in range(start_e, end_e):
        wg = loaded.pop(f"{l_prefix}{experts_rel}.{e}.{g_suf}")
        wu = loaded.pop(f"{l_prefix}{experts_rel}.{e}.{u_suf}")
        wd = loaded.pop(f"{l_prefix}{experts_rel}.{e}.{d_suf}")
        gu_list.append(torch.cat([wg, wu], dim=0))
        dn_list.append(wd)
    w_gu_st = torch.stack(gu_list, dim=0)
    w_dn_st = torch.stack(dn_list, dim=0)
    if str(device) != "cpu":
        w_gu_st = w_gu_st.to(device=device, dtype=dtype)
        w_dn_st = w_dn_st.to(device=device, dtype=dtype)
    return w_gu_st, w_dn_st


def _populate_layer_bf16_from_reader(
    layer_mod: nn.Module,
    reader: SafetensorShardReader,
    l_prefix: str,
    valid_layer_keys: set[str],
    device: str,
    dtype: torch.dtype,
) -> Tuple[int, List[str]]:
    """Populates a decoder layer's BF16 weights from `reader` (supporting both 3D and per-expert 2D MoE shards)."""
    raw_tensors = reader.get_prefix_tensors(
        l_prefix, strip_prefix=True, device=device, dtype=dtype, allowed_rel_keys=valid_layer_keys
    )
    loaded_bytes = 0
    loaded_keys: List[str] = []
    for rel_k, t_gpu in raw_tensors.items():
        loaded_bytes += int(t_gpu.numel() * t_gpu.element_size())
        _set_tensor_by_qualname(layer_mod, rel_k, t_gpu)
        loaded_keys.append(rel_k)
    del raw_tensors

    moe_info = _locate_layer_moe_components(layer_mod)
    if moe_info is not None and moe_info["has_stacked_3d_params"]:
        exp_mod = moe_info["experts_mod"]
        if getattr(exp_mod, "gate_up_proj", None) is not None and exp_mod.gate_up_proj.device.type == "meta":
            E = int(moe_info["num_experts"])
            w_gu, w_dn = _read_moe_expert_chunk(
                reader, l_prefix, moe_info["experts_rel"], 0, E, device=device, dtype=dtype
            )
            loaded_bytes += int(w_gu.numel() * w_gu.element_size() + w_dn.numel() * w_dn.element_size())
            gu_k = f"{moe_info['experts_rel']}.gate_up_proj"
            dn_k = f"{moe_info['experts_rel']}.down_proj"
            _set_tensor_by_qualname(layer_mod, gu_k, w_gu)
            _set_tensor_by_qualname(layer_mod, dn_k, w_dn)
            loaded_keys.extend([gu_k, dn_k])
    return loaded_bytes, loaded_keys


@torch.no_grad()
def _compile_native_mtp_from_reader(
    reader: SafetensorShardReader,
    src_dir: str,
    out_dir: str,
    attn_quant: str,
    attn_group_size: int,
    shared_mlp_quant: str,
    expert_group_size: int,
    expert_chunk_size: int,
    tlb_slots: int,
    device: str,
    dtype: torch.dtype,
    lut_pool: SharedQuantLUTPool,
) -> Optional[Dict[str, Any]]:
    """Stream-compiles native HF `mtp.*` weights (`Qwen3-Next` / `DeepSeek-V3`) into `mtp_core.safetensors` + `mtp_experts.safetensors`."""
    dev = torch.device(device)
    fc_key = "mtp.fc.weight" if reader.has_key("mtp.fc.weight") else "mtp.eh_proj.weight"
    if not reader.has_key(fc_key):
        return None
    emb_norm_key = "mtp.pre_fc_norm_embedding.weight" if reader.has_key("mtp.pre_fc_norm_embedding.weight") else "mtp.enorm.weight"
    hid_norm_key = "mtp.pre_fc_norm_hidden.weight" if reader.has_key("mtp.pre_fc_norm_hidden.weight") else "mtp.hnorm.weight"
    out_norm_key = "mtp.norm.weight" if reader.has_key("mtp.norm.weight") else "mtp.shared_head.norm.weight"
    if not (reader.has_key(emb_norm_key) and reader.has_key(hid_norm_key) and reader.has_key(out_norm_key)):
        return None

    print("[StreamMoECompile] Detected native HF `mtp.*` weights — compiling CUDA-Graph MTP drafter...", flush=True)
    # Build a fresh 1-layer skeleton cloned from a full-attention decoder layer
    tmp_model, _ = _create_empty_skeleton(src_dir, dtype=dtype, skip_multimodal=True)
    dec_layers = resolve_decoder_layers(tmp_model)
    full_attn_lyr = next((l for l in dec_layers if hasattr(l, "self_attn") and l.self_attn is not None), dec_layers[-1])
    valid_layer_keys = {k for k, _ in full_attn_lyr.named_parameters()} | {k for k, _ in full_attn_lyr.named_buffers()}

    # Load non-expert MTP layer tensors (`mtp.layers.0.*`) onto GPU
    l_prefix = "mtp.layers.0."
    moe_info = _locate_layer_moe_components(full_attn_lyr)
    exp_rel = moe_info["experts_rel"] if moe_info is not None else None
    non_exp_keys = {k for k in valid_layer_keys if not (exp_rel and (k == exp_rel or k.startswith(f"{exp_rel}.")))}

    raw_layer = reader.get_prefix_tensors(
        l_prefix, strip_prefix=True, device=device, dtype=dtype, allowed_rel_keys=non_exp_keys
    )
    for rel_k, t_gpu in raw_layer.items():
        _set_tensor_by_qualname(full_attn_lyr, rel_k, t_gpu)
    del raw_layer
    _move_cpu_buffers_to_device(full_attn_lyr, dev)
    full_attn_lyr.eval()

    q_meta = _quantize_layer_self_attn(full_attn_lyr, attn_quant, attn_group_size, device, dtype, lut_pool)
    if moe_info is not None and shared_mlp_quant != "none" and moe_info.get("shared_mlp_mod") is not None:
        q_meta.update(
            _quantize_layer_mlp_linear(
                full_attn_lyr,
                moe_info["shared_mlp_mod"],
                moe_info["shared_mlp_rel"],
                shared_mlp_quant,
                attn_group_size,
                device,
                dtype,
                lut_pool,
            )
        )

    has_moe_experts = False
    exp_meta: Dict[str, Any] = {}
    if moe_info is not None and exp_rel is not None:
        E = int(moe_info["num_experts"])
        D = int(moe_info["d_model"])
        I = int(moe_info["d_ffn"])
        layout = ContiguousExpertPageLayout.compute(d_model=D, d_ffn=I, group_size=expert_group_size)
        pages_cpu = torch.zeros((E, layout.page_bytes), dtype=torch.uint8, device="cpu")
        for s_e in range(0, E, max(1, int(expert_chunk_size))):
            e_end = min(E, s_e + max(1, int(expert_chunk_size)))
            w_gu, w_dn = _read_moe_expert_chunk(reader, l_prefix, exp_rel, s_e, e_end, device=device, dtype=dtype)
            ch_pages, _ = quantize_pack_moe_experts_uint4(
                w_gu, w_dn, group_size=expert_group_size, compute_device=device, expert_chunk_size=e_end - s_e
            )
            pages_cpu[s_e:e_end].copy_(ch_pages)
            del w_gu, w_dn, ch_pages
        save_file(
            {"pages": pages_cpu, "freq_rank": torch.arange(E, dtype=torch.int32)},
            os.path.join(out_dir, "mtp_experts.safetensors"),
        )
        del pages_cpu
        has_moe_experts = True
        exp_meta = {
            "experts_file": "mtp_experts.safetensors",
            "experts_rel": exp_rel,
            "num_experts": E,
            "moe_d_ffn": I,
            "expert_layout": layout.to_dict(),
            "tlb_slots": min(E, max(16, int(tlb_slots))),
        }

    # Export `mtp_core.safetensors` keeping `pre_fc_norm_*`, `fc`, and `norm` strictly unquantized BF16!
    core_tensors: Dict[str, torch.Tensor] = {
        "pre_fc_norm_embedding.weight": reader.get_tensor(emb_norm_key, device="cpu", dtype=dtype),
        "pre_fc_norm_hidden.weight": reader.get_tensor(hid_norm_key, device="cpu", dtype=dtype),
        "fc.weight": reader.get_tensor(fc_key, device="cpu", dtype=dtype),
        "norm.weight": reader.get_tensor(out_norm_key, device="cpu", dtype=dtype),
    }
    layer_cpu = _export_module_cpu_tensors(full_attn_lyr, skip_submodules=(exp_rel,) if exp_rel else ())
    for k_l, t_l in layer_cpu.items():
        core_tensors[f"layer.{k_l}"] = t_l
    save_file(core_tensors, os.path.join(out_dir, "mtp_core.safetensors"))
    del core_tensors, layer_cpu

    cfg_obj = getattr(tmp_model.config, "text_config", tmp_model.config)
    sa = full_attn_lyr.self_attn
    d_model = int(getattr(cfg_obj, "hidden_size", 2048))
    h_q = int(getattr(sa, "num_heads", None) or getattr(cfg_obj, "num_attention_heads", 16))
    h_kv = int(getattr(sa, "num_key_value_heads", None) or getattr(cfg_obj, "num_key_value_heads", 2))
    d_h = int(getattr(sa, "head_dim", d_model // max(1, h_q)))
    arch_str = str(getattr(cfg_obj, "model_type", "qwen3_next"))
    unit_off = "qwen3_next" in arch_str.lower() or "gemma" in arch_str.lower()

    meta: Dict[str, Any] = {
        "format_version": "tmmu_mtp_v1",
        "arch_kind": arch_str,
        "d_model": d_model,
        "num_q_heads": h_q,
        "num_kv_heads": h_kv,
        "head_dim": d_h,
        "rms_norm_eps": float(getattr(cfg_obj, "rms_norm_eps", 1e-6)),
        "unit_offset_norm": unit_off,
        "act_kind": "silu" if "qwen" in arch_str.lower() or "deepseek" in arch_str.lower() else "gelu_tanh",
        "quant_linears": q_meta,
        "has_moe_experts": has_moe_experts,
        **exp_meta,
    }
    with open(os.path.join(out_dir, "mtp_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    del tmp_model, full_attn_lyr
    gc.collect()
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[StreamMoECompile] Native MTP layer compiled -> mtp_core.safetensors (MoE={has_moe_experts}).", flush=True)
    return meta


@torch.no_grad()
def compile_streaming_moe_to_nvm(
    model_id_or_path: str,
    nvm_dir: str,
    bundle: ContrastiveCalibrationBundle,
    tlb_slots: int = 48,
    expert_group_size: int = 64,
    expert_chunk_size: int = 16,
    attn_quant: str = "int8",
    attn_group_size: int = 64,
    shared_mlp_quant: str = "int8",
    lm_head_quant: str = "none",
    calib_batch_size: int = 4,
    max_calib_tokens: int = 4096,
    offload_ple: bool = True,
    save_standalone_ple: bool = True,
    skip_multimodal: bool = True,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Streams a Sparse MoE `.safetensors` checkpoint layer-by-layer into a Zero-Skeleton `nvm_moe_v1` cartridge.

    Peak GPU VRAM is strictly bounded to `< 2.5 GB` even on 80B-120B MoE checkpoints:
      - Never loads full BF16 expert banks (`experts` stay on `0`-byte `meta` while capturing exact router
        frequencies + input channel salience `x_ms` on `Logic_Calib + Fact_Calib`).
      - Streams and packs `expert_chunk_size` (`16`) experts at a time into 4KB-aligned contiguous `UINT4` pages.
      - Advances student hidden states layer-by-layer using `SparseMoETLBExperts` so downstream router statistics
        reflect upstream quantized activations (`0` eval/test split leakage!).
    """
    if attn_quant not in VALID_ATTN_QUANT_MODES:
        raise ValueError(f"Invalid attn_quant '{attn_quant}'. Expected {VALID_ATTN_QUANT_MODES}.")
    if shared_mlp_quant not in VALID_BOUNDARY_MLP_QUANT_MODES:
        raise ValueError(f"Invalid shared_mlp_quant '{shared_mlp_quant}'. Expected {VALID_BOUNDARY_MLP_QUANT_MODES}.")
    if lm_head_quant not in VALID_LM_HEAD_QUANT_MODES:
        raise ValueError(f"Invalid lm_head_quant '{lm_head_quant}'. Expected {VALID_LM_HEAD_QUANT_MODES}.")

    t_start = time.perf_counter()
    dev = torch.device(device)
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    src_dir = resolve_hf_snapshot_dir(model_id_or_path)
    out_dir = os.path.abspath(nvm_dir)
    layers_out_dir = os.path.join(out_dir, "layers")
    os.makedirs(layers_out_dir, exist_ok=True)
    _copy_tokenizer_and_configs(src_dir, out_dir)

    reader = SafetensorShardReader(src_dir)
    model, stripped_towers = _create_empty_skeleton(src_dir, dtype=dtype, skip_multimodal=skip_multimodal)
    if torch.cuda.is_available() and dev.type == "cuda":
        if any(m.__class__.__name__ == "Qwen3NextGatedDeltaNet" for m in model.modules()):
            from runtime.qwen3_next_kernels import install_qwen3_next_fast_gdn

            install_qwen3_next_fast_gdn(model)
    decoder_layers = resolve_decoder_layers(model)
    num_total_layers = len(decoder_layers)
    layers_qualname = _find_module_qualname(model, decoder_layers)
    layers_prefix = f"{layers_qualname}."
    lut_pool = SharedQuantLUTPool(device=dev, dtype=dtype)

    # 1. Optional PLE host mmap + Preamble/Head tensors
    ple_qualname = _resolve_ple_module_path(model)
    ple_prefix = f"{ple_qualname}." if ple_qualname else None
    ple_meta: Optional[Dict[str, Any]] = None
    if offload_ple and ple_qualname is not None and reader.has_key(f"{ple_qualname}.weight"):
        ple_parent, ple_attr = _get_parent_and_attr(model, ple_qualname)
        orig_ple = getattr(ple_parent, ple_attr)
        raw_scale = getattr(orig_ple, "embed_scale", 1.0)
        ple_scale_float = float(raw_scale.item()) if isinstance(raw_scale, torch.Tensor) else float(raw_scale)
        ple_pad_idx = getattr(orig_ple, "padding_idx", None)

        ple_src_file = reader.get_file_for_key(f"{ple_qualname}.weight")
        ple_dst_path = os.path.join(out_dir, "ple_host.safetensors")
        if save_standalone_ple:
            w_ple_tmp = reader.get_tensor(f"{ple_qualname}.weight", device="cpu", dtype=dtype)
            save_file({"weight": w_ple_tmp}, ple_dst_path)
            del w_ple_tmp
            ple_mmap_w = mmap_safetensor_tensor(ple_dst_path, "weight")
            ple_source_mode = "standalone"
        else:
            ple_mmap_w = mmap_safetensor_tensor(ple_src_file, f"{ple_qualname}.weight")
            ple_source_mode = "source_ref"

        ple_offloader = DeterministicPLEOffloader.from_cpu_tensor(
            ple_mmap_w, embed_scale=ple_scale_float, padding_idx=ple_pad_idx
        )
        setattr(ple_parent, ple_attr, ple_offloader)
        del orig_ple
        ple_meta = {
            "qualname": ple_qualname,
            "embed_scale": ple_scale_float,
            "padding_idx": ple_pad_idx,
            "storage": ple_source_mode,
            "source_file": ple_src_file if not save_standalone_ple else "ple_host.safetensors",
            "source_key": f"{ple_qualname}.weight" if not save_standalone_ple else "weight",
            "offloaded_vram_mb": ple_offloader.offloaded_vram_mb,
        }

    skipped_prefixes = tuple(stripped_towers)
    valid_model_keys = {k for k, _ in model.named_parameters()} | {k for k, _ in model.named_buffers()}
    base_cpu_tensors: Dict[str, torch.Tensor] = {}
    for key in reader.keys():
        if key.startswith(layers_prefix):
            continue
        if ple_meta is not None and ple_prefix is not None and key.startswith(ple_prefix):
            continue
        if any(key == sp or key.startswith(sp + ".") for sp in skipped_prefixes):
            continue
        if key not in valid_model_keys:
            continue
        t_gpu = reader.get_tensor(key, device=device, dtype=dtype)
        _set_tensor_by_qualname(model, key, t_gpu)
        base_cpu_tensors[key] = t_gpu.detach().cpu().contiguous()

    _move_cpu_buffers_to_device(model, dev, exclude_prefixes=(layers_qualname,))
    tie_word_embeddings = False
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight") and model.lm_head.weight.device.type == "meta":
        model.tie_weights()
        tie_word_embeddings = True

    lm_head_quant_meta: Optional[Dict[str, Any]] = None
    if lm_head_quant != "none" and not tie_word_embeddings and hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        qlm = PhaseCQuantLinear.from_linear(
            model.lm_head,
            quant_mode=lm_head_quant,
            group_size=attn_group_size,
            target_device=device,
            scale_dtype=dtype,
            lut_pool=lut_pool,
        )
        model.lm_head = qlm
        base_cpu_tensors.pop("lm_head.weight", None)
        base_cpu_tensors.pop("lm_head.bias", None)
        base_cpu_tensors["lm_head.q_weight"] = qlm.q_weight.detach().cpu().contiguous()
        base_cpu_tensors["lm_head.scale"] = qlm.scale.detach().cpu().contiguous()
        if qlm.bias is not None:
            base_cpu_tensors["lm_head.bias"] = qlm.bias.detach().cpu().contiguous()
        lm_head_quant_meta = {
            "in_features": qlm.in_features,
            "out_features": qlm.out_features,
            "quant_mode": qlm.quant_mode,
            "group_size": qlm.group_size,
            "has_bias": qlm.bias is not None,
            "has_wmin": qlm.wmin is not None,
        }

    save_file(base_cpu_tensors, os.path.join(out_dir, "base_resident.safetensors"))
    del base_cpu_tensors

    # 2. Capture exact Preamble envelopes (`H_0`, RoPE, causal masks) on `Logic_Calib + Fact_Calib`
    envelopes: List[_BatchCallEnvelope] = []
    orig_meta_layers = [decoder_layers[i] for i in range(num_total_layers)]
    spy_layers = [_PreambleSpyLayer(orig_meta_layers[i], i) for i in range(num_total_layers)]
    for i in range(num_total_layers):
        decoder_layers[i] = spy_layers[i]
    model.eval()

    def _record_split_envelopes(split: TokenizedSplit, kind: str) -> None:
        tok_seen = 0
        n_rows = split.input_ids.shape[0]
        for s in range(0, n_rows, calib_batch_size):
            if tok_seen >= max_calib_tokens:
                break
            e = min(n_rows, s + calib_batch_size)
            b_ids = split.input_ids[s:e].to(dev)
            b_mask = split.attention_mask[s:e].to(dev)
            env = _BatchCallEnvelope(split_kind=kind, attn_mask_2d_cpu=split.attention_mask[s:e].cpu())
            for spy in spy_layers:
                spy.active_envelope = env
            try:
                _ = model(input_ids=b_ids, attention_mask=b_mask, use_cache=False, logits_to_keep=1)
            except TypeError:
                _ = model(input_ids=b_ids, attention_mask=b_mask, use_cache=False)
            for spy in spy_layers:
                spy.active_envelope = None
            assert env.h0_cpu is not None
            envelopes.append(env)
            tok_seen += int(b_mask.sum().item())

    _record_split_envelopes(bundle.logic_calib, "logic")
    _record_split_envelopes(bundle.fact_calib, "fact")

    for i in range(num_total_layers):
        decoder_layers[i] = orig_meta_layers[i]
    del spy_layers, orig_meta_layers

    peak_preamble_vram_mb = (
        torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        if (torch.cuda.is_available() and dev.type == "cuda")
        else 0.0
    )

    # Evict Preamble so layer-loop VRAM is strictly single-layer O(1)
    if hasattr(model, "lm_head"):
        if isinstance(model.lm_head, PhaseCQuantLinear):
            _evict_module_to_meta(model.lm_head)
        elif tie_word_embeddings:
            model.lm_head.weight = nn.Parameter(torch.empty(0, dtype=dtype, device="meta"), requires_grad=False)
    for top_name, top_mod in model.named_children():
        if top_name == layers_qualname.split(".")[0]:
            continue
        _evict_module_to_meta(top_mod)
    lm_container, _ = _get_parent_and_attr(model, layers_qualname)
    for _, child_mod in lm_container.named_children():
        if child_mod is decoder_layers or isinstance(child_mod, DeterministicPLEOffloader):
            continue
        _evict_module_to_meta(child_mod)
    gc.collect()
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    # 3. Streaming Layer-by-Layer MoE Compilation (`0 .. num_total_layers - 1`)
    H_student_cpu: List[torch.Tensor] = [env.h0_cpu for env in envelopes]  # type: ignore[misc]
    kv_student_cpu: List[UserDict] = [UserDict() for _ in envelopes]
    layer_manifest: Dict[str, Any] = {}
    moe_layer_indices: List[int] = []
    moe_layer_stats: List[Dict[str, Any]] = []
    peak_layer_vram_mb = 0.0

    def _call_single_layer(
        layer_mod: nn.Module,
        idx: int,
        env: _BatchCallEnvelope,
        h_in_cpu: torch.Tensor,
        kv_cpu: UserDict,
    ) -> torch.Tensor:
        memo_gpu: Dict[Tuple[Any, ...], torch.Tensor] = {}
        args_gpu = _tree_to_device_dedup(env.layer_args_cpu[idx], dev, memo_gpu)
        kwargs_gpu = _tree_to_device_dedup(env.layer_kwargs_cpu[idx], dev, memo_gpu)
        kv_work = UserDict(kv_cpu)
        h_gpu = h_in_cpu.to(device=dev, dtype=dtype, non_blocking=True)
        out_gpu = _unwrap_layer_output(
            layer_mod(h_gpu, *args_gpu, shared_kv_states=kv_work, **kwargs_gpu)
        )
        for k_type, pair in kv_work.items():
            if isinstance(pair, tuple) and len(pair) == 2:
                kv_cpu[k_type] = (pair[0].detach().cpu(), pair[1].detach().cpu())
        return out_gpu.detach().cpu()

    for idx in range(num_total_layers):
        t_layer = time.perf_counter()
        layer_mod = decoder_layers[idx]
        l_prefix = f"{layers_prefix}{idx}."
        moe_info = _locate_layer_moe_components(layer_mod)

        valid_all = {k for k, _ in layer_mod.named_parameters()} | {k for k, _ in layer_mod.named_buffers()}
        if moe_info is not None:
            exp_rel = moe_info["experts_rel"]
            non_exp_keys = {k for k in valid_all if not (k == exp_rel or k.startswith(exp_rel + "."))}
        else:
            non_exp_keys = valid_all

        raw_non_exp = reader.get_prefix_tensors(
            l_prefix, strip_prefix=True, device=device, dtype=dtype, allowed_rel_keys=non_exp_keys
        )
        for rel_k, t_gpu in raw_non_exp.items():
            _set_tensor_by_qualname(layer_mod, rel_k, t_gpu)
        del raw_non_exp
        _move_cpu_buffers_to_device(layer_mod, dev)
        layer_mod.eval()

        q_meta = _quantize_layer_self_attn(layer_mod, attn_quant, attn_group_size, device, dtype, lut_pool)

        if moe_info is None:
            # Dense non-MoE layer (e.g., hybrid early dense layers)
            if shared_mlp_quant != "none":
                dense_mlp = get_target_mlp(model, idx)
                dense_mlp_rel = _find_module_qualname(layer_mod, dense_mlp)
                q_meta.update(
                    _quantize_layer_mlp_linear(
                        layer_mod, dense_mlp, dense_mlp_rel, shared_mlp_quant, attn_group_size, device, dtype, lut_pool
                    )
                )
            if idx < num_total_layers - 1:
                for b_i, env in enumerate(envelopes):
                    H_student_cpu[b_i] = _call_single_layer(
                        layer_mod, idx, env, H_student_cpu[b_i], kv_student_cpu[b_i]
                    )
            bnd_tensors = _export_module_cpu_tensors(layer_mod)
            bnd_file = f"layer_{idx:03d}_boundary.safetensors"
            save_file(bnd_tensors, os.path.join(layers_out_dir, bnd_file))
            del bnd_tensors
            layer_manifest[str(idx)] = {
                "compiled_moe": False,
                "compiled_mlp": False,
                "boundary_file": f"layers/{bnd_file}",
                "quant_linears": q_meta,
            }
            tag = "DENSE"
        else:
            # Quantize resident Shared MLP if present (`Gemma4TextDecoderLayer.mlp` or `Qwen3Next.mlp.shared_expert`)
            if shared_mlp_quant != "none" and moe_info["shared_mlp_mod"] is not None:
                q_meta.update(
                    _quantize_layer_mlp_linear(
                        layer_mod,
                        moe_info["shared_mlp_mod"],
                        moe_info["shared_mlp_rel"],
                        shared_mlp_quant,
                        attn_group_size,
                        device,
                        dtype,
                        lut_pool,
                    )
                )

            E = int(moe_info["num_experts"])
            D = int(moe_info["d_model"])
            I = int(moe_info["d_ffn"])
            experts_mod = moe_info["experts_mod"]
            experts_rel = moe_info["experts_rel"]

            # A. Capture exact calibration router expert frequencies + input salience `x_ms` while `experts_mod` stays on `meta`!
            x_sq_sum = torch.zeros((D,), dtype=torch.float64, device=dev)
            exp_counts = torch.zeros((E,), dtype=torch.int64, device=dev)
            n_valid_tokens = 0
            cur_env_ref: List[Optional[_BatchCallEnvelope]] = [None]

            def _moe_calib_stop_pre_hook(_mod: nn.Module, args: Tuple[Any, ...]) -> None:
                nonlocal n_valid_tokens
                env_cur = cur_env_ref[0]
                if env_cur is not None and len(args) >= 2:
                    x_in = args[0].detach().reshape(-1, D)
                    tk_idx = args[1].detach().reshape(x_in.shape[0], -1)
                    mask_flat = env_cur.attn_mask_2d_cpu.reshape(-1).to(device=x_in.device, dtype=torch.bool)
                    vx = x_in[mask_flat].float()
                    vi = tk_idx[mask_flat].reshape(-1).long()
                    vi = vi[(vi >= 0) & (vi < E)]
                    if vx.numel() > 0:
                        x_sq_sum.add_(vx.pow(2).sum(dim=0).double())
                        n_valid_tokens += int(vx.shape[0])
                    if vi.numel() > 0:
                        exp_counts.scatter_add_(0, vi, torch.ones_like(vi, dtype=torch.int64))
                raise _StopAtMLPPreHook()

            p_hook = experts_mod.register_forward_pre_hook(_moe_calib_stop_pre_hook)
            try:
                for b_i, env in enumerate(envelopes):
                    cur_env_ref[0] = env
                    memo_gpu: Dict[Tuple[Any, ...], torch.Tensor] = {}
                    args_gpu = _tree_to_device_dedup(env.layer_args_cpu[idx], dev, memo_gpu)
                    kwargs_gpu = _tree_to_device_dedup(env.layer_kwargs_cpu[idx], dev, memo_gpu)
                    kv_tmp = UserDict(kv_student_cpu[b_i])
                    h_gpu = H_student_cpu[b_i].to(device=dev, dtype=dtype, non_blocking=True)
                    try:
                        _ = layer_mod(h_gpu, *args_gpu, shared_kv_states=kv_tmp, **kwargs_gpu)
                    except _StopAtMLPPreHook:
                        pass
            finally:
                p_hook.remove()
                cur_env_ref[0] = None

            x_ms = (
                (x_sq_sum / float(max(1, n_valid_tokens))).float().clamp_min(1e-8)
                if n_valid_tokens > 0
                else None
            )
            # Sort experts descending by calibration activation count (stable tie-break by expert index)
            tie_break = torch.arange(E, device=dev, dtype=torch.float64) * 1e-6
            freq_rank = torch.argsort(exp_counts.double() - tie_break, descending=True).to(
                device="cpu", dtype=torch.int32
            ).contiguous()
            top_tlb_cov = (
                100.0 * float(exp_counts[freq_rank[: min(tlb_slots, E)].to(dev).long()].sum().item())
                / float(max(1, int(exp_counts.sum().item())))
            )

            # B. Stream and pack `E` experts in bounded GPU chunks (`expert_chunk_size = 16`)
            layout = ContiguousExpertPageLayout.compute(d_model=D, d_ffn=I, group_size=expert_group_size)
            expert_pages_cpu = torch.empty((E, layout.page_bytes), dtype=torch.uint8, device="cpu")
            for s_e in range(0, E, expert_chunk_size):
                e_e = min(E, s_e + expert_chunk_size)
                w_gu_c, w_dn_c = _read_moe_expert_chunk(
                    reader, l_prefix, experts_rel, s_e, e_e, device=device, dtype=dtype
                )
                pages_c, _ = quantize_pack_moe_experts_uint4(
                    gate_up_weight=w_gu_c,
                    down_weight=w_dn_c,
                    group_size=expert_group_size,
                    input_salience_ms=x_ms,
                    compute_device=device,
                    expert_chunk_size=expert_chunk_size,
                )
                expert_pages_cpu[s_e:e_e].copy_(pages_c)
                del w_gu_c, w_dn_c, pages_c

            _evict_module_to_meta(experts_mod)

            # C. Advance Student hidden state `H_student_cpu` using compiled `SparseMoETLBExperts`
            moe_vm = SparseMoETLBExperts(
                num_experts=E,
                d_model=D,
                d_ffn=I,
                layout=layout,
                host_expert_pages=expert_pages_cpu,
                lut_pool=lut_pool,
                act_fn=moe_info["act_fn"],
                device=dev,
                tlb_slots=min(E, max(int(tlb_slots), 32)),
                expert_freq_rank=freq_rank,
                mode="vm_paged",
                tlb_strategy="exact_demand",
                layer_idx=idx,
                use_fused_kernels=True,
            )
            exp_parent, exp_attr = _get_parent_and_attr(layer_mod, experts_rel)
            setattr(exp_parent, exp_attr, moe_vm)

            if idx < num_total_layers - 1:
                for b_i, env in enumerate(envelopes):
                    H_student_cpu[b_i] = _call_single_layer(
                        layer_mod, idx, env, H_student_cpu[b_i], kv_student_cpu[b_i]
                    )

            moe_core_tensors, moe_page_tensors, moe_meta = moe_vm.export_serialized_core_and_pages()
            moe_meta["tlb_slots"] = int(min(E, max(1, tlb_slots)))
            moe_vm.release_tlb()
            moe_vm.tlb_cache.host_expert_pages = torch.empty((0, 0), dtype=torch.uint8, device="cpu")
            setattr(exp_parent, exp_attr, nn.Identity())

            non_exp_tensors = _export_module_cpu_tensors(layer_mod, skip_submodules=(experts_rel,))
            combined_core: Dict[str, torch.Tensor] = {}
            for k_ne, t_ne in non_exp_tensors.items():
                combined_core[f"layer.{k_ne}"] = t_ne
            for k_mc, t_mc in moe_core_tensors.items():
                combined_core[f"moe.{k_mc}"] = t_mc

            core_file = f"layer_{idx:03d}_core.safetensors"
            pages_file = f"layer_{idx:03d}_pages.safetensors"
            save_file(combined_core, os.path.join(layers_out_dir, core_file))
            save_file(moe_page_tensors, os.path.join(layers_out_dir, pages_file))
            del non_exp_tensors, moe_core_tensors, moe_page_tensors, combined_core, expert_pages_cpu, moe_vm

            l_stat = {
                "layer_idx": idx,
                "num_experts": E,
                "d_model": D,
                "d_ffn": I,
                "page_bytes": layout.page_bytes,
                "layer_pages_mb": (E * layout.page_bytes) / (1024 ** 2),
                "calib_tokens": n_valid_tokens,
                "static_top_tlb_coverage_pct": top_tlb_cov,
            }
            moe_layer_indices.append(idx)
            moe_layer_stats.append(l_stat)
            layer_manifest[str(idx)] = {
                "compiled_moe": True,
                "compiled_mlp": False,
                "experts_rel_name": experts_rel,
                "core_file": f"layers/{core_file}",
                "pages_file": f"layers/{pages_file}",
                "moe_meta": moe_meta,
                "quant_linears": q_meta,
                "stats": l_stat,
            }
            tag = f"MOE E={E} static_cov@{tlb_slots}={top_tlb_cov:.1f}%"

        if torch.cuda.is_available() and dev.type == "cuda":
            cur_peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
            peak_layer_vram_mb = max(peak_layer_vram_mb, cur_peak)

        for env in envelopes:
            env.layer_args_cpu.pop(idx, None)
            env.layer_kwargs_cpu.pop(idx, None)
        _evict_module_to_meta(layer_mod)
        decoder_layers[idx] = nn.Identity()  # type: ignore[assignment]
        del layer_mod
        gc.collect()
        if os.name != "nt":
            try:
                import ctypes

                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
        if torch.cuda.is_available() and dev.type == "cuda":
            torch.cuda.empty_cache()

        dt_l = time.perf_counter() - t_layer
        print(
            f"[StreamMoECompile] Layer {idx:02d}/{num_total_layers - 1:02d} ({tag}, attn={attn_quant}, "
            f"shared_mlp={shared_mlp_quant}) flushed in {dt_l:.2f}s | Peak Compile VRAM={peak_layer_vram_mb:.1f} MB"
        )

    # Optional: compile native Hugging Face `mtp.*` layer (`Qwen3-Next` / `DeepSeek-V3`) if present in shards!
    mtp_meta: Optional[Dict[str, Any]] = None
    if reader.has_key("mtp.fc.weight") or reader.has_key("mtp.eh_proj.weight"):
        mtp_meta = _compile_native_mtp_from_reader(
            reader=reader,
            src_dir=src_dir,
            out_dir=out_dir,
            attn_quant=attn_quant,
            attn_group_size=attn_group_size,
            shared_mlp_quant=shared_mlp_quant,
            expert_group_size=expert_group_size,
            expert_chunk_size=expert_chunk_size,
            tlb_slots=min(64, int(tlb_slots)),
            device=device,
            dtype=dtype,
            lut_pool=lut_pool,
        )

    total_disk_bytes = 0
    pages_disk_bytes = 0
    ple_disk_bytes = 0
    for root_d, _, files in os.walk(out_dir):
        for fn in files:
            fp = os.path.join(root_d, fn)
            sz = os.path.getsize(fp)
            total_disk_bytes += sz
            if fn.endswith("_pages.safetensors"):
                pages_disk_bytes += sz
            elif fn == "ple_host.safetensors":
                ple_disk_bytes += sz

    compile_wall_s = time.perf_counter() - t_start
    manifest = {
        "format_version": "nvm_moe_v1",
        "source_model": model_id_or_path,
        "num_total_layers": num_total_layers,
        "compiled_layer_indices": moe_layer_indices,
        "layers_qualname": layers_qualname,
        "stripped_multimodal_towers": stripped_towers,
        "tie_word_embeddings": tie_word_embeddings,
        "attn_quant": attn_quant,
        "attn_group_size": attn_group_size,
        "shared_mlp_quant": shared_mlp_quant,
        "lm_head_quant": lm_head_quant,
        "lm_head_quant_meta": lm_head_quant_meta,
        "expert_group_size": expert_group_size,
        "tlb_slots": tlb_slots,
        "ple_meta": ple_meta,
        "mtp_meta": mtp_meta,
        "compile_telemetry": {
            "compile_wall_s": compile_wall_s,
            "peak_preamble_vram_mb": peak_preamble_vram_mb,
            "peak_single_layer_vram_mb": peak_layer_vram_mb,
            "peak_compile_vram_mb": max(peak_preamble_vram_mb, peak_layer_vram_mb),
            "total_disk_mb": total_disk_bytes / (1024 ** 2),
            "core_resident_disk_mb": (total_disk_bytes - pages_disk_bytes - ple_disk_bytes) / (1024 ** 2),
            "pages_disk_mb": pages_disk_bytes / (1024 ** 2),
            "ple_disk_mb": ple_disk_bytes / (1024 ** 2),
        },
        "layers": layer_manifest,
        "compiled_layer_stats": moe_layer_stats,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    del model, envelopes, H_student_cpu, kv_student_cpu, lut_pool
    gc.collect()
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.empty_cache()

    return manifest


@torch.no_grad()
def load_nvm_cartridge(
    nvm_dir: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    mode: str = "vm_paged",
    tlb_strategy: str = "speculative_tlb",
    page_storage: str = "pinned_ram",
    tlb_slots_override: Optional[int] = None,
    decode_req_pages_override: Optional[int] = None,
    use_phase_c_kernels: bool = True,
    load_tokenizer: bool = True,
) -> Tuple[
    nn.Module,
    Any,
    Dict[int, BitTieredTLBMLP],
    SharedQuantLUTPool,
    Optional[DeterministicPLEOffloader],
    Dict[str, Any],
]:
    """Cold-boots a `.nvm` Neural VM cartridge directly onto GPU with zero dense BF16 allocation."""
    if page_storage not in VALID_PAGE_STORAGE_MODES:
        raise ValueError(f"Invalid page_storage '{page_storage}'. Expected {VALID_PAGE_STORAGE_MODES}.")

    t0 = time.perf_counter()
    cart_dir = os.path.abspath(nvm_dir)
    with open(os.path.join(cart_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dev = torch.device(device)
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    tokenizer = None
    if load_tokenizer:
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(cart_dir, local_files_only=True)
        except Exception:
            src_snap = resolve_hf_snapshot_dir(manifest["source_model"])
            tokenizer = AutoTokenizer.from_pretrained(src_snap, local_files_only=True)

    model, _ = _create_empty_skeleton(
        cart_dir,
        dtype=dtype,
        skip_multimodal=bool(manifest.get("stripped_multimodal_towers")),
    )
    decoder_layers = resolve_decoder_layers(model)
    layers_qualname = manifest["layers_qualname"]
    lut_pool = SharedQuantLUTPool(device=dev, dtype=dtype)

    # 1. Load `base_resident.safetensors` onto GPU (handling optional quantized untied `lm_head`)
    base_path = os.path.join(cart_dir, "base_resident.safetensors")
    lm_head_qmeta = manifest.get("lm_head_quant_meta")
    with safe_open(base_path, framework="pt", device="cpu") as sf:
        base_keys_set = set(sf.keys())
        consumed_base: set[str] = set()
        if lm_head_qmeta is not None and "lm_head.q_weight" in base_keys_set:
            qlm = PhaseCQuantLinear(
                in_features=int(lm_head_qmeta["in_features"]),
                out_features=int(lm_head_qmeta["out_features"]),
                quant_mode=lm_head_qmeta["quant_mode"],
                q_weight=sf.get_tensor("lm_head.q_weight").to(device=dev, non_blocking=True),
                scale=sf.get_tensor("lm_head.scale").to(device=dev, non_blocking=True),
                wmin=(
                    sf.get_tensor("lm_head.wmin").to(device=dev, non_blocking=True)
                    if lm_head_qmeta.get("has_wmin")
                    else None
                ),
                bias=(
                    sf.get_tensor("lm_head.bias").to(device=dev, non_blocking=True)
                    if lm_head_qmeta.get("has_bias")
                    else None
                ),
                group_size=int(lm_head_qmeta.get("group_size", 64)),
                lut_pool=lut_pool,
            )
            model.lm_head = qlm
            consumed_base.update({"lm_head.q_weight", "lm_head.scale", "lm_head.wmin", "lm_head.bias"})

        for key in base_keys_set:
            if key in consumed_base:
                continue
            t = sf.get_tensor(key)
            t_gpu = (
                t.to(device=dev, dtype=dtype, non_blocking=True).contiguous()
                if t.is_floating_point()
                else t.to(device=dev, non_blocking=True).contiguous()
            )
            _set_tensor_by_qualname(model, key, t_gpu)

    # 2. Attach `DeterministicPLEOffloader` via zero-copy OS mmap (`~1 ms` load, `0 MB` VRAM!)
    ple_offloader: Optional[DeterministicPLEOffloader] = None
    ple_meta = manifest.get("ple_meta")
    if ple_meta is not None:
        ple_file = (
            os.path.join(cart_dir, ple_meta["source_file"])
            if ple_meta["storage"] == "standalone"
            else ple_meta["source_file"]
        )
        ple_w_mmap = mmap_safetensor_tensor(ple_file, ple_meta["source_key"])
        ple_offloader = DeterministicPLEOffloader.from_cpu_tensor(
            ple_w_mmap,
            embed_scale=float(ple_meta["embed_scale"]),
            padding_idx=ple_meta["padding_idx"],
        )
        ple_parent, ple_attr = _get_parent_and_attr(model, ple_meta["qualname"])
        setattr(ple_parent, ple_attr, ple_offloader)

    fmt_version = manifest.get("format_version", "nvm_v1")
    shared_dma_stream = (
        torch.cuda.Stream(device=dev) if (torch.cuda.is_available() and dev.type == "cuda") else None
    )

    if fmt_version == "nvm_moe_v1":
        # Map dense Phase C-E `tlb_strategy` aliases smoothly to MoE strategies
        if tlb_strategy in ("speculative_tlb", "amortized_spec_tlb", "overlap_shared_mlp"):
            eff_moe_strategy = "overlap_shared_mlp"
        elif tlb_strategy in ("sync_tlb", "exact_demand"):
            eff_moe_strategy = "exact_demand"
        else:
            eff_moe_strategy = tlb_strategy

        moe_experts_by_layer: Dict[int, Any] = {}
        num_total_layers = int(manifest["num_total_layers"])
        for idx in range(num_total_layers):
            l_info = manifest["layers"][str(idx)]
            layer_mod = decoder_layers[idx]
            quant_linears_meta = l_info.get("quant_linears", {})

            if not l_info.get("compiled_moe", False):
                bnd_path = os.path.join(cart_dir, l_info["boundary_file"])
                with safe_open(bnd_path, framework="pt", device="cpu") as sf:
                    bnd_sd = {k: sf.get_tensor(k) for k in sf.keys()}

                consumed: set[str] = set()
                for q_rel, q_spec in quant_linears_meta.items():
                    qlin = PhaseCQuantLinear(
                        in_features=int(q_spec["in_features"]),
                        out_features=int(q_spec["out_features"]),
                        quant_mode=q_spec["quant_mode"],
                        q_weight=bnd_sd[f"{q_rel}.q_weight"].to(device=dev, non_blocking=True),
                        scale=bnd_sd[f"{q_rel}.scale"].to(device=dev, non_blocking=True),
                        wmin=bnd_sd[f"{q_rel}.wmin"].to(device=dev, non_blocking=True) if q_spec["has_wmin"] else None,
                        bias=bnd_sd[f"{q_rel}.bias"].to(device=dev, non_blocking=True) if q_spec["has_bias"] else None,
                        group_size=int(q_spec["group_size"]),
                        lut_pool=lut_pool,
                    )
                    q_parent, q_attr = _get_parent_and_attr(layer_mod, q_rel)
                    setattr(q_parent, q_attr, qlin)
                    for suf in ("q_weight", "scale", "wmin", "bias"):
                        consumed.add(f"{q_rel}.{suf}")

                for k, t_cpu in bnd_sd.items():
                    if k in consumed:
                        continue
                    t_gpu = (
                        t_cpu.to(device=dev, dtype=dtype, non_blocking=True).contiguous()
                        if t_cpu.is_floating_point()
                        else t_cpu.to(device=dev, non_blocking=True).contiguous()
                    )
                    _set_tensor_by_qualname(layer_mod, k, t_gpu)
                del bnd_sd
            else:
                core_path = os.path.join(cart_dir, l_info["core_file"])
                pages_path = os.path.join(cart_dir, l_info["pages_file"])
                with safe_open(core_path, framework="pt", device="cpu") as sf:
                    layer_sd: Dict[str, torch.Tensor] = {}
                    moe_core_sd: Dict[str, torch.Tensor] = {}
                    for k in sf.keys():
                        if k.startswith("layer."):
                            layer_sd[k[6:]] = sf.get_tensor(k)
                        elif k.startswith("moe."):
                            moe_core_sd[k[4:]] = sf.get_tensor(k)

                consumed = set()
                for q_rel, q_spec in quant_linears_meta.items():
                    qlin = PhaseCQuantLinear(
                        in_features=int(q_spec["in_features"]),
                        out_features=int(q_spec["out_features"]),
                        quant_mode=q_spec["quant_mode"],
                        q_weight=layer_sd[f"{q_rel}.q_weight"].to(device=dev, non_blocking=True),
                        scale=layer_sd[f"{q_rel}.scale"].to(device=dev, non_blocking=True),
                        wmin=layer_sd[f"{q_rel}.wmin"].to(device=dev, non_blocking=True) if q_spec["has_wmin"] else None,
                        bias=layer_sd[f"{q_rel}.bias"].to(device=dev, non_blocking=True) if q_spec["has_bias"] else None,
                        group_size=int(q_spec["group_size"]),
                        lut_pool=lut_pool,
                    )
                    q_parent, q_attr = _get_parent_and_attr(layer_mod, q_rel)
                    setattr(q_parent, q_attr, qlin)
                    for suf in ("q_weight", "scale", "wmin", "bias"):
                        consumed.add(f"{q_rel}.{suf}")

                for k, t_cpu in layer_sd.items():
                    if k in consumed:
                        continue
                    t_gpu = (
                        t_cpu.to(device=dev, dtype=dtype, non_blocking=True).contiguous()
                        if t_cpu.is_floating_point()
                        else t_cpu.to(device=dev, non_blocking=True).contiguous()
                    )
                    _set_tensor_by_qualname(layer_mod, k, t_gpu)

                if page_storage == "nvme_mmap":
                    expert_pages_host = mmap_safetensor_tensor(pages_path, "expert_pages")
                else:
                    with safe_open(pages_path, framework="pt", device="cpu") as pf:
                        expert_pages_host = pf.get_tensor("expert_pages").contiguous()
                    if torch.cuda.is_available() and dev.type == "cuda":
                        try:
                            expert_pages_host = expert_pages_host.pin_memory()
                        except Exception:
                            pass

                moe_info = _locate_layer_moe_components(layer_mod)
                act_fn = moe_info["act_fn"] if moe_info is not None else nn.GELU(approximate="tanh")
                moe_vm = SparseMoETLBExperts.from_serialized_buffers(
                    core_tensors=moe_core_sd,
                    expert_pages_host=expert_pages_host,
                    meta=l_info["moe_meta"],
                    lut_pool=lut_pool,
                    act_fn=act_fn,
                    device=dev,
                    mode=mode,
                    tlb_strategy=eff_moe_strategy,
                    tlb_slots_override=tlb_slots_override,
                    shared_dma_stream=shared_dma_stream,
                    layer_idx=idx,
                    use_fused_kernels=use_phase_c_kernels,
                )
                exp_parent, exp_attr = _get_parent_and_attr(layer_mod, l_info["experts_rel_name"])
                setattr(exp_parent, exp_attr, moe_vm)
                if eff_moe_strategy in ("overlap_shared_mlp", "lookahead_overlap"):
                    install_moe_intra_layer_overlap(layer_mod)
                moe_experts_by_layer[idx] = moe_vm
                del layer_sd, moe_core_sd

        _move_cpu_buffers_to_device(model, dev)
        if manifest.get("tie_word_embeddings", False):
            model.tie_weights()

        l2_info: Dict[str, Any] = {"l2_slots_per_layer": 0, "total_l2_gb": 0.0, "all_mapped": False}
        cuda_graph_mode = "disabled"
        if use_phase_c_kernels and torch.cuda.is_available() and dev.type == "cuda":
            if moe_experts_by_layer:
                ws_moe = get_shared_moe_workspace(dev)
                ws_moe.ensure_capacity(
                    d_model=max(m.d_model for m in moe_experts_by_layer.values()),
                    d_ffn=max(m.d_ffn for m in moe_experts_by_layer.values()),
                    page_bytes=max(m.layout.page_bytes for m in moe_experts_by_layer.values()),
                )
                l2_info = auto_configure_moe_l2_zero_copy(moe_experts_by_layer)
            from runtime.qwen3_next_kernels import install_fast_moe_core_kernels, install_qwen3_next_fast_gdn
            from runtime.cuda_graph_decode import CUDAGraphDecodeRunner

            if any(m.__class__.__name__ == "Qwen3NextGatedDeltaNet" for m in model.modules()):
                install_qwen3_next_fast_gdn(model)
            else:
                install_fast_moe_core_kernels(model)

            graph_runner = CUDAGraphDecodeRunner(model, moe_experts_by_layer)
            if graph_runner.is_supported:
                model._nvm_graph_runner = graph_runner  # type: ignore[attr-defined]
                cuda_graph_mode = graph_runner.active_graph_mode

        model.eval()
        if torch.cuda.is_available() and dev.type == "cuda":
            torch.cuda.synchronize(dev)

        cold_boot_s = time.perf_counter() - t0
        alloc_mb = torch.cuda.memory_allocated(dev) / (1024 ** 2) if (torch.cuda.is_available() and dev.type == "cuda") else 0.0
        peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2) if (torch.cuda.is_available() and dev.type == "cuda") else 0.0
        manifest["last_cold_boot"] = {
            "cold_boot_s": cold_boot_s,
            "mode": mode,
            "tlb_strategy": eff_moe_strategy,
            "page_storage": page_storage,
            "cuda_allocated_mb": alloc_mb,
            "cuda_peak_mb": peak_mb,
            "l2_slots_per_layer": l2_info["l2_slots_per_layer"],
            "l2_ram_gb": l2_info["total_l2_gb"],
            "all_experts_mapped": l2_info["all_mapped"],
            "cuda_graph_mode": cuda_graph_mode,
        }
        return model, tokenizer, moe_experts_by_layer, lut_pool, ple_offloader, manifest

    # 3. Setup shared DMA stream & `SharedBatchTLBController` across compiled layers (`nvm_v1`)
    compiled_indices = [int(i) for i in manifest["compiled_layer_indices"]]
    eff_req_pages = (
        int(decode_req_pages_override)
        if decode_req_pages_override is not None
        else int(manifest["decode_req_pages"])
    )
    batch_controller = (
        SharedBatchTLBController(
            num_layers=len(compiled_indices),
            decode_req_pages=eff_req_pages,
            device=dev,
            dma_stream=shared_dma_stream,
        )
        if compiled_indices
        else None
    )
    compiled_slot_map = {l_idx: slot for slot, l_idx in enumerate(compiled_indices)}

    # 4. Stream per-layer shards directly into `decoder_layers[idx]`
    phase_b_mlps: Dict[int, BitTieredTLBMLP] = {}
    num_total_layers = int(manifest["num_total_layers"])

    for idx in range(num_total_layers):
        l_info = manifest["layers"][str(idx)]
        layer_mod = decoder_layers[idx]
        quant_linears_meta: Dict[str, Dict[str, Any]] = l_info.get("quant_linears", {})

        if not l_info["compiled_mlp"]:
            bnd_path = os.path.join(cart_dir, l_info["boundary_file"])
            with safe_open(bnd_path, framework="pt", device="cpu") as sf:
                bnd_sd = {k: sf.get_tensor(k) for k in sf.keys()}

            consumed: set[str] = set()
            for q_rel, q_spec in quant_linears_meta.items():
                qlin = PhaseCQuantLinear(
                    in_features=int(q_spec["in_features"]),
                    out_features=int(q_spec["out_features"]),
                    quant_mode=q_spec["quant_mode"],
                    q_weight=bnd_sd[f"{q_rel}.q_weight"].to(device=dev, non_blocking=True),
                    scale=bnd_sd[f"{q_rel}.scale"].to(device=dev, non_blocking=True),
                    wmin=bnd_sd[f"{q_rel}.wmin"].to(device=dev, non_blocking=True) if q_spec["has_wmin"] else None,
                    bias=bnd_sd[f"{q_rel}.bias"].to(device=dev, non_blocking=True) if q_spec["has_bias"] else None,
                    group_size=int(q_spec["group_size"]),
                    lut_pool=lut_pool,
                )
                q_parent, q_attr = _get_parent_and_attr(layer_mod, q_rel)
                setattr(q_parent, q_attr, qlin)
                for suf in ("q_weight", "scale", "wmin", "bias"):
                    consumed.add(f"{q_rel}.{suf}")

            for k, t_cpu in bnd_sd.items():
                if k in consumed:
                    continue
                t_gpu = (
                    t_cpu.to(device=dev, dtype=dtype, non_blocking=True).contiguous()
                    if t_cpu.is_floating_point()
                    else t_cpu.to(device=dev, non_blocking=True).contiguous()
                )
                _set_tensor_by_qualname(layer_mod, k, t_gpu)
            del bnd_sd
        else:
            core_path = os.path.join(cart_dir, l_info["core_file"])
            pages_path = os.path.join(cart_dir, l_info["pages_file"])
            with safe_open(core_path, framework="pt", device="cpu") as sf:
                layer_sd: Dict[str, torch.Tensor] = {}
                mlp_core_sd: Dict[str, torch.Tensor] = {}
                for k in sf.keys():
                    if k.startswith("layer."):
                        layer_sd[k[6:]] = sf.get_tensor(k)
                    elif k.startswith("mlp."):
                        mlp_core_sd[k[4:]] = sf.get_tensor(k)

            consumed = set()
            for q_rel, q_spec in quant_linears_meta.items():
                qlin = PhaseCQuantLinear(
                    in_features=int(q_spec["in_features"]),
                    out_features=int(q_spec["out_features"]),
                    quant_mode=q_spec["quant_mode"],
                    q_weight=layer_sd[f"{q_rel}.q_weight"].to(device=dev, non_blocking=True),
                    scale=layer_sd[f"{q_rel}.scale"].to(device=dev, non_blocking=True),
                    wmin=layer_sd[f"{q_rel}.wmin"].to(device=dev, non_blocking=True) if q_spec["has_wmin"] else None,
                    bias=layer_sd[f"{q_rel}.bias"].to(device=dev, non_blocking=True) if q_spec["has_bias"] else None,
                    group_size=int(q_spec["group_size"]),
                    lut_pool=lut_pool,
                )
                q_parent, q_attr = _get_parent_and_attr(layer_mod, q_rel)
                setattr(q_parent, q_attr, qlin)
                for suf in ("q_weight", "scale", "wmin", "bias"):
                    consumed.add(f"{q_rel}.{suf}")

            for k, t_cpu in layer_sd.items():
                if k in consumed:
                    continue
                t_gpu = (
                    t_cpu.to(device=dev, dtype=dtype, non_blocking=True).contiguous()
                    if t_cpu.is_floating_point()
                    else t_cpu.to(device=dev, non_blocking=True).contiguous()
                )
                _set_tensor_by_qualname(layer_mod, k, t_gpu)

            # Cold Specialist Pages: either `pinned_ram` or zero-RAM `nvme_mmap`!
            if page_storage == "nvme_mmap":
                q4_pages_host = mmap_safetensor_tensor(pages_path, "q4_pages")
            else:
                with safe_open(pages_path, framework="pt", device="cpu") as pf:
                    q4_pages_host = pf.get_tensor("q4_pages").contiguous()
                if torch.cuda.is_available() and dev.type == "cuda":
                    try:
                        q4_pages_host = q4_pages_host.pin_memory()
                    except Exception:
                        pass

            orig_mlp = get_target_mlp(model, idx)
            act_fn = getattr(orig_mlp, "act_fn", nn.GELU(approximate="tanh"))
            b_mlp = BitTieredTLBMLP.from_serialized_buffers(
                core_tensors=mlp_core_sd,
                q4_pages_host=q4_pages_host,
                meta=l_info["mlp_meta"],
                lut_pool=lut_pool,
                act_fn=act_fn,
                device=dev,
                mode=mode,
                tlb_strategy=tlb_strategy,
                tlb_slots_override=tlb_slots_override,
                decode_req_pages_override=decode_req_pages_override,
                shared_dma_stream=shared_dma_stream,
                batch_controller=batch_controller,
                controller_slot_idx=compiled_slot_map[idx],
                use_phase_c_kernels=use_phase_c_kernels,
            )
            mlp_parent, mlp_attr = _get_parent_and_attr(layer_mod, l_info["mlp_rel_name"])
            setattr(mlp_parent, mlp_attr, b_mlp)
            phase_b_mlps[idx] = b_mlp
            del layer_sd, mlp_core_sd

    _move_cpu_buffers_to_device(model, dev)
    if manifest.get("tie_word_embeddings", False):
        model.tie_weights()

    # Pre-size shared Phase C CUDA workspace once across all compiled layer widths
    if use_phase_c_kernels and phase_b_mlps and torch.cuda.is_available() and dev.type == "cuda" and dtype == torch.bfloat16:
        ws = get_shared_phase_c_workspace(dev)
        ws.ensure_capacity(
            d_model=max(m.d_model for m in phase_b_mlps.values()),
            n_hot=max(m.n_hot for m in phase_b_mlps.values()),
            n_cold=max(m.n_cold for m in phase_b_mlps.values()),
            c_act=max(m.tlb_cache.tlb_slots * m.page_size for m in phase_b_mlps.values()),
            bridge_rank=max((int(m.U_bridge.shape[1]) if m.U_bridge.ndim == 2 else 0) for m in phase_b_mlps.values()),
            num_pages=max(m.num_pages for m in phase_b_mlps.values()),
        )
        for m in phase_b_mlps.values():
            m.set_phase_c_enabled(True)

    model.eval()
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.synchronize(dev)

    cold_boot_s = time.perf_counter() - t0
    alloc_mb = torch.cuda.memory_allocated(dev) / (1024 ** 2) if (torch.cuda.is_available() and dev.type == "cuda") else 0.0
    peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2) if (torch.cuda.is_available() and dev.type == "cuda") else 0.0

    manifest["last_cold_boot"] = {
        "cold_boot_s": cold_boot_s,
        "mode": mode,
        "tlb_strategy": tlb_strategy,
        "page_storage": page_storage,
        "cuda_allocated_mb": alloc_mb,
        "cuda_peak_mb": peak_mb,
    }
    return model, tokenizer, phase_b_mlps, lut_pool, ple_offloader, manifest


class _FinalHiddenInjectorLayer(nn.Module):
    """Zero-FLOP pass-through layer that optionally injects precomputed `H_L` at Layer 0 for exact head loss."""

    def __init__(self, orig_layer: nn.Module) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        for attr in ("layer_type", "attention_type", "is_sliding", "block_type"):
            if hasattr(orig_layer, attr):
                setattr(self, attr, getattr(orig_layer, attr))
        self.injected_hidden: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.injected_hidden is not None:
            return self.injected_hidden
        return hidden_states


@torch.no_grad()
def evaluate_streaming_dense_bf16_ppl(
    model_id_or_path: str,
    splits: Dict[str, TokenizedSplit],
    batch_size: int = 2,
    skip_multimodal: bool = True,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Computes exact 0-ulp unmodified Dense BF16 Perplexity & Next-Token Accuracy in O(1) single-layer VRAM.

    Enables exact ground-truth teacher evaluation on 30B-70B+ checkpoints that exceed single-GPU VRAM and System RAM.
    """
    t0 = time.perf_counter()
    dev = torch.device(device)
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    src_dir = resolve_hf_snapshot_dir(model_id_or_path)
    reader = SafetensorShardReader(src_dir)
    model, stripped_towers = _create_empty_skeleton(src_dir, dtype=dtype, skip_multimodal=skip_multimodal)
    if torch.cuda.is_available() and dev.type == "cuda":
        if any(m.__class__.__name__ == "Qwen3NextGatedDeltaNet" for m in model.modules()):
            from runtime.qwen3_next_kernels import install_qwen3_next_fast_gdn

            install_qwen3_next_fast_gdn(model)
    decoder_layers = resolve_decoder_layers(model)
    num_total_layers = len(decoder_layers)
    layers_qualname = _find_module_qualname(model, decoder_layers)
    layers_prefix = f"{layers_qualname}."

    # Attach PLE via mmap if architecture has PLE
    ple_qualname = _resolve_ple_module_path(model)
    ple_prefix = f"{ple_qualname}." if ple_qualname else None
    if ple_qualname is not None and reader.has_key(f"{ple_qualname}.weight"):
        ple_parent, ple_attr = _get_parent_and_attr(model, ple_qualname)
        orig_ple = getattr(ple_parent, ple_attr)
        raw_scale = getattr(orig_ple, "embed_scale", 1.0)
        ple_scale_float = float(raw_scale.item()) if isinstance(raw_scale, torch.Tensor) else float(raw_scale)
        ple_pad_idx = getattr(orig_ple, "padding_idx", None)
        ple_mmap = mmap_safetensor_tensor(reader.get_file_for_key(f"{ple_qualname}.weight"), f"{ple_qualname}.weight")
        setattr(
            ple_parent,
            ple_attr,
            DeterministicPLEOffloader.from_cpu_tensor(ple_mmap, embed_scale=ple_scale_float, padding_idx=ple_pad_idx),
        )
        del orig_ple

    skipped_prefixes = tuple(stripped_towers)
    valid_model_keys = {k for k, _ in model.named_parameters()} | {k for k, _ in model.named_buffers()}
    base_keys: List[str] = []
    for key in reader.keys():
        if key.startswith(layers_prefix):
            continue
        if ple_prefix is not None and key.startswith(ple_prefix):
            continue
        if any(key == sp or key.startswith(sp + ".") for sp in skipped_prefixes):
            continue
        if key not in valid_model_keys:
            continue
        base_keys.append(key)
        _set_tensor_by_qualname(model, key, reader.get_tensor(key, device=device, dtype=dtype))

    _move_cpu_buffers_to_device(model, dev, exclude_prefixes=(layers_qualname,))
    tie_word_embeddings = False
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight") and model.lm_head.weight.device.type == "meta":
        model.tie_weights()
        tie_word_embeddings = True

    # 1. Record Preamble envelopes across all eval batches
    orig_meta_layers = [decoder_layers[i] for i in range(num_total_layers)]
    spy_layers = [_PreambleSpyLayer(orig_meta_layers[i], i) for i in range(num_total_layers)]
    for i in range(num_total_layers):
        decoder_layers[i] = spy_layers[i]
    model.eval()

    batch_records: List[Tuple[str, torch.Tensor, torch.Tensor, _BatchCallEnvelope]] = []
    for split_name, split in splits.items():
        n_rows = split.input_ids.shape[0]
        for s in range(0, n_rows, batch_size):
            e = min(n_rows, s + batch_size)
            b_ids_cpu = split.input_ids[s:e].cpu()
            b_mask_cpu = split.attention_mask[s:e].cpu()
            env = _BatchCallEnvelope(split_kind=split_name, attn_mask_2d_cpu=b_mask_cpu)
            for spy in spy_layers:
                spy.active_envelope = env
            try:
                _ = model(input_ids=b_ids_cpu.to(dev), attention_mask=b_mask_cpu.to(dev), use_cache=False, logits_to_keep=1)
            except TypeError:
                _ = model(input_ids=b_ids_cpu.to(dev), attention_mask=b_mask_cpu.to(dev), use_cache=False)
            for spy in spy_layers:
                spy.active_envelope = None
            assert env.h0_cpu is not None
            batch_records.append((split_name, b_ids_cpu, b_mask_cpu, env))

    for i in range(num_total_layers):
        decoder_layers[i] = orig_meta_layers[i]
    del spy_layers, orig_meta_layers

    # Evict Preamble while streaming 0 .. L-1
    if tie_word_embeddings and hasattr(model, "lm_head"):
        model.lm_head.weight = nn.Parameter(torch.empty(0, dtype=dtype, device="meta"), requires_grad=False)
    for top_name, top_mod in model.named_children():
        if top_name == layers_qualname.split(".")[0]:
            continue
        _evict_module_to_meta(top_mod)
    lm_container, _ = _get_parent_and_attr(model, layers_qualname)
    for _, child_mod in lm_container.named_children():
        if child_mod is decoder_layers or isinstance(child_mod, DeterministicPLEOffloader):
            continue
        _evict_module_to_meta(child_mod)
    gc.collect()
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Stream unmodified Dense BF16 layers 0 .. L-1
    H_cur_cpu: List[torch.Tensor] = [rec[3].h0_cpu for rec in batch_records]  # type: ignore[misc]
    kv_cur_cpu: List[UserDict] = [UserDict() for _ in batch_records]
    for idx in range(num_total_layers):
        layer_mod = decoder_layers[idx]
        l_prefix = f"{layers_prefix}{idx}."
        valid_layer_keys = {k for k, _ in layer_mod.named_parameters()} | {k for k, _ in layer_mod.named_buffers()}
        _populate_layer_bf16_from_reader(layer_mod, reader, l_prefix, valid_layer_keys, device, dtype)
        _move_cpu_buffers_to_device(layer_mod, dev)
        layer_mod.eval()

        for b_i, (_, _, _, env) in enumerate(batch_records):
            memo_gpu: Dict[Tuple[Any, ...], torch.Tensor] = {}
            args_gpu = _tree_to_device_dedup(env.layer_args_cpu[idx], dev, memo_gpu)
            kwargs_gpu = _tree_to_device_dedup(env.layer_kwargs_cpu[idx], dev, memo_gpu)
            kv_work = UserDict(kv_cur_cpu[b_i])
            out_gpu = _unwrap_layer_output(
                layer_mod(H_cur_cpu[b_i].to(device=dev, dtype=dtype, non_blocking=True), *args_gpu, shared_kv_states=kv_work, **kwargs_gpu)
            )
            for k_type, pair in kv_work.items():
                if isinstance(pair, tuple) and len(pair) == 2:
                    kv_cur_cpu[b_i][k_type] = (pair[0].detach().cpu(), pair[1].detach().cpu())
            H_cur_cpu[b_i] = out_gpu.detach().cpu()

        _evict_module_to_meta(layer_mod)
        gc.collect()
        if torch.cuda.is_available() and dev.type == "cuda":
            torch.cuda.empty_cache()

    # 3. Create fresh head skeleton (`embed_tokens`, `norm`, `lm_head` + non-persistent `embed_scale`/`rotary_emb`)
    del model
    model, _ = _create_empty_skeleton(src_dir, dtype=dtype, skip_multimodal=skip_multimodal)
    decoder_layers = resolve_decoder_layers(model)
    for key in base_keys:
        _set_tensor_by_qualname(model, key, reader.get_tensor(key, device=device, dtype=dtype))
    _move_cpu_buffers_to_device(model, dev, exclude_prefixes=(layers_qualname,))
    if tie_word_embeddings:
        model.tie_weights()

    inject_layers = [_FinalHiddenInjectorLayer(decoder_layers[i]) for i in range(num_total_layers)]
    for i in range(num_total_layers):
        decoder_layers[i] = inject_layers[i]
    model.eval()

    acc_stats: Dict[str, Dict[str, Any]] = {
        sn: {"nll": 0.0, "tokens": 0, "correct": 0, "preds": []} for sn in splits
    }
    for b_i, (split_name, b_ids_cpu, b_mask_cpu, _) in enumerate(batch_records):
        b_ids = b_ids_cpu.to(dev)
        b_mask = b_mask_cpu.to(dev)
        inject_layers[0].injected_hidden = H_cur_cpu[b_i].to(device=dev, dtype=dtype)
        outputs = model(input_ids=b_ids, attention_mask=b_mask, use_cache=False)
        inject_layers[0].injected_hidden = None
        logits = outputs.logits

        # Row-by-row FP32 cross-entropy keeps peak vocab buffer < 260 MiB even for vocab=262,144
        preds_batch: List[torch.Tensor] = []
        for r in range(logits.shape[0]):
            sl_r = logits[r, :-1, :].float()
            lbl_r = b_ids[r, 1:]
            msk_r = b_mask[r, 1:].bool()
            pr_r = sl_r.argmax(dim=-1)
            losses_r = F.cross_entropy(sl_r, lbl_r, reduction="none")
            acc_stats[split_name]["nll"] += float(losses_r[msk_r].sum().item())
            acc_stats[split_name]["tokens"] += int(msk_r.sum().item())
            acc_stats[split_name]["correct"] += int((pr_r[msk_r] == lbl_r[msk_r]).sum().item())
            preds_batch.append(pr_r.cpu())
            del sl_r, losses_r
        acc_stats[split_name]["preds"].append(torch.cat([p.unsqueeze(0) for p in preds_batch], dim=0))
        del outputs, logits

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        if (torch.cuda.is_available() and dev.type == "cuda")
        else 0.0
    )
    results: Dict[str, Any] = {
        "wall_s": time.perf_counter() - t0,
        "peak_vram_mb": peak_vram_mb,
        "splits": {},
    }
    for sn, st in acc_stats.items():
        toks = max(1, int(st["tokens"]))
        mean_loss = float(st["nll"]) / toks
        results["splits"][sn] = {
            "loss": mean_loss,
            "perplexity": math.exp(min(mean_loss, 20.0)),
            "next_token_acc_pct": 100.0 * (int(st["correct"]) / toks),
            "tokens": int(st["tokens"]),
            "preds": torch.cat(st["preds"], dim=0),
        }

    del model, batch_records, H_cur_cpu, kv_cur_cpu
    gc.collect()
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.empty_cache()
    return results


class _OffloadedMMapDecoderLayer(nn.Module):
    """Streams an offloaded decoder layer's BF16 weights from host shard across PCIe on each forward call."""

    def __init__(
        self,
        orig_layer: nn.Module,
        reader: SafetensorShardReader,
        l_prefix: str,
        valid_layer_keys: set[str],
        device: torch.device,
        dtype: torch.dtype,
        pcie_counter: List[int],
    ) -> None:
        super().__init__()
        self.orig_layer = orig_layer
        self.reader = reader
        self.l_prefix = l_prefix
        self.valid_layer_keys = valid_layer_keys
        self.device = device
        self.dtype = dtype
        self.pcie_counter = pcie_counter
        self.gradient_checkpointing = False
        for attr in ("layer_type", "attention_type", "is_sliding", "block_type"):
            if hasattr(orig_layer, attr):
                setattr(self, attr, getattr(orig_layer, attr))

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        step_bytes, loaded_keys = _populate_layer_bf16_from_reader(
            self.orig_layer,
            self.reader,
            self.l_prefix,
            self.valid_layer_keys,
            str(self.device),
            self.dtype,
        )
        self.pcie_counter[0] += step_bytes
        out = self.orig_layer(hidden_states, *args, **kwargs)
        for rel_k in loaded_keys:
            _set_tensor_by_qualname(self.orig_layer, rel_k, torch.empty(0, dtype=self.dtype, device="meta"))
        return out


@torch.no_grad()
def measure_layer_offload_kv_decode(
    model_id_or_path: str,
    prompt_ids: torch.Tensor,
    num_gpu_layers: int = 18,
    num_decode_tokens: int = 8,
    warmup_steps: int = 2,
    skip_multimodal: bool = True,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Benchmarks real GPU + Host Layer Offloading (`use_cache=True`) on models exceeding 24 GB VRAM.

    Keeps `embed_tokens` + `lm_head` + `num_gpu_layers` decoder layers resident on GPU, and streams the
    remaining `L - num_gpu_layers` dense BF16 layers over PCIe from read-only mmap (`0 MB` swap thrash).
    """
    dev = torch.device(device)
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    src_dir = resolve_hf_snapshot_dir(model_id_or_path)
    reader = SafetensorShardReader(src_dir)
    model, stripped_towers = _create_empty_skeleton(src_dir, dtype=dtype, skip_multimodal=skip_multimodal)
    if torch.cuda.is_available() and dev.type == "cuda":
        if any(m.__class__.__name__ == "Qwen3NextGatedDeltaNet" for m in model.modules()):
            from runtime.qwen3_next_kernels import install_qwen3_next_fast_gdn

            install_qwen3_next_fast_gdn(model)
    decoder_layers = resolve_decoder_layers(model)
    num_total_layers = len(decoder_layers)
    layers_qualname = _find_module_qualname(model, decoder_layers)
    layers_prefix = f"{layers_qualname}."

    ple_qualname = _resolve_ple_module_path(model)
    ple_prefix = f"{ple_qualname}." if ple_qualname else None
    if ple_qualname is not None and reader.has_key(f"{ple_qualname}.weight"):
        ple_parent, ple_attr = _get_parent_and_attr(model, ple_qualname)
        orig_ple = getattr(ple_parent, ple_attr)
        raw_scale = getattr(orig_ple, "embed_scale", 1.0)
        ple_scale_float = float(raw_scale.item()) if isinstance(raw_scale, torch.Tensor) else float(raw_scale)
        ple_pad_idx = getattr(orig_ple, "padding_idx", None)
        ple_mmap = mmap_safetensor_tensor(reader.get_file_for_key(f"{ple_qualname}.weight"), f"{ple_qualname}.weight")
        setattr(
            ple_parent,
            ple_attr,
            DeterministicPLEOffloader.from_cpu_tensor(ple_mmap, embed_scale=ple_scale_float, padding_idx=ple_pad_idx),
        )
        del orig_ple

    skipped_prefixes = tuple(stripped_towers)
    valid_model_keys = {k for k, _ in model.named_parameters()} | {k for k, _ in model.named_buffers()}
    for key in reader.keys():
        if key.startswith(layers_prefix):
            continue
        if ple_prefix is not None and key.startswith(ple_prefix):
            continue
        if any(key == sp or key.startswith(sp + ".") for sp in skipped_prefixes):
            continue
        if key not in valid_model_keys:
            continue
        _set_tensor_by_qualname(model, key, reader.get_tensor(key, device=device, dtype=dtype))

    pcie_counter = [0]
    eff_gpu_layers = min(max(0, int(num_gpu_layers)), num_total_layers)
    for idx in range(num_total_layers):
        layer_mod = decoder_layers[idx]
        l_prefix = f"{layers_prefix}{idx}."
        valid_layer_keys = {k for k, _ in layer_mod.named_parameters()} | {k for k, _ in layer_mod.named_buffers()}
        if idx < eff_gpu_layers:
            _populate_layer_bf16_from_reader(layer_mod, reader, l_prefix, valid_layer_keys, device, dtype)
            _move_cpu_buffers_to_device(layer_mod, dev)
            layer_mod.eval()
        else:
            _move_cpu_buffers_to_device(layer_mod, dev)
            layer_mod.eval()
            decoder_layers[idx] = _OffloadedMMapDecoderLayer(
                layer_mod, reader, l_prefix, valid_layer_keys, dev, dtype, pcie_counter
            )

    _move_cpu_buffers_to_device(model, dev, exclude_prefixes=(layers_qualname,))
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight") and model.lm_head.weight.device.type == "meta":
        model.tie_weights()
    model.eval()

    resident_alloc_mb = (
        torch.cuda.memory_allocated(dev) / (1024 ** 2)
        if (torch.cuda.is_available() and dev.type == "cuda")
        else 0.0
    )

    inp = prompt_ids[:1, :64].to(dev)
    out = model(input_ids=inp, use_cache=True)
    past_kv = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    for _ in range(warmup_steps):
        out = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    pcie_counter[0] = 0
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t_dec0 = time.perf_counter()
    for _ in range(num_decode_tokens):
        out = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elapsed = max(1e-9, time.perf_counter() - t_dec0)

    peak_alloc_mb = (
        torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        if (torch.cuda.is_available() and dev.type == "cuda")
        else 0.0
    )
    ms_per_tok = (elapsed * 1000.0) / max(1, num_decode_tokens)
    tok_per_s = num_decode_tokens / elapsed
    pcie_mb_per_tok = (pcie_counter[0] / (1024 ** 2)) / max(1, num_decode_tokens)

    del model, past_kv, out
    gc.collect()
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "num_gpu_layers": eff_gpu_layers,
        "num_offloaded_layers": num_total_layers - eff_gpu_layers,
        "resident_cuda_mb": resident_alloc_mb,
        "peak_cuda_mb": peak_alloc_mb,
        "decode_ms_per_tok": ms_per_tok,
        "decode_tok_per_s": tok_per_s,
        "pcie_mb_per_token": pcie_mb_per_tok,
    }
