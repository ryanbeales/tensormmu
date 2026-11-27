"""Hardware Pre-Flight Safety, GPU Fit Estimation & Auto-Tiering (`Linux` + `Windows`, Consumer RTX -> AWS H100/H200).

Inspects NVIDIA GPU compute capability (`sm_80+`), total/free VRAM, container-aware System RAM (cgroups v1/v2),
`/dev/shm` capacity in Docker/Kubernetes, and disk headroom *before* downloading, compiling, or loading models.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from runtime.moe_tlb_kernels import ContiguousExpertPageLayout, query_available_system_ram_bytes


@dataclass
class HostHardwareSnapshot:
    os_name: str
    cuda_available: bool
    device_index: int
    gpu_name: str
    compute_capability: Tuple[int, int]
    total_vram_gb: float
    free_vram_gb: float
    total_ram_gb: float
    avail_ram_gb: float
    shm_total_gb: Optional[float]
    free_disk_gb: float
    disk_checked_path: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["compute_capability"] = f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"
        return d


@dataclass
class FitAssessment:
    ok_to_proceed: bool
    status: str  # "ok" | "warn" | "insufficient_vram" | "insufficient_disk" | "unsupported_gpu"
    model_label: str
    architecture: str
    hardware: HostHardwareSnapshot
    core_resident_vram_gb: float
    min_viable_vram_gb: float
    recommended_vram_gb: float
    all_resident_vram_gb: float
    kv_per_1k_tokens_mb: float
    recommended_vm_mode: str
    recommended_tlb_slots: Optional[int]
    max_experts_per_layer: int
    estimated_download_gb: float = 0.0
    estimated_cartridge_gb: float = 0.0
    estimated_compile_peak_vram_gb: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hardware"] = self.hardware.to_dict()
        return d

    def format_banner(self) -> str:
        hw = self.hardware
        cc = f"sm_{hw.compute_capability[0]}{hw.compute_capability[1]}"
        lines = [
            "=" * 78,
            f"  TensorMMU Hardware Safety Check — {self.model_label} ({self.architecture})",
            "=" * 78,
            f"  GPU        : {hw.gpu_name} ({cc}) | Free VRAM: {hw.free_vram_gb:.2f} / {hw.total_vram_gb:.2f} GiB",
            f"  System RAM : {hw.avail_ram_gb:.2f} / {hw.total_ram_gb:.2f} GiB available | Free Disk: {hw.free_disk_gb:.1f} GiB",
            f"  VRAM Tiers : Core={self.core_resident_vram_gb:.2f} GiB | Min={self.min_viable_vram_gb:.2f} GiB | "
            f"Recommended={self.recommended_vram_gb:.2f} GiB | All-Resident={self.all_resident_vram_gb:.2f} GiB",
            f"  KV Growth  : {self.kv_per_1k_tokens_mb:.1f} MiB / 1K tokens | Auto Config: mode={self.recommended_vm_mode}, "
            f"tlb_slots={self.recommended_tlb_slots}/{self.max_experts_per_layer or 'dense'}",
        ]
        if self.estimated_compile_peak_vram_gb > 0:
            lines.append(
                f"  Compile Est: Peak GPU={self.estimated_compile_peak_vram_gb:.2f} GiB | "
                f"HF Download~{self.estimated_download_gb:.1f} GiB + Cartridge~{self.estimated_cartridge_gb:.1f} GiB"
            )
        for w in self.warnings:
            lines.append(f"  [WARNING]  {w}")
        for e in self.errors:
            lines.append(f"  [ERROR]    {e}")
        lines.append("=" * 78)
        return "\n".join(lines)


def _parse_device_index(device: str) -> int:
    if ":" in device:
        try:
            return int(device.split(":")[1])
        except ValueError:
            return 0
    return 0


def probe_host_hardware(device: str = "cuda:0", disk_path: str = ".") -> HostHardwareSnapshot:
    dev_idx = _parse_device_index(device)
    avail_ram_b, total_ram_b = query_available_system_ram_bytes()

    check_dir = os.path.abspath(disk_path)
    while not os.path.exists(check_dir):
        parent = os.path.dirname(check_dir)
        if parent == check_dir:
            check_dir = os.getcwd()
            break
        check_dir = parent
    try:
        free_disk_b = shutil.disk_usage(check_dir).free
    except Exception:
        free_disk_b = 0

    shm_gb: Optional[float] = None
    if os.name != "nt" and os.path.isdir("/dev/shm"):
        try:
            shm_gb = shutil.disk_usage("/dev/shm").total / float(1024 ** 3)
        except Exception:
            shm_gb = None

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return HostHardwareSnapshot(
            os_name=os.name,
            cuda_available=False,
            device_index=dev_idx,
            gpu_name="CPU (No CUDA GPU Detected)",
            compute_capability=(0, 0),
            total_vram_gb=0.0,
            free_vram_gb=0.0,
            total_ram_gb=total_ram_b / float(1024 ** 3),
            avail_ram_gb=avail_ram_b / float(1024 ** 3),
            shm_total_gb=shm_gb,
            free_disk_gb=free_disk_b / float(1024 ** 3),
            disk_checked_path=check_dir,
        )

    props = torch.cuda.get_device_properties(dev_idx)
    try:
        free_vram_b, total_vram_b = torch.cuda.mem_get_info(dev_idx)
    except Exception:
        total_vram_b = int(props.total_memory)
        free_vram_b = total_vram_b

    return HostHardwareSnapshot(
        os_name=os.name,
        cuda_available=True,
        device_index=dev_idx,
        gpu_name=str(props.name),
        compute_capability=(int(props.major), int(props.minor)),
        total_vram_gb=total_vram_b / float(1024 ** 3),
        free_vram_gb=free_vram_b / float(1024 ** 3),
        total_ram_gb=total_ram_b / float(1024 ** 3),
        avail_ram_gb=avail_ram_b / float(1024 ** 3),
        shm_total_gb=shm_gb,
        free_disk_gb=free_disk_b / float(1024 ** 3),
        disk_checked_path=check_dir,
    )


def _extract_text_cfg_dict(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw_cfg.get("text_config"), dict):
        merged = dict(raw_cfg)
        merged.update(raw_cfg["text_config"])
        return merged
    return raw_cfg


def _estimate_kv_and_gdn_from_config(cfg: Dict[str, Any]) -> Tuple[float, float]:
    """Returns `(kv_per_1k_tokens_mb, static_gdn_state_gb)` from HF `config.json`."""
    tc = _extract_text_cfg_dict(cfg)
    n_layers = int(tc.get("num_hidden_layers", 32))
    h_kv = int(tc.get("num_key_value_heads") or tc.get("num_attention_heads", 8))
    hidden = int(tc.get("hidden_size", 4096))
    n_q = int(tc.get("num_attention_heads", 16))
    head_dim = int(tc.get("head_dim") or (hidden // max(1, n_q)))

    # Qwen3-Next has 1 full-attention layer per `full_attention_interval` (e.g. 12/48) + 36 GDN layers
    full_interval = int(tc.get("full_attention_interval", 0) or 0)
    layer_types = tc.get("layer_types")
    num_kv_shared = int(tc.get("num_kv_shared_layers", 0) or 0)

    if full_interval > 1:
        n_attn_layers = n_layers // full_interval
        n_gdn_layers = n_layers - n_attn_layers
    elif isinstance(layer_types, list) and layer_types:
        n_attn_layers = max(1, len(layer_types) - num_kv_shared)
        n_gdn_layers = 0
    else:
        n_attn_layers = max(1, n_layers - num_kv_shared)
        n_gdn_layers = 0

    # Gemma 4 uses `global_head_dim` (e.g. 512) + `num_global_key_value_heads` on global attention layers
    glob_head_dim = int(tc.get("global_head_dim") or head_dim)
    glob_h_kv = int(tc.get("num_global_key_value_heads") or h_kv)
    if isinstance(layer_types, list) and layer_types:
        kv_bytes_per_tok = 0
        non_shared = max(1, len(layer_types) - num_kv_shared)
        for lt in layer_types[:non_shared]:
            if "full" in str(lt) or "global" in str(lt):
                kv_bytes_per_tok += 2 * glob_h_kv * glob_head_dim * 2
            else:
                kv_bytes_per_tok += 2 * h_kv * head_dim * 2
    else:
        kv_bytes_per_tok = n_attn_layers * 2 * h_kv * head_dim * 2

    kv_per_1k_mb = (kv_bytes_per_tok * 1024) / float(1024 ** 2)

    gdn_state_bytes = 0
    if n_gdn_layers > 0:
        hv = int(tc.get("linear_num_value_heads", 32))
        kd = int(tc.get("linear_key_head_dim", 128))
        vd = int(tc.get("linear_value_head_dim", 128))
        # `rec_slab` (1x) + `step_rec_slab` (8x speculative rollback) in float32
        gdn_state_bytes = n_gdn_layers * (1 + 8) * hv * kd * vd * 4

    return kv_per_1k_mb, gdn_state_bytes / float(1024 ** 3)


def _check_gpu_common(hw: HostHardwareSnapshot, warnings: List[str], errors: List[str]) -> None:
    if not hw.cuda_available:
        errors.append(
            "No NVIDIA CUDA GPU detected (`torch.cuda.is_available() == False`). "
            "TensorMMU requires an Ampere (`sm_80`) or newer NVIDIA GPU on Linux or Windows."
        )
        return
    if hw.compute_capability < (8, 0):
        cc = f"sm_{hw.compute_capability[0]}{hw.compute_capability[1]}"
        errors.append(
            f"Unsupported GPU architecture `{hw.gpu_name}` (`{cc}`). TensorMMU requires `sm_80+` "
            "(NVIDIA Ampere RTX 30-series / A10G / A100, Ada RTX 40-series / L40S, Hopper H100/H200, or Blackwell RTX 50-series) "
            "for native `bfloat16` + NVRTC execution. Note on AWS: use `g5` / `g6e` / `p4d` / `p5`, NOT `g4dn` (Turing T4 `sm_75`)."
        )
    if hw.shm_total_gb is not None and hw.shm_total_gb < 0.5:
        warnings.append(
            f"Container `/dev/shm` is very small ({hw.shm_total_gb * 1024:.0f} MiB). "
            "Run Docker with `--shm-size=4g` (or `--ipc=host`) on Linux/AWS to avoid shared-memory bottlenecks."
        )


def assess_cartridge_fit(
    cartridge_dir: str,
    device: str = "cuda:0",
    requested_vm_mode: str = "vm_paged",
    requested_tlb_slots: Optional[int] = None,
) -> FitAssessment:
    """Evaluates whether an existing `.nvm`/`.tmmu` cartridge fits on `device` and auto-tunes `tlb_slots`."""
    cart_dir = os.path.abspath(cartridge_dir)
    manifest_path = os.path.join(cart_dir, "manifest.json")
    cfg_path = os.path.join(cart_dir, "config.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    raw_cfg: Dict[str, Any] = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

    hw = probe_host_hardware(device=device, disk_path=cart_dir)
    warnings: List[str] = []
    errors: List[str] = []
    _check_gpu_common(hw, warnings, errors)

    tc = _extract_text_cfg_dict(raw_cfg)
    arch_list = raw_cfg.get("architectures") or [tc.get("model_type", "unknown")]
    arch_name = str(arch_list[0]) if arch_list else "unknown"
    model_label = os.path.basename(cart_dir)

    kv_per_1k_mb, gdn_state_gb = _estimate_kv_and_gdn_from_config(raw_cfg)
    base_4k_kv_gb = (kv_per_1k_mb * 4.0) / 1024.0

    ct = manifest.get("compile_telemetry", {})
    core_disk_gb = float(ct.get("core_resident_disk_mb", 0.0)) / 1024.0
    if core_disk_gb <= 0.05:
        base_f = os.path.join(cart_dir, "base_resident.safetensors")
        core_disk_gb = (os.path.getsize(base_f) / float(1024 ** 3)) if os.path.isfile(base_f) else 2.5

    # IncludeEagle/Lookahead heads + NVRTC/CUDA context/workspace (~1.35 GiB) + 4K initial KV + GDN slabs
    runtime_overhead_gb = 1.35 + base_4k_kv_gb + gdn_state_gb
    core_vram_gb = core_disk_gb + runtime_overhead_gb

    is_moe = manifest.get("format_version") == "nvm_moe_v1"
    if is_moe:
        compiled_indices = manifest.get("compiled_layer_indices", [])
        layers_meta = manifest.get("layers", {})
        first_moe = layers_meta.get(str(compiled_indices[0]), {}) if compiled_indices else {}
        num_experts = int(first_moe.get("num_experts") or tc.get("num_experts", 128))
        top_k = int(tc.get("num_experts_per_tok") or tc.get("top_k_experts") or 8)

        bytes_per_slot_all_layers = 0
        for l_idx in compiled_indices:
            li = layers_meta.get(str(l_idx), {})
            p_bytes = int(li.get("page_bytes", 0))
            if p_bytes <= 0:
                lay = ContiguousExpertPageLayout.compute(
                    int(tc.get("hidden_size", 2048)),
                    int(tc.get("moe_intermediate_size", 512)),
                    int(manifest.get("expert_group_size", 64)),
                )
                p_bytes = lay.page_bytes
            bytes_per_slot_all_layers += p_bytes

        gb_per_slot = bytes_per_slot_all_layers / float(1024 ** 3)
        total_expert_gb = num_experts * gb_per_slot
        min_slots = min(num_experts, max(16, top_k * 2))
        min_viable_gb = core_vram_gb + min_slots * gb_per_slot
        all_res_gb = core_vram_gb + total_expert_gb

        # Reserve KV + graph pool headroom (hybrid/linear-attn models with <=40 MiB/1K KV need far less headroom!)
        if kv_per_1k_mb <= 40.0:
            kv_headroom_gb = max(1.8, (kv_per_1k_mb * 16.0) / 1024.0 + 1.2)
        else:
            kv_headroom_gb = max(3.0, min(12.0, hw.free_vram_gb * 0.14), (kv_per_1k_mb * 12.0) / 1024.0)
        kv_32k_headroom_gb = max(kv_headroom_gb, (kv_per_1k_mb * 32.0) / 1024.0 + 1.5)
        usable_for_slots_gb = max(0.0, hw.free_vram_gb - core_vram_gb - kv_headroom_gb)
        max_safe_slots = int(usable_for_slots_gb // max(1e-6, gb_per_slot)) if gb_per_slot > 0 else num_experts
        max_safe_slots = max(min_slots, min(num_experts, max_safe_slots))

        if requested_vm_mode == "all_resident":
            if hw.cuda_available and all_res_gb + 1.5 > hw.free_vram_gb:
                warnings.append(
                    f"`vm_mode=all_resident` requires ~{all_res_gb:.1f} GiB VRAM, which exceeds safe free VRAM "
                    f"({hw.free_vram_gb:.1f} GiB). Automatically falling back to `vm_paged` (`tlb_slots={max_safe_slots}`)."
                )
                rec_mode = "vm_paged"
                rec_slots = max_safe_slots
            else:
                rec_mode = "all_resident"
                rec_slots = num_experts
        elif requested_tlb_slots is not None:
            hard_cap_slots = int(max(0.0, hw.free_vram_gb - core_vram_gb - 1.0) // max(1e-6, gb_per_slot))
            if hw.cuda_available and requested_tlb_slots > hard_cap_slots and hard_cap_slots >= min_slots:
                warnings.append(
                    f"Requested `tlb_slots={requested_tlb_slots}` (~{core_vram_gb + requested_tlb_slots * gb_per_slot:.1f} GiB) "
                    f"exceeds safe free VRAM ({hw.free_vram_gb:.1f} GiB). Clamping to `tlb_slots={hard_cap_slots}`."
                )
                rec_slots = hard_cap_slots
            else:
                rec_slots = max(1, min(num_experts, int(requested_tlb_slots)))
            rec_mode = "all_resident" if rec_slots >= num_experts else "vm_paged"
        else:
            # Smart auto-tiering across consumer GPUs (12-24 GB) and large AWS GPUs (48-141 GB)!
            if hw.cuda_available and all_res_gb + kv_32k_headroom_gb <= hw.free_vram_gb:
                # Fits 100% in GPU HBM with full 32K KV headroom (e.g. AWS L40S 48 GB / A100 80 GB / H100 80 GB)!
                rec_mode = "all_resident"
                rec_slots = num_experts
            else:
                rec_mode = "vm_paged"
                # Check if remaining experts easily fit in L2 Zero-Copy DDR5 (e.g. Gemma-4-26B on 24 GB + 32 GB RAM)
                if total_expert_gb <= 14.0 and hw.avail_ram_gb >= (total_expert_gb * 0.65 + 6.0):
                    # Keep L1 moderate (~38-50% of experts) so plenty of VRAM stays free for 32K KV cache,
                    # while L2 Zero-Copy maps 100% of remaining experts for `full_graph`!
                    balanced = max(min_slots, min(max_safe_slots, max(int(manifest.get("tlb_slots", 48)), num_experts * 3 // 8)))
                    rec_slots = balanced
                else:
                    rec_slots = max_safe_slots

        rec_vram_gb = core_vram_gb + rec_slots * gb_per_slot
        if hw.cuda_available and hw.free_vram_gb < min_viable_gb:
            errors.append(
                f"Model `{model_label}` will NOT fit on `{hw.gpu_name}`: minimum viable VRAM is "
                f"{min_viable_gb:.2f} GiB (core trunk {core_vram_gb:.2f} GiB + {min_slots} min TLB slots), "
                f"but only {hw.free_vram_gb:.2f} GiB is free out of {hw.total_vram_gb:.2f} GiB."
            )
        elif hw.cuda_available and total_expert_gb > (rec_slots * gb_per_slot + hw.avail_ram_gb):
            warnings.append(
                f"Expert pages ({total_expert_gb:.1f} GiB) exceed combined L1 VRAM + free System RAM "
                f"({hw.avail_ram_gb:.1f} GiB); cold misses will stream from NVMe SSD (ensure fast PCIe 4.0/5.0 NVMe)."
            )
    else:
        # Dense-decompiled `nvm_v1` cartridge
        layers_meta = manifest.get("layers", {})
        first_l = next((v for v in layers_meta.values() if v.get("compiled")), {})
        num_experts = int(first_l.get("num_pages", 64))
        p_bytes = int(first_l.get("bytes_per_page", 512 * 1024))
        n_comp = len(manifest.get("compiled_layer_indices", []))
        gb_per_slot = (n_comp * p_bytes) / float(1024 ** 3)
        min_slots = min(num_experts, int(manifest.get("decode_req_pages", 8)) * 2)
        min_viable_gb = core_vram_gb + min_slots * gb_per_slot
        all_res_gb = core_vram_gb + num_experts * gb_per_slot
        rec_slots = int(requested_tlb_slots or manifest.get("tlb_slots", 24))
        rec_mode = requested_vm_mode
        rec_vram_gb = core_vram_gb + rec_slots * gb_per_slot
        if hw.cuda_available and hw.free_vram_gb < min_viable_gb:
            errors.append(
                f"Cartridge `{model_label}` requires >= {min_viable_gb:.2f} GiB VRAM, "
                f"but `{hw.gpu_name}` only has {hw.free_vram_gb:.2f} GiB free."
            )

    if errors:
        status = "unsupported_gpu" if not hw.cuda_available or hw.compute_capability < (8, 0) else "insufficient_vram"
    elif warnings:
        status = "warn"
    else:
        status = "ok"

    return FitAssessment(
        ok_to_proceed=len(errors) == 0,
        status=status,
        model_label=model_label,
        architecture=arch_name,
        hardware=hw,
        core_resident_vram_gb=round(core_vram_gb, 2),
        min_viable_vram_gb=round(min_viable_gb, 2),
        recommended_vram_gb=round(rec_vram_gb, 2),
        all_resident_vram_gb=round(all_res_gb, 2),
        kv_per_1k_tokens_mb=round(kv_per_1k_mb, 2),
        recommended_vm_mode=rec_mode,
        recommended_tlb_slots=rec_slots,
        max_experts_per_layer=num_experts,
        warnings=warnings,
        errors=errors,
    )


def fetch_or_load_hf_config(model_id_or_path: str) -> Dict[str, Any]:
    """Loads `config.json` from a local directory or fetches ONLY `config.json` (~2 KB) from Hugging Face Hub."""
    if os.path.isdir(model_id_or_path):
        cfg_file = os.path.join(model_id_or_path, "config.json")
        if os.path.isfile(cfg_file):
            with open(cfg_file, "r", encoding="utf-8") as f:
                return json.load(f)
    from huggingface_hub import hf_hub_download

    local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    cfg_file = hf_hub_download(repo_id=model_id_or_path, filename="config.json", local_files_only=local_only)
    with open(cfg_file, "r", encoding="utf-8") as f:
        return json.load(f)


def assess_compile_fit(
    model_id_or_path: str,
    out_dir: str,
    device: str = "cuda:0",
) -> FitAssessment:
    """Pre-checks GPU compile VRAM, post-compile serving VRAM, and disk space *before* downloading/compiling."""
    raw_cfg = fetch_or_load_hf_config(model_id_or_path)
    hw = probe_host_hardware(device=device, disk_path=out_dir)
    warnings: List[str] = []
    errors: List[str] = []
    _check_gpu_common(hw, warnings, errors)

    tc = _extract_text_cfg_dict(raw_cfg)
    arch_list = raw_cfg.get("architectures") or [tc.get("model_type", "unknown")]
    arch_name = str(arch_list[0]) if arch_list else "unknown"

    n_layers = int(tc.get("num_hidden_layers", 32))
    d_model = int(tc.get("hidden_size", 4096))
    vocab = int(tc.get("vocab_size", 151936))
    num_experts = int(tc.get("num_experts") or tc.get("n_routed_experts") or 0)
    top_k = int(tc.get("num_experts_per_tok") or tc.get("top_k_experts") or 8)
    moe_inter = int(tc.get("moe_intermediate_size") or tc.get("intermediate_size", 2048))
    shared_inter = int(tc.get("shared_expert_intermediate_size") or tc.get("intermediate_size", 2048))
    tied_emb = bool(tc.get("tie_word_embeddings", False))

    kv_per_1k_mb, gdn_state_gb = _estimate_kv_and_gdn_from_config(raw_cfg)

    # Estimate parameter counts for BF16 download vs UINT4/INT8 compiled `.tmmu` cartridge
    emb_params = vocab * d_model * (1 if tied_emb else 2)
    attn_params_per_layer = 4 * d_model * d_model
    if num_experts > 1:
        routed_params_per_layer = num_experts * 3 * d_model * moe_inter
        shared_params_per_layer = 3 * d_model * shared_inter
        total_params = emb_params + n_layers * (attn_params_per_layer + shared_params_per_layer + routed_params_per_layer)
        # Single-layer streaming compile loads 1 BF16 layer onto GPU at a time (`O(1)` VRAM!)
        single_layer_bf16_gb = ((attn_params_per_layer + shared_params_per_layer + routed_params_per_layer) * 2) / float(
            1024 ** 3
        )
        compile_peak_vram_gb = max(
            (emb_params * 2) / float(1024 ** 3) + 1.2,
            single_layer_bf16_gb * 1.25 + 1.2,
        )
        lay = ContiguousExpertPageLayout.compute(d_model, moe_inter, 64)
        gb_per_slot = (n_layers * lay.page_bytes) / float(1024 ** 3)
        # Trunk: bf16 embed + uint4 lm_head + uint4/int8 attn & shared_mlp
        core_disk_gb = (
            (vocab * d_model * 2)
            + (0 if tied_emb else int(vocab * d_model * 0.56))
            + n_layers * int((attn_params_per_layer + shared_params_per_layer) * 0.65)
        ) / float(1024 ** 3)
        core_vram_gb = core_disk_gb + 1.35 + gdn_state_gb + (kv_per_1k_mb * 4.0) / 1024.0
        min_slots = min(num_experts, max(16, top_k * 2))
        min_viable_gb = core_vram_gb + min_slots * gb_per_slot
        all_res_gb = core_vram_gb + num_experts * gb_per_slot
        est_cart_gb = core_disk_gb + all_res_gb - core_vram_gb
    else:
        ffn_inter = int(tc.get("intermediate_size", d_model * 4))
        ffn_params_per_layer = 3 * d_model * ffn_inter
        total_params = emb_params + n_layers * (attn_params_per_layer + ffn_params_per_layer)
        compile_peak_vram_gb = max(
            (emb_params * 2) / float(1024 ** 3) + 1.2,
            ((attn_params_per_layer + ffn_params_per_layer) * 2) / float(1024 ** 3) * 1.35 + 1.2,
        )
        core_vram_gb = ((emb_params * 2 + n_layers * attn_params_per_layer) / float(1024 ** 3)) + 1.5
        min_viable_gb = core_vram_gb + 1.5
        all_res_gb = (total_params * 0.6) / float(1024 ** 3)
        est_cart_gb = all_res_gb

    est_dl_gb = (total_params * 2) / float(1024 ** 3)
    is_local_src = os.path.isdir(model_id_or_path)
    needed_disk_gb = est_cart_gb + (0.0 if is_local_src else est_dl_gb) + 5.0

    if hw.free_disk_gb > 0 and hw.free_disk_gb < needed_disk_gb:
        errors.append(
            f"Insufficient free disk space at `{hw.disk_checked_path}`: need ~{needed_disk_gb:.1f} GiB "
            f"({'local source + ' if is_local_src else f'~{est_dl_gb:.1f} GiB HF download + '}~{est_cart_gb:.1f} GiB cartridge), "
            f"but only {hw.free_disk_gb:.1f} GiB is free."
        )

    if hw.cuda_available and compile_peak_vram_gb > hw.free_vram_gb:
        warnings.append(
            f"Estimated single-layer BF16 compile peak ({compile_peak_vram_gb:.1f} GiB) is close to or above "
            f"free GPU VRAM ({hw.free_vram_gb:.1f} GiB); compile may spill or require closing other GPU processes."
        )
    if hw.cuda_available and min_viable_gb > hw.total_vram_gb:
        warnings.append(
            f"Heads-up: after compilation, serving `{model_id_or_path}` requires at least ~{min_viable_gb:.1f} GiB VRAM, "
            f"which exceeds `{hw.gpu_name}` ({hw.total_vram_gb:.1f} GiB). You can compile here, or serve on a larger GPU."
        )

    rec_mode = "all_resident" if (hw.cuda_available and all_res_gb + 4.0 <= hw.free_vram_gb) else "vm_paged"
    rec_slots = num_experts if rec_mode == "all_resident" else (min(num_experts, 192) if num_experts > 0 else None)

    if errors:
        status = "insufficient_disk" if any("disk" in e.lower() for e in errors) else "unsupported_gpu"
    elif warnings:
        status = "warn"
    else:
        status = "ok"

    return FitAssessment(
        ok_to_proceed=len(errors) == 0,
        status=status,
        model_label=str(model_id_or_path),
        architecture=arch_name,
        hardware=hw,
        core_resident_vram_gb=round(core_vram_gb, 2),
        min_viable_vram_gb=round(min_viable_gb, 2),
        recommended_vram_gb=round(min(hw.total_vram_gb, max(min_viable_gb, all_res_gb)), 2),
        all_resident_vram_gb=round(all_res_gb, 2),
        kv_per_1k_tokens_mb=round(kv_per_1k_mb, 2),
        recommended_vm_mode=rec_mode,
        recommended_tlb_slots=rec_slots,
        max_experts_per_layer=num_experts,
        estimated_download_gb=0.0 if is_local_src else round(est_dl_gb, 1),
        estimated_cartridge_gb=round(est_cart_gb, 1),
        estimated_compile_peak_vram_gb=round(compile_peak_vram_gb, 2),
        warnings=warnings,
        errors=errors,
    )
