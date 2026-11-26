r"""TensorMMU (`tmmu`) Unified CLI: Hardware Check, Streaming Model Compilation & Agent Server.

Examples:
  # 1. Pre-flight check whether a Hugging Face model or local cartridge fits your GPU:
  tmmu check Qwen/Qwen3-Next-80B-A3B-Instruct
  tmmu check checkpoints/gemma-4-26B-A4B-it-nvm-moe.nvm

  # 2. Download & stream-compile a Hugging Face checkpoint (< 5 GB peak compile VRAM):
  tmmu compile google/gemma-4-26B-A4B-it -o checkpoints/gemma-4-26B-A4B-it.tmmu

  # 3. Start OpenAI + Anthropic (`/v1/messages`) + K8s (`/livez`, `/readyz`) server:
  #    If `--model` is a Hugging Face repo ID not yet compiled, the HTTP server starts immediately,
  #    responds `503 Service Unavailable` on `/readyz` & `/v1/messages` while downloading/compiling,
  #    and flips to `200 OK` automatically once loaded!
  tmmu serve --model checkpoints/gemma-4-26B-A4B-it-nvm-moe.nvm --port 8000
  tmmu serve --model Qwen/Qwen3-Next-80B-A3B-Instruct --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import List, Optional

from aiohttp import web

from runtime.openai_server import (
    DEFAULT_AGENT_MAX_TOKENS,
    DEFAULT_THINKING_BUDGET,
    TensorMMUEngine,
    create_tensormmu_web_app,
)
from tensormmu import __version__
from tensormmu.compiler import (
    download_and_compile_model,
    find_existing_cartridge,
    is_compiled_cartridge,
)
from tensormmu.hardware_safety import assess_cartridge_fit, assess_compile_fit


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int_opt(name: str) -> Optional[int]:
    val = os.environ.get(name)
    if val is None or not val.strip():
        return None
    return int(val.strip())


def pick_default_cartridge(checkpoints_dir: str) -> Optional[str]:
    preferred = [
        "gemma-4-26B-A4B-it.tmmu",
        "gemma-4-26B-A4B-it-nvm-moe.nvm",
        "Qwen3-Next-80B-A3B-Instruct.tmmu",
        "Qwen3-Next-80B-A3B-Instruct-nvm-moe.nvm",
        "gemma-4-E4B-it-attn-int8.nvm",
        "gemma-4-31B-it-nvm-hybrid.nvm",
    ]
    for name in preferred:
        cand = os.path.join(checkpoints_dir, name)
        if is_compiled_cartridge(cand):
            return os.path.abspath(cand)
    if os.path.isdir(checkpoints_dir):
        for entry in sorted(os.listdir(checkpoints_dir)):
            cand = os.path.join(checkpoints_dir, entry)
            if (entry.endswith(".tmmu") or entry.endswith(".nvm")) and is_compiled_cartridge(cand):
                return os.path.abspath(cand)
    return None


def _cmd_check(args: argparse.Namespace) -> int:
    target = args.target
    existing = find_existing_cartridge(target, checkpoints_dir=args.checkpoints_dir)
    if existing is not None:
        fit = assess_cartridge_fit(
            cartridge_dir=existing,
            device=args.device,
            requested_vm_mode=args.vm_mode,
            requested_tlb_slots=args.tlb_slots,
        )
    else:
        fit = assess_compile_fit(
            model_id_or_path=target,
            out_dir=args.checkpoints_dir,
            device=args.device,
        )

    if args.json:
        print(json.dumps(fit.to_dict(), indent=2), flush=True)
    else:
        print(fit.format_banner(), flush=True)
    return 0 if fit.ok_to_proceed else 2


def _cmd_compile(args: argparse.Namespace) -> int:
    try:
        cart_dir = download_and_compile_model(
            model_id_or_path=args.model,
            out_dir=args.output,
            checkpoints_dir=args.checkpoints_dir,
            device=args.device,
            tlb_slots=args.tlb_slots,
            attn_quant=args.attn_quant,
            shared_mlp_quant=args.shared_mlp_quant,
            lm_head_quant=args.lm_head_quant,
            expert_group_size=args.expert_group_size,
            calibrate_drafter=args.calibrate_drafter,
            calibrate_lookahead=args.calibrate_lookahead,
            force=args.force,
            recompile=args.recompile,
        )
        print(f"\n[TensorMMU] Ready to serve: tmmu serve --model \"{cart_dir}\"", flush=True)
        return 0
    except Exception as exc:
        print(f"[TensorMMU ERROR] {exc}", file=sys.stderr, flush=True)
        return 1


def _cmd_serve(args: argparse.Namespace) -> int:
    os.environ["TMMU_L2_WINA_RATIO"] = str(max(0.20, min(1.0, float(args.l2_wina_ratio))))
    os.environ["TMMU_L2_WINA_BITS"] = str(3 if int(args.l2_wina_bits) == 3 else 4)
    os.environ["TMMU_COLD_DROP_WEIGHT"] = str(max(0.0, min(0.20, float(args.cold_drop_weight))))
    model_spec = args.model or args.nvm_dir or os.environ.get("TMMU_MODEL", "").strip()
    if not model_spec:
        picked = pick_default_cartridge(args.checkpoints_dir)
        if picked is None:
            print(
                "[TensorMMU ERROR] No `--model` / `TMMU_MODEL` specified and no compiled cartridges found in "
                f"`{os.path.abspath(args.checkpoints_dir)}`.\n"
                "Example: tmmu serve --model google/gemma-4-26B-A4B-it",
                file=sys.stderr,
                flush=True,
            )
            return 1
        model_spec = picked

    existing_cart = find_existing_cartridge(model_spec, checkpoints_dir=args.checkpoints_dir)
    if existing_cart is None and args.no_compile_if_missing:
        print(
            f"[TensorMMU ERROR] Cartridge `{model_spec}` not found and `--no-compile-if-missing` is set.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    hint_name = os.path.basename(existing_cart) if existing_cart else model_spec.rstrip("/\\").split("/")[-1]
    engine = TensorMMUEngine(
        nvm_dir=existing_cart,
        checkpoints_dir=args.checkpoints_dir,
        device=args.device,
        vm_mode=args.vm_mode,
        tlb_slots=args.tlb_slots,
        tlb_strategy=args.tlb_strategy,
        page_storage=args.page_storage,
        draft_mode=args.draft_mode,
        confidence_gated_moe=not args.no_confidence_gated_moe,
        auto_swap_model=args.auto_swap_model,
        default_max_tokens=args.max_tokens,
        default_thinking_budget=args.thinking_budget,
        force_unsafe_load=args.force,
        defer_load=True,
        target_model_hint=hint_name,
    )

    app = create_tensormmu_web_app(
        engine=engine,
        api_key=args.api_key,
        fail_livez_on_error=args.fail_livez_on_error,
    )

    def _bootstrap_worker() -> None:
        try:
            cart_to_load = existing_cart
            if cart_to_load is None:
                cart_to_load = download_and_compile_model(
                    model_id_or_path=model_spec,
                    checkpoints_dir=args.checkpoints_dir,
                    device=args.device,
                    tlb_slots=args.tlb_slots,
                    calibrate_drafter=args.calibrate_drafter,
                    calibrate_lookahead=args.calibrate_lookahead,
                    force=args.force,
                    on_progress=lambda st, msg, pct: engine.set_bootstrap_state(st, detail=msg, progress_pct=pct),
                )
            engine.load_cartridge(
                nvm_dir=cart_to_load,
                vm_mode=args.vm_mode,
                tlb_slots=args.tlb_slots,
                tlb_strategy=args.tlb_strategy,
                page_storage=args.page_storage,
                draft_mode=args.draft_mode,
            )
            st = engine.status_dict()
            print(
                f"\n[TensorMMU READY] Serving `{st['active_model']}` on http://{args.host}:{args.port}\n"
                f"  OpenAI Endpoint    : http://{args.host}:{args.port}/v1/chat/completions\n"
                f"  Anthropic Endpoint : http://{args.host}:{args.port}/v1/messages\n"
                f"  K8s Probes         : /livez (200) | /readyz (200) | Playground: / | Architecture: /about\n"
                f"  Agent Defaults     : max_tokens={engine.default_max_tokens} | thinking_budget={engine.default_thinking_budget}\n"
                f"  CUDA Graph Mode    : {st.get('cuda_graph_mode')} | Allocated VRAM: {st.get('cuda_allocated_mb')} MiB\n",
                flush=True,
            )
        except Exception as exc:
            err_str = str(exc)
            engine.set_bootstrap_state("failed", detail=err_str, error=err_str)
            print(f"\n[TensorMMU BOOTSTRAP FAILED] {err_str}", file=sys.stderr, flush=True)
            if args.fail_fast:
                os._exit(1)

    async def _on_startup(_app: web.Application) -> None:
        loop = asyncio.get_running_loop()
        _app["bootstrap_task"] = loop.run_in_executor(None, _bootstrap_worker)

    app.on_startup.append(_on_startup)

    print("=" * 78, flush=True)
    print(f"  TensorMMU v{__version__} — HTTP/SSE Server Starting on http://{args.host}:{args.port}", flush=True)
    print(f"  Target Model : {existing_cart or model_spec}", flush=True)
    print("  Readiness    : /livez=200 immediately; /readyz & /v1/messages return 503 until model is loaded", flush=True)
    print("=" * 78, flush=True)

    web.run_app(app, host=args.host, port=args.port, print=None)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmmu",
        description="TensorMMU: Hardware-Inspired GPU Virtual Memory & TLB Inference Server for Coding Agents.",
    )
    parser.add_argument("--version", action="version", version=f"tensormmu {__version__}")
    sub = parser.add_subparsers(dest="subcommand")

    # 1. `tmmu check`
    p_chk = sub.add_parser("check", help="Pre-flight check whether a HF model or compiled cartridge fits this GPU.")
    p_chk.add_argument("target", type=str, help="Hugging Face model ID (e.g. Qwen/Qwen3-Next-80B-A3B-Instruct) or cartridge path.")
    p_chk.add_argument("--checkpoints-dir", type=str, default=os.environ.get("TMMU_CHECKPOINTS_DIR", "checkpoints"))
    p_chk.add_argument("--device", type=str, default=os.environ.get("TMMU_DEVICE", "cuda:0"))
    p_chk.add_argument("--vm-mode", type=str, default="vm_paged", choices=["vm_paged", "all_resident"])
    p_chk.add_argument("--tlb-slots", type=int, default=None)
    p_chk.add_argument("--json", action="store_true", help="Emit machine-readable JSON fit report.")

    # 2. `tmmu compile`
    p_cmp = sub.add_parser("compile", help="Download (if needed) & stream-compile a Hugging Face model into a `.tmmu` cartridge.")
    p_cmp.add_argument("model", type=str, help="Hugging Face repo ID or local snapshot directory.")
    p_cmp.add_argument("-o", "--output", type=str, default=None, help="Output `.tmmu` cartridge directory.")
    p_cmp.add_argument("--checkpoints-dir", type=str, default=os.environ.get("TMMU_CHECKPOINTS_DIR", "checkpoints"))
    p_cmp.add_argument("--device", type=str, default=os.environ.get("TMMU_DEVICE", "cuda:0"))
    p_cmp.add_argument("--tlb-slots", type=int, default=_env_int_opt("TMMU_TLB_SLOTS"))
    p_cmp.add_argument("--attn-quant", type=str, default="int8", choices=["none", "int8", "int4", "hybrid_int8_int4"])
    p_cmp.add_argument("--shared-mlp-quant", type=str, default="int8", choices=["none", "int8", "int4"])
    p_cmp.add_argument("--lm-head-quant", type=str, default="auto", choices=["auto", "none", "int8"])
    p_cmp.add_argument("--expert-group-size", type=int, default=64)
    p_cmp.add_argument("--calibrate-drafter", action="store_true", help="Calibrate EagleLiteDraftHead after compiling.")
    p_cmp.add_argument("--calibrate-lookahead", action="store_true", help="Calibrate Cross-Layer Lookahead Router Bank.")
    p_cmp.add_argument("--recompile", action="store_true", help="Recompile even if target cartridge already exists.")
    p_cmp.add_argument("--force", action="store_true", default=_env_bool("TMMU_FORCE"), help="Proceed even if GPU/disk safety check warns/blocks.")

    # 3. `tmmu serve`
    p_srv = sub.add_parser("serve", help="Start OpenAI + Anthropic + Kubernetes-ready TensorMMU inference server.")
    p_srv.add_argument(
        "--model",
        type=str,
        default=None,
        help="Compiled `.tmmu`/`.nvm` path OR Hugging Face repo ID (auto-downloads & compiles in background if missing!).",
    )
    p_srv.add_argument("--nvm-dir", type=str, default=None, help="Alias for `--model` (backwards compatibility).")
    p_srv.add_argument("--checkpoints-dir", type=str, default=os.environ.get("TMMU_CHECKPOINTS_DIR", "checkpoints"))
    p_srv.add_argument("--host", type=str, default=os.environ.get("TMMU_HOST", "127.0.0.1"))
    p_srv.add_argument("--port", type=int, default=int(os.environ.get("TMMU_PORT", "8000")))
    p_srv.add_argument("--device", type=str, default=os.environ.get("TMMU_DEVICE", "cuda:0"))
    p_srv.add_argument(
        "--vm-mode",
        type=str,
        default=os.environ.get("TMMU_VM_MODE", "vm_paged"),
        choices=["vm_paged", "all_resident"],
    )
    p_srv.add_argument("--tlb-slots", type=int, default=_env_int_opt("TMMU_TLB_SLOTS"))
    p_srv.add_argument(
        "--l2-wina-ratio",
        type=float,
        default=float(os.environ.get("TMMU_L2_WINA_RATIO", "0.50")),
        help="Active neuron keep ratio for L2 Zero-Copy `UINT4-WINA`/`UINT3-WINA` (default: 0.50; set 1.0 to disable).",
    )
    p_srv.add_argument(
        "--l2-wina-bits",
        type=int,
        choices=[3, 4],
        default=int(os.environ.get("TMMU_L2_WINA_BITS", "4")),
        help="Bit-width for L2 Zero-Copy WINA pages: 4 (`UINT4-WINA`) or 3 (`UINT3-WINA`, -30%% RAM & +43%% L2 capacity).",
    )
    p_srv.add_argument(
        "--cold-drop-weight",
        type=float,
        default=float(os.environ.get("TMMU_COLD_DROP_WEIGHT", "0.06")),
        help="Drop unmapped L3 NVMe cold experts below this normalized router weight (default: 0.06; 0.0 = exact, 0.09 = max speed).",
    )
    p_srv.add_argument("--tlb-strategy", type=str, default=os.environ.get("TMMU_TLB_STRATEGY") or None)
    p_srv.add_argument(
        "--page-storage",
        type=str,
        default=os.environ.get("TMMU_PAGE_STORAGE") or None,
        choices=["nvme_mmap", "pinned_ram", "pageable_ram"],
    )
    p_srv.add_argument(
        "--draft-mode",
        type=str,
        default=os.environ.get("TMMU_DRAFT_MODE", "two_tier"),
        choices=["two_tier", "mtp_only", "pld_only", "eagle_only", "none"],
    )
    p_srv.add_argument("--no-confidence-gated-moe", action="store_true")
    p_srv.add_argument("--auto-swap-model", action="store_true", default=_env_bool("TMMU_AUTO_SWAP_MODEL"))
    p_srv.add_argument("--max-tokens", type=int, default=DEFAULT_AGENT_MAX_TOKENS, help="Default max_tokens when omitted by client (default: 32768).")
    p_srv.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET, help="Default thinking token cap when reasoning is enabled (default: 6144).")
    p_srv.add_argument("--api-key", type=str, default=os.environ.get("TMMU_API_KEY") or None)
    p_srv.add_argument("--no-compile-if-missing", action="store_true", help="Do not download/compile if `--model` is not yet compiled.")
    p_srv.add_argument("--calibrate-drafter", action="store_true")
    p_srv.add_argument("--calibrate-lookahead", action="store_true")
    p_srv.add_argument("--fail-livez-on-error", action="store_true", default=_env_bool("TMMU_FAIL_LIVEZ_ON_ERROR"), help="Return HTTP 500 on `/livez` if bootstrap fails so K8s restarts the pod.")
    p_srv.add_argument("--fail-fast", action="store_true", default=_env_bool("TMMU_FAIL_FAST"), help="Exit process immediately if bootstrap/safety check fails.")
    p_srv.add_argument("--force", action="store_true", default=_env_bool("TMMU_FORCE"), help="Override GPU/disk pre-flight safety block.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.subcommand == "check":
        return _cmd_check(args)
    if args.subcommand == "compile":
        return _cmd_compile(args)
    if args.subcommand == "serve":
        return _cmd_serve(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
