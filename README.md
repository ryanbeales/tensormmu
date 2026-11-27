# TensorMMU (`tmmu`)

**Hardware-Inspired GPU Virtual Memory, TLB Paging & Zero-Copy Inference Server for Sparse-MoE LLMs and Coding Agents.**

> **Target Platforms:** Linux & Windows with NVIDIA GPUs (`sm_80+`: RTX 30/40/50-series, A10G, L4, L40S, A100, H100, H200, Blackwell). *(macOS/Metal is not targeted at this time.)*

Modern CPUs do not require an entire binary to fit in L1/L2 cache before executing an instruction — they rely on a **Memory Management Unit (MMU)**, **Translation Lookaside Buffer (TLB)**, and tiered caches (`L1 -> L2 -> RAM -> NVMe`).

**TensorMMU** applies that exact computer-architecture design to Sparse Mixture-of-Experts (`MoE`) LLMs such as **`Qwen/Qwen3-Next-80B-A3B-Instruct`** (`512` experts/layer, `1.95%` active per token) and **`google/gemma-4-26B-A4B-it`** (`128` experts/layer, `6.25%` active per token):

- **L1 GPU VRAM TLB Cache (`1,008 GB/s` on RTX 4090 / `3.35 TB/s` on H100):** Keeps the always-active trunk + hottest `K` 4 KB-aligned `UINT4` expert pages resident on GPU, with hybrid-KV VRAM reclamation (`186–191` L1 slots on 24 GB) and persistent warm TLB profiles (`warm_tlb_profile.json`).
- **L2 Zero-Copy Pinned DDR5 (`cuMemHostAlloc` + `UINT4-WINA` / `UINT3-WINA`):** Allocates exact-byte device-mapped pinned RAM (zero PyTorch power-of-2 slab waste) and runs Gate-First conditional neuron-major PCIe fetch (`UINT3-WINA` shrinks L2 pages `-30%` to `1.316 MiB`, fitting `206` experts/layer in `12.7 GiB` DDR5).
- **L3 NVMe Pinned DMA Ring + Hardware-Tier Cold Expert Dropping:** Streams unmapped cold misses behind Shared-MLP compute with Cross-Layer `INT8` Lookahead Routers, while CUDA-Graphed `prune_cold_l3_routes` skips tiny-weight unmapped L3 outliers (`slot == -1`) on GPU with zero syncs.
- **CUDA-Graphed Native `mtp.*` + `PLD` Two-Tier Speculative Decoding:** Loads official Hugging Face `mtp.*` heads (e.g., `Qwen3-Next-80B`'s 1,553 MTP tensors) in `+190 MiB` VRAM, achieving `~85–92%` draft acceptance (`2.3–2.6 tok/step`) with dynamic warm `K=2/K=3` gating.
- **Agent-Ready OpenAI + Anthropic Endpoints & Native Tool Calling:** Serves `/v1/chat/completions` **and** `/v1/messages` (Claude Code, Anthropic SDK, Oh My Pi, Cline, Cursor) with structured `tool_calls` / `tool_use`, `max_tokens=32768` + `thinking_token_budget=6144` defaults, and `O(1)`-overhead growable KV caches.

---

## Measured Single-GPU Performance (1× RTX 4090 24 GB + 32 GB DDR5)

| Model | Total / Active | Cartridge Size | Allocated GPU VRAM | TTFT | Decode Throughput | GSM8K CoT |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **`Qwen3-Next-80B-A3B-Instruct`** *(Turbo Warm)* | 80B / 3.0B (`512E` top-10) | 44.0 GB | **18.80 GiB** (`186` L1 + `206` L2 `UINT3-WINA` + `120` L3) | `2.6 s` | **39.7 tok/s** (**`62.3` peak**) | **100%** (5/5) |
| **`Gemma-4-26B-A4B-it`** | 26B / 3.8B (`128E` top-8) | 15.3 GB | **8.15 GiB** (`48` L1 + `80` L2) | `523 ms` | **49.1 – 55.2 tok/s** | **100%** (5/5) |
| *Naive BF16 Layer Offloading* | 80B / 3.0B | 160 GB | 22.40 GiB | `14.2 s` | *0.38 tok/s* | 100% |

### How We Took `Qwen3-Next-80B` from `16.8 -> 62.3 tok/s` (`3.22×` Peak Speedup)

Each optimization below was benchmarked back-to-back on the exact same 5-problem GSM8K suite on a single RTX 4090 (24 GB) + 32 GB DDR5 PC:

| Milestone | L1 / L2 Slots | Mean `tok/s` | Peak `tok/s` | L3 NVMe Misses | `TLB` Hit % | GSM8K |
| :--- | :---: | ---: | ---: | ---: | ---: | :---: |
| **1. Baseline `UINT4`** (`L2=94`, no WINA / MTP / Lookahead) | `170 / 94` | `16.79` (`1.00×`) | `19.37` (`1.00×`) | — | `92.8%` | `5/5` |
| **2. + Exact-Byte `cuMemHostAlloc` L2 + Lookahead + `UINT4-WINA` + Native `mtp.*` (`K=1`)** | `170 / 137` | `21.55` (`1.28×`) | `29.83` (`1.54×`) | `14,377` | `96.6%` | `5/5` |
| **3. + CUDA-Graphed Cold L3 Drop (`9%`) + Dynamic Warm `K=2` MTP** | `170 / 137` | `28.18` (`1.68×`) | `38.46` (`1.99×`) | `5,920` (`-59%`) | `98.3%` | `5/5` |
| **4. + `UINT3-WINA 65%` Cold L2 (`1.316 MiB/exp`, `-30%` RAM)** | `170 / 196` | `29.46` (`1.75×`) | `40.32` (`2.08×`) | `3,497` (`-76%`) | `99.0%` | `5/5` |
| **5. + Hybrid-KV VRAM Reclaim + All-Hit Fast Piecewise + `two_tier` (Cold)** | `191 / 206` | `31.20` (`1.86×`) | `48.25` (`2.49×`) | `3,106` | `99.4%` | `4/5`* |
| **6. + Persistent Warm TLB Profile (`warm_tlb_profile.json` Restart)** | **`186 / 206`** | **`39.66` (`2.36×`)** | **`62.34` (`3.22×`)** | **`1,125` (`-92%`)** | **`99.8%`** | **`5/5`** |

- **Exact-Byte `cuMemHostAlloc` Flexible L2:** Eliminates PyTorch `CachingHostAllocator`'s power-of-2 slab rounding (which silently wasted ~2× pinned RAM per layer) and scales L2 dynamically to free host RAM instead of a fixed cap.
- **`UINT4-WINA` & `UINT3-WINA` Gate-First Conditional Fetch:** Computes `gate_proj` first on L2 Zero-Copy hits, scores neuron salience (`|SiLU(gate)| · ||W_down[:,j]||`), and fetches only the top-`k` `(up_j, down_j)` neuron slabs over PCIe. In 3-bit mode (`--l2-wina-bits 3`), exact `32-weights-into-3-uint32` packing shrinks L2 pages `-30%` (`1.875 -> 1.316 MiB`), expanding L2 capacity `+43%` (`137 -> 196–206` experts/layer).
- **CUDA-Graphed Hardware-Tier Cold L3 Expert Dropping (`--cold-drop-weight`):** Checks `gpu_expert_to_slot` live inside the CUDA Graph with zero extra syncs and drops an expert **only** if it is both physically unmapped on slow L3 NVMe (`slot == -1`) and below the normalized router-weight threshold (never touching L1/L2 hits or top-1), renormalizing remaining weights.
- **CUDA-Graphed Native `mtp.*` + `PLD` (`two_tier`):** Uses the model's official multi-token prediction head (`+190 MiB` VRAM) with a rolling L3-miss EMA gate that automatically unlocks `K=2` when warm (`K=3` when 100% L1+L2 mapped), averaging `2.32 tok/step`.
- **Hybrid-KV VRAM Reclaim + All-Hit Fast Piecewise + Persistent Warm TLB:** Reclaims unused VRAM on low-KV hybrid models (`GatedDeltaNet`, `24 MiB / 1K`), skips Python LRU/set churn on all-hit layers, and persists warmed L1+L2 expert ranks across restarts (`warm_tlb_profile.json`).

```powershell
# Default conservative serve (UINT4-WINA 50%, cold-drop 6%):
tmmu serve --model checkpoints/Qwen3-Next-80B-A3B-Instruct.tmmu --port 8000

# Turbo 80B mode (UINT3-WINA 65% + ColdDrop 9%, up to 62.3 tok/s on RTX 4090 + 32 GB RAM):
tmmu serve --model checkpoints/Qwen3-Next-80B-A3B-Instruct.tmmu --l2-wina-bits 3 --l2-wina-ratio 0.65 --cold-drop-weight 0.09
```

Open `docs/index.html` (or visit `http://localhost:8000/about` when running the server) for the full architecture walkthrough and Dense vs. Sparse-MoE findings.

---

## Quickstart (Python CLI)

### 1. Install

```powershell
# Create virtual environment and install PyTorch CUDA + TensorMMU
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux:
# source .venv/bin/activate

pip install -r requirements.txt
pip install --no-build-isolation --no-deps -e .
```

### 2. Pre-Flight GPU & Disk Safety Check (`tmmu check`)

Before downloading 50–160 GB of weights, verify that your NVIDIA GPU (`sm_80+`), free VRAM, System RAM / cgroup limit, and disk have headroom — and see the auto-selected `tlb_slots` tier:

```powershell
tmmu check google/gemma-4-26B-A4B-it
tmmu check Qwen/Qwen3-Next-80B-A3B-Instruct
```

If a model cannot safely fit on your GPU (or if you accidentally launch on an unsupported pre-Ampere instance such as AWS `g4dn` Turing T4 `sm_75`), TensorMMU prints an explicit breakdown of `Core VRAM`, `Minimum Viable VRAM`, and `Recommended VRAM` and blocks before OOMing (override with `--force` only if intentional).

### 3. Stream-Compile & Serve (`tmmu compile` / `tmmu serve`)

Streaming compilation processes **one transformer layer at a time** (`< 2.0 GiB` peak VRAM for 26B, `< 4.8 GiB` peak VRAM for 80B):

```powershell
# Compile explicitly:
tmmu compile google/gemma-4-26B-A4B-it -o checkpoints/gemma-4-26B-A4B-it.tmmu

# Serve OpenAI + Anthropic + Playground UI:
tmmu serve --model checkpoints/gemma-4-26B-A4B-it.tmmu --port 8000
```

Or pass a Hugging Face repo ID directly to `tmmu serve` — the HTTP server binds **immediately**, returns `503 Service Unavailable` (`Retry-After: 10`) with live progress on `/v1/messages` and `/v1/chat/completions` while downloading and compiling in the background, and flips `/readyz` from `503` to `200 OK` automatically once loaded:

```powershell
tmmu serve --model google/gemma-4-26B-A4B-it --host 0.0.0.0 --port 8000
```

---

## Docker & Kubernetes Serving

### Docker Compose (Download + Compile + Serve)

```bash
# Downloads & stream-compiles TMMU_MODEL into ./checkpoints on first run,
# answering 200 on /livez and 503 on /readyz & /v1/messages until ready:
TMMU_MODEL=google/gemma-4-26B-A4B-it docker compose up --build
```

### Direct `docker run` (GHCR Package or Local Build)

```bash
# Pull prebuilt image from GitHub Container Registry (or run `docker build -t tensormmu:latest .`):
docker pull ghcr.io/ryanbeales/tensormmu:latest

docker run --gpus all --shm-size=4g -p 8000:8000 \
  -v "$(pwd)/checkpoints:/models/cartridges" \
  -v "$(pwd)/hf_cache:/models/hf_cache" \
  -e TMMU_MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct" \
  ghcr.io/ryanbeales/tensormmu:latest
```

### Kubernetes Probes (`deploy/k8s-deployment.yaml`)

| Endpoint | Purpose | While Downloading / Compiling / Loading | When Model Loaded |
| :--- | :--- | :--- | :--- |
| `GET /livez` (`/healthz`) | K8s `livenessProbe` | `200 OK` (`{"alive": true, "ready": false, "state": "compiling", ...}`) | `200 OK` |
| `GET /readyz` | K8s `readinessProbe` | `503 Service Unavailable` + `Retry-After: 10` | `200 OK` |
| `POST /v1/messages` | Anthropic Messages + Tools | `503` (`overloaded_error`, includes `state` & `progress_pct`) | `200 OK` / SSE |
| `POST /v1/chat/completions` | OpenAI Chat + Tools | `503` (`model_not_ready`, includes `state` & `progress_pct`) | `200 OK` / SSE |

See [`deploy/k8s-deployment.yaml`](deploy/k8s-deployment.yaml) for a complete Kubernetes `Deployment` + `Service` manifest with `/dev/shm` memory mount and cgroup-aware L2 pinned RAM sizing.

---

## Using TensorMMU with Coding Agents

### Agent `max_tokens` & Thinking Budget Tuning

Following the findings in [Tuning a Local Coding Agent](https://doug.sh/posts/tuning-a-local-coding-agent-oh-my-pi/), leaving `max_tokens` at `4,096` or `8,192` causes coding agents to truncate file-write tool calls mid-JSON whenever reasoning consumes 5,000–6,000 tokens. TensorMMU solves both halves of this problem out of the box:

1. **Default `max_tokens = 32,768`** (configurable via `--max-tokens` or `TMMU_DEFAULT_MAX_TOKENS`), backed by **layer-by-layer growable static KV buffers** (`4K -> 8K -> 16K -> 32K`) so short turns allocate `< 900 MiB` KV VRAM while long code-generation turns smoothly expand up to `32K`.
2. **Bounded `thinking_token_budget = 6,144`** (configurable via `--thinking-budget`, OpenAI `reasoning_effort: "low"|"medium"|"high"`, or Anthropic `thinking: {"type": "enabled", "budget_tokens": 4096}`), automatically closing `<think>` / `<|channel>thought` when the reasoning budget is reached so `26,000+` tokens remain available for tool calls and code.

### Claude Code / Anthropic SDK (`/v1/messages`)

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_API_KEY="tmmu-local"

claude --model gemma-4-26B-A4B-it-nvm-moe
```

### OpenAI SDK Tool Calling (`/v1/chat/completions`)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="tmmu-local")

resp = client.chat.completions.create(
    model="gemma-4-26B-A4B-it-nvm-moe",
    messages=[{"role": "user", "content": "Run git status in the workspace using run_shell."}],
    tools=[{
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command in the workspace",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }],
)
print(resp.choices[0].message.tool_calls)
```

---

## Scaling Up on AWS GPU Instances

TensorMMU automatically scales `vm_mode`, L1 `tlb_slots`, and L2 Zero-Copy pinned RAM based on detected GPU HBM and host RAM — using the **exact same `.tmmu` / `.nvm` cartridge**:

- **`g5.2xlarge` / `g6.2xlarge` (24 GB A10G / L4):** Runs 26B–80B Sparse-MoE models in `vm_paged` mode (`L1 + L2 Zero-Copy + L3 NVMe`). *(Avoid `g4dn` T4 `sm_75`, which lacks BF16/Ampere tensor features — `tmmu check` will catch and warn immediately.)*
- **`g6e.2xlarge` / `g6e.4xlarge` (48 GB L40S + 64–128 GB RAM):** Automatically promotes 26B models to `all_resident` in HBM/GDDR6, and maps **100% of `Qwen3-Next-80B`'s 44 GB cartridge across L1 VRAM + L2 Zero-Copy** (`full_graph` mode, zero NVMe faults!).
- **`p4d` / `p5` (80 GB A100 / 80 GB H100 / 141 GB H200):** Automatically promotes `80B` cartridges to `all_resident` inside HBM, or unlocks **200B–400B+ Sparse-MoE models** (`Qwen3-235B-A22B`, DeepSeek-class MoEs) on a single GPU via `vm_paged` tiering!

---

## Related Work, Alternatives & Architectural Lineage

### Alternative Local MoE & Hybrid Inference Engines
Depending on your hardware and model size, several strong alternatives explore complementary points in the design space:

- **[Strata (`Niko1221/Strata`)](https://github.com/Niko1221/Strata), [PolyStrata (`simplexai-labs/PolyStrata`)](https://github.com/simplexai-labs/PolyStrata) & [MoMoE (`aflin/momoe`)](https://github.com/aflin/momoe):** Specialized C++/`llama.cpp` forks that hit higher peak `tok/s` (`~95–120+` on `Qwen3.8-Flash-Next` with `64 GB+` RAM) by combining aggressive `2–3 bit` (`IQ2_XS` / `IQ3`) GGUF quants, native Multi-Token Prediction (`MTP`) drafting, and **concurrent CPU `AVX-512` compute** for non-cached experts (with `Coaade-Inc/strataflow` exploring GGML per-expert streaming). **Where TensorMMU differs:** TensorMMU runs **100% of experts on the GPU** at lossless `4-bit` (`UINT4` group-64) compiled directly from Hugging Face `.safetensors` — paging across VRAM, zero-copy DDR5, and NVMe inside CUDA Graphs (even on `32 GB` RAM PCs or CPU-constrained Kubernetes pods) and scaling straight onto `48–141 GB` AWS GPUs.
- **[Colibrì (`JustVugg/colibri`)](https://github.com/JustVugg/colibri):** A pure-C hierarchical inference engine that also treats `NVMe -> System RAM -> GPU VRAM` as a demand-paged hierarchy for Sparse-MoE models, using explicit async C file I/O (`pread` / `io_uring`) and persistent `.coli_usage` hot-store pinning. **When to prefer Colibrì:** ultra-giant `400B–744B` MoEs (e.g., `GLM-5`, full `DeepSeek-V3/R1`) on disk-bound consumer PCs with zero Python/PyTorch runtime dependencies. **Where TensorMMU differs:** TensorMMU targets maximum single-GPU speed (`28–55 tok/s`) and full OpenAI/Anthropic agent serving in the `26B–120B+` regime (and `200B+` on AWS HBM GPUs) via `cudaHostAllocMapped` L2 Zero-Copy reads, Full/Piecewise CUDA Graphs, and native tool-calling endpoints.
- **[Raymond's Adaptive KV Streaming (`RaymondHuang210129/llama.cpp-adaptive-kv-streaming`)](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming):** Orthogonal and complementary — pages **long-context KV cache blocks** (`128K–256K` tokens) from System RAM through a shared GPU VRAM ring buffer when model weights already fit in VRAM, whereas TensorMMU pages **MoE model weights** while growing a GPU-resident KV cache on demand (`4K -> 32K`).
- **`llama.cpp` (`-ot` expert offload) & `KTransformers`:** Offload cold MoE experts to **host CPU cores** (`AVX-512` / `AMX`) instead of paging expert weights across PCIe onto the GPU. Great on workstation/server CPUs with 8–12 channel DDR5, whereas TensorMMU executes 100% of active experts on NVIDIA GPU SMs via L1 TLB + L2 Zero-Copy so ordinary 2-channel DDR5 desktops aren't bottlenecked by CPU GEMV threads.
- **Unified-Memory Engines (`Halogen Flash Server`, `MLX`):** Achieve `~37–45+ tok/s` on `80B` MoEs when the entire model sits in `128 GB+` unified memory (e.g., AMD Strix Halo or Apple M-series Ultra) with no PCIe boundary. TensorMMU brings tiered memory to discrete NVIDIA GPUs (`24 GB` RTX up to `80–141 GB` H100/H200) where weights exceed GPU VRAM.

### Google & DeepSeek Conditional / Neural Memory (and Why We Renamed from `NeuralVM`)
TensorMMU sits directly alongside recent architectural work from **Google** and **DeepSeek** on decoupling massive static knowledge capacity from active per-token GPU compute:

- **DeepSeek *Engram* (*Conditional Memory via Scalable Lookup*):** Introduces *conditional memory* alongside MoE conditional compute — storing massive static N-gram knowledge tables in host CPU DRAM/NVMe and prefetching deterministic `O(1)` hash lookups over PCIe with minimal latency overhead.
- **Google *Per-Layer Embeddings (`PLE`)* & *Titans* (*Neural Long-Term Memory*):** Google's Gemma ecosystem (`PLE`, supported directly in TensorMMU's compiler via zero-VRAM `ple_host.safetensors` offloading) shortcuts per-layer token knowledge lookup from host memory, while Google's *Titans* explores test-time-trained neural long-term memory modules.
- **From `NeuralVM` to `TensorMMU`:** This project began under the working name **`NeuralVM`** (*Neural Virtual Machine*) with the goal of decompiling *dense* models into hot reasoning trunks + swappable knowledge pages. Our Phases A–E experiments (preserved in this repo) showed that post-hoc *dense* FFN paging breaks down because dense layers activate 100% of channels per token — whereas **native Sparse-MoE routing (`1.95%–6.25%` active weights) and deterministic host lookup tables (`PLE` / `Engram`)** provide true hardware locality where a **GPU Memory Management Unit (`TensorMMU`) + TLB + L1/L2/L3 cache hierarchy** runs losslessly at interactive agent speeds.

---

## Repository Layout

- `tensormmu/` — Unified CLI (`tmmu check`, `tmmu compile`, `tmmu serve`), GPU/RAM/cgroup/disk safety checks (`hardware_safety.py`), streaming compiler wrapper (`compiler.py`), and OpenAI + Anthropic tool-call protocol engine (`tool_parser.py`).
- `runtime/` — NVRTC CUDA MoE-TLB, `UINT4-WINA`/`UINT3-WINA` & exact-byte `cuMemHostAlloc` Zero-Copy kernels (`moe_tlb_kernels.py`), 3-tier TLB cache & Cold L3 Drop manager (`moe_expert_tlb.py`), growable Full/Piecewise CUDA Graph decode runner (`cuda_graph_decode.py`), CUDA-graphed Native `mtp.*` drafter (`native_mtp.py`), GatedDeltaNet fused kernels (`qwen3_next_kernels.py`), speculative decoding (`speculative_vm.py`), and HTTP/SSE server (`openai_server.py`).
- `decompiler/` — `O(1)`-VRAM single-layer streaming `.safetensors -> .tmmu/.nvm` serializer (`nvm_serializer.py`).
- `docs/index.html` — Standalone technical explanation website (also served at `GET /about`).
- `benchmarks/` — Historical Phase 1–H research & benchmark reproduction scripts plus archived RTX 4090 JSON telemetry (`benchmarks/results/`).
- `Dockerfile`, `docker-compose.yml`, `deploy/k8s-deployment.yaml` — Container & Kubernetes deployment manifests.

## License

[MIT](LICENSE)
