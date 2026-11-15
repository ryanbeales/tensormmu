# TensorMMU Historical Research & Benchmark Suite (`benchmarks/`)

None of the scripts in this folder are required by the `tensormmu` (`tmmu`) CLI, server, or Docker image — they are preserved here as reproducible research history showing how the architecture evolved from dense FFN decompilation to CUDA-graphed Sparse-MoE virtual memory:

| Phase | Script(s) | Summary & Outcome |
| :--- | :--- | :--- |
| **Phases 1–5 (Dense SVD Hypothesis)** | `run_ablation.py`, `run_phase2_cascade.py`, `run_phase3_full_mlp.py`, `run_phase4_giant_killer.py`, `run_phase5_cartridge_transplant.py`, `config.py` | Initial *NeuralVM* experiments testing contrastive SVD hot-skeleton + cold-neuron decomposition on dense `Gemma 4 E2B / E4B` MLPs. |
| **Phases B–E (Dense Scaling up to 31B)** | `run_phase_b_comparison.py`, `run_phase_c2_streaming_nvm.py`, `run_phase_d_31b_scale.py`, `run_phase_e_31b_amortized.py` | Built the `O(1)` single-layer streaming `.safetensors` compiler and tested dense paging on `Gemma-4-31B` — proving the negative result that 100%-active dense SwiGLU layers lack the activation locality required for hardware paging. |
| **Phase F (Zero-Skeleton Sparse-MoE VM)** | `run_phase_f_moe_vm.py` | Pivoted to native Sparse-MoE (`Gemma-4-26B-A4B` & `Qwen3-Next-80B-A3B`) with 4 KB-aligned contiguous `UINT4` expert pages, GPU TLB caching, and intra-layer Shared-MLP DMA overlap (`100%` GSM8K parity). |
| **Phase G (Lookahead & Confidence-Gated Spec)** | `run_phase_g_lookahead_moe.py` | Added Cross-Layer `INT8` Lookahead Routers, Probationary Draft-LRU, and Confidence-Gated Speculative Paging. |
| **Phase H (3-Tier Zero-Copy & CUDA Graphs)** | `run_phase_h_speedup.py` | Added L2 `cudaHostAllocMapped` Zero-Copy DDR5, 32-slot NVMe Pinned DMA Ring, and Full/Piecewise CUDA Graphs — reaching **`49–55 tok/s`** (`Gemma-4-26B` at `8.15 GiB` VRAM) and **`28.4–34.5 tok/s`** (`Qwen3-Next-80B` at `18.8 GiB` VRAM) on a single RTX 4090. |

Measured JSON telemetry from these runs is archived under [`benchmarks/results/`](results/).

To run any benchmark from the repository root:

```powershell
python benchmarks/run_phase_h_speedup.py --help
```
