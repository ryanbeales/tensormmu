"""TensorMMU (`tmmu`): Hardware-Inspired Virtual Memory & TLB Inference Engine for Sparse-MoE & Large LLMs."""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

__version__ = "0.1.0"


def ensure_torch_cuda_nvrtc_utils() -> None:
    """Backfills `_cuda_load_module`, `_get_gpu_rtc_library`, and `_get_gpu_runtime_library` on PyTorch < 2.7."""
    try:
        import torch
        import torch.cuda._utils as cu_utils
    except Exception:
        return

    if hasattr(cu_utils, "_cuda_load_module") and hasattr(cu_utils, "_get_gpu_rtc_library"):
        return

    def _get_gpu_runtime_library() -> ctypes.CDLL:
        if sys.platform == "win32":
            return ctypes.CDLL("nvcuda.dll")
        return ctypes.CDLL("libcuda.so.1")

    def _check_cuda(result: int) -> None:
        if result == 0:
            return
        err_str = ctypes.c_char_p()
        libcuda = _get_gpu_runtime_library()
        libcuda.cuGetErrorString(result, ctypes.byref(err_str))
        msg = err_str.value.decode("utf-8", errors="replace") if err_str.value is not None else f"code {result}"
        raise RuntimeError(f"CUDA driver error: {msg}")

    def _get_gpu_rtc_library() -> ctypes.CDLL:
        cuda_ver = str(getattr(torch.version, "cuda", None) or "12.4")
        major = int(cuda_ver.split(".")[0])
        candidates: List[str] = []
        if sys.platform == "win32":
            candidates.append(f"nvrtc64_{major}0_0.dll")
        else:
            candidates.extend([f"libnvrtc.so.{major}", "libnvrtc.so"])
            try:
                import nvidia.cuda_nvrtc.lib as nvrtc_pkg

                pkg_dir = os.path.dirname(nvrtc_pkg.__file__)
                candidates.extend(sorted(glob.glob(os.path.join(pkg_dir, "libnvrtc.so*")), reverse=True))
            except Exception:
                pass
            candidates.extend(sorted(glob.glob("/usr/local/cuda*/lib64/libnvrtc.so*"), reverse=True))

        for lib_name in candidates:
            try:
                return ctypes.CDLL(lib_name)
            except OSError:
                continue
        raise OSError(f"Could not load NVRTC shared library (tried {candidates})")

    class _CudaKernel:
        def __init__(self, func: ctypes.c_void_p, module: ctypes.c_void_p) -> None:
            self.func = func
            self.module = module
            self._max_shared_mem_bytes = 0

        def __call__(
            self,
            grid: Tuple[int, int, int] = (1, 1, 1),
            block: Tuple[int, int, int] = (1, 1, 1),
            args: Optional[List[Any]] = None,
            shared_mem: int = 0,
            stream: Optional[Any] = None,
        ) -> None:
            libcuda = _get_gpu_runtime_library()
            if not args:
                args = []
            processed_ptrs: List[ctypes.c_void_p] = []
            c_args: List[Any] = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    if not arg.is_cuda and not (arg.is_cpu and arg.is_pinned()):
                        raise ValueError("All tensor arguments must be CUDA tensors or pinned CPU tensors")
                    ptr = ctypes.c_void_p(arg.data_ptr())
                    processed_ptrs.append(ptr)
                    c_args.append(ctypes.byref(ptr))
                elif isinstance(arg, int):
                    c_int = ctypes.c_int(arg)
                    c_args.append(ctypes.byref(c_int))
                elif isinstance(arg, float):
                    c_double = ctypes.c_double(arg)
                    c_args.append(ctypes.byref(c_double))
                else:
                    raise TypeError(f"Unsupported kernel argument type: {type(arg)}")

            c_args_array = (ctypes.c_void_p * len(c_args))()
            for i, c_arg in enumerate(c_args):
                c_args_array[i] = ctypes.cast(c_arg, ctypes.c_void_p)

            if stream is None:
                stream = torch.cuda.current_stream()
            stream_handle = getattr(stream, "_as_parameter_", None)
            if stream_handle is None:
                stream_handle = ctypes.c_void_p(int(stream.cuda_stream))

            _check_cuda(
                libcuda.cuLaunchKernel(
                    self.func,
                    int(grid[0]),
                    int(grid[1]),
                    int(grid[2]),
                    int(block[0]),
                    int(block[1]),
                    int(block[2]),
                    int(shared_mem),
                    stream_handle,
                    c_args_array,
                    None,
                )
            )

        def set_shared_memory_config(self, shared_mem_bytes: int) -> None:
            if shared_mem_bytes < 48 * 1024:
                self._max_shared_mem_bytes = shared_mem_bytes
                return
            libcuda = _get_gpu_runtime_library()
            cudaFuncAttributeMaxDynamicSharedMemorySize = 8
            _check_cuda(
                libcuda.cuFuncSetAttribute(
                    self.func,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    int(shared_mem_bytes),
                )
            )
            self._max_shared_mem_bytes = shared_mem_bytes

    class _CudaModule:
        def __init__(self, module: ctypes.c_void_p) -> None:
            self._module = module
            self._kernels: Dict[str, _CudaKernel] = {}

        def __getattr__(self, name: str) -> _CudaKernel:
            if name in self._kernels:
                return self._kernels[name]
            libcuda = _get_gpu_runtime_library()
            func = ctypes.c_void_p()
            _check_cuda(libcuda.cuModuleGetFunction(ctypes.byref(func), self._module, name.encode("utf-8")))
            kernel = _CudaKernel(func, self._module)
            self._kernels[name] = kernel
            return kernel

    def _cuda_load_module(
        ptx: Union[str, bytes], kernel_names: Optional[List[str]] = None
    ) -> Union[_CudaModule, Dict[str, _CudaKernel]]:
        libcuda = _get_gpu_runtime_library()
        if isinstance(ptx, str):
            ptx = ptx.encode("utf-8")
        module = ctypes.c_void_p()
        stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            _check_cuda(libcuda.cuModuleLoadData(ctypes.byref(module), ptx))
        if not kernel_names:
            return _CudaModule(module)
        kernels: Dict[str, _CudaKernel] = {}
        for name in kernel_names:
            func = ctypes.c_void_p()
            _check_cuda(libcuda.cuModuleGetFunction(ctypes.byref(func), module, name.encode("utf-8")))
            kernels[name] = _CudaKernel(func, module)
        return kernels

    if not hasattr(cu_utils, "_get_gpu_runtime_library"):
        setattr(cu_utils, "_get_gpu_runtime_library", _get_gpu_runtime_library)
    if not hasattr(cu_utils, "_get_gpu_rtc_library"):
        setattr(cu_utils, "_get_gpu_rtc_library", _get_gpu_rtc_library)
    if not hasattr(cu_utils, "_check_cuda"):
        setattr(cu_utils, "_check_cuda", _check_cuda)
    if not hasattr(cu_utils, "_CudaKernel"):
        setattr(cu_utils, "_CudaKernel", _CudaKernel)
    if not hasattr(cu_utils, "_CudaModule"):
        setattr(cu_utils, "_CudaModule", _CudaModule)
    if not hasattr(cu_utils, "_cuda_load_module"):
        setattr(cu_utils, "_cuda_load_module", _cuda_load_module)


ensure_torch_cuda_nvrtc_utils()

__all__ = [
    "__version__",
    "ensure_torch_cuda_nvrtc_utils",
]
