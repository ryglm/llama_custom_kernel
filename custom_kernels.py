# ---------------------------------------------------------
# custom_kernels.py
# Custom CUDA kernels for optimized operations in transformer models.
# ---------------------------------------------------------
import os
import types
import pathlib
import shutil
from typing import Optional

import torch
from torch.utils.cpp_extension import load

_VERBOSE = os.getenv("CUSTOM_KERNELS_VERBOSE", "0") not in ("0", "false", "False", "")
def _log(msg: str) -> None:
    if _verbose_enabled():
        print(f"[custom_kernels] {msg}")

def _verbose_enabled() -> bool:
    return _VERBOSE

def _get_eps(module) -> float:
    # HF has used both .eps and .variance_epsilon in various versions
    return float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-5)))

def _rmsnorm_ref(hidden_states: torch.Tensor,
                 weight: torch.Tensor,
                 eps: float) -> torch.Tensor:
    """
    Version-independent RMSNorm reference:
      y = x * rsqrt(mean(x^2, -1, keepdim=True) + eps) * weight
    Works for fp16/bf16/fp32 by upcasting to fp32 and casting back.
    """
    x_dtype = hidden_states.dtype
    x = hidden_states.float()
    w = weight.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(var + float(eps))
    y = y * w
    return y.to(x_dtype)

def _maybe_set_cuda_home_from_nvcc() -> None:
    # Convenience: if CUDA_HOME is missing but nvcc is on PATH, infer it.
    if "CUDA_HOME" in os.environ:
        return
    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_root = pathlib.Path(nvcc).parent.parent
        os.environ["CUDA_HOME"] = str(cuda_root)
        _log(f"CUDA_HOME inferred from nvcc: {cuda_root}")

_ext = None
_ext_tried = False  # avoid retry spam if build fails

def _build_ext():
    """
    Try to build/load the RMSNorm CUDA extension from files.
    Returns the loaded module or None if unavailable.
    Never raises to the caller; we always fall back cleanly.
    """
    global _ext, _ext_tried
    if _ext is not None:
        return _ext
    if _ext_tried:
        return None
    _ext_tried = True

    if not torch.cuda.is_available():
        _log("CUDA is not available; will use PyTorch RMSNorm fallback.")
        return None

    try:
        _maybe_set_cuda_home_from_nvcc()
        base_dir = pathlib.Path(__file__).resolve().parent
        cpp_path = base_dir / "rmsnorm_binding.cpp"
        cu_path = base_dir / "rmsnorm_kernel.cu"
        if not cpp_path.exists() or not cu_path.exists():
            _log("Missing rmsnorm_binding.cpp or rmsnorm_kernel.cu; using fallback.")
            return None

        _log("Building/loading rmsnorm_cuda_ext from source files...")
        _ext = load(
            name="rmsnorm_cuda_ext",
            sources=[str(cpp_path), str(cu_path)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=_verbose_enabled(),
        )
        _log("rmsnorm_cuda_ext loaded successfully.")
        return _ext
    except Exception as e:
        _log(f"Extension build failed: {e}. Using fallback RMSNorm.")
        return None

def _fast_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for HF's LlamaRMSNorm.forward.
    Tries the CUDA kernel; on any issue, cleanly falls back to _rmsnorm_ref.
    """
    try:
        eps = _get_eps(self)
        weight = self.weight

        # Kernel supports CUDA + fp16/fp32 (bf16 -> fallback)
        if (
            hidden_states.is_cuda
            and weight.is_cuda
            and hidden_states.dtype in (torch.float16, torch.float32)
        ):
            ext = _build_ext()
            if ext is not None:
                # Ensure contiguity for a view; cheap if already contiguous
                x = hidden_states.contiguous()
                w = weight.contiguous()
                x_flat = x.view(-1, x.shape[-1])  # (rows, hidden)
                out_flat = ext.rmsnorm_forward(x_flat, w, float(eps))
                return out_flat.view_as(x)
    except Exception as e:
        _log(f"Kernel path failed: {e}. Falling back.")

    # Safe reference path (works on CPU/CUDA, all dtypes)
    return _rmsnorm_ref(hidden_states, self.weight, _get_eps(self))

def patch_llama_rmsnorm(model) -> int:
    """
    Swap all LlamaRMSNorm.forward with a CUDA kernel-backed version.
    Returns how many modules were patched.
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        target_type = LlamaRMSNorm
    except Exception:
        # Fallback to name check if import path changes in HF versions
        target_type = None

    patched = 0
    for m in model.modules():
        try:
            if (target_type and isinstance(m, target_type)) or (m.__class__.__name__ == "LlamaRMSNorm"):
                # Bind our forward into the module instance
                m.forward = types.MethodType(_fast_rmsnorm_forward, m)
                patched += 1
        except Exception as e:
            _log(f"Failed to patch a module: {e}")
            continue

    if patched == 0:
        _log("No LlamaRMSNorm modules found to patch.")
    else:
        _log(f"Patched {patched} LlamaRMSNorm layers.")
    return patc

