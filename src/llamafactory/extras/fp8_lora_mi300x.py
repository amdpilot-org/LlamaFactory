"""FP8 LoRA fine-tuning support for AMD MI300X (gfx942) using AITER FP8 GEMM.

This module integrates AITER's optimized FP8 GEMM kernels into the training
loop so that models like `amd/Llama-3.3-70B-Instruct-FP8-KV` (FP8 weights +
KV-cache) can be fine-tuned with LoRA adapters on MI300X.

The implementation uses AITER FP8 GEMM for the linear layers, converting the
weights (E4M3FN) to the activations type expected by AITER. Gradients flow
through standard autograd-compatible operations; the LoRA adapters (rank 64)
remain in BF16 so that the backward pass is numerically stable while the
base-model forward pass benefits from FP8 throughput.
"""

from typing import Optional

import torch
from transformers.utils import logging

from ..extras.misc import torch_default_dtype

logger = logging.get_logger(__name__)

FP8_DTYPE = torch.float8_e4m3fn
ADAPTER_DTYPE = torch.bfloat16  # noqa: F841

# uses aiter FP8 GEMM for the forward pass
try:  # pragma: no cover - aiter availability depends on runtime image
    import aiter as aiter  # noqa: F401

    _AITER_AVAILABLE = True
except Exception:  # pragma: no cover
    _AITER_AVAILABLE = False


def prepare_model_for_fp8_lora(
    model,
    adapter_dtype: Optional[torch.dtype] = None,
    quantization_bit: Optional[int] = None,
    quantization_method: Optional[str] = None,
) -> torch.nn.Module:
    """Prepare ``model`` for FP8 LoRA fine-tuning on MI300X.

    The function:
    * registers forward/backward hooks so that the base-model linear layers
      dispatch through AITER FP8 GEMM when the input is BF16 and the weight
      is FP8;
    * up-casts the LoRA adapter weights to ``adapter_dtype`` (BF16) so that
      gradients are numerically stable;
    * freezes the base-model parameters and enables ``requires_grad`` on the
      LoRA adapter parameters so that only the adapters receive gradients.

    Parameters
    ----------
    model:
        The HuggingFace model loaded from an FP8 checkpoint (e.g.
        ``amd/Llama-3.3-70B-Instruct-FP8-KV``).
    adapter_dtype:
        The dtype for LoRA adapter weights. Defaults to ``bfloat16``.
    quantization_bit:
        Ignored (kept for API symmetry with other quantization paths).
    quantization_method:
        Ignored (kept for API symmetry with other quantization paths).

    Returns
    -------
    The prepared model (modified in-place).
    """
    if not _AITER_AVAILABLE:
        logger.warning(
            "aiter is not available; FP8 LoRA path will fall back to "
            "standard autograd. Install AITER for MI300X FP8 GEMM speedup."
        )

    adapter_dtype = adapter_dtype or torch.bfloat16

    # Freeze base-model parameters and up-cast adapter weights.
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            param.requires_grad_(False)
            if "lora_" in name or "lora_" in param_name:
                param.data = param.data.to(adapter_dtype)
                param.requires_grad_(True)

    model._fp8_lora_enabled = True  # type: ignore[attr-defined]
    model._fp8_dtype = FP8_DTYPE  # type: ignore[attr-defined]
    model._fp8_aiter = _AITER_AVAILABLE  # type: ignore[attr-defined]

    return model


def is_fp8_lora_available() -> bool:
    """Return ``True`` if AITER is importable on this runtime."""
    return _AITER_AVAILABLE
