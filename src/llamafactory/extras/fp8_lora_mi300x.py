# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""FP8 LoRA preparation for AMD MI300X.

Wraps ``nn.Linear`` layers so that the *base* forward path uses FP8 GEMM
(via AITER when available, falling back to native torch ``float8_e4m3fn``
matmuls) while keeping LoRA adapter weights and master weights in BF16.
"""

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from . import logging


logger = logging.get_logger(__name__)


_AITER_FP8_GEMM = None

def _get_aiter_fp8_gemm():
    r"""Lazy-load AITER FP8 GEMM, returning ``None`` if unavailable."""
    global _AITER_FP8_GEMM
    if _AITER_FP8_GEMM is not None:
        return _AITER_FP8_GEMM

    try:
        import aiter
        if hasattr(aiter, "fp8_gemm"):
            _AITER_FP8_GEMM = aiter.fp8_gemm
            logger.info_rank0("Using AITER FP8 GEMM for MI300X.")
            return _AITER_FP8_GEMM
    except Exception:
        pass

    _AITER_FP8_GEMM = False
    return None


class FP8LinearWrapper(nn.Module):
    r"""Wrapper that casts input/weight to FP8 for GEMM and accumulates in BF16.

    The base ``weight`` is kept in BF16 (or FP32) as the *master* weight.
    On each forward we dynamically quantise to ``float8_e4m3fn``, call the
    FP8 GEMM backend, then dequantize back to BF16.  The original ``bias``
    and any LoRA adapters attached to the wrapped module remain in BF16.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self._wrapped = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        # Keep a FP8 buffer for the quantised weight so we don't re-quantise
        # on every forward when gradients aren't needed (not implemented here
        # for simplicity, but the shape is registered).
        self.register_buffer("weight_fp8", None, persistent=False)

    @property
    def weight(self):
        return self._wrapped.weight

    @property
    def bias(self):
        return self._wrapped.bias

    def _maybe_quantise_weight(self) -> torch.Tensor:
        w = self._wrapped.weight
        if w.dtype == torch.float8_e4m3fn:
            return w
        # Dynamic per-tensor quantisation to E4M3
        fp8 = w.to(torch.float8_e4m3fn)
        return fp8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep LoRA path in BF16; only the base linear GEMM goes through FP8
        orig_dtype = x.dtype
        if orig_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            x = x.to(torch.bfloat16)

        fp8_gemm = _get_aiter_fp8_gemm()
        w_fp8 = self._maybe_quantise_weight()

        if fp8_gemm is not None and fp8_gemm is not False:
            try:
                out = fp8_gemm(x, w_fp8.t(), self._wrapped.bias)
                return out.to(orig_dtype)
            except Exception as exc:
                logger.warning_rank0_once(
                    f"AITER FP8 GEMM failed ({exc}), falling back to native FP8 matmul."
                )

        # Native fallback: cast activations to FP8, matmul in FP32/BF16
        x_fp8 = x.to(torch.float8_e4m3fn)
        # torch.matmul supports float8 on MI300X (gfx942) via hipBLASLt
        out = torch.matmul(x_fp8, w_fp8.t().to(torch.float8_e4m3fn))
        if self._wrapped.bias is not None:
            out = out + self._wrapped.bias
        return out.to(orig_dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, fp8= True"


def _replace_linear_for_fp8(
    module: nn.Module,
    prefix: str = "",
    target_names: set[str] | None = None,
) -> None:
    r"""Recursively replace ``nn.Linear`` layers with ``FP8LinearWrapper``."""
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if target_names is None or full_name in target_names:
                setattr(module, name, FP8LinearWrapper(child))
                logger.debug_rank0(f"Wrapped {full_name} with FP8LinearWrapper.")
        else:
            _replace_linear_for_fp8(child, full_name, target_names)


def prepare_model_for_fp8_lora(
    model: "PreTrainedModel",
    lora_config: Any | None = None,
    target_modules: set[str] | None = None,
) -> "PreTrainedModel":
    r"""Prepare a model for FP8 LoRA fine-tuning on AMD MI300X.

    This function wraps the *base* ``nn.Linear`` weights so their forward
    GEMM runs in FP8 (via AITER if available, native torch fallback
    otherwise) while keeping the LoRA adapter weights and master weights
    in BF16.

    Args:
        model: The pretrained model to wrap.
        lora_config: Optional PEFT LoRA config; if provided, its
            ``target_modules`` are used to decide which layers to wrap.
            When ``None``, all ``nn.Linear`` layers are wrapped.
        target_modules: Explicit set of fully-qualified module names to wrap.
            Overrides ``lora_config`` when provided.

    Returns:
        The same ``model`` instance (modified in-place).
    """
    if target_modules is None and lora_config is not None:
        target_modules = getattr(lora_config, "target_modules", None)
        if target_modules is not None:
            if isinstance(target_modules, str):
                target_modules = {target_modules}
            else:
                target_modules = set(target_modules)

    logger.info_rank0(
        f"Preparing model for FP8 LoRA (target_modules={target_modules}) on MI300X."
    )
    _replace_linear_for_fp8(model, target_names=target_modules)
    logger.info_rank0("FP8 LoRA preparation complete.")
    return model
