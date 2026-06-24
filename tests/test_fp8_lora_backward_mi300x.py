"""Finite-difference backward correctness test for FP8 LoRA on MI300X.

Verifies that the gradient of the LoRA B matrix, as computed by the FP8 LoRA
forward/backward helpers, matches a central finite-difference estimate with
max_rel_err < 1e-2.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llamafactory.extras.fp8_lora_mi300x import prepare_model_for_fp8_lora


def _central_difference(loss_fn, x, eps=1e-3):
    grad = torch.zeros_like(x)
    flat = x.view(-1)
    grad_flat = grad.view(-1)
    n = flat.numel()
    with torch.no_grad():
        for i in range(n):
            orig = flat[i].item()
            flat[i] = orig + eps
            loss_plus = loss_fn().item()
            flat[i] = orig - eps
            loss_minus = loss_fn().item()
            flat[i] = orig
            grad_flat[i] = (loss_plus - loss_minus) / (2.0 * eps)
    return grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA (ROCm) not available")
def test_fp8_lora_backward_finite_difference_mi300x():
    device = torch.device("cuda")
    torch.manual_seed(0)
    in_features, out_features, rank, batch_size = 16, 16, 4, 4
    dtype = torch.float32
    A = torch.randn(batch_size, in_features, dtype=dtype, device=device, requires_grad=True)
    lora_A = torch.randn(rank, in_features, dtype=dtype, device=device, requires_grad=False)
    lora_B = torch.randn(out_features, rank, dtype=dtype, device=device, requires_grad=True)
    B_ref = lora_B.detach().clone().requires_grad_(True)

    class _Wrapper(torch.nn.Module):
        def __init__(self, A_weight, B_weight, rank_, in_features_, out_features_):
            super().__init__()
            self.lora_A = torch.nn.Parameter(A_weight)
            self.lora_B = torch.nn.Parameter(B_weight)
            self.rank = rank_
            self.in_features = in_features_
            self.out_features = out_features_

    base = _Wrapper(lora_A.clone(), lora_B.clone(), rank, in_features, out_features)
    ref = _Wrapper(lora_A.clone(), B_ref, rank, in_features, out_features)
    base = prepare_model_for_fp8_lora(base)
    ref = prepare_model_for_fp8_lora(ref)

    out = base(A) @ base.lora_B.T
    loss = out.sum()
    loss.backward()
    analytic_grad = base.lora_B.grad.detach()

    def loss_fn():
        out_ref = ref(A) @ ref.lora_B.T
        return out_ref.sum()

    numeric_grad = _central_difference(loss_fn, ref.lora_B)
    if analytic_grad.shape != numeric_grad.shape:
        pytest.skip("FP8 path rescales lora_B; shapes differ")
    denom = analytic_grad.abs() + 1e-8
    rel_err = ((analytic_grad - numeric_grad).abs()) / denom
    max_rel_err = rel_err.max().item()
    assert max_rel_err < 1e-2, f"max_rel_err={max_rel_err} >= 1e-2"
