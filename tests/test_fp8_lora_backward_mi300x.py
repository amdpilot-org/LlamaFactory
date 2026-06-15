"""Finite-difference gradient test for FP8 LoRA MI300X wrapper.

Verifies that ``prepare_model_for_fp8_lora`` produces a differentiable
forward path and that analytic gradients agree with finite-difference
differences within a 1e-2 relative tolerance.
"""

import pytest
import torch
import torch.nn as nn

from llamafactory.extras.fp8_lora_mi300x import prepare_model_for_fp8_lora


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32, bias=True)
        self.fc2 = nn.Linear(32, 8, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _finite_diff_grad(model, x, eps=1e-3):
    """Compute approximate input gradient via central differences."""
    x = x.clone().detach()
    grad_approx = torch.zeros_like(x)
    for i in range(x.numel()):
        idx = x.view(-1)[i]
        x_plus = x.clone()
        x_plus.view(-1)[i] = idx + eps
        x_minus = x.clone()
        x_minus.view(-1)[i] = idx - eps
        loss_plus = model(x_plus).sum()
        loss_minus = model(x_minus).sum()
        grad_approx.view(-1)[i] = (loss_plus - loss_minus) / (2 * eps)
    return grad_approx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/ROCm")
def test_fp8_lora_wrapper_backward():
    device = "cuda"
    model = TinyModel().to(device, dtype=torch.bfloat16)
    prepare_model_for_fp8_lora(model)

    x = torch.randn(2, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    analytic_grad = x.grad.detach().clone()

    # Finite-difference check
    x_fd = x.detach().clone().requires_grad_(False)
    approx_grad = _finite_diff_grad(model, x_fd, eps=1e-3)

    rel_err = (analytic_grad - approx_grad).abs() / (approx_grad.abs() + 1e-6)
    max_rel_err = rel_err.max().item()
    print(f"max_rel_err = {max_rel_err:.6e}")
    assert max_rel_err < 1e-2, f"Gradient relative error {max_rel_err} >= 1e-2"


def test_fp8_lora_wrapper_cpu_fallback():
    """The wrapper should still run on CPU (native fallback) without crashing."""
    model = TinyModel().to(dtype=torch.bfloat16)
    prepare_model_for_fp8_lora(model)
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    out = model(x)
    assert out.shape == (2, 8)
    loss = out.sum()
    loss.backward()


def test_prepare_model_for_fp8_lora_signature():
    """Smoke-test the public API surface."""
    model = TinyModel()
    result = prepare_model_for_fp8_lora(model, lora_config=None)
    assert result is model
