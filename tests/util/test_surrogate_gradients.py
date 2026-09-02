from __future__ import annotations

import math

import torch

from pyrokinesis.util import (
    atan_surrogate,
    default_spike_grad,
    fast_sigmoid_surrogate,
    sigmoid_surrogate,
    straight_through_surrogate,
    triangular_surrogate,
)


def test_all_surrogates_hard_threshold_forward():
    x = torch.tensor([-3.0, -0.1, 0.0, 0.1, 3.0])
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
    for fn in (
        straight_through_surrogate,
        sigmoid_surrogate(),
        atan_surrogate(),
        triangular_surrogate(),
        fast_sigmoid_surrogate(),
    ):
        assert torch.equal(fn(x), expected)


def test_surrogates_hard_threshold_without_grad():
    x = torch.tensor([-1.0, 1.0])
    with torch.no_grad():
        for fn in (
            sigmoid_surrogate(),
            atan_surrogate(),
            triangular_surrogate(),
            fast_sigmoid_surrogate(),
        ):
            assert torch.equal(fn(x), torch.tensor([0.0, 1.0]))


def test_straight_through_backward_is_identity():
    x = torch.tensor([-2.0, 0.0, 1.5], requires_grad=True)
    straight_through_surrogate(x - 0.5).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_default_spike_grad_is_straight_through():
    assert default_spike_grad is straight_through_surrogate
    x = torch.tensor([-1.0, 1.0])
    assert torch.equal(default_spike_grad(x), torch.tensor([0.0, 1.0]))


def test_sigmoid_surrogate_backward_matches_derivative():
    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    sigmoid_surrogate(beta=2.0)(x).sum().backward()
    expected = torch.sigmoid(2.0 * x.detach()) * (1 - torch.sigmoid(2.0 * x.detach()))
    assert torch.allclose(x.grad, expected, atol=1e-6)


def test_atan_surrogate_backward_formula():
    beta = 2.0
    x = torch.tensor([-0.5, 0.0, 0.5], requires_grad=True)
    atan_surrogate(beta=beta)(x).sum().backward()
    scale = math.pi / 2 * beta * x.detach()
    expected = beta / 2 / (1 + scale ** 2)
    assert torch.allclose(x.grad, expected, atol=1e-6)


def test_triangular_surrogate_backward_formula():
    beta = 1.0
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    triangular_surrogate(beta=beta)(x).sum().backward()
    expected = (1 - beta * torch.abs(x.detach())).clamp(min=0.0)
    assert torch.allclose(x.grad, expected)


def test_fast_sigmoid_surrogate_backward_formula():
    beta = 1.0
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    fast_sigmoid_surrogate(beta=beta)(x).sum().backward()
    expected = beta / 2 / (1 + torch.abs(beta * x.detach())) ** 2
    assert torch.allclose(x.grad, expected)


def test_surrogate_beta_widens_gradient_window():
    # Larger beta -> steeper surrogate -> narrower gradient support near 0.
    x = torch.tensor([-0.6, 0.0, 0.6], requires_grad=True)
    sigmoid_surrogate(beta=1.0)(x).sum().backward()
    wide = x.grad.clone()
    x.grad = None

    sigmoid_surrogate(beta=20.0)(x).sum().backward()
    steep = x.grad.clone()

    assert steep[0].item() < wide[0].item()
    assert steep[2].item() < wide[2].item()


def test_surrogates_work_as_spike_grad():
    from pyrokinesis.snn import LIF

    m = LIF(beta=0.5, threshold=1.0, spike_grad=sigmoid_surrogate(beta=10.0))
    spk, (_mem,) = m.step_state(torch.tensor([2.0]), (torch.zeros(1),))
    assert torch.allclose(spk, torch.tensor([1.0]))