from __future__ import annotations

import math

import pytest
import torch

from pyrokinesis.snn.neurons.SRM0 import SRM0

B, F = 4, 8


def test_srm0_step_hand_computed():
    m = SRM0(tau_mem=10.0, tau_ref=2.0, threshold=1.0, dt=1.0)
    state = m.initial_state((1,))

    spk, (mem, adapt) = m.step_state(torch.tensor([2.0]), state)
    assert torch.allclose(spk, torch.tensor([1.0]))
    assert torch.allclose(mem, torch.tensor([1.0]), atol=1e-5)
    assert torch.allclose(adapt, torch.tensor([1.0]))

    spk, (mem, adapt) = m.step_state(torch.tensor([0.0]), (mem, adapt))
    assert torch.allclose(spk, torch.tensor([0.0]))
    assert torch.allclose(mem, torch.tensor([math.exp(-0.1)]), atol=1e-5)
    assert torch.allclose(adapt, torch.tensor([math.exp(-0.5)]), atol=1e-5)


def test_srm0_parameter_defaults():
    m = SRM0()
    assert m.tau_mem.item() == pytest.approx(10.0)
    assert m.tau_ref.item() == pytest.approx(2.0)
    assert m.threshold.item() == 1.0


def test_srm0_refractory_suppresses_repeated_firing():
    # After a spike the refractory trace pushes the threshold up.
    m = SRM0(tau_mem=100.0, tau_ref=100.0, threshold=1.0, dt=1.0)
    state = m.initial_state((1,))

    spk, (mem, adapt) = m.step_state(torch.tensor([2.0]), state)
    assert torch.allclose(spk, torch.tensor([1.0]))
    assert torch.allclose(adapt, torch.tensor([1.0]))

    spk, (mem, adapt) = m.step_state(torch.tensor([0.0]), (mem, adapt))
    assert torch.allclose(spk, torch.tensor([0.0]))


def test_srm0_fires_under_strong_input():
    m = SRM0(init_hidden=False)
    state = m.initial_state((B, F))
    total = 0
    for _ in range(100):
        spk, state = m.step_state(torch.full((B, F), 5.0), state)
        total += spk.sum().item()
    assert total > 0


def test_srm0_strong_input_stays_finite():
    m = SRM0(init_hidden=False)
    state = m.initial_state((B, F))
    for _ in range(500):
        spk, state = m.step_state(torch.full((B, F), 1e3), state)
    assert torch.isfinite(spk).all()
    for s in state:
        assert torch.isfinite(s).all()


def test_srm0_sequence_matches_eager_scan():
    m = SRM0(init_hidden=True).compile_sequence_scan(mode="default")
    eager = SRM0(init_hidden=True)
    x_seq = torch.randn(10, B, F)

    out = m.forward_sequence(x_seq)
    ref = eager.forward_sequence(x_seq)
    assert torch.allclose(out, ref, atol=1e-5)
