from __future__ import annotations

import pytest
import torch

from crematorium.snn.neurons.ALIF import ALIF

B, F = 4, 8


def test_alif_step_hand_computed():
    alif = ALIF(beta=0.5, threshold=1.0, beta_a=0.5, tau_a=0.9)
    state = alif.initial_state((1,))

    spk, (mem, adapt) = alif.step_state(torch.tensor([1.5]), state)
    assert torch.allclose(spk, torch.tensor([1.0]))
    assert torch.allclose(mem, torch.tensor([0.5]))
    assert torch.allclose(adapt, torch.tensor([1.0]))

    spk, (mem, adapt) = alif.step_state(torch.tensor([0.5]), (mem, adapt))
    assert torch.allclose(spk, torch.tensor([0.0]))
    assert torch.allclose(mem, torch.tensor([0.75]))
    assert torch.allclose(adapt, torch.tensor([0.9]))


def test_alif_parameter_defaults():
    alif = ALIF()
    assert alif.beta.item() == pytest.approx(0.9)
    assert alif.threshold.item() == 1.0
    assert alif.beta_a.item() == pytest.approx(0.5)
    assert alif.tau_a.item() == pytest.approx(0.9)


def test_alif_adaptation_raises_threshold():
    # A spike raises the effective threshold, suppressing the next spike.
    alif = ALIF(beta=1.0, threshold=1.0, beta_a=1.0, tau_a=1.0)
    state = alif.initial_state((1,))

    spk, (mem, adapt) = alif.step_state(torch.tensor([2.0]), state)
    assert torch.allclose(spk, torch.tensor([1.0]))
    assert torch.allclose(adapt, torch.tensor([1.0]))

    # Without adaptation this would fire again; with adapt=1 the effective
    # threshold is 2.0 and mem=2.0 is exactly at it.
    spk, (mem, adapt) = alif.step_state(torch.tensor([0.0]), (mem, adapt))
    assert torch.allclose(spk, torch.tensor([0.0]))


def test_alif_fires_under_strong_input():
    alif = ALIF(init_hidden=False)
    state = alif.initial_state((B, F))
    total = 0
    for _ in range(100):
        spk, state = alif.step_state(torch.full((B, F), 5.0), state)
        total += spk.sum().item()
    assert total > 0


def test_alif_strong_input_stays_finite():
    alif = ALIF(init_hidden=False)
    state = alif.initial_state((B, F))
    for _ in range(500):
        spk, state = alif.step_state(torch.full((B, F), 1e3), state)
    assert torch.isfinite(spk).all()
    for s in state:
        assert torch.isfinite(s).all()


def test_alif_sequence_matches_eager_scan():
    alif = ALIF(init_hidden=True).compile_sequence_scan(mode="default")
    eager = ALIF(init_hidden=True)
    x_seq = torch.randn(10, B, F)

    out = alif.forward_sequence(x_seq)
    ref = eager.forward_sequence(x_seq)
    assert torch.allclose(out, ref, atol=1e-5)
