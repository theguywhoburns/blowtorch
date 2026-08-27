from __future__ import annotations

import pytest
import torch

from crematorium.snn.neurons.TwoCompartment import TwoCompartment

B, F = 4, 8


def test_two_compartment_step_hand_computed():
    m = TwoCompartment(gL=0.1, g_c=1.0, C=1.0, EL=0.0, threshold=1.0, dt=0.1)
    state = m.initial_state((1,))

    spk, (v_soma, v_dend) = m.step_state(torch.tensor([5.0]), state)
    assert torch.allclose(spk, torch.tensor([0.0]))
    assert torch.allclose(v_soma, torch.tensor([0.5]))
    # Coupling is explicit: both updates read the pre-step potentials, so a
    # silent dendrite starts the simulation at rest.
    assert torch.allclose(v_dend, torch.tensor([0.0]))

    # Coupling pulls the dendrite up once the soma is depolarized.
    spk, (v_soma, v_dend) = m.step_state(torch.tensor([5.0]), (v_soma, v_dend))
    assert v_dend.item() > 0.0


def test_two_compartment_parameter_defaults():
    m = TwoCompartment()
    assert m.gL.item() == pytest.approx(0.1)
    assert m.g_c.item() == pytest.approx(1.0)
    assert m.C.item() == pytest.approx(1.0)
    assert m.EL.item() == 0.0
    assert m.threshold.item() == 1.0
    assert m.dt == 0.01


def test_two_compartment_fires_under_strong_input():
    m = TwoCompartment(init_hidden=False)
    state = m.initial_state((B, F))
    total = 0
    for _ in range(100):
        spk, state = m.step_state(torch.full((B, F), 100.0), state)
        total += spk.sum().item()
    assert total > 0


def test_two_compartment_soma_reset_after_spike():
    m = TwoCompartment(init_hidden=False, dt=0.1)
    state = m.initial_state((1,))
    for _ in range(100):
        spk, state = m.step_state(torch.tensor([11.0]), state)
        if spk.item() == 1.0:
            # x * dt = 1.1 exceeds threshold 1.0 marginally, so the subtract
            # reset leaves v_soma at 0.1.
            assert state[0].item() < 1.0
            return
    pytest.fail("expected TwoCompartment to fire")


def test_two_compartment_strong_input_stays_finite():
    m = TwoCompartment(init_hidden=False)
    state = m.initial_state((B, F))
    for _ in range(500):
        spk, state = m.step_state(torch.full((B, F), 1e3), state)
    assert torch.isfinite(spk).all()
    for s in state:
        assert torch.isfinite(s).all()


def test_two_compartment_sequence_matches_eager_scan():
    m = TwoCompartment(init_hidden=True).compile_sequence_scan(mode="default")
    eager = TwoCompartment(init_hidden=True)
    x_seq = torch.randn(10, B, F)

    out = m.forward_sequence(x_seq)
    ref = eager.forward_sequence(x_seq)
    assert torch.allclose(out, ref, atol=1e-5)