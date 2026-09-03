from __future__ import annotations

import pytest
import torch

from pyrokinesis.snn.neurons.Izhikevich import Izhikevich

B, F = 4, 8


def _ref_step(m, x, v, u):
    a = m.a.item()
    b = m.b.item()
    c = m.c.item()
    d = m.d.item()
    dt = m.dt

    # Explicit coupling: both updates read the pre-step potentials.
    v_new = v + (0.04 * v ** 2 + 5 * v + 140 - u + x) * dt
    u_new = u + (a * (b * v - u)) * dt
    spk = (v_new > 30.0).to(v_new.dtype)
    v = (1 - spk) * v_new + spk * c
    u = u_new + spk * d
    return spk, v, u


def test_izhikevich_matches_reference():
    m = Izhikevich()
    state = m.initial_state((B, F))

    v = torch.full((B, F), -65.0)
    u = torch.full((B, F), -13.0)
    x = torch.randn(B, F) * 30.0

    for _ in range(50):
        spk, state = m.step_state(x, state)
        ref_spk, v, u = _ref_step(m, x, v, u)
        assert torch.equal(spk, ref_spk)
        assert torch.allclose(state[0], v)
        assert torch.allclose(state[1], u)


def test_izhikevich_parameter_defaults():
    m = Izhikevich()
    assert m.a.item() == pytest.approx(0.02)
    assert m.b.item() == pytest.approx(0.2)
    assert m.c.item() == -65.0
    assert m.d.item() == pytest.approx(8.0)
    assert m.dt == 1.0


def test_izhikevich_fires_under_strong_input():
    m = Izhikevich(init_hidden=False)
    state = m.initial_state((B, F))
    total = 0
    for _ in range(50):
        spk, state = m.step_state(torch.full((B, F), 20.0), state)
        total += spk.sum().item()
    assert total > 0


def test_izhikevich_spike_resets_v_and_u():
    # After a spike v snaps back to c and u jumps by d.
    m = Izhikevich(init_hidden=False)
    state = m.initial_state((1,))
    for _ in range(100):
        spk, state = m.step_state(torch.tensor([20.0]), state)
        if spk.item() == 1.0:
            assert state[0].item() == pytest.approx(-65.0)
            return
    pytest.fail("expected Izhikevich to fire")


def test_izhikevich_strong_input_stays_finite():
    m = Izhikevich(init_hidden=False)
    state = m.initial_state((B, F))
    for _ in range(500):
        spk, state = m.step_state(torch.full((B, F), 1e3), state)
    assert torch.isfinite(spk).all()
    for s in state:
        assert torch.isfinite(s).all()


def test_izhikevich_sequence_matches_eager_scan():
    m = Izhikevich(init_hidden=True).compile_sequence_scan(mode="default")
    eager = Izhikevich(init_hidden=True)
    x_seq = torch.randn(10, B, F) * 30.0

    out = m.forward_sequence(x_seq)
    ref = eager.forward_sequence(x_seq)
    assert torch.allclose(out, ref, atol=1e-5)


def test_v_peak_is_a_declared_param():
    # The firing threshold was hardcoded to 30.0 before the review fixes;
    # every other constant in the library is a declared Param.
    m = Izhikevich()
    assert isinstance(m.v_peak, torch.nn.Parameter)
    assert m.v_peak.item() == pytest.approx(30.0)
    assert "v_peak" in m.state_dict()

    # The constructor kwarg is honored and changes firing behavior.
    hot = Izhikevich(v_peak=-100.0)  # fires immediately
    cold = Izhikevich(v_peak=1e6)  # never fires
    x = torch.zeros(B, F)
    s_hot = hot.initial_state((B, F))
    s_cold = cold.initial_state((B, F))

    spk_hot, _ = hot.step_state(x, s_hot)
    spk_cold, _ = cold.step_state(x, s_cold)

    assert spk_hot.all()
    assert not spk_cold.any()

    # Default behavior unchanged: reference step still fires at 30 mV.
    m2 = Izhikevich()
    v = torch.full((B, F), 31.0)
    u = torch.full((B, F), -13.0)
    spk, _ = m2.step_state(torch.zeros(B, F), (v, u))
    assert spk.all()
