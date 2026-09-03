from __future__ import annotations

import math
from itertools import pairwise

import pytest
import torch

from crematorium.snn.neurons.MCN import MCN
from crematorium.util import atan_surrogate, default_spike_grad

B, F = 4, 8


def test_mcn_fig3f_parameter_defaults():
    m = MCN()
    assert m.tau_B.item() == pytest.approx(2.0)
    assert m.tau_A.item() == pytest.approx(2.0)
    assert m.tau_L.item() == pytest.approx(4.0)
    assert m.gB.item() == pytest.approx(1.0)
    assert m.gA.item() == pytest.approx(1.0)
    assert m.gL.item() == pytest.approx(1.0)
    assert m.threshold.item() == pytest.approx(0.8)
    assert m.spike_grad is default_spike_grad


def test_mcn_multi_input_metadata():
    m = MCN()
    assert m._cr_input_names == ("x_b", "x_a")
    assert m._cr_primary_input_index == 0

    state = m.initial_state_like((torch.randn(2, 3), torch.randn(2, 7)))
    assert state[0].shape == (2, 3)
    assert state[1].shape == (2, 7)
    assert state[2].shape == (2, 3)


def test_mcn_step_hand_computed():
    m = MCN()
    state = m.initial_state_like((torch.zeros(1), torch.zeros(1)))

    # tau_B = tau_A = 2: V = 0.5*V + 0.5*x.
    # m = 1 - 1/4 - 1/4 - 1/4 = 0.25, so u = 0.25*u + 0.25*V_b + 0.25*V_a.
    spk, (V_b, V_a, u) = m.step_state(
        (torch.tensor([2.0]), torch.tensor([4.0])),
        state,
    )
    assert spk.item() == 0.0
    assert torch.allclose(V_b, torch.tensor([1.0]))
    assert torch.allclose(V_a, torch.tensor([2.0]))
    assert torch.allclose(u, torch.tensor([0.75]))


def test_mcn_purity_each_state_reads_only_pre_step_values():
    m = MCN()
    xb = torch.tensor([1.5])
    xa = torch.tensor([1.0])
    zero = m.initial_state_like((xb, xa))

    _, (V_b, V_a, u) = m.step_state((xb, xa), zero)
    assert torch.allclose(V_b, torch.tensor([0.75]))
    assert torch.allclose(V_a, torch.tensor([0.5]))
    assert torch.allclose(u, torch.tensor([0.3125]))

    # Dendritic updates read only their own pre-step potential + input.
    _, (V_b2, _, _) = m.step_state(
        (xb, xa), (torch.tensor([0.0]), torch.tensor([9.0]), torch.tensor([9.0]))
    )
    assert torch.allclose(V_b2, V_b)
    _, (_, V_a2, _) = m.step_state(
        (xb, xa), (torch.tensor([9.0]), torch.tensor([0.0]), torch.tensor([9.0]))
    )
    assert torch.allclose(V_a2, V_a)

    # The soma integrates the current-step dendrites with the pre-step u.
    _, (_, _, u2) = m.step_state(
        (xb, xa), (torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.1]))
    )
    assert torch.allclose(u2, torch.tensor([0.25 * 0.1 + 0.25 * 0.75 + 0.25 * 0.5]))


def test_mcn_fig3f_periodic_firing():
    m = MCN(init_hidden=True)
    xb = torch.full((30, 1, 1), 1.5)
    xa = torch.full((30, 1, 1), 1.0)

    spk = m.forward_sequence((xb, xa))

    times = (spk > 0).nonzero(as_tuple=True)[0].tolist()
    assert len(times) >= 3
    diffs = [b - a for a, b in pairwise(times)]
    assert len(set(diffs)) == 1


def test_mcn_soma_subtract_reset_after_spike():
    m = MCN(init_hidden=False)
    xb = torch.tensor([1.5])
    xa = torch.tensor([1.0])
    state = m.initial_state_like((xb, xa))

    for _ in range(8):
        spk, (V_b, V_a, u) = m.step_state((xb, xa), state)
        state = (V_b, V_a, u)
        if spk.item() == 1.0:
            assert u.item() < 0.8
            return
    pytest.fail("expected MCN to fire")


def test_mcn_surrogate_matches_paper_formula():
    # Paper eq. (31): d(spk)/du = 2*tau_L / (4 + (pi*tau_L*u)**2).
    tau_L = 4.0
    x = torch.linspace(-0.4, 0.4, 9, requires_grad=True)
    atan_surrogate(beta=tau_L)(x).sum().backward()

    expected = 2 * tau_L / (4 + (math.pi * tau_L * x.detach()) ** 2)
    assert torch.allclose(x.grad, expected, atol=1e-6)


def test_mcn_compiled_sequence_matches_eager():
    m = MCN(init_hidden=True).compile_sequence_scan(mode="default")
    eager = MCN(init_hidden=True)
    xb = torch.randn(10, B, F)
    xa = torch.randn(10, B, F)

    out = m.forward_sequence((xb, xa))
    ref = eager.forward_sequence((xb, xa))
    assert torch.allclose(out, ref, atol=1e-5)


def test_mcn_fast_sequence_matches_eager():
    m = MCN(init_hidden=True).fast_sequence_()
    eager = MCN(init_hidden=True)
    xb = torch.randn(10, B, F)
    xa = torch.randn(10, B, F)

    out = m.forward_sequence((xb, xa))
    ref = eager.forward_sequence((xb, xa))
    assert torch.allclose(out, ref, atol=1e-5)


def test_mcn_hidden_explicit_equivalence():
    torch.manual_seed(0)
    hidden = MCN(init_hidden=True)
    explicit = MCN(init_hidden=False)
    state = explicit.initial_state_like((torch.randn(B, F), torch.randn(B, F)))

    for _ in range(5):
        xb = torch.randn(B, F)
        xa = torch.randn(B, F)
        h = hidden((xb, xa))
        e, state = explicit.step_state((xb, xa), state)
        assert torch.equal(h, e)
        assert torch.allclose(hidden._buffers["u"], state[2], atol=1e-6)


def test_mcn_strong_input_stays_finite():
    m = MCN(init_hidden=False)
    state = m.initial_state_like((torch.randn(B, F), torch.randn(B, F)))
    for _ in range(500):
        spk, state = m.step_state(
            (torch.full((B, F), 1e3), torch.full((B, F), 1e3)),
            state,
        )
    assert torch.isfinite(spk).all()
    for s in state:
        assert torch.isfinite(s).all()
