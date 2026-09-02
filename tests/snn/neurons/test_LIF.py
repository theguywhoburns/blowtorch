from __future__ import annotations

import inspect

import pytest
import torch

from pyrokinesis.snn import Reset, SnnModule, subtract_reset
from pyrokinesis.snn.neurons.LIF import LIF

B, F = 4, 8
T = 5


class _ZeroLIF(LIF):
    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        mem = SnnModule.StateSpec(reset=Reset.zero())


class _HardZeroLIF(LIF):
    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        mem = SnnModule.StateSpec(reset=Reset.hard_zero())


class _NoResetLIF(LIF):
    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        mem = SnnModule.StateSpec(reset=Reset.none())


def test_lif_step_hand_computed():
    lif = LIF(beta=0.5, threshold=1.0)
    mem = torch.zeros(1)

    spk, (mem,) = lif.step_state(torch.tensor([1.5]), (mem,))
    assert torch.allclose(spk, torch.tensor([1.0]))
    assert torch.allclose(mem, torch.tensor([0.5]))

    spk, (mem,) = lif.step_state(torch.tensor([0.5]), (mem,))
    assert torch.allclose(spk, torch.tensor([0.0]))
    assert torch.allclose(mem, torch.tensor([0.75]))

    spk, (mem,) = lif.step_state(torch.tensor([1.0]), (mem,))
    assert torch.allclose(spk, torch.tensor([1.0]))
    assert torch.allclose(mem, torch.tensor([0.375]))


def test_lif_parameter_defaults():
    lif = LIF()
    assert lif.beta.item() == pytest.approx(0.9)
    assert lif.threshold.item() == 1.0
    assert isinstance(lif.beta, torch.nn.Parameter)
    assert isinstance(lif.threshold, torch.nn.Parameter)
    assert lif.beta.requires_grad is False
    assert lif.threshold.requires_grad is False
    beta, threshold = lif.constrained()
    assert beta.item() == pytest.approx(0.9)
    assert threshold.item() == 1.0


def test_lif_beta_constraint_unit_interval():
    m1 = LIF(beta=2.0, learnable_beta=True)
    assert m1.constrained()[0].item() == pytest.approx(1.0)
    m2 = LIF(beta=-0.5, learnable_beta=True)
    assert m2.constrained()[0].item() == pytest.approx(0.0)
    m3 = LIF(beta=0.9, learnable_beta=True)
    assert m3.constrained()[0].item() == pytest.approx(0.9)


def test_lif_threshold_constraint_positive():
    m1 = LIF(threshold=0.0, learnable_threshold=True)
    assert m1.constrained()[1].item() == pytest.approx(1e-6)
    m2 = LIF(threshold=-2.0, learnable_threshold=True)
    assert m2.constrained()[1].item() == pytest.approx(1e-6)
    m3 = LIF(threshold=1.0, learnable_threshold=True)
    assert m3.constrained()[1].item() == pytest.approx(1.0)


def test_lif_constraints_skipped_when_fixed():
    m = LIF(beta=2.0, threshold=0.5)
    beta, threshold = m.constrained()
    assert beta.item() == pytest.approx(2.0)
    assert threshold.item() == pytest.approx(0.5)

    mem = torch.zeros(1)
    _, (mem,) = m.step_state(torch.tensor([0.5]), (mem,))
    assert torch.allclose(mem, torch.tensor([0.5]))


def test_lif_reset_kind_subtract_default():
    lif = LIF(beta=0.5, threshold=1.0)
    mem = torch.zeros(1)
    _, (mem,) = lif.step_state(torch.tensor([1.5]), (mem,))
    assert torch.allclose(mem, torch.tensor([0.5]))
    assert torch.allclose(subtract_reset(torch.tensor(1.5), torch.tensor(1.0), torch.tensor(1.0)), torch.tensor(0.5))


def test_lif_reset_kind_zero():
    lif = _ZeroLIF(beta=0.5, threshold=1.0)
    mem = torch.zeros(1)
    _, (mem,) = lif.step_state(torch.tensor([1.5]), (mem,))
    assert torch.allclose(mem, torch.tensor([0.0]))


def test_lif_reset_kind_hard_zero():
    lif = _HardZeroLIF(beta=0.5, threshold=1.0)
    mem = torch.zeros(1)
    spk, (mem,) = lif.step_state(torch.tensor([1.5]), (mem,))
    assert spk.item() == 1.0
    assert mem.item() == 0.0


def test_lif_reset_kind_none():
    lif = _NoResetLIF(beta=0.5, threshold=1.0)
    mem = torch.zeros(1)
    _, (mem,) = lif.step_state(torch.tensor([1.5]), (mem,))
    assert torch.allclose(mem, torch.tensor([1.5]))


def test_lif_spike_grad_override():
    lif = LIF(beta=0.5, threshold=1.0, spike_grad=lambda x: torch.ones_like(x))
    mem = torch.zeros(1)
    spk, (mem,) = lif.step_state(torch.tensor([0.1]), (mem,))
    assert spk.item() == 1.0
    assert torch.allclose(mem, torch.tensor([-0.9]))

    lif0 = LIF(beta=0.5, threshold=1.0, spike_grad=lambda x: torch.zeros_like(x))
    mem = torch.zeros(1)
    spk, (mem,) = lif0.step_state(torch.tensor([0.5]), (mem,))
    assert spk.item() == 0.0
    assert torch.allclose(mem, torch.tensor([0.5]))


def test_lif_spike_grad_override_and_declarative_reset():
    lif = LIF(beta=0.5, threshold=1.0, spike_grad=lambda x: torch.ones_like(x))
    assert lif._pk_reset_exprs[0].kind == "subtract"
    assert lif._pk_reset_exprs[0].target == "threshold"

    mem = torch.zeros(1)
    spk, (mem,) = lif.step_state(torch.tensor([0.1]), (mem,))
    assert spk.item() == 1.0
    assert torch.allclose(mem, torch.tensor([-0.9]))


def test_lif_reset_uses_constrained_threshold():
    # A learnable out-of-range threshold is clamped by clamp_positive; the
    # reset must subtract the constrained value (what _step spikes against),
    # not the raw parameter.
    lif = LIF(
        beta=0.0,
        threshold=-2.0,
        learnable_threshold=True,
        spike_grad=lambda x: torch.ones_like(x),
    )
    assert lif.constrained()[1].item() == pytest.approx(1e-6)

    mem = torch.zeros(1)
    spk, (mem,) = lif.step_state(torch.tensor([0.1]), (mem,))
    assert spk.item() == 1.0
    assert torch.allclose(mem, torch.tensor([0.1 - 1e-6]))


def test_lif_learnable_gradients_flow():
    lif = LIF(init_hidden=False, learnable_beta=True, learnable_threshold=True)
    mem = lif.initial_state((B, F))
    x = torch.randn(B, F)
    spk, (mem,) = lif.step_state(x, mem)
    spk.mean().backward()
    assert lif.beta.grad is not None and torch.isfinite(lif.beta.grad).all()
    assert lif.threshold.grad is not None and torch.isfinite(lif.threshold.grad).all()


def test_lif_input_gradient_flows():
    lif = LIF(init_hidden=False, beta=0.5, threshold=1.0)
    mem = lif.initial_state((B, F))
    x = torch.randn(B, F, requires_grad=True)
    spk, (mem,) = lif.step_state(x, mem)
    spk.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_lif_spike_output_binary():
    lif = LIF(init_hidden=True)
    out = lif(torch.randn(B, F))
    assert torch.equal(out, out.bool().to(out.dtype))


def test_lif_hidden_explicit_equivalence():
    torch.manual_seed(0)
    hidden = LIF(init_hidden=True)
    explicit = LIF(init_hidden=False)
    state = explicit.initial_state((B, F))

    for _ in range(T):
        x = torch.randn(B, F)
        h_spk = hidden(x)
        e_spk, state = explicit.step_state(x, state)
        assert torch.allclose(hidden._buffers["mem"], state[0], atol=1e-6)
        assert torch.equal(h_spk, e_spk)


def test_lif_hidden_sequence_equals_explicit_sequence():
    torch.manual_seed(0)
    hidden = LIF(init_hidden=True)
    explicit = LIF(init_hidden=False)
    x_seq = torch.randn(T, B, F)

    h_seq = hidden.forward_sequence(x_seq)
    e_seq, final = explicit.forward_sequence(x_seq)
    assert torch.allclose(h_seq, e_seq, atol=1e-6)
    assert torch.allclose(hidden._buffers["mem"], final, atol=1e-6)


def test_lif_dtype_follows_input():
    lif = LIF(init_hidden=True)
    lif(torch.randn(B, F, dtype=torch.float64))
    assert lif._buffers["mem"].dtype == torch.float64
    assert lif._buffers["spk"].dtype == torch.float64

    lif2 = LIF(init_hidden=True)
    lif2(torch.randn(B, F, dtype=torch.float16))
    assert lif2._buffers["mem"].dtype == torch.float16
    assert lif2._buffers["spk"].dtype == torch.float16


def test_lif_hidden_spk_buffer_non_differentiable():
    lif = LIF(init_hidden=True)
    x = torch.randn(B, F, requires_grad=True)
    lif(x)
    assert lif._buffers["spk"].requires_grad is False
    assert lif._buffers["mem"].requires_grad is True

    lif2 = LIF(init_hidden=False)
    state = lif2.initial_state((B, F))
    spk, (state,) = lif2.step_state(x, state)
    assert spk.requires_grad is True


def test_lif_no_grad_forward_is_hard_threshold():
    lif = LIF(init_hidden=False, beta=0.5, threshold=1.0)
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 3.0])
    with torch.no_grad():
        spk, (_mem,) = lif.step_state(x, (torch.zeros(5),))
    expected = (x - 1.0 > 0).to(x.dtype)
    assert torch.equal(spk, expected)
    assert spk.requires_grad is False
    assert spk.grad_fn is None


def test_lif_repr():
    r = repr(LIF(init_hidden=True))
    assert r.startswith("LIF")
    assert "init_hidden=True" in r

    r_size = repr(LIF(size=16))
    assert "size=16" in r_size

    r_none = repr(LIF())
    assert "size=" not in r_none


def test_lif_signature_full():
    sig = inspect.signature(LIF)
    expected = [
        "self",
        "size",
        "init_hidden",
        "validate",
        "beta",
        "learnable_beta",
        "force_learn_beta",
        "beta_constraint",
        "threshold",
        "learnable_threshold",
        "force_learn_threshold",
        "threshold_constraint",
        "spike_grad",
        "kwargs",
    ]
    assert list(sig.parameters) == expected


def test_lif_size_validation():
    for bad in (0, -5, 3.5):
        with pytest.raises(ValueError, match="size must be a positive int"):
            LIF(size=bad)

    lif = LIF(size=None, init_hidden=True)
    out = lif(torch.randn(2, 3))
    assert out.shape == (2, 3)


def test_lif_integrate_over_threshold_with_beta():
    lif = LIF(init_hidden=False, beta=1.0, threshold=3.0)
    mem = torch.zeros(1)

    spk, (mem,) = lif.step_state(torch.tensor([1.0]), (mem,))
    assert spk.item() == 0.0
    assert torch.allclose(mem, torch.tensor([1.0]))

    spk, (mem,) = lif.step_state(torch.tensor([1.0]), (mem,))
    assert spk.item() == 0.0
    assert torch.allclose(mem, torch.tensor([2.0]))

    spk, (mem,) = lif.step_state(torch.tensor([1.0]), (mem,))
    assert spk.item() == 0.0
    assert torch.allclose(mem, torch.tensor([3.0]))

    spk, (mem,) = lif.step_state(torch.tensor([1.0]), (mem,))
    assert spk.item() == 1.0
    assert torch.allclose(mem, torch.tensor([1.0]))


def test_lif_extreme_params_stays_finite():
    for beta in (0.0, 1e-6, 1.0):
        lif = LIF(init_hidden=False, beta=beta, threshold=1e-6)
        state = lif.initial_state((B, F))
        spk = None
        for _ in range(500):
            spk, state = lif.step_state(torch.full((B, F), 1e6), state)
        assert torch.isfinite(state[0]).all()
        assert torch.isfinite(spk).all()


def test_lif_strong_input_stays_finite():
    lif = LIF(init_hidden=False, beta=0.99, threshold=1.0)
    state = lif.initial_state((B, F))
    for _ in range(2000):
        spk, state = lif.step_state(torch.full((B, F), 1e6), state)
    assert torch.isfinite(state[0]).all()
    assert torch.isfinite(spk).all()

def test_lif_matches_norse_reference():
    pytest.importorskip("norse")
    from norse.torch.functional.lif import (
        LIFFeedForwardState,
        LIFParameters,
        lif_feed_forward_step,
    )

    torch.manual_seed(0)
    T = 100
    beta, threshold = 0.9, 1.0
    m = LIF(init_hidden=False, beta=beta, threshold=threshold)

    p = LIFParameters(
        tau_syn_inv=torch.as_tensor(1.0),
        tau_mem_inv=torch.as_tensor(1.0 - beta),
        v_leak=torch.as_tensor(0.0),
        v_th=torch.as_tensor(threshold),
        v_reset=torch.as_tensor(0.0),
        method="super",
    )

    # Sub-threshold: with the norse input scaled by 1/(1-beta) the two
    # discrete recurrences are identical, so membranes match exactly.
    x_seq = torch.randn(T, B, F) * 0.02
    state = m.initial_state((B, F))
    ns = LIFFeedForwardState(v=state[0], i=torch.zeros(B, F))
    mems_o, mems_n = [], []
    for t in range(T):
        z, state = m.step_state(x_seq[t], state)
        zz, ns = lif_feed_forward_step(x_seq[t] / (1.0 - beta), ns, p, dt=1.0)
        assert torch.equal(z, zz)
        mems_o.append(state[0])
        mems_n.append(ns.v)
    assert torch.allclose(torch.stack(mems_o), torch.stack(mems_n), atol=1e-6)

    # Firing: trajectories agree until the first spike (ours uses a subtract
    # reset, norse a set reset, so post-spike evolution legitimately differs).
    x_seq = torch.randn(T, B, F) * 5.0
    state = m.initial_state((B, F))
    ns = LIFFeedForwardState(v=state[0], i=torch.zeros(B, F))
    first_o = first_n = None
    for t in range(T):
        z, state = m.step_state(x_seq[t], state)
        zz, ns = lif_feed_forward_step(x_seq[t] / (1.0 - beta), ns, p, dt=1.0)
        if first_o is None and z.any():
            first_o = t
        if first_n is None and zz.any():
            first_n = t
    assert first_o == first_n
    assert first_o is not None
