import pytest
import torch

from blowtorch.snn import HH, SnnModule

B, F = 4, 8
T = 50


def test_hh_declares_state_specs():
    m = HH()
    assert len(m._bt_state_specs) == 4
    assert not getattr(m, "_bt_reset_exprs", {})
    assert [s.default for s in m._bt_state_specs] == [
        -65.0,
        0.0529,
        0.5961,
        0.3177,
    ]


def test_hh_rejects_invalid_substeps():
    with pytest.raises(ValueError, match="substeps"):
        HH(substeps=0)
    with pytest.raises(ValueError, match="substeps"):
        HH(substeps=1.5)


def test_hh_rejects_invalid_rate_cap():
    with pytest.raises(ValueError, match="rate_cap"):
        HH(rate_cap=0.0)
    with pytest.raises(ValueError, match="rate_cap"):
        HH(rate_cap=-1.0)


def test_hh_stays_at_rest_without_input():
    m = HH(init_hidden=False)
    state = m.initial_state((1,))
    mem0, m0, h0, n0 = (s.item() for s in state)
    for _ in range(200):
        spk, state = m.step_state(torch.zeros(1, 1), state)
    assert not spk.any()
    assert torch.allclose(state[0], torch.tensor([mem0]), atol=5e-3)
    assert torch.allclose(state[1], torch.tensor([m0]), atol=1e-3)
    assert torch.allclose(state[2], torch.tensor([h0]), atol=1e-3)
    assert torch.allclose(state[3], torch.tensor([n0]), atol=1e-3)


def test_hh_fires_under_strong_input():
    m = HH(init_hidden=False, dt=0.05)
    state = m.initial_state((1,))
    fired = 0
    peak = -1e9
    for _ in range(200):
        spk, state = m.step_state(torch.tensor([[60.0]]), state)
        if spk.item() > 0:
            fired += 1
        peak = max(peak, state[0].item())
    assert fired > 0
    assert peak > 0.0


def test_hh_substeps_preserve_dynamics():
    torch.manual_seed(0)
    D, S = 0.02, 2
    x_seq = torch.randn(T, B, F) * 2.0

    coarse = HH(init_hidden=False, dt=D, substeps=S)
    single = HH(init_hidden=False, dt=D / S, substeps=1)
    y_seq = x_seq.repeat_interleave(S, dim=0)

    out_c = coarse.forward_sequence(x_seq)
    out_s = single.forward_sequence(y_seq)

    assert torch.equal(out_c[0], out_s[0][S - 1 :: S])
    for a, b in zip(out_c[1:], out_s[1:]):
        assert torch.equal(a, b)


def test_hh_hidden_explicit_equivalence():
    torch.manual_seed(0)
    hidden = HH(init_hidden=True)
    explicit = HH(init_hidden=False)
    state = explicit.initial_state((B, F))

    for _ in range(10):
        x = torch.randn(B, F) * 5.0
        h_spk = hidden(x)
        e_spk, state = explicit.step_state(x, state)
        assert torch.equal(h_spk, e_spk)
        for i, name in enumerate(("mem", "m", "h", "n")):
            assert torch.allclose(hidden._buffers[name], state[i], atol=1e-6)


def test_hh_compiled_sequence_matches_eager():
    torch._dynamo.reset()
    torch.manual_seed(0)
    x_seq = torch.randn(T, B, F) * 5.0

    compiled = HH(init_hidden=False).compile_sequence_scan(mode="default")
    eager = HH(init_hidden=False)

    out_c = compiled.forward_sequence(x_seq)
    out_e = eager.forward_sequence(x_seq)
    assert isinstance(out_c, tuple) and isinstance(out_e, tuple)
    for a, b in zip(out_c, out_e):
        assert torch.allclose(a, b, atol=1e-5)


def test_hh_gradients_flow_through_gates_and_params():
    m = HH(
        init_hidden=False,
        learnable_gNa=True,
        learnable_gK=True,
        learnable_threshold=True,
    )
    state = m.initial_state((B, F))
    for _ in range(T):
        spk, state = m.step_state(torch.randn(B, F) * 5.0, state)
    spk.mean().backward()
    assert m.gNa.grad is not None and torch.isfinite(m.gNa.grad).all()
    assert m.gK.grad is not None and torch.isfinite(m.gK.grad).all()
    assert m.threshold.grad is not None and torch.isfinite(m.threshold.grad).all()


def _assert_states_finite(state, spk):
    for s in state:
        assert torch.isfinite(s).all()
    assert torch.isfinite(spk).all()


def test_hh_strong_input_stays_finite():
    m = HH(init_hidden=False)
    state = m.initial_state((B, F))
    for _ in range(200):
        spk, state = m.step_state(torch.full((B, F), 1e6), state)
    _assert_states_finite(state, spk)


def test_hh_large_dt_stays_finite():
    m = HH(init_hidden=False, dt=1.0)
    state = m.initial_state((B, F))
    for _ in range(200):
        spk, state = m.step_state(torch.full((B, F), 1e6), state)
    _assert_states_finite(state, spk)


def test_hh_many_substeps_stays_finite():
    m = HH(init_hidden=False, dt=0.5, substeps=16)
    state = m.initial_state((B, F))
    for _ in range(200):
        spk, state = m.step_state(torch.full((B, F), 1e6), state)
    _assert_states_finite(state, spk)


def test_hh_long_rollout_stays_finite():
    m = HH(init_hidden=False, dt=0.1)
    state = m.initial_state((B, F))
    x = torch.randn(B, F) * 5.0
    for _ in range(2000):
        spk, state = m.step_state(x, state)
    _assert_states_finite(state, spk)


def test_hh_strong_input_gradients_finite():
    m = HH(
        init_hidden=False,
        dt=0.5,
        learnable_gNa=True,
        learnable_gK=True,
        learnable_threshold=True,
    )
    state = m.initial_state((B, F))
    spk = None
    for _ in range(50):
        spk, state = m.step_state(torch.full((B, F), 1e6), state)
    (spk.sum() + state[0].sum()).backward()
    assert m.gNa.grad is not None and torch.isfinite(m.gNa.grad).all()
    assert m.gK.grad is not None and torch.isfinite(m.gK.grad).all()
    assert m.threshold.grad is not None and torch.isfinite(m.threshold.grad).all()


def test_hh_spike_grad_default_is_used():
    assert isinstance(HH().spike_grad, object)
    m = HH(init_hidden=False)
    assert callable(m.spike_grad)


def test_hh_importable_from_snn():
    from blowtorch.snn import HH as snn_HH

    assert snn_HH is HH
    assert HH.__mro__[1] is SnnModule


def _hh_rate_ref(x: torch.Tensor, a: float, c: float) -> torch.Tensor:
    mask = x.abs() < 1e-4
    d = torch.where(mask, 1.0, 1.0 - torch.exp(-x / c))
    return torch.where(mask, a * c, a * x / d)


def test_hh_matches_manual_reference():
    torch.manual_seed(0)
    T = 50
    x_seq = torch.randn(T, B, F) * 5.0

    m = HH(init_hidden=False, dt=0.02, substeps=2)
    gNa, gK, gL = m.gNa.item(), m.gK.item(), m.gL.item()
    ENa, EK, EL = m.ENa.item(), m.EK.item(), m.EL.item()
    C, threshold = m.C.item(), m.threshold.item()
    dt = m.dt / m.substeps

    state = m.initial_state((B, F))
    spks_o = []
    for t in range(T):
        z, state = m.step_state(x_seq[t], state)
        spks_o.append(z)
    spks_o = torch.stack(spks_o)

    mem, mm, hh, nn = m.initial_state((B, F))
    spks_r = []
    for t in range(T):
        for _ in range(m.substeps):
            INa = gNa * (mm ** 3) * hh * (mem - ENa)
            IK = gK * (nn ** 4) * (mem - EK)
            IL = gL * (mem - EL)
            mem = mem + (x_seq[t] - INa - IK - IL) / C * dt

            am = _hh_rate_ref(mem + 40, 0.1, 10.0)
            bm = 4.0 * torch.exp(-(mem + 65) / 18)
            ah = 0.07 * torch.exp(-(mem + 65) / 20)
            bh = 1.0 / (1 + torch.exp(-(mem + 35) / 10))
            an = _hh_rate_ref(mem + 55, 0.01, 10.0)
            bn = 0.125 * torch.exp(-(mem + 65) / 80)

            mm = torch.clamp(mm + (am * (1 - mm) - bm * mm) * dt, 0.0, 1.0)
            hh = torch.clamp(hh + (ah * (1 - hh) - bh * hh) * dt, 0.0, 1.0)
            nn = torch.clamp(nn + (an * (1 - nn) - bn * nn) * dt, 0.0, 1.0)

        spk = (mem - threshold > 0).to(mem.dtype)
        spks_r.append(spk)

    assert torch.equal(spks_o, torch.stack(spks_r))
    assert torch.allclose(mem, state[0], atol=1e-5)
    assert torch.allclose(mm, state[1], atol=1e-5)
    assert torch.allclose(hh, state[2], atol=1e-5)
    assert torch.allclose(nn, state[3], atol=1e-5)