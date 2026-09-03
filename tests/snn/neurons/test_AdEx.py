from __future__ import annotations

import pytest
import torch

from crematorium import clamp_positive
from crematorium.snn import AdEx, SnnModule

B, F = 4, 8
T = 5


class _AdExReference(SnnModule):
    class Params:
        tau_m = SnnModule.Param(10.0, constraint=clamp_positive)
        tau_w = SnnModule.Param(100.0, constraint=clamp_positive)
        V_rest = SnnModule.Param(0.0)
        V_reset = SnnModule.Param(0.0)
        V_T = SnnModule.Param(1.0, constraint=clamp_positive)
        delta_T = SnnModule.Param(0.5, constraint=clamp_positive)
        a = SnnModule.Param(0.1, constraint=clamp_positive)
        b = SnnModule.Param(0.2, constraint=clamp_positive)

    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        mem = SnnModule.StateSpec()
        adapt = SnnModule.StateSpec()

    def _step(self, x, mem, adapt):
        tau_m = self.constrain("tau_m")
        tau_w = self.constrain("tau_w")
        V_rest = self.constrain("V_rest")
        V_T = self.constrain("V_T")
        delta_T = self.constrain("delta_T")
        a = self.constrain("a")
        dv = (
            -(mem - V_rest) + delta_T * torch.exp((mem - V_T) / delta_T) - adapt + x
        ) / tau_m
        dw = (a * (mem - V_rest) - adapt) / tau_w
        mem = mem + dv
        adapt = adapt + dw
        spk = self.spike_grad(mem - V_T)
        return spk, mem, adapt


def test_adex_declares_multi_state_resets():
    m = AdEx()
    assert m._cr_reset_exprs[0].kind == "set"
    assert m._cr_reset_exprs[0].target == "V_reset"
    assert m._cr_reset_exprs[1].kind == "add"
    assert m._cr_reset_exprs[1].target == "b"


def test_adex_reset_semantics_match_reference():
    torch.manual_seed(0)
    m = AdEx(init_hidden=False)
    x_seq = torch.randn(T, B, F) * 5.0
    spk_seq, final_mem, final_adapt = m.forward_sequence(x_seq)

    ref = _AdExReference(init_hidden=False)
    mem, adapt = ref.initial_state((B, F))
    tau_m = ref.constrain("tau_m")
    tau_w = ref.constrain("tau_w")
    V_rest = ref.constrain("V_rest")
    V_reset = ref.constrain("V_reset")
    V_T = ref.constrain("V_T")
    delta_T = ref.constrain("delta_T")
    a = ref.constrain("a")
    b = ref.constrain("b")

    ref_spks = []
    for t in range(T):
        x = x_seq[t]
        dv = (
            -(mem - V_rest) + delta_T * torch.exp((mem - V_T) / delta_T) - adapt + x
        ) / tau_m
        dw = (a * (mem - V_rest) - adapt) / tau_w
        mem = mem + dv
        adapt = adapt + dw
        spk = ref.spike_grad(mem - V_T)
        ref_spks.append(spk)
        mem = mem.masked_fill(spk > 0, V_reset)
        adapt = adapt + b * spk

    ref_spks = torch.stack(ref_spks)
    assert torch.allclose(spk_seq, ref_spks, atol=1e-6)
    assert torch.allclose(final_mem, mem, atol=1e-6)
    assert torch.allclose(final_adapt, adapt, atol=1e-6)


def test_adex_hidden_explicit_equivalence():
    torch.manual_seed(0)
    hidden = AdEx(init_hidden=True)
    explicit = AdEx(init_hidden=False)
    state = explicit.initial_state((B, F))

    for _ in range(T):
        x = torch.randn(B, F) * 5.0
        h_spk = hidden(x)
        e_spk, state = explicit.step_state(x, state)
        assert torch.equal(h_spk, e_spk)
        assert torch.allclose(hidden._buffers["mem"], state[0], atol=1e-6)
        assert torch.allclose(hidden._buffers["adapt"], state[1], atol=1e-6)


def test_adex_compiled_sequence_matches_eager():
    torch._dynamo.reset()
    torch.manual_seed(0)
    x_seq = torch.randn(T, B, F) * 5.0

    compiled = AdEx(init_hidden=False).compile_sequence_scan(mode="default")
    eager = AdEx(init_hidden=False)

    out_c = compiled.forward_sequence(x_seq)
    out_e = eager.forward_sequence(x_seq)
    assert isinstance(out_c, tuple) and isinstance(out_e, tuple)
    assert torch.allclose(out_c[0], out_e[0], atol=1e-5)
    assert torch.allclose(out_c[1], out_e[1], atol=1e-5)
    assert torch.allclose(out_c[2], out_e[2], atol=1e-5)


def test_adex_gradients_flow_through_both_states():
    m = AdEx(init_hidden=False, learnable_a=True, learnable_b=True)
    state = m.initial_state((B, F))
    spk = None
    for _ in range(T):
        spk, state = m.step_state(torch.randn(B, F) * 5.0, state)
    spk.mean().backward()
    assert m.a.grad is not None and torch.isfinite(m.a.grad).all()
    assert m.b.grad is not None and torch.isfinite(m.b.grad).all()


def test_adex_large_input_stays_finite():
    m = AdEx(init_hidden=False)
    state = m.initial_state((B, F))
    for _ in range(T):
        spk, state = m.step_state(torch.full((B, F), 1e3), state)
    assert torch.isfinite(state[0]).all()
    assert torch.isfinite(state[1]).all()
    assert torch.isfinite(spk).all()


def test_adex_extreme_membrane_stays_finite():
    m = AdEx(init_hidden=False)
    mem, adapt = m.initial_state((B, F))
    mem = torch.full_like(mem, 1e3)
    adapt = torch.full_like(adapt, 1e3)
    spk, (mem, adapt) = m.step_state(torch.full((B, F), 1e3), (mem, adapt))
    assert torch.isfinite(mem).all()
    assert torch.isfinite(adapt).all()
    assert torch.isfinite(spk).all()


def test_adex_large_input_gradients_finite():
    m = AdEx(init_hidden=False, learnable_delta_T=True, learnable_tau_m=True)
    state = m.initial_state((B, F))
    spk = None
    for _ in range(T):
        spk, state = m.step_state(torch.full((B, F), 1e3), state)
    (spk.sum() + state[0].sum() + state[1].sum()).backward()
    assert m.delta_T.grad is not None and torch.isfinite(m.delta_T.grad).all()
    assert m.tau_m.grad is not None and torch.isfinite(m.tau_m.grad).all()


def test_adex_matches_norse_lif_adex():
    pytest.importorskip("norse")
    from norse.torch.functional.lif_adex import (
        LIFAdExFeedForwardState,
        LIFAdExParameters,
        lif_adex_feed_forward_step,
    )

    torch.manual_seed(0)
    T = 60
    x_seq = torch.randn(T, B, F) * 2.0

    m = AdEx(init_hidden=False)
    p = LIFAdExParameters(
        adaptation_current=torch.as_tensor(m.a.item()),
        adaptation_spike=torch.as_tensor(m.b.item()),
        delta_T=torch.as_tensor(m.delta_T.item()),
        tau_ada_inv=torch.as_tensor(1.0 / m.tau_w.item()),
        tau_syn_inv=torch.as_tensor(1.0),
        tau_mem_inv=torch.as_tensor(1.0 / m.tau_m.item()),
        v_leak=torch.as_tensor(m.V_rest.item()),
        v_th=torch.as_tensor(m.V_T.item()),
        v_reset=torch.as_tensor(m.V_reset.item()),
        method="super",
    )

    state = m.initial_state((B, F))
    ns = LIFAdExFeedForwardState(
        v=state[0],
        i=torch.zeros(B, F),
        a=state[1],
    )
    spks_o, spks_n = [], []
    mems_o, mems_n = [], []
    adas_o, adas_n = [], []
    for t in range(T):
        z, state = m.step_state(x_seq[t], state)
        zz, ns = lif_adex_feed_forward_step(x_seq[t], ns, p, dt=1.0)
        spks_o.append(z)
        spks_n.append(zz)
        mems_o.append(state[0])
        mems_n.append(ns.v)
        adas_o.append(state[1])
        adas_n.append(ns.a)

    assert torch.equal(torch.stack(spks_o), torch.stack(spks_n))
    assert torch.allclose(torch.stack(mems_o), torch.stack(mems_n), atol=1e-5)
    assert torch.allclose(torch.stack(adas_o), torch.stack(adas_n), atol=1e-5)
