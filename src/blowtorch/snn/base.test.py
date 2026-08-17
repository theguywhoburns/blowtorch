from __future__ import annotations

import inspect

import pytest
import torch

from blowtorch import BlowtorchModule
from blowtorch.snn import (
    Reset,
    SnnModule,
    default_spike_grad,
    hard_zero_reset,
    no_reset,
    subtract_reset,
    zero_reset,
)

B, F = 4, 8


class _Probe(SnnModule):
    """Records which spike_grad it called via the return value."""

    class Specs:
        o = SnnModule.OutputSpec(differentiable=False)
        s = SnnModule.StateSpec()

    def _step(self, x, s):
        return self.spike_grad(x - 1), s


class _ParamSnn(SnnModule):
    """SnnModule subclass with a declared param (for signature checks)."""

    class Params:
        beta: float = SnnModule.Param(0.5)

    class Specs:
        o = SnnModule.OutputSpec()
        s = SnnModule.StateSpec()

    def _step(self, x, s):
        return x, x


class _ProbeParams:
    p = SnnModule.Param(2.0)


class _ResetProbe(SnnModule):
    """Single-output/single-state probe; reset configured by subclass Specs."""

    Params = _ProbeParams

    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        mem = SnnModule.StateSpec()

    def _step(self, x, mem):
        return self.spike_grad(x), mem


class _NoResetProbe(_ResetProbe):
    class Specs:
        mem = SnnModule.StateSpec(reset=Reset.none())


class _SubtractResetProbe(_ResetProbe):
    class Specs:
        mem = SnnModule.StateSpec(reset=Reset.subtract(_ProbeParams.p))


class _ZeroResetProbe(_ResetProbe):
    class Specs:
        mem = SnnModule.StateSpec(reset=Reset.zero())


class _HardZeroResetProbe(_ResetProbe):
    class Specs:
        mem = SnnModule.StateSpec(reset=Reset.hard_zero())


class _SetResetProbe(_ResetProbe):
    class Specs:
        mem = SnnModule.StateSpec(reset=Reset.set(_ProbeParams.p))


class _AddResetProbe(_ResetProbe):
    class Specs:
        mem = SnnModule.StateSpec(reset=Reset.add(_ProbeParams.p))


class _StringTargetProbe(_ResetProbe):
    class Specs:
        mem = SnnModule.StateSpec(reset=Reset.subtract("p"))


def _custom_reset(mem, spk):
    return mem * (1 + spk)


class _CustomResetProbe(_ResetProbe):
    class Specs:
        mem = SnnModule.StateSpec(reset=Reset.custom(_custom_reset))

    def _custom_reset(self, mem, spk):
        return _custom_reset(mem, spk)


def test_reset_mechanism_pure_functions():
    mem = torch.tensor(1.5)
    spk = torch.tensor(1.0)
    th = torch.tensor(1.0)

    assert torch.allclose(
        subtract_reset(mem, spk, th),
        torch.addcmul(mem, spk, th, value=-1.0),
    )
    assert torch.allclose(subtract_reset(mem, spk, th), torch.tensor(0.5))

    assert torch.equal(zero_reset(mem, spk, th), mem * (1.0 - spk))
    assert torch.equal(hard_zero_reset(mem, spk, th), mem.masked_fill(spk > 0, 0.0))
    assert torch.equal(no_reset(mem, spk, th), mem)


def test_hard_zero_reset_matches_zero_reset_on_binary():
    mem = torch.tensor([[0.5, 2.0, 1.0], [3.0, 0.0, 1.5]])
    spk = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    th = torch.tensor(1.0)
    assert torch.equal(
        hard_zero_reset(mem, spk, th),
        zero_reset(mem, spk, th),
    )


def test_declarative_subtract_matches_utility():
    m = _SubtractResetProbe()
    mem = torch.tensor(1.5)
    spk = torch.tensor(1.0)
    _, (next_mem,) = m.step_state(spk, (mem,))
    assert torch.allclose(
        next_mem,
        subtract_reset(mem, spk, torch.tensor(2.0)),
    )


def test_default_spike_grad_is_default_spike_grad():
    m = SnnModule()
    assert m.spike_grad is default_spike_grad


def test_snn_module_is_blowtorch_module():
    assert issubclass(SnnModule, BlowtorchModule)


def test_signature_includes_spike_grad_only():
    sig = inspect.signature(_ParamSnn)
    names = list(sig.parameters)
    assert names[-2:] == ["spike_grad", "kwargs"]
    assert "reset_mechanism" not in names
    assert sig.parameters["spike_grad"].default is None
    assert sig.parameters["spike_grad"].kind == inspect.Parameter.KEYWORD_ONLY


def test_default_spike_grad_hard_threshold():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 3.0])
    out = default_spike_grad(x)
    assert torch.equal(out, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))
    assert out.dtype == x.dtype


def test_default_spike_grad_no_grad_path():
    with torch.no_grad():
        out = default_spike_grad(torch.tensor([-1.0, 1.0]))
    assert torch.equal(out, torch.tensor([0.0, 1.0]))
    assert out.requires_grad is False
    assert out.grad_fn is None


def test_default_spike_grad_straight_through_backward():
    x = torch.tensor([-1.0, 2.0], requires_grad=True)
    out = default_spike_grad(x - 1)
    out.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x.grad))


def test_spike_grad_override_used():
    probe = _Probe(init_hidden=True, spike_grad=lambda x: torch.ones_like(x))
    out = probe(torch.randn(B, F))
    assert torch.equal(out, torch.ones_like(out))


def test_no_reset_mechanism_attribute_and_default_spike_grad():
    m = SnnModule()
    assert not hasattr(m, "reset_mechanism")
    assert m.spike_grad is default_spike_grad


# ----------------------------------------------------------------------
# Declarative reset system
# ----------------------------------------------------------------------


def test_reset_none_leaves_state_unchanged():
    m = _NoResetProbe()
    spk, (next_mem,) = m.step_state(
        torch.tensor([-1.0, 2.0]),
        (torch.full((2,), 3.0),),
    )
    assert torch.equal(spk, torch.tensor([0.0, 1.0]))
    assert torch.equal(next_mem, torch.full((2,), 3.0))


def test_reset_subtract():
    m = _SubtractResetProbe()
    spk, (next_mem,) = m.step_state(
        torch.tensor([-1.0, 2.0]),
        (torch.full((2,), 3.0),),
    )
    assert torch.equal(spk, torch.tensor([0.0, 1.0]))
    assert torch.allclose(next_mem, torch.tensor([3.0, 1.0]))


def test_reset_zero():
    m = _ZeroResetProbe()
    _, (next_mem,) = m.step_state(
        torch.tensor([-1.0, 2.0]),
        (torch.full((2,), 3.0),),
    )
    assert torch.allclose(next_mem, torch.tensor([3.0, 0.0]))


def test_reset_hard_zero():
    m = _HardZeroResetProbe()
    _, (next_mem,) = m.step_state(
        torch.tensor([-1.0, 2.0]),
        (torch.full((2,), 3.0),),
    )
    assert torch.allclose(next_mem, torch.tensor([3.0, 0.0]))


def test_reset_set():
    m = _SetResetProbe()
    _, (next_mem,) = m.step_state(
        torch.tensor([-1.0, 2.0]),
        (torch.full((2,), 3.0),),
    )
    assert torch.allclose(next_mem, torch.tensor([3.0, 2.0]))


def test_reset_add():
    m = _AddResetProbe()
    _, (next_mem,) = m.step_state(
        torch.tensor([-1.0, 2.0]),
        (torch.full((2,), 3.0),),
    )
    assert torch.allclose(next_mem, torch.tensor([3.0, 5.0]))


def test_reset_custom():
    m = _CustomResetProbe()
    _, (next_mem,) = m.step_state(
        torch.tensor([-1.0, 2.0]),
        (torch.full((2,), 3.0),),
    )
    assert torch.allclose(next_mem, torch.tensor([3.0, 6.0]))


def test_reset_target_string_name():
    m = _StringTargetProbe()
    _, (next_mem,) = m.step_state(
        torch.tensor([-1.0, 2.0]),
        (torch.full((2,), 3.0),),
    )
    assert torch.allclose(next_mem, torch.tensor([3.0, 1.0]))


def test_reset_target_param_spec_object():
    m = _SubtractResetProbe()
    assert m._bt_reset_exprs[0].target is _ProbeParams.p


def test_reset_unknown_param_name_raises():
    class _Bad(_ResetProbe):
        class Specs:
            mem = SnnModule.StateSpec(reset=Reset.subtract("nope"))

    with pytest.raises(ValueError, match="Unknown param name"):
        _Bad()


def test_reset_unknown_param_spec_raises():
    class _Bad(_ResetProbe):
        class Specs:
            mem = SnnModule.StateSpec(reset=Reset.subtract(SnnModule.Param(5.0)))

    with pytest.raises(ValueError, match="not found in Params"):
        _Bad()


def test_state_spec_without_reset_leaves_state_unchanged():
    m = _ResetProbe()
    mem = torch.full((2,), 3.0)
    spk, (next_mem,) = m.step_state(torch.tensor([-1.0, 2.0]), (mem,))
    assert torch.equal(spk, torch.tensor([0.0, 1.0]))
    assert torch.equal(next_mem, mem)


def test_reset_hidden_explicit_equivalence():
    torch.manual_seed(0)
    hidden = _SubtractResetProbe(init_hidden=True)
    explicit = _SubtractResetProbe(init_hidden=False)
    state = explicit.initial_state((B, F))

    for _ in range(5):
        x = torch.randn(B, F)
        h_spk = hidden(x)
        e_spk, state = explicit.step_state(x, state)
        assert torch.equal(h_spk, e_spk)
        assert torch.allclose(hidden._buffers["mem"], state[0], atol=1e-6)