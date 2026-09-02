import torch
import pytest

import torch.nn as nn

from pyrokinesis import PyroModule
from pyrokinesis.nn import Sequential
from pyrokinesis.snn import LIF


def test_sequential_rejects_bad_construction():
    with pytest.raises(ValueError, match="at least one layer"):
        Sequential()

    with pytest.raises(TypeError, match=r"must be nn.Module"):
        Sequential(LIF(), "not-a-module")

    with pytest.raises(ValueError, match="init_hidden=True"):
        Sequential(LIF(init_hidden=True), LIF())


def test_sequential_explicit_matches_manual_chain():
    torch.manual_seed(0)
    T, B, F = 5, 3, 8

    net = Sequential(LIF(), LIF())
    l1, l2 = net.layer0, net.layer1

    state = net.initial_state((B, F))
    x_seq = torch.randn(T, B, F)

    seq_out, *final = net.forward_sequence(x_seq, state)

    refs = []
    m1, m2 = state
    for t in range(T):
        out1 = l1.forward(x_seq[t], m1)
        spk1, m1 = out1[0], out1[1]

        out2 = l2.forward(spk1, m2)
        spk2, m2 = out2[0], out2[1]

        refs.append(spk2)

    ref = torch.stack(refs)

    assert torch.allclose(seq_out, ref, atol=1e-6)
    assert torch.allclose(final[0], m1, atol=1e-6)
    assert torch.allclose(final[1], m2, atol=1e-6)


class _TwoOutput(PyroModule):
    class Specs:
        a = PyroModule.OutputSpec()
        b = PyroModule.OutputSpec()
        s = PyroModule.StateSpec()

    def _step(self, x: torch.Tensor, s: torch.Tensor):
        return x, x, s


def test_sequential_rejects_multi_output_layer():
    with pytest.raises(ValueError, match="single-output"):
        Sequential(_TwoOutput())


def test_sequential_hidden_and_sequence():
    torch.manual_seed(0)
    T, B, F = 6, 4, 8

    net = Sequential(LIF(), LIF(), init_hidden=True)

    x = torch.randn(B, F)
    out = net.forward(x)
    assert out.shape == (B, F)
    assert torch.equal(out, out.bool().to(out.dtype))

    assert len(net._pk_state_names) == 2
    assert net._buffers[net._pk_state_names[0]].shape == (B, F)
    assert net._buffers[net._pk_state_names[1]].shape == (B, F)

    x_seq = torch.randn(T, B, F)
    seq = net.forward_sequence(x_seq)
    assert seq.shape == (T, B, F)
    assert torch.equal(seq, seq.bool().to(seq.dtype))


def test_sequential_mixed_stateless():
    torch.manual_seed(0)
    T, B = 5, 4

    net = Sequential(nn.Linear(4, 8), LIF())
    state = net.initial_state((B, 4))
    assert len(state) == 1
    assert state[0].shape == (B, 8)

    x_seq = torch.randn(T, B, 4)
    seq, *_final = net.forward_sequence(x_seq, state)
    assert seq.shape == (T, B, 8)

    net2 = Sequential(nn.Linear(4, 8), LIF(), init_hidden=True)
    out = net2.forward(torch.randn(B, 4))
    assert out.shape == (B, 8)
    assert net2._buffers[net2._pk_state_names[0]].shape == (B, 8)


def test_sequential_state_factories_match_hidden_alloc():
    net = Sequential(nn.Linear(4, 8), LIF(), init_hidden=True)
    net.allocate_like(torch.randn(3, 4))

    state = net.initial_state((3, 4))

    stored = tuple(net._buffers[n] for n in net._pk_state_names)

    for t, s in zip(state, stored, strict=True):
        assert tuple(t.shape) == tuple(s.shape)


def test_sequential_hidden_shape_change_raises():
    net = Sequential(LIF(), init_hidden=True)
    net(torch.randn(3, 4))

    with pytest.raises(ValueError, match="must stay fixed in hidden mode"):
        net(torch.randn(5, 4))


def test_sequential_compiled_matches_eager():
    torch.manual_seed(0)
    T, B = 8, 4

    eager = Sequential(nn.Linear(4, 8), LIF(), LIF())
    net = Sequential(nn.Linear(4, 8), LIF(), LIF())
    net.load_state_dict(eager.state_dict())

    x_seq = torch.randn(T, B, 4)
    e_out, *ef = eager.forward_sequence(x_seq, eager.initial_state((B, 4)))

    net.fast_sequence_()
    c_out, *cf = net.forward_sequence(x_seq, net.initial_state((B, 4)))

    assert torch.allclose(c_out, e_out, atol=1e-6)
    for a, b in zip(ef, cf, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_sequential_compiled_hidden_matches_eager():
    torch.manual_seed(0)
    T, B = 6, 4

    eager = Sequential(nn.Linear(4, 8), LIF(), init_hidden=True)
    net = Sequential(nn.Linear(4, 8), LIF(), init_hidden=True)
    net.load_state_dict(eager.state_dict())

    x_seq = torch.randn(T, B, 4)

    e_out = eager.forward_sequence(x_seq)
    assert isinstance(e_out, torch.Tensor)

    net.fast_sequence_()
    c_out = net.forward_sequence(x_seq)

    assert isinstance(c_out, torch.Tensor)
    assert c_out.shape == (T, B, 8)
    assert torch.allclose(c_out, e_out, atol=1e-6)


def test_sequential_fast_sequence_disables_child_validation():
    net = Sequential(LIF(), nn.Linear(4, 4), LIF())
    assert net.validate is True
    assert net.layer0.validate is True
    assert net.layer2.validate is True

    net.fast_sequence_(compile_scan=False)

    assert net.validate is False
    assert net.layer0.validate is False
    assert net.layer2.validate is False


def test_sequential_trainable():
    torch.manual_seed(0)
    T, B = 6, 4

    net = Sequential(nn.Linear(4, 8), LIF())
    opt = torch.optim.Adam(net.parameters(), lr=0.01)

    x_seq = torch.randn(T, B, 4)
    target = torch.zeros(T, B, 8)

    ys, *_ = net.forward_sequence(x_seq, net.initial_state((B, 4)))
    loss = torch.nn.functional.mse_loss(ys, target)
    loss.backward()

    assert net.layer0.weight.grad is not None
    assert net.layer0.weight.grad.abs().sum() > 0

    w0 = net.layer0.weight.clone()
    opt.step()

    assert not torch.equal(w0, net.layer0.weight)


def test_sequential_state_dict_roundtrip():
    torch.manual_seed(0)

    net = Sequential(nn.Linear(4, 8), LIF(beta=0.7), LIF())
    sd = net.state_dict()
    assert "layer0.weight" in sd
    assert "layer1.beta" in sd

    net2 = Sequential(nn.Linear(4, 8), LIF(beta=0.7), LIF())
    net2.load_state_dict(sd)

    x_seq = torch.randn(2, 3, 4)
    s1 = net.initial_state((3, 4))
    s2 = net2.initial_state((3, 4))
    o1, *_ = net.forward_sequence(x_seq, s1)
    o2, *_ = net2.forward_sequence(x_seq, s2)

    assert torch.allclose(o1, o2, atol=1e-6)


def test_sequential_extra_state_roundtrip():
    h = Sequential(LIF(), init_hidden=True)
    h.forward(torch.randn(3, 4))

    extra = h.get_extra_state()

    h2 = Sequential(LIF(), init_hidden=True)
    h2.set_extra_state(extra)

    assert h2._pk_allocated
    assert torch.allclose(
        h2._buffers[h2._pk_state_names[0]],
        h._buffers[h._pk_state_names[0]],
    )