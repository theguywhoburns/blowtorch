"""Tests for Sequential container mutation: __delitem__, __setitem__, _layers setter.

Covers hidden-mode buffer lifecycle (orphaned buffers purged, _pk_allocated
reset on state-name change) and _modules bookkeeping (no orphaned children).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pyrokinesis.nn import Sequential
from pyrokinesis.snn import AdEx, LIF


# ---------------------------------------------------------------------------
# __delitem__ in hidden mode
# ---------------------------------------------------------------------------


def test_delitem_hidden_purges_orphaned_buffers():
    """After deleting a layer in hidden mode, orphaned buffers and
    _non_persistent_buffers_set entries must be removed, and
    _pk_allocated must reset to False."""
    torch.manual_seed(0)
    net = Sequential(LIF(), LIF(), init_hidden=True)
    net.forward(torch.randn(3, 4))  # allocate + populate buffers

    # Record the buffer names before deletion
    old_state_names = list(net._pk_state_names)
    assert len(old_state_names) == 2  # l0_mem, l1_mem

    del net[0]

    # State registry rebuilt: only l0_mem remains
    assert net._pk_state_names == ("l0_mem",)

    # Orphaned buffer name must be gone
    for name in old_state_names:
        if name not in net._pk_state_names:
            assert name not in net._buffers
            assert name not in net._non_persistent_buffers_set

    # _pk_allocated must be False so the next forward re-allocates
    assert net._pk_allocated is False


def test_delitem_hidden_state_correct_after_delete():
    """Surviving layer must produce valid output after delete — not crash
    and not read stale state from a removed layer."""
    torch.manual_seed(0)
    net = Sequential(LIF(), LIF(), init_hidden=True)

    x = torch.randn(3, 4)
    out1 = net.forward(x)
    assert out1.shape == (3, 4)

    # Record that a buffer existed for the surviving layer
    _ = net._buffers["l1_mem"]

    del net[0]
    # _pk_allocated is False → forward will re-allocate
    assert net._pk_allocated is False

    # Same batch shape → forward must not crash
    out2 = net.forward(x)
    assert out2.shape == (3, 4)
    assert net._pk_allocated is True

    # The single surviving layer should produce deterministic output
    # (spikes are binary, output is valid)
    assert torch.equal(out2, out2.bool().to(out2.dtype))


def test_delitem_explicit_no_state_corruption():
    """__delitem__ in explicit mode (init_hidden=False) must not corrupt
    output shapes or parameter identity."""
    torch.manual_seed(0)
    net = Sequential(LIF(), LIF())

    state = net.initial_state((3, 4))
    x = torch.randn(3, 4)

    out1, *_ = net.forward(x, *state)
    assert out1.shape == (3, 4)

    # Record surviving layer's identity
    surviving_layer = net.layer1
    del net[0]

    # Output shape must still be correct for the single-layer network
    state2 = net.initial_state((3, 4))
    out2, *_ = net.forward(x, *state2)
    assert out2.shape == (3, 4)

    # The surviving layer must be the same object (weights preserved)
    assert net.layer0 is surviving_layer

    # Running it twice with the same initial state → same output
    state3 = net.initial_state((3, 4))
    out3, *_ = net.forward(x, *state3)
    assert torch.allclose(out2, out3, atol=1e-6)


# ---------------------------------------------------------------------------
# __setitem__ with state arity changes
# ---------------------------------------------------------------------------


def test_setitem_state_arity_change_reallocates():
    """Replacing LIF (1 state) with AdEx (2 states) must reset
    _pk_allocated so the new states get allocated on next forward."""
    torch.manual_seed(0)
    net = Sequential(LIF(), init_hidden=True)
    net.forward(torch.randn(3, 4))  # allocate
    assert net._pk_allocated is True
    assert net._pk_state_names == ("l0_mem",)

    net[0] = AdEx()

    # Registry rebuilt: 2 states now
    assert net._pk_state_names == ("l0_mem", "l0_adapt")
    assert net._pk_allocated is False

    # Forward must not crash — new state gets allocated
    out = net.forward(torch.randn(3, 4))
    assert out.shape == (3, 4)
    assert net._pk_allocated is True


def test_setitem_same_arity_no_reset():
    """Replacing LIF with another LIF (same arity) must rebuild the
    registry but NOT reset _pk_allocated — existing buffers are valid
    for the same state names. Forward must work."""
    torch.manual_seed(0)
    net = Sequential(LIF(), init_hidden=True)
    net.forward(torch.randn(3, 4))
    assert net._pk_allocated is True

    # Replace with a new LIF (same arity = 1 state)
    net[0] = LIF(beta=0.5)
    assert net._pk_state_names == ("l0_mem",)

    # State names unchanged → _pk_allocated stays True (buffers still valid)
    assert net._pk_allocated is True

    out = net.forward(torch.randn(3, 4))
    assert out.shape == (3, 4)
    assert net._pk_allocated is True


def test_setattr_layer_arity_change_reallocates():
    """Replacing a layer via net.layer0 = ... (setattr path) must also
    handle state arity changes correctly."""
    torch.manual_seed(0)
    net = Sequential(LIF(), init_hidden=True)
    net.forward(torch.randn(3, 4))

    net.layer0 = AdEx()

    assert net._pk_state_names == ("l0_mem", "l0_adapt")
    assert net._pk_allocated is False

    out = net.forward(torch.randn(3, 4))
    assert out.shape == (3, 4)
    assert net._pk_allocated is True


# ---------------------------------------------------------------------------
# _modules bookkeeping
# ---------------------------------------------------------------------------


def test_layers_setter_cleans_modules():
    """Assigning net._layers = [...] must remove stale layer entries from
    _modules so children(), parameters(), and state_dict() stay correct."""
    torch.manual_seed(0)
    net = Sequential(LIF(), LIF(), LIF())

    assert len(list(net.children())) == 3
    assert len(list(net.parameters())) == 6  # LIF has 2 params; 3 layers

    net._layers = [LIF()]

    assert len(list(net.children())) == 1
    # LIF has 2 params: beta + threshold
    assert len(list(net.parameters())) == 2


def test_delitem_cleans_modules():
    """del net[0] must remove the corresponding layer from _modules."""
    net = Sequential(LIF(), LIF(), LIF())

    del net[0]

    assert len(list(net.children())) == 2
    # Re-registration: layer0 and layer1 exist, no layer2
    assert "layer0" in net._modules
    assert "layer1" in net._modules
    assert "layer2" not in net._modules


def test_setitem_no_orphaned_modules():
    """net[0] = LIF() must not create extra entries in _modules."""
    net = Sequential(LIF(), LIF(), LIF())

    net[0] = LIF()

    assert len(list(net.children())) == 3
    assert "layer0" in net._modules
    assert "layer1" in net._modules
    assert "layer2" in net._modules


def test_setitem_negative_index_replaces_last():
    """net[-1] = ... must replace the last layer, not register layer-1."""
    net = Sequential(nn.Linear(4, 4), LIF())

    net[-1] = nn.Linear(4, 4)

    assert sorted(net._modules) == ["layer0", "layer1"]
    assert isinstance(net.layer1, nn.Linear)
    assert sum(p.numel() for p in net.parameters()) == 2 * (4 * 4 + 4)


def test_layers_setter_state_dict_correct():
    """After _layers reassignment in hidden mode, state_dict() must contain
    only the new layers' keys — no stale hidden-mode keys."""
    torch.manual_seed(0)
    net = Sequential(LIF(), LIF(), init_hidden=True)
    net.forward(torch.randn(3, 4))

    old_keys = set(net.state_dict().keys())
    # Should have layer0.beta, layer0.threshold, layer1.beta, layer1.threshold
    assert "layer0.beta" in old_keys
    assert "layer1.beta" in old_keys

    net._layers = [LIF()]
    new_keys = set(net.state_dict().keys())

    # Must contain exactly the new single layer's params
    assert "layer0.beta" in new_keys
    assert "layer0.threshold" in new_keys

    # Stale keys must be gone
    assert "layer1.beta" not in new_keys
    assert "layer1.threshold" not in new_keys
