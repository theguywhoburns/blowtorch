"""Regression: MCN + learnable_tau_L + atan_surrogate freezes beta at construction.

The MCN docstring says: if you change tau_L, pass spike_grad=atan_surrogate(tau_L).
A user who does that AND sets learnable_tau_L=True follows the docs perfectly —
then tau_L drifts every optimizer step while the backward pass silently keeps
the construction-time beta. Neither loud fail nor user error: stale math.

PASSES if either fix lands:
  A) surrogate reads live tau_L (bound method / dynamic beta), or
  B) construction raises loudly for learnable tau_L + frozen closure.
FAILS on current dev (silent stale gradient).

Run:  pytest tests/snn/neurons/test_mcn_stale_surrogate.py -q
      python tests/snn/neurons/test_mcn_stale_surrogate.py   (exit 1 if stale)
"""
import sys

import torch

from pyrokinesis.snn import MCN
from pyrokinesis.util import atan_surrogate

TAU_L0 = 4.0     # construction-time value, baked into the closure
TAU_L_NOW = 8.0  # post-training value
X_VAL = 0.3      # probe point; any nonzero works


def probe_grad(fn, x_val=X_VAL):
    """d(surrogate)/dx at x_val via a real backward pass."""
    x = torch.tensor([x_val], requires_grad=True)
    fn(x).backward(torch.ones(1))
    return x.grad.detach().clone()


def surrogate_tracks_or_raises() -> bool:
    try:
        mcn = MCN(learnable_tau_L=True, spike_grad=atan_surrogate(TAU_L0))
    except Exception as e:  # fix B: PEBCAK-loud at construction
        print(f"FIXED (loud fail at construction): {type(e).__name__}: {e}")
        return True

    with torch.no_grad():
        mcn.tau_L.data.fill_(TAU_L_NOW)  # simulate optimizer drift

    actual = probe_grad(mcn.spike_grad)          # what training actually gets
    live = probe_grad(atan_surrogate(TAU_L_NOW))  # what the paper wants now

    if torch.allclose(actual, live, atol=1e-6):
        print("FIXED (surrogate tracks live tau_L)")
        return True

    frozen = probe_grad(atan_surrogate(TAU_L0))
    if not torch.allclose(actual, frozen, atol=1e-6):
        raise AssertionError(
            f"gradient matches neither live nor frozen beta: {actual} — "
            f"something else changed, re-read the code"
        )
    print(
        f"STALE: tau_L={TAU_L_NOW} but backward uses beta={TAU_L0}: "
        f"grad={actual.item():.6f}, correct={live.item():.6f}"
    )
    return False


def test_mcn_surrogate_not_stale():
    assert surrogate_tracks_or_raises(), (
        "stale surrogate gradient under learnable tau_L: silent wrong math, "
        "violates the library's own loud-fail philosophy"
    )


if __name__ == "__main__":
    sys.exit(0 if surrogate_tracks_or_raises() else 1)
