"""Gradient parity across the three execution paths (review fix #1).

The library's pitch is "gradients + compile": the manual per-step loop, the
eager sequence scan, and the compiled sequence scan must produce identical
parameter and input gradients. Nothing previously pinned this, so an
``index_copy_``/inductor regression could have shipped silently. The
eager-vs-compiled scan here exercises the chunked ``index_copy_`` scatter
against the flat compiled loop; the step loop is the reference path users
fall back to when debugging.
"""

from __future__ import annotations

import pytest
import torch

from crematorium.snn import LIF

B, F, T = 3, 5, 6


def _make_inputs() -> torch.Tensor:
    g = torch.Generator().manual_seed(1234)
    return torch.randn(T, B, F, generator=g)


def _fresh_net() -> LIF:
    torch.manual_seed(0)
    return LIF(learnable_beta=True, learnable_threshold=True)


def _loss(ys: torch.Tensor, tail: tuple[torch.Tensor, ...]) -> torch.Tensor:
    # Spike sum drives the surrogate path; the final membrane state drives
    # the recurrent beta path, so both gradient routes are exercised.
    return ys.sum() + tail[-1].abs().sum()


def _grads(mode: str) -> dict[str, torch.Tensor]:
    net = _fresh_net()
    x_seq = _make_inputs().requires_grad_(True)

    if mode == "loop":
        state = net.initial_state((B, F))
        ys: list[torch.Tensor] = []
        for t in range(T):
            out = net.forward(x_seq[t], *state)
            assert isinstance(out, tuple)
            ys.append(out[0])
            state = tuple(out[1:])
        loss = _loss(torch.stack(ys), state)
    elif mode == "eager":
        result = net.forward_sequence(x_seq, None)
        assert isinstance(result, tuple)
        loss = _loss(result[0], (result[-1],))
    elif mode == "compiled":
        net.compile_sequence_scan(mode="default")
        result = net.forward_sequence(x_seq, None)
        assert isinstance(result, tuple)
        loss = _loss(result[0], (result[-1],))
    else:  # pragma: no cover
        raise ValueError(mode)

    loss.backward()

    return {
        "beta": net.beta.grad,
        "threshold": net.threshold.grad,
        "x": x_seq.grad,
    }


@pytest.mark.parametrize(
    "mode", ["loop", "eager", "compiled"], ids=["loop", "eager", "compiled"]
)
def test_gradients_exist_and_are_meaningful(mode: str):
    g = _grads(mode)

    for name in ("beta", "threshold", "x"):
        assert g[name] is not None, f"{mode}: {name} grad is None"
        assert torch.isfinite(g[name]).all(), f"{mode}: {name} grad not finite"

    # A zero beta gradient would mean the loss never touched the recurrent
    # path and the parity assertions below would be vacuous.
    assert g["beta"].abs().sum() > 0, f"{mode}: beta grad is all zeros"


@pytest.mark.parametrize("name", ["beta", "threshold", "x"])
def test_loop_and_eager_scan_gradients_match(name: str):
    g_loop = _grads("loop")
    g_eager = _grads("eager")

    assert torch.allclose(g_loop[name], g_eager[name], rtol=1e-6, atol=1e-8), (
        f"step-loop and eager-scan {name} gradients diverge: "
        f"max |diff| = {(g_loop[name] - g_eager[name]).abs().max()}"
    )


@pytest.mark.parametrize("name", ["beta", "threshold", "x"])
def test_eager_and_compiled_scan_gradients_match(name: str):
    g_eager = _grads("eager")
    g_compiled = _grads("compiled")

    assert torch.allclose(g_eager[name], g_compiled[name], rtol=1e-5, atol=1e-7), (
        f"eager-scan and compiled-scan {name} gradients diverge: "
        f"max |diff| = {(g_eager[name] - g_compiled[name]).abs().max()}"
    )


def test_compiled_scan_outputs_match_eager_scan():
    net_e = _fresh_net()
    net_c = _fresh_net()
    x_seq = _make_inputs()

    eager = net_e.forward_sequence(x_seq, None)
    net_c.compile_sequence_scan(mode="default")
    compiled = net_c.forward_sequence(x_seq, None)

    assert torch.allclose(eager[0], compiled[0], rtol=1e-6, atol=1e-7)
    assert torch.allclose(eager[-1], compiled[-1], rtol=1e-6, atol=1e-7)
