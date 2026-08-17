from __future__ import annotations

import math
from typing import Callable

import torch

from blowtorch.base import Tensor

__all__ = [
    "default_spike_grad",
    "straight_through_surrogate",
    "sigmoid_surrogate",
    "atan_surrogate",
    "triangular_surrogate",
    "fast_sigmoid_surrogate",
]


def _hard_threshold(x: Tensor) -> Tensor:
    return (x > 0).to(x.dtype)


class _StraightThroughSpike(torch.autograd.Function):
    """
    Forward: hard threshold.

    Backward: identity (straight-through estimator).
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return _hard_threshold(x)

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> Tensor:
        return grad_outputs[0]


def straight_through_surrogate(x: Tensor) -> Tensor:
    """
    Hard threshold forward, straight-through identity backward.

    The default spike function: ``(x > 0)`` in the forward pass, and the
    unmodified gradient in the backward pass.
    """
    if torch.is_grad_enabled():
        return _StraightThroughSpike.apply(x)

    return _hard_threshold(x)


#: Alias kept for backwards compatibility; the default ``spike_grad``.
default_spike_grad = straight_through_surrogate


class _SigmoidSpike(torch.autograd.Function):
    """
    Forward: hard threshold.

    Backward: derivative of ``sigmoid(beta * x)``, i.e.
    ``sigmoid(beta * x) * (1 - sigmoid(beta * x))``.
    """

    @staticmethod
    def forward(ctx, x: Tensor, beta: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.beta = beta
        return _hard_threshold(x)

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> tuple[Tensor, None]:
        (x,) = ctx.saved_tensors
        sig = torch.sigmoid(ctx.beta * x)
        return grad_outputs[0] * sig * (1 - sig), None


def sigmoid_surrogate(beta: float = 1.0) -> Callable[[Tensor], Tensor]:
    """
    Sigmoid surrogate: hard threshold forward, sigmoid-derivative backward.

    Backward gradient: ``sigmoid(beta * x) * (1 - sigmoid(beta * x))``.
    Larger ``beta`` makes the surrogate steeper (narrower gradient window).
    """

    def surrogate(x: Tensor) -> Tensor:
        if torch.is_grad_enabled():
            return _SigmoidSpike.apply(x, beta)

        return _hard_threshold(x)

    return surrogate


class _AtanSpike(torch.autograd.Function):
    """
    Forward: hard threshold.

    Backward: derivative of ``atan(beta * x * pi / 2)``:
    ``beta / 2 / (1 + (pi / 2 * beta * x) ** 2)``.
    """

    @staticmethod
    def forward(ctx, x: Tensor, beta: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.beta = beta
        return _hard_threshold(x)

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> tuple[Tensor, None]:
        (x,) = ctx.saved_tensors
        scale = math.pi / 2 * ctx.beta * x
        return grad_outputs[0] * (ctx.beta / 2 / (1 + scale * scale)), None


def atan_surrogate(beta: float = 1.0) -> Callable[[Tensor], Tensor]:
    """
    Arctangent surrogate (ATan): hard threshold forward, ATan backward.

    Backward gradient: ``beta / 2 / (1 + (pi / 2 * beta * x) ** 2)``.
    Larger ``beta`` makes the surrogate steeper.
    """

    def surrogate(x: Tensor) -> Tensor:
        if torch.is_grad_enabled():
            return _AtanSpike.apply(x, beta)

        return _hard_threshold(x)

    return surrogate


class _TriangularSpike(torch.autograd.Function):
    """
    Forward: hard threshold.

    Backward: triangular / piecewise-linear gradient:
    ``max(0, 1 - beta * |x|)``.
    """

    @staticmethod
    def forward(ctx, x: Tensor, beta: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.beta = beta
        return _hard_threshold(x)

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> tuple[Tensor, None]:
        (x,) = ctx.saved_tensors
        return grad_outputs[0] * (1 - ctx.beta * torch.abs(x)).clamp(min=0), None


def triangular_surrogate(beta: float = 1.0) -> Callable[[Tensor], Tensor]:
    """
    Triangular surrogate: hard threshold forward, triangular backward.

    Backward gradient: ``max(0, 1 - beta * |x|)``. ``beta`` controls the
    width of the gradient window (gradients vanish past ``|x| > 1 / beta``).
    """

    def surrogate(x: Tensor) -> Tensor:
        if torch.is_grad_enabled():
            return _TriangularSpike.apply(x, beta)

        return _hard_threshold(x)

    return surrogate


class _FastSigmoidSpike(torch.autograd.Function):
    """
    Forward: hard threshold.

    Backward: derivative of the fast sigmoid ``x / (1 + |x|)``:
    ``beta / 2 / (1 + |beta * x|) ** 2``.
    """

    @staticmethod
    def forward(ctx, x: Tensor, beta: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.beta = beta
        return _hard_threshold(x)

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> tuple[Tensor, None]:
        (x,) = ctx.saved_tensors
        return (
            grad_outputs[0] * (ctx.beta / 2 / (1 + torch.abs(ctx.beta * x)) ** 2),
            None,
        )


def fast_sigmoid_surrogate(beta: float = 1.0) -> Callable[[Tensor], Tensor]:
    """
    Fast-sigmoid surrogate: hard threshold forward, fast-sigmoid backward.

    Backward gradient: ``beta / 2 / (1 + |beta * x|) ** 2``. Larger ``beta``
    makes the surrogate steeper.
    """

    def surrogate(x: Tensor) -> Tensor:
        if torch.is_grad_enabled():
            return _FastSigmoidSpike.apply(x, beta)

        return _hard_threshold(x)

    return surrogate