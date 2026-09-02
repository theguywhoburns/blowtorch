from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from pyrokinesis import ParamSpec, StateSpec, Tensor

if TYPE_CHECKING:
    from .PyroModule import SnnModule

__all__ = [
    "Reset",
    "ResetHandler",
    "ResetSpec",
    "hard_zero_reset",
    "no_reset",
    "subtract_reset",
    "zero_reset",
]


def subtract_reset(mem: Tensor, spk: Tensor, threshold: Tensor) -> Tensor:
    """
    Subtract threshold from fired neurons.
    """
    return torch.addcmul(mem, spk, threshold, value=-1.0)


def zero_reset(mem: Tensor, spk: Tensor, threshold: Tensor) -> Tensor:
    """
    Reset fired neurons to zero multiplicatively.
    """
    return mem * (1.0 - spk)


def hard_zero_reset(mem: Tensor, spk: Tensor, threshold: Tensor) -> Tensor:
    """
    Hard-reset fired neurons to zero using a mask.
    """
    return mem.masked_fill(spk > 0, 0.0)


def no_reset(mem: Tensor, spk: Tensor, threshold: Tensor) -> Tensor:
    """
    No reset.
    """
    return mem


@dataclass(frozen=True)
class ResetSpec:
    """
    Declares how a StateSpec is reset when the output fires.

    Produced by the ``Reset`` factory; see there for the kinds and their
    exact semantics.
    """
    kind: str
    target: str | ParamSpec | None = None
    custom_fn: str | Callable | None = None


class Reset:
    """
    Factory for declarative reset specs.

    Used as a ``StateSpec`` extra, e.g.::

        class Specs:
            spk = SnnModule.OutputSpec(differentiable=False)
            mem = SnnModule.StateSpec(reset=Reset.subtract("threshold"))

    The target can be either the name of a ``Params`` entry (a string,
    validated against the module's Params at init) or a ``ParamSpec`` object
    itself (e.g. a module-level sentinel).

    Reset kinds:

    ``none()``
        Do nothing on spike.
    ``subtract("param")``
        ``state = state - spk * param`` (LIF-style soft reset).
    ``zero()``
        ``state = state * (1 - spk)`` (multiplicative hard reset).
    ``hard_zero()``
        ``state = state.masked_fill(spk > 0, 0)`` (masked hard reset).
    ``set("param")``
        ``state = (1 - spk) * state + spk * param`` (reset to a fixed value).
    ``add("param")``
        ``state = state + spk * param`` (inject a fixed amount, e.g. AdEx
        adaptation).
    ``custom(fn)``
        Apply a per-spike method ``fn(self, state, spk)``. ``fn`` may be the
        method name as a string (``Reset.custom("my_reset")``) or the bound
        method itself. A lambda or a name that is not a method on the module
        is rejected at construction time.
    """

    @staticmethod
    def none() -> ResetSpec:
        return ResetSpec("none")

    @staticmethod
    def subtract(param: str | ParamSpec) -> ResetSpec:
        return ResetSpec("subtract", param)

    @staticmethod
    def zero() -> ResetSpec:
        return ResetSpec("zero")

    @staticmethod
    def hard_zero() -> ResetSpec:
        return ResetSpec("hard_zero")

    @staticmethod
    def set(param: str | ParamSpec) -> ResetSpec:
        return ResetSpec("set", param)

    @staticmethod
    def add(param: str | ParamSpec) -> ResetSpec:
        return ResetSpec("add", param)

    @staticmethod
    def custom(fn: str | Callable) -> ResetSpec:
        return ResetSpec("custom", custom_fn=fn)


class ResetHandler:
    """
    Records the ResetSpec for each state during module construction.
    """

    @staticmethod
    def apply(module: "SnnModule", state_index: int, spec: StateSpec, reset_spec: ResetSpec) -> None:
        if not hasattr(module, "_pk_reset_exprs"):
            module._pk_reset_exprs = {}
        module._pk_reset_exprs[state_index] = reset_spec

    def __init__(self, module: "SnnModule", state_index: int, spec: StateSpec, reset_spec: ResetSpec) -> None:
        self.apply(module, state_index, spec, reset_spec)
