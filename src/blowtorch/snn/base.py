from __future__ import annotations

import inspect
import types
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from blowtorch.base import (
    BlowtorchModule,
    ParamSpec,
    StateSpec,
    Tensor,
    extend_specs,
    identity,
)

__all__ = [
    "SnnModule",
    "Reset",
    "ResetSpec",
    "subtract_reset",
    "zero_reset",
    "hard_zero_reset",
    "no_reset",
    "default_spike_grad",
    "straight_through_surrogate",
]


# Reset mechanisms (public pure utilities; resets are declared via Specs)


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


# Declarative reset spec


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
        if not hasattr(module, "_bt_reset_exprs"):
            module._bt_reset_exprs = {}
        module._bt_reset_exprs[state_index] = reset_spec

    def __init__(self, module: "SnnModule", state_index: int, spec: StateSpec, reset_spec: ResetSpec) -> None:
        self.apply(module, state_index, spec, reset_spec)


# Default surrogate spike lives in blowtorch.util.surrogate_gradients.

from blowtorch.util.surrogate_gradients import (
    default_spike_grad,
    straight_through_surrogate,
)


@extend_specs(reset=ResetHandler)
class SnnModule(BlowtorchModule):
    """
    Spike-specific behavior on top of BlowtorchModule.

    Adds:
      - ``spike_grad``: surrogate spike function, ``step(x)`` fires when the
        last output crosses zero. Defaults to ``default_spike_grad``
        (hard threshold forward, straight-through identity backward).
        Pass a custom callable via ``spike_grad=...`` at construction to
        select a different surrogate (e.g. a smooth tanh or sigmoid
        approximation).
      - declarative per-state resets via ``StateSpec(reset=...)``. The reset
        target is a Params name (string, validated at init) or a ``ParamSpec``
        object. The framework applies resets to the pre-reset state returned
        by ``_step`` before exposing it, in both hidden and explicit modes.
        Resets are opt-in: by default no state is reset unless a
        ``StateSpec(reset=...)`` is declared.
    """

    spike_grad: Callable[[Tensor], Tensor]

    _bt_reset_exprs: dict[int, ResetSpec]

    @classmethod
    def _extra_init_params(cls) -> list[inspect.Parameter]:
        return [
            inspect.Parameter(
                "spike_grad",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[Callable[[Tensor], Tensor]],
            ),
        ]

    def __init__(
        self,
        *,
        spike_grad: Optional[Callable[[Tensor], Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.spike_grad = (
            spike_grad
            if spike_grad is not None
            else default_spike_grad
        )

    def _install_reset_fn(self) -> None:
        """
        Code-generate ``_bt_apply_resets(pre_state, spk) -> tuple`` from the
        ResetSpecs recorded on ``_bt_reset_exprs`` during construction.
        """
        reset_exprs = getattr(self, "_bt_reset_exprs", {})

        if not reset_exprs:
            self._bt_apply_resets = lambda pre_state, spk: pre_state
            return

        lines: list[str] = []

        for i in range(len(self._bt_state_specs)):
            lines.append(f"state_{i} = pre_state[{i}]")

        for i, reset_spec in reset_exprs.items():
            if reset_spec.kind == "none":
                continue

            if reset_spec.kind in ("subtract", "set", "add"):
                param_name = self._resolve_param_name(reset_spec.target)
                param_value = self._constraint_expr(param_name)

                if reset_spec.kind == "subtract":
                    lines.append(f"state_{i} = state_{i} - spk * {param_value}")
                elif reset_spec.kind == "set":
                    lines.append(
                        f"state_{i} = (1 - spk) * state_{i} + spk * {param_value}"
                    )
                else:
                    lines.append(f"state_{i} = state_{i} + spk * {param_value}")
            elif reset_spec.kind == "zero":
                lines.append(f"state_{i} = state_{i} * (1 - spk)")
            elif reset_spec.kind == "hard_zero":
                lines.append(f"state_{i} = state_{i}.masked_fill(spk > 0, 0.0)")
            elif reset_spec.kind == "custom":
                fn = reset_spec.custom_fn

                if isinstance(fn, str):
                    fn_name = fn
                elif callable(fn):
                    fn_name = getattr(fn, "__name__", None)
                    if fn_name is None or fn_name == "<lambda>":
                        raise ValueError(
                            "Reset.custom requires a named method or a "
                            f"method-name string, got {fn!r}"
                        )
                else:
                    raise ValueError(
                        "Reset.custom requires a method-name string or a "
                        f"named callable, got {fn!r}"
                    )

                target = getattr(self, fn_name, None)

                if not callable(target):
                    raise ValueError(
                        f"Reset.custom target {fn_name!r} is not a method on "
                        f"{type(self).__name__}"
                    )

                lines.append(f"state_{i} = self.{fn_name}(state_{i}, spk)")
            else:
                raise ValueError(f"Unknown reset kind {reset_spec.kind!r}")

        lines.append(
            "return ("
            + ", ".join(f"state_{i}" for i in range(len(self._bt_state_specs)))
            + ",)"
        )

        # Safety: the body is exec'd, but every interpolated fragment is
        # framework-controlled — param names (validated as non-keyword
        # identifiers at class creation), state indices derived from specs, and
        # custom reset fn names checked to be actual attributes above.
        src = f"def _bt_apply_resets(self, pre_state, spk):\n    " + "\n    ".join(lines)

        ns: dict[str, Any] = {}
        exec(src, ns)

        self._bt_apply_resets = types.MethodType(ns["_bt_apply_resets"], self)

    def _constraint_expr(self, param_name: str) -> str:
        """
        Expression resolving ``param_name`` to its constrained value.

        Matches the accessor ``_bt_constrained`` generates in
        ``BlowtorchModule._install_constrained`` so resets act on the same
        values ``_step`` spikes against.
        """
        param_names = tuple(self._bt_param_specs.keys())
        idx = param_names.index(param_name)
        constraint = self._bt_constraints[idx]

        if constraint is identity:
            return f"self.{param_name}"

        return f"self._bt_constraint_{idx}(self.{param_name})"

    def _resolve_param_name(self, target: str | ParamSpec | None) -> str:
        if isinstance(target, str):
            if target not in self._bt_param_specs:
                raise ValueError(f"Unknown param name {target!r}")
            return target

        for name, spec in self._bt_param_specs.items():
            if spec is target:
                return name

        raise ValueError("Reset target ParamSpec not found in Params")