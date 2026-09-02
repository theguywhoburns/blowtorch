from __future__ import annotations

import inspect
import types
from typing import Any, Callable, Optional

from pyrokinesis import (
    PyroModule,
    ParamSpec,
    StepOutput,
    Tensor,
    extend_specs,
    identity,
)
from pyrokinesis.util.surrogate_gradients import (
    default_spike_grad,
    straight_through_surrogate,
)

from .reset import ResetHandler, ResetSpec

__all__ = [
    "SnnModule",
    "default_spike_grad",
    "straight_through_surrogate",
]


@extend_specs(reset=ResetHandler)
class SnnModule(PyroModule):
    """
    Spike-specific behavior on top of PyroModule.

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

    SNN step contract: ``_step`` must return a tuple whose first element is
    the spike output (used to trigger declarative resets); the remaining
    elements are the pre-reset state tensors.
    """

    spike_grad: Callable[[Tensor], Tensor]

    _pk_reset_exprs: dict[int, ResetSpec]

    @classmethod
    def _pk_extra_init_params(cls) -> list[inspect.Parameter]:
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

    def _pk_process_spec_extensions(self) -> None:
        """
        Let the base class dispatch generic StateSpec extras, then install the
        SNN-specific reset machinery.
        """
        super()._pk_process_spec_extensions()
        self._pk_install_reset_fn()

    def _pk_post_step(self, out: StepOutput) -> StepOutput:
        """
        Apply declarative resets to the state after ``_step``.

        Assumes the first tensor in the tuple is the spike output; the
        remaining tensors are the pre-reset states.
        """
        if isinstance(out, tuple) and len(out) > 0:
            spk = out[0]
            pre_state = out[1:]
            next_state = self._pk_apply_resets(pre_state, spk)
            return (spk, *next_state)
        return out

    def _pk_install_reset_fn(self) -> None:
        """
        Code-generate ``_pk_apply_resets(pre_state, spk) -> tuple`` from the
        ResetSpecs recorded on ``_pk_reset_exprs`` during construction.
        """
        reset_exprs = getattr(self, "_pk_reset_exprs", {})

        if not reset_exprs:
            self._pk_apply_resets = lambda pre_state, spk: pre_state
            return

        lines: list[str] = []

        for i in range(len(self._pk_state_specs)):
            lines.append(f"state_{i} = pre_state[{i}]")

        for i, reset_spec in reset_exprs.items():
            if reset_spec.kind == "none":
                continue

            if reset_spec.kind in ("subtract", "set", "add"):
                param_name = self._pk_resolve_param_name(reset_spec.target)
                param_value = self._pk_constraint_expr(param_name)

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
            + ", ".join(f"state_{i}" for i in range(len(self._pk_state_specs)))
            + ",)"
        )

        # Safety: the body is exec'd, but every interpolated fragment is
        # framework-controlled — param names (validated as non-keyword
        # identifiers at class creation), state indices derived from specs, and
        # custom reset fn names checked to be actual attributes above.
        src = "def _pk_apply_resets(self, pre_state, spk):\n    " + "\n    ".join(lines)

        ns: dict[str, Any] = {}
        exec(src, ns)

        self._pk_apply_resets = types.MethodType(ns["_pk_apply_resets"], self)

    def _pk_constraint_expr(self, param_name: str) -> str:
        """
        Expression resolving ``param_name`` to its constrained value.

        Matches the accessor ``_pk_constrained`` generates in
        ``PyroModule._install_constrained`` so resets act on the same
        values ``_step`` spikes against.
        """
        param_names = tuple(self._pk_param_specs.keys())
        idx = param_names.index(param_name)
        constraint = self._pk_constraints[idx]

        if constraint is identity:
            return f"self.{param_name}"

        return f"self._pk_constraint_{idx}(self.{param_name})"

    def _pk_resolve_param_name(self, target: str | ParamSpec | None) -> str:
        if isinstance(target, str):
            if target not in self._pk_param_specs:
                raise ValueError(f"Unknown param name {target!r}")
            return target

        for name, spec in self._pk_param_specs.items():
            if spec is target:
                return name

        raise ValueError(
            "Reset target ParamSpec not found in Params: a ParamSpec target "
            "must be the exact object declared in this module's Params "
            f"(targets are matched by identity; prefer the param name as a "
            f"string, e.g. Reset.subtract({sorted(self._pk_param_specs)[0]!r}))"
        )