from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import types
from typing import Any

from crematorium import ParamSpec, StateSpec, StepOutput, identity
from crematorium.module.mixins.states import StateMixin

_CR_RESET_CACHE: dict[str, Any] = {}

if TYPE_CHECKING:
    from .module import SnnModule

__all__ = [
    "Reset",
    "ResetHandler",
    "ResetMixin",
    "ResetSpec",
]


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
    def apply(
        module: "SnnModule", state_index: int, spec: StateSpec, reset_spec: ResetSpec
    ) -> None:
        if not hasattr(module, "_cr_reset_exprs"):
            module._cr_reset_exprs = {}
        module._cr_reset_exprs[state_index] = reset_spec

    def __init__(
        self,
        module: "SnnModule",
        state_index: int,
        spec: StateSpec,
        reset_spec: ResetSpec,
    ) -> None:
        self.apply(module, state_index, spec, reset_spec)


class ResetMixin(StateMixin):
    """Declarative per-state reset mixin — SNN-specific, not generic CrModule.

    Extends ``StateMixin`` so spec metadata resolves by inheritance, not by
    ``SnnModule`` base order. Contributes two frozen hooks (collected once
    per class in ``CrModule.__init_subclass__``, never resolved per call):
    ``_cr_hook_specs__rst`` installs the reset fn after spec dispatch,
    ``_cr_hook_post__rst`` applies resets in the post-step chain. Diamond
    is safe without reinit guards: neither mixin defines ``__init__`` (C3
    keeps ``StateMixin`` once in the MRO) and both dispatch and install
    are idempotent.
    """

    _cr_reset_exprs: dict[int, ResetSpec]

    def _cr_hook_specs__rst(self) -> None:
        self._cr_install_reset_fn()  # type: ignore[attr-defined]

    def _cr_hook_post__rst(self, out: StepOutput) -> StepOutput:
        if isinstance(out, tuple) and len(out) > 0:
            # Position-aware split: _step returns (outputs..., states...),
            # so resets apply to out[n_outputs:], never to extra outputs.
            n_out = len(self._cr_output_names)  # type: ignore[attr-defined]
            spk = out[0]
            pre_state = out[n_out:]
            return (*out[:n_out], *self._cr_apply_resets(pre_state, spk))  # type: ignore[attr-defined]
        return out

    def _cr_install_reset_fn(self) -> None:
        reset_exprs = getattr(self, "_cr_reset_exprs", {})
        if not reset_exprs:
            self._cr_apply_resets = lambda pre_state, spk: pre_state  # type: ignore[attr-defined]
            return
        lines: list[str] = []
        for i in range(len(self._cr_state_specs)):  # type: ignore[attr-defined]
            lines.append(f"state_{i} = pre_state[{i}]")
        for i, reset_spec in reset_exprs.items():
            if reset_spec.kind == "none":
                continue
            if reset_spec.kind in ("subtract", "set", "add"):
                param_name = self._cr_resolve_param_name(reset_spec.target)  # type: ignore[attr-defined]
                param_value = self._cr_constraint_expr(param_name)  # type: ignore[attr-defined]
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
                            f"Reset.custom requires a named method or a method-name string, got {fn!r}"
                        )
                else:
                    raise ValueError(
                        f"Reset.custom requires a method-name string or a named callable, got {fn!r}"
                    )
                target = getattr(self, fn_name, None)
                if not callable(target):
                    raise ValueError(
                        f"Reset.custom target {fn_name!r} is not a method on {type(self).__name__}"
                    )
                lines.append(f"state_{i} = self.{fn_name}(state_{i}, spk)")
            else:
                raise ValueError(f"Unknown reset kind {reset_spec.kind!r}")
        lines.append(
            "return ("
            + ", ".join(f"state_{i}" for i in range(len(self._cr_state_specs)))
            + ",)"
        )  # type: ignore[attr-defined]
        src = "def _cr_apply_resets(self, pre_state, spk):\n    " + "\n    ".join(lines)
        cached = _CR_RESET_CACHE.get(src)
        if cached is not None:
            self._cr_apply_resets = types.MethodType(cached, self)  # type: ignore[attr-defined]
            return
        ns: dict[str, object] = {}
        exec(src, ns)
        _CR_RESET_CACHE[src] = ns["_cr_apply_resets"]
        self._cr_apply_resets = types.MethodType(ns["_cr_apply_resets"], self)  # type: ignore[attr-defined]

    def _cr_constraint_expr(self, param_name: str) -> str:
        param_names = tuple(self._cr_param_specs.keys())  # type: ignore[attr-defined]
        idx = param_names.index(param_name)
        constraint = self._cr_constraints[idx]  # type: ignore[attr-defined]
        if constraint is identity:
            return f"self.{param_name}"
        return f"self._cr_constraint_{idx}(self.{param_name})"

    def _cr_resolve_param_name(self, target: str | ParamSpec | None) -> str:
        if isinstance(target, str):
            if target not in self._cr_param_specs:  # type: ignore[attr-defined]
                raise ValueError(f"Unknown param name {target!r}")
            return target
        for name, spec in self._cr_param_specs.items():  # type: ignore[attr-defined]
            if spec is target:
                return name
        # Example name for the hint must not IndexError on a module with an
        # empty Params block.
        param_names = tuple(self._cr_param_specs.keys())  # type: ignore[attr-defined]
        example = param_names[0] if param_names else "<param-name>"
        raise ValueError(
            "Reset target ParamSpec not found in Params: a ParamSpec target "
            "must be the exact object declared in this module's Params "
            "(targets are matched by identity; prefer the param name as a "
            f"string, e.g. Reset.subtract({example!r}))"
        )
