from __future__ import annotations

import inspect
from typing import Any, ClassVar

from ..specs import Constraint, ParamSpec, Tensor


class ParamMixin:
    """Declared learnable/fixed parameters and the constrained accessor."""

    _cr_param_specs: ClassVar[dict[str, ParamSpec]] = {}
    _cr_param_annotations: ClassVar[dict[str, Any]] = {}
    _cr_constraints: tuple[Constraint, ...]
    _cr_param_constraint_map: dict[str, Constraint | None]

    @classmethod
    def _cr_extra_init_params(cls) -> list[inspect.Parameter]:
        """
        Domain bases can extend the generated constructor signature.
        """
        return []

    def constrain(self, name: str) -> Tensor:
        """
        Return a single constrained parameter by ``Params`` name.

        One dict lookup plus one attribute lookup plus one constraint call;
        only the requested parameter is resolved, so ``_step`` bodies pay
        for what they use and never need positional placeholders for
        params consumed elsewhere (e.g. by declarative resets). Prefer
        this in ``_step``.
        """
        try:
            constraint = self._cr_param_constraint_map[name]
        except KeyError:
            raise KeyError(
                f"unknown Param {name!r}; "
                f"valid: {sorted(self._cr_param_specs)}"
            ) from None
        value = getattr(self, name)
        return value if constraint is None else constraint(value)

    # NOTE: `constrained()` (tuple in Params declaration order) was removed.
    # It was fully dependent on declaration order: inserting a Param silently
    # shifted every positional unpacking downstream of it, and callers that
    # skipped reset-only params needed silent `_` placeholders that shifted
    # too. Use `constrain(name)` per parameter, or `constrained_dict()` for
    # the whole mapping.
    # def constrained(self) -> tuple[Tensor, ...]:
    #     return self._cr_constrained_fn()

    def constrained_dict(self) -> dict[str, Tensor]:
        """
        Constrained parameters keyed by Params declaration name.

        Same values as ``constrain(name)`` per name, so inserting a Param
        cannot silently shift values into the wrong variable. Prefer
        ``constrain(name)`` in ``_step`` — this builds the whole mapping
        every call.
        """
        return {name: self.constrain(name) for name in self._cr_param_specs}
