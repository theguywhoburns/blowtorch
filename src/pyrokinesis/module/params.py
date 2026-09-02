from __future__ import annotations

import inspect
from typing import Any, Callable, ClassVar

from .specs import Constraint, ParamSpec, Tensor


class ParamMixin:
    """Declared learnable/fixed parameters and the constrained accessor."""

    _pk_param_specs: ClassVar[dict[str, ParamSpec]] = {}
    _pk_param_annotations: ClassVar[dict[str, Any]] = {}
    _pk_constraints: tuple[Constraint, ...]
    _pk_constrained_fn: Callable[..., tuple[Tensor, ...]]

    @classmethod
    def _pk_extra_init_params(cls) -> list[inspect.Parameter]:
        """
        Domain bases can extend the generated constructor signature.
        """
        return []

    def constrained(self) -> tuple[Tensor, ...]:
        """
        Return constrained parameters in Params declaration order.

        Hot path:
          - no strings
          - no dict lookups
          - no metadata resolution

        The returned expression is frozen at init time.
        """
        return self._pk_constrained_fn()