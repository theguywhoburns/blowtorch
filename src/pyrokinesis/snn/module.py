from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from pyrokinesis import PyroModule, Tensor, extend_specs
from pyrokinesis.util.surrogate_gradients import (
    default_spike_grad,
    straight_through_surrogate,
)

from .reset import ResetHandler, ResetMixin

__all__ = [
    "SnnModule",
    "default_spike_grad",
    "straight_through_surrogate",
]


@extend_specs(reset=ResetHandler)
class SnnModule(ResetMixin, PyroModule):
    """
    Spike-specific behavior on top of PyroModule.

    Adds:
      - ``spike_grad``: surrogate spike function, ``step(x)`` fires when the
        last output crosses zero. Defaults to ``default_spike_grad``
        (hard threshold forward, straight-through identity backward).
        Pass a custom callable via ``spike_grad=...`` at construction to
        select a different surrogate (e.g. a smooth tanh or sigmoid
        approximation).
      - declarative per-state resets via ``StateSpec(reset=...)`` (provided by
        ``ResetMixin``). The reset target is a Params name (string, validated
        at init) or a ``ParamSpec`` object. The framework applies resets to the
        pre-reset state returned by ``_step`` before exposing it, in both hidden
        and explicit modes. Resets are opt-in: by default no state is reset
        unless a ``StateSpec(reset=...)`` is declared.

    SNN step contract: ``_step`` must return a tuple whose first element is
    the spike output (used to trigger declarative resets); the remaining
    elements are the pre-reset state tensors.
    """

    spike_grad: Callable[[Tensor], Tensor]

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
