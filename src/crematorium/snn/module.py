from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from crematorium import CrModule, Tensor, extend_specs
from crematorium.util.surrogate_gradients import (
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
class SnnModule(ResetMixin, CrModule):
    """
    Spike-specific behavior on top of CrModule.

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
    def _cr_extra_init_params(cls) -> list[inspect.Parameter]:
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

        self.spike_grad = spike_grad if spike_grad is not None else default_spike_grad

        # A frozen explicit surrogate silently stops tracking a learnable
        # param it was derived from (e.g. atan_surrogate(tau_L)): refuse
        # loudly. Params opt in declaratively via frozen_surrogate=True.
        if spike_grad is not None:
            for name, spec in self._cr_param_specs.items():
                if spec.frozen_surrogate and getattr(self, name).requires_grad:
                    raise ValueError(
                        f"{type(self).__name__} got an explicit spike_grad "
                        f"with learnable {name!r}: frozen-beta surrogates "
                        f"(e.g. atan_surrogate({name})) silently stop "
                        f"tracking {name} as it trains. Keep {name} fixed, "
                        f"or pass a callable that reads the live value "
                        f"(e.g. a bound method using "
                        f"self.constrain({name!r}))."
                    )
