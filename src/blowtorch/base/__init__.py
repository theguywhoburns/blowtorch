from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .collection import collect_metadata, generate_signature
from .constants import ConstantMixin
from .construction import (
    build_params,
    install_constrained,
    remaining_kwargs_error,
)
from .forward import ForwardMixin
from .inputs import InputMixin
from .params import ParamMixin
from .repr import ReprMixin
from .scan import (
    SequenceScanMixin,
    _SEQUENCE_SCAN_CHUNK,
    sequence_scan,
    set_sequence_scan_chunk,
)
from .serialization import SerializationMixin
from .specs import (
    Constant,
    ConstantSpec,
    Constraint,
    Input,
    InputSpec,
    InputTensor,
    OutputSpec,
    Param,
    ParamSpec,
    StateSpec,
    StepOutput,
    Tensor,
    clamp_positive,
    clamp_unit_interval,
    extend_specs,
    identity,
)
from .states import StateMixin
from .validation import (
    ValidationMixin,
    get_validation,
    no_validation,
    set_validation,
)

__all__ = [
    "BlowtorchModule",
    "Tensor",
    "StepOutput",
    "InputTensor",
    "Param",
    "ParamSpec",
    "Constant",
    "ConstantSpec",
    "Input",
    "InputSpec",
    "OutputSpec",
    "StateSpec",
    "extend_specs",
    "identity",
    "clamp_unit_interval",
    "clamp_positive",
    "set_sequence_scan_chunk",
    "set_validation",
    "get_validation",
    "no_validation",
]


# Generic Blowtorch module

class BlowtorchModule(
    InputMixin,
    ParamMixin,
    ConstantMixin,
    StateMixin,
    ForwardMixin,
    SequenceScanMixin,
    ValidationMixin,
    SerializationMixin,
    ReprMixin,
    nn.Module,
):
    """
    Generic declarative stateful step module.

    Subclasses declare:

        class Inputs:
            ...

        class Params:
            ...

        class Specs:
            ...

        def _step(self, x, *state):
            ...

    BlowtorchModule handles:
      - parameter creation
      - learnable / force_learn behavior (an explicit ``learnable_<param>=False``
        overrides a spec-level ``force_learn=True``)
      - constraints
      - hot-path constrained accessor generation
      - hidden / explicit dispatch
      - hidden buffer allocation
      - explicit state factories
      - detach
      - basic sequence scan

    BlowtorchModule is a pure state-threading engine: it makes no assumptions
    about the semantics of the tensors returned by ``_step`` (no notion of a
    "spike" output or of resetting state). Subclasses may hook the raw step
    output through the ``_post_step`` method, which is applied before the
    output is returned or stored; ``SnnModule`` overrides it to apply
    declarative per-state resets.

    Declaring inputs:

      The ``Inputs`` class is optional. Without it, the module behaves as if it
      had a single implicit input ``x``::

          x: Tensor

      With it, each attribute declares one step input, either annotation-only::

          class Inputs:
              x: Tensor
              inh: Tensor

      or with an ``InputSpec`` value for extra semantics::

          class Inputs:
              x: Tensor = BlowtorchModule.Input(primary=True)
              inh: Tensor

      ``_step`` receives the declared inputs positionally, in declaration order
      (``def _step(self, x, inh, *state)``). Public call sites accept a single
      tensor (single-input modules only), a tuple/list of tensors, or a dict
      keyed by input name, in declared order. Internally inputs are always a
      canonical ``tuple[Tensor, ...]``.

      Primary input: the input whose shape/device/dtype hidden buffers and
      ``StateSpec(shape="input")``/``shape=None`` follow. If no input is marked
      ``primary=True`` the first declared input is primary; more than one
      primary input raises at class creation.

      State shapes may reference any input by name::

          class Specs:
              v = BlowtorchModule.StateSpec(shape="inh")

      Input names must be valid non-keyword Python identifiers and must not
      collide with parameter, constant, output, or state names.

    Execution modes:
      - ``init_hidden=True``: the module owns its state. Buffers are allocated
        lazily on first input and returned by ``forward(x)``; pass
        ``allocate_like(x)`` (or ``allocate_like((x, inh))``) ahead of time
        when the first call must not allocate (e.g. before ``torch.compile`` or
        CUDA-graph capture). This is the convenient mode for single-step loops
        over one input stream.
      - ``init_hidden=False`` (default): the caller owns state. ``forward`` is
        a pure function of ``(x, *state)`` returning ``(out, *next_state)``;
        use ``initial_state(...)`` or ``state_factory()`` to seed it. This is
        the mode to use with ``forward_sequence``, which threads state across
        the scan itself.

      Multi-input ``forward_sequence``::

          module.forward_sequence((x_seq, inh_seq), state)

      Hidden-mode multi-input::

          module = Module(init_hidden=True)
          module.allocate_like((x0, inh0))
          out_seq = module.forward_sequence((x_seq, inh_seq))

      ``torch.compile`` caveat: the compiled scan expects a stable input
      structure. Prefer tuple inputs over dicts and keep the input arity fixed,
      since changing it may trigger a recompile.

    Validation: ``validate=True`` (default) checks step-output arity and state
    shapes against the specs on every call. Override per module with the
    ``validate=...`` init parameter, or set the global default with
    ``set_validation(...)`` / ``no_validation()``. Once ``validate=`` is set on
    a module it no longer follows the global toggle.
    """

    # Namespaced declarative helpers.
    Param = Param
    Constant = Constant
    Input = Input
    InputSpec = InputSpec
    OutputSpec = OutputSpec
    StateSpec = StateSpec

    # Class construction / metadata

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        collect_metadata(cls)
        generate_signature(cls)

    # Construction

    def __init__(
        self,
        *,
        size: Optional[int] = None,
        init_hidden: bool = False,
        validate: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """
        ``size`` is metadata only: it is stored (and shown in ``repr``) but does
        not influence allocation, which derives from the input shape or each
        ``StateSpec.shape``. ``init_hidden=True`` owns the state as hidden
        buffers; ``init_hidden=False`` requires the caller to pass state.
        """
        super().__init__()

        if size is not None:
            if not isinstance(size, int) or size <= 0:
                raise ValueError(
                    f"{type(self).__name__} size must be a positive int, got {size!r}"
                )

        self.size = size
        self.init_hidden = init_hidden
        self._validate_override = validate

        self._bt_allocated: bool = False
        self._bt_compiled_sequence: Optional[Callable[..., Any]] = None

        remaining, constraint_fns = build_params(self, kwargs)

        if remaining:
            raise remaining_kwargs_error(self, remaining)

        self._bt_constraints: tuple[Constraint, ...] = tuple(constraint_fns)
        install_constrained(self)
        self._process_spec_extensions()