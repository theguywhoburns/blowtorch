from __future__ import annotations

from typing import Any, Callable, ClassVar, Optional, Self

import torch.nn as nn

from .collection import collect_metadata, generate_signature
from .construction import (
    build_params,
    _cr_install_constrained,
    remaining_kwargs_error,
)
from .mixins import (
    ConstantMixin,
    ForwardMixin,
    InputMixin,
    ParamMixin,
    ReprMixin,
    SequenceScanMixin,
    SerializationMixin,
    StateMixin,
    ValidationMixin,
    _SEQUENCE_SCAN_CHUNK,
    sequence_scan,
    set_sequence_scan_chunk,
)
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
from .mixins.validation import (
    get_validation,
    no_validation,
    set_validation,
)

__all__ = [
    "_SEQUENCE_SCAN_CHUNK",
    "Constant",
    "ConstantSpec",
    "CrModule",
    "Input",
    "InputSpec",
    "InputTensor",
    "OutputSpec",
    "Param",
    "ParamSpec",
    "StateSpec",
    "StepOutput",
    "Tensor",
    "clamp_positive",
    "clamp_unit_interval",
    "extend_specs",
    "get_validation",
    "identity",
    "no_validation",
    "sequence_scan",
    "set_sequence_scan_chunk",
    "set_validation",
]


# Generic crematorium module


class CrModule(
    SequenceScanMixin,
    SerializationMixin,
    ReprMixin,
    ForwardMixin,
    StateMixin,
    ValidationMixin,
    ConstantMixin,
    ParamMixin,
    InputMixin,
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

    CrModule handles:
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

    CrModule is a pure state-threading engine: it makes no assumptions
    about the semantics of the tensors returned by ``_step``. Subclasses may hook the raw step
    output through frozen ``_cr_hook_post__*`` chain entries, which run at the end of
    ``_cr_forward_explicit`` before the output is returned or stored;
    ``SnnModule`` contributes one to apply declarative per-state resets.

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
              x: Tensor = CrModule.Input(primary=True)
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
              v = CrModule.StateSpec(shape="inh")

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

    # Frozen hook chains (collected once per class, child-first; a subclass
    # redefining the same hook name replaces the parent entry). Hot paths
    # iterate these tuples instead of resolving super()/getattr per call.
    _cr_hook_post_steps: ClassVar[tuple[Callable[..., Any], ...]] = ()
    _cr_hook_specs_steps: ClassVar[tuple[Callable[..., Any], ...]] = ()

    # Class construction / metadata

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        collect_metadata(cls)
        generate_signature(cls)
        # Collect frozen hook chains. Iterate MRO child-first so that if a
        # subclass redefines the same _cr_hook_<point>__<tag> method, it
        # *replaces* the parent's entry (replace semantics, not additive).
        post_hooks: dict[str, Callable[..., Any]] = {}
        spec_hooks: dict[str, Callable[..., Any]] = {}
        for klass in cls.__mro__:  # child-first
            for name, fn in klass.__dict__.items():
                if name.startswith("_cr_hook_post__"):
                    post_hooks.setdefault(name, fn)
                elif name.startswith("_cr_hook_specs__"):
                    spec_hooks.setdefault(name, fn)
        cls._cr_hook_post_steps = tuple(post_hooks.values())
        cls._cr_hook_specs_steps = tuple(spec_hooks.values())

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

        self._cr_allocated: bool = False
        self._cr_compiled_sequence: Optional[Callable[..., Any]] = None

        remaining, constraint_fns = build_params(self, kwargs)

        if remaining:
            raise remaining_kwargs_error(self, remaining)

        self._cr_constraints: tuple[Constraint, ...] = tuple(constraint_fns)
        _cr_install_constrained(self)
        self._cr_process_spec_extensions()

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """
        Deepcopy everything except the compiled scan.

        ``_cr_compiled_sequence`` is a closure over the original module;
        functions are atomic under ``copy.deepcopy``, so without this the
        copy would silently route its compiled path through the original's
        dynamics. The copy recompiles lazily on the next
        ``compile_sequence_scan()`` / ``fast_sequence_()`` call.
        """
        from copy import deepcopy

        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        new.__dict__.update(deepcopy(self.__dict__, memo))
        new._cr_compiled_sequence = None
        return new

    def __copy__(self) -> Self:
        """
        Shallow-copy, but drop the compiled scan (same reason as
        ``__deepcopy__``: the closure binds the original module).
        """
        cls = type(self)
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        new._cr_compiled_sequence = None
        return new
