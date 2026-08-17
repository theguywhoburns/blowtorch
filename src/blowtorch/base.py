from __future__ import annotations

import inspect
import keyword
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional, TypeVar, Union, overload

import torch
import torch.nn as nn

__all__ = [
    "BlowtorchModule",
    "Param",
    "ParamSpec",
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


Tensor = torch.Tensor
StepOutput = tuple[Tensor, ...]

Constraint = Callable[[Tensor], Tensor]

# In eager scans, batch this many steps into a single index_copy_ scatter so
# peak memory stays at input + output (no per-step (B, F) list held for stack).
_SEQUENCE_SCAN_CHUNK = 8


def set_sequence_scan_chunk(chunk: int) -> None:
    """
    Set the eager-scan chunk size used by ``forward_sequence``.

    Chunking batches several steps into one ``index_copy_`` scatter. Larger
    values reduce dispatch overhead but increase peak memory; the optimal size
    depends on hardware, dtype, and batch/feature dims. Pass ``1`` to disable
    chunking.
    """
    global _SEQUENCE_SCAN_CHUNK

    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
        raise ValueError(f"scan chunk must be a positive int, got {chunk!r}")

    _SEQUENCE_SCAN_CHUNK = chunk


# Constraints

def identity(t: Tensor) -> Tensor:
    """No constraint."""
    return t


def clamp_unit_interval(t: Tensor) -> Tensor:
    """Clamp to [0, 1]."""
    return torch.clamp(t, 0.0, 1.0)


def clamp_positive(t: Tensor) -> Tensor:
    """Clamp to be strictly positive."""
    return torch.clamp(t, min=1e-6)


# Global validation toggle

_GLOBAL_VALIDATE = True


def set_validation(enabled: bool) -> None:
    """Set the global default for validation."""
    global _GLOBAL_VALIDATE
    _GLOBAL_VALIDATE = bool(enabled)


def get_validation() -> bool:
    """Return the current global validation default."""
    return _GLOBAL_VALIDATE


@contextmanager
def no_validation():
    """Context manager that disables global validation temporarily.

    Modules constructed with ``validate=None`` follow this toggle, so
    wrapping a hot loop in ``no_validation()`` skips their per-forward checks.
    """
    prev = get_validation()
    set_validation(False)
    try:
        yield
    finally:
        set_validation(prev)


def _floating_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    State tensors should generally be floating point.
    If an integer dtype is requested, promote to default float dtype.
    """
    return dtype if dtype.is_floating_point else torch.get_default_dtype()


# Declarative parameter / state specs

@dataclass(frozen=True)
class ParamSpec:
    default: Any = None
    learnable: bool = False
    force_learn: bool = False
    constraint: Constraint = identity
    dtype: Any = None


T = TypeVar("T")


@overload
def Param(
    default: Any = None,
    *,
    learnable: bool = False,
    force_learn: bool = False,
    constraint: Constraint = identity,
    dtype: None = None,
) -> Any: ...


@overload
def Param(
    default: Any = None,
    *,
    learnable: bool = False,
    force_learn: bool = False,
    constraint: Constraint = identity,
    dtype: type[T],
) -> T: ...


@overload
def Param(
    default: Any = None,
    *,
    learnable: bool = False,
    force_learn: bool = False,
    constraint: Constraint = identity,
    dtype: Any = None,
) -> Any: ...


def Param(
    default: Any = None,
    *,
    learnable: bool = False,
    force_learn: bool = False,
    constraint: Constraint = identity,
    dtype: Any = None,
) -> Any:
    """
    Declarative parameter field.

    With ``dtype=`` the call is typed as that Python type, so writing

        beta = BlowtorchModule.Param(0.9, dtype=float)

    makes the assigned attribute a ``float`` for static type checkers.
    Returns Any otherwise.
    """
    return ParamSpec(
        default=default,
        learnable=learnable,
        force_learn=force_learn,
        constraint=constraint,
        dtype=dtype,
    )


@dataclass(frozen=True)
class ConstantSpec:
    """
    Declares a non-learnable hyperparameter on the module.

    Unlike ``Param``, a constant is never registered as an ``nn.Parameter``;
    it is exposed as a plain attribute and constructor kwarg.
    """
    default: Any = None
    validate: Optional[Callable[[Any], None]] = None
    dtype: Any = None


@overload
def Constant(
    default: Any = None,
    *,
    validate: Optional[Callable[[Any], None]] = None,
    dtype: None = None,
) -> Any: ...


@overload
def Constant(
    default: Any = None,
    *,
    validate: Optional[Callable[[Any], None]] = None,
    dtype: type[T],
) -> T: ...


@overload
def Constant(
    default: Any = None,
    *,
    validate: Optional[Callable[[Any], None]] = None,
    dtype: Any = None,
) -> Any: ...


def Constant(
    default: Any = None,
    *,
    validate: Optional[Callable[[Any], None]] = None,
    dtype: Any = None,
) -> Any:
    """
    Declarative non-learnable hyperparameter field.

    With ``dtype=`` the call is typed as that Python type, so writing

        dt = BlowtorchModule.Constant(0.01, dtype=float)

    makes the assigned attribute a ``float`` for static type checkers.
    Returns Any otherwise.

    ``validate``, if given, must raise ``ValueError`` on an invalid value; the
    framework re-raises with the module and field name attached.
    """
    return ConstantSpec(
        default=default,
        validate=validate,
        dtype=dtype,
    )


@dataclass(frozen=True)
class OutputSpec:
    """
    Declares an output tensor returned by `_step`.

    Outputs are not passed back into `_step` as recurrent state.
    """
    default: float | Callable[[nn.Module], float] = 0.0
    differentiable: bool = True


@dataclass(frozen=True)
class StateSpec:
    """
    Declares a recurrent state tensor passed into/out of `_step`.

    `shape` controls the state buffer's shape. It defaults to "input" (same
    shape as the step input, so a (B, F) input yields a (B, F) state). Pass an
    explicit tuple to decouple state shape from input shape:

        mem = BlowtorchModule.StateSpec(shape=(F,))   # per-feature state

    or `None` for a scalar-shaped state.
    """
    default: float | Callable[[nn.Module], float] = 0.0
    differentiable: bool = True
    shape: str | tuple[int, ...] | None = "input"
    extras: dict = field(default_factory=dict)

    def __init__(
        self,
        default: float | Callable[[nn.Module], float] = 0.0,
        differentiable: bool = True,
        shape: str | tuple[int, ...] | None = "input",
        **extras: Any,
    ) -> None:
        object.__setattr__(self, "default", default)
        object.__setattr__(self, "differentiable", differentiable)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "extras", extras)


Spec = Union[OutputSpec, StateSpec]


# Generic Blowtorch module

class BlowtorchModule(nn.Module):
    """
    Generic declarative stateful step module.

    Subclasses declare:

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
      - reset / detach
      - basic sequence scan

    Execution modes:
      - ``init_hidden=True``: the module owns its state. Buffers are allocated
        lazily on first input and returned by ``forward(x)``; pass
        ``allocate_like(x)`` ahead of time when the first call must not
        allocate (e.g. before ``torch.compile`` or CUDA-graph capture). This is
        the convenient mode for single-step loops over one input stream.
      - ``init_hidden=False`` (default): the caller owns state. ``forward`` is
        a pure function of ``(x, *state)`` returning ``(out, *next_state)``;
        use ``initial_state(...)`` or ``state_factory()`` to seed it. This is
        the mode to use with ``forward_sequence``, which threads state across
        the scan itself.

    Validation: ``validate=True`` (default) checks step-output arity and state
    shapes against the specs on every call. Override per module with the
    ``validate=...`` init parameter, or set the global default with
    ``set_validation(...)`` / ``no_validation()``. Once ``validate=`` is set on
    a module it no longer follows the global toggle.
    """

    # Namespaced declarative helpers.
    Param = Param
    Constant = Constant
    OutputSpec = OutputSpec
    StateSpec = StateSpec

    # Metadata collected at class creation time.
    _bt_param_specs: ClassVar[dict[str, ParamSpec]] = {}
    _bt_param_annotations: ClassVar[dict[str, Any]] = {}
    _bt_constant_specs: ClassVar[dict[str, ConstantSpec]] = {}
    _bt_constant_annotations: ClassVar[dict[str, Any]] = {}

    _bt_spec_entries: ClassVar[tuple[tuple[str, Spec], ...]] = ()

    _bt_output_names: ClassVar[tuple[str, ...]] = ()
    _bt_state_names: ClassVar[tuple[str, ...]] = ()

    _bt_output_specs: ClassVar[tuple[OutputSpec, ...]] = ()
    _bt_state_specs: ClassVar[tuple[StateSpec, ...]] = ()

    _bt_spec_extensions: ClassVar[dict[str, Callable[..., Any]]] = {}

    # ------------------------------------------------------------------
    # Class construction / metadata
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        cls._bt_param_specs, cls._bt_param_annotations, cls._bt_constant_specs, cls._bt_constant_annotations = cls._collect_params()
        cls._bt_spec_entries = cls._collect_specs()

        cls._bt_output_names = tuple(
            name
            for name, spec in cls._bt_spec_entries
            if isinstance(spec, OutputSpec)
        )

        cls._bt_state_names = tuple(
            name
            for name, spec in cls._bt_spec_entries
            if isinstance(spec, StateSpec)
        )

        cls._bt_output_specs = tuple(
            spec
            for _, spec in cls._bt_spec_entries
            if isinstance(spec, OutputSpec)
        )

        cls._bt_state_specs = tuple(
            spec
            for _, spec in cls._bt_spec_entries
            if isinstance(spec, StateSpec)
        )

        cls._generate_runtime_type_hints()

    @classmethod
    def _collect_params(
        cls,
    ) -> tuple[
        dict[str, ParamSpec],
        dict[str, Any],
        dict[str, ConstantSpec],
        dict[str, Any],
    ]:
        """
        Collect ParamSpecs and ConstantSpecs from nested `Params` classes
        across the MRO.
        """
        params: dict[str, ParamSpec] = {}
        annotations: dict[str, Any] = {}
        constants: dict[str, ConstantSpec] = {}
        constant_annotations: dict[str, Any] = {}

        for klass in reversed(cls.__mro__):
            for scope in (klass, klass.__dict__.get("Params", None)):
                if scope is None:
                    continue

                scope_annotations = getattr(scope, "__annotations__", {})

                for name, value in vars(scope).items():
                    if isinstance(value, ParamSpec):
                        params[name] = value

                        if name in scope_annotations:
                            annotations[name] = scope_annotations[name]
                    elif isinstance(value, ConstantSpec):
                        constants[name] = value

                        if name in scope_annotations:
                            constant_annotations[name] = scope_annotations[name]

        return params, annotations, constants, constant_annotations

    @classmethod
    def _collect_specs(cls) -> tuple[tuple[str, Spec], ...]:
        """
        Collect OutputSpec / StateSpec entries from nested `Specs` classes.
        """
        entries: dict[str, Spec] = {}

        for klass in reversed(cls.__mro__):
            specs_cls = vars(klass).get("Specs", None)
            if specs_cls is None:
                continue

            for name, value in vars(specs_cls).items():
                if isinstance(value, (OutputSpec, StateSpec)):
                    entries[name] = value

        return tuple(entries.items())

    # ------------------------------------------------------------------
    # Runtime typed hints / signature generation
    # ------------------------------------------------------------------

    @classmethod
    def _extra_init_params(cls) -> list[inspect.Parameter]:
        """
        Domain bases can extend the generated constructor signature.
        """
        return []

    @classmethod
    def _generate_runtime_type_hints(cls) -> None:
        """
        Generate a runtime __signature__ for help(), inspect, notebooks, etc.

        NOTE:
        This gives runtime introspection. Full static type-checker inference
        later requires a pyright/mypy plugin, dataclass_transform, or stubs.
        """
        sig_params: list[inspect.Parameter] = [
            inspect.Parameter(
                "self",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ),
            inspect.Parameter(
                "size",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[int],
            ),
            inspect.Parameter(
                "init_hidden",
                inspect.Parameter.KEYWORD_ONLY,
                default=False,
                annotation=bool,
            ),
            inspect.Parameter(
                "validate",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[bool],
            ),
        ]

        for name, spec in cls._bt_param_specs.items():
            ann = spec.dtype if spec.dtype is not None else cls._bt_param_annotations.get(name, Any)

            sig_params.extend(
                [
                    inspect.Parameter(
                        name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=spec.default,
                        annotation=ann,
                    ),
                    inspect.Parameter(
                        f"learnable_{name}",
                        inspect.Parameter.KEYWORD_ONLY,
                        default=spec.learnable,
                        annotation=bool,
                    ),
                    inspect.Parameter(
                        f"force_learn_{name}",
                        inspect.Parameter.KEYWORD_ONLY,
                        default=spec.force_learn,
                        annotation=bool,
                    ),
                    inspect.Parameter(
                        f"{name}_constraint",
                        inspect.Parameter.KEYWORD_ONLY,
                        default=spec.constraint,
                        annotation=Constraint,
                    ),
                ]
            )

        for name, spec in cls._bt_constant_specs.items():
            ann = (
                spec.dtype
                if spec.dtype is not None
                else cls._bt_constant_annotations.get(name, Any)
            )
            sig_params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=spec.default,
                    annotation=ann,
                )
            )

        sig_params.extend(cls._extra_init_params())

        sig_params.append(
            inspect.Parameter(
                "kwargs",
                inspect.Parameter.VAR_KEYWORD,
                annotation=Any,
            )
        )

        cls.__signature__ = inspect.Signature(sig_params)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        size: Optional[int] = None,
        init_hidden: bool = False,
        validate: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
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

        constraint_fns: list[Constraint] = []

        _UNSET = object()

        for name, spec in self._bt_param_specs.items():
            value = kwargs.pop(name, spec.default)

            force_learn_kwarg = kwargs.pop(
                f"force_learn_{name}",
                _UNSET,
            )

            learnable_kwarg = kwargs.pop(
                f"learnable_{name}",
                _UNSET,
            )

            constraint = kwargs.pop(
                f"{name}_constraint",
                spec.constraint,
            )

            # force_learn=True means always learnable. An explicit
            # force_learn_<name>=True wins over everything; an explicit
            # learnable_<name>=False overrides spec-level force_learn.
            force_learn = (
                bool(force_learn_kwarg)
                if force_learn_kwarg is not _UNSET
                else spec.force_learn
            )

            if learnable_kwarg is not _UNSET:
                learnable = bool(learnable_kwarg)
                if force_learn_kwarg is not _UNSET and bool(force_learn_kwarg):
                    learnable = True
            else:
                learnable = force_learn or spec.learnable

            if value is None:
                raise ValueError(
                    f"{type(self).__name__} parameter {name!r} has no value or default"
                )

            tensor = torch.as_tensor(
                value,
                dtype=spec.dtype if isinstance(spec.dtype, torch.dtype) else None,
            )

            if not tensor.is_floating_point():
                tensor = tensor.to(torch.get_default_dtype())

            self.register_parameter(
                name,
                nn.Parameter(
                    tensor,
                    requires_grad=learnable,
                ),
            )

            # Fixed parameters skip constraint work entirely.
            constraint_fns.append(constraint if learnable else identity)

        for name, spec in self._bt_constant_specs.items():
            value = kwargs.pop(name, spec.default)

            if spec.validate is not None:
                try:
                    spec.validate(value)
                except ValueError as exc:
                    raise ValueError(
                        f"{type(self).__name__} {name}: {exc}"
                    ) from None

            if isinstance(spec.dtype, type):
                value = spec.dtype(value)

            setattr(self, name, value)

        if kwargs:
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword arguments: "
                f"{sorted(kwargs)}"
            )

        self._bt_constraints: tuple[Constraint, ...] = tuple(constraint_fns)
        self._install_constrained()
        self._process_spec_extensions()

    def _process_spec_extensions(self) -> None:
        """
        Dispatch each StateSpec extra through the handler registry.

        Handlers run after specs are resolved so they can rely on class
        metadata (``_bt_state_specs``, ``_bt_param_specs``) and on instance
        parameters already being registered.
        """
        for i, spec in enumerate(self._bt_state_specs):
            for key, value in spec.extras.items():
                handler = self._bt_spec_extensions.get(key)
                if handler is not None:
                    handler(self, i, spec, value)

        self._install_reset_fn()

    def _install_reset_fn(self) -> None:
        """
        Install the state-reset function applied after ``_step``.

        Base modules have no resets; SNN subclasses override this to
        code-generate reset expressions from their Specs.
        """
        self._bt_apply_resets = lambda pre_state, spk: pre_state

    # ------------------------------------------------------------------
    # Validation flag
    # ------------------------------------------------------------------

    @property
    def validate(self) -> bool:
        """
        Runtime validation flag.
        If constructed with validate=None, follows the global toggle.
        If an explicit bool was passed, that value wins.
        """
        return (
            get_validation()
            if self._validate_override is None
            else self._validate_override
        )

    @validate.setter
    def validate(self, value: bool) -> None:
        """Allow dynamic toggling: `module.validate = False`."""
        self._validate_override = bool(value)

    # ------------------------------------------------------------------
    # Hot-path constrained parameter accessor
    # ------------------------------------------------------------------

    def constrained(self) -> tuple[Tensor, ...]:
        """
        Return constrained parameters in Params declaration order.

        Hot path:
          - no strings
          - no dict lookups
          - no metadata resolution

        The returned expression is frozen at init time.
        """
        return self._bt_constrained_fn(self)

    def _install_constrained(self) -> None:
        """
        Freeze the constrained-parameter return expression.

        Example generated function for LIF:

            def _bt_constrained(self):
                return (self._bt_constraint_0(self.beta), self.threshold)
        """
        exprs: list[str] = []

        # Safety: the generated body below is exec'd, so every interpolated
        # name must come from module-controlled sources. Param names are
        # restricted to non-keyword Python identifiers (validated here and at
        # class creation); constraint fns are attributes set via setattr from
        # the declarative ParamSpecs, never from user strings. Nothing from
        # user input can reach the exec namespace.
        param_names = tuple(self._bt_param_specs.keys())

        for i, (name, constraint) in enumerate(
            zip(param_names, self._bt_constraints)
        ):
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(
                    f"Param name {name!r} must be a valid non-keyword Python identifier"
                )

            if constraint is identity:
                exprs.append(f"self.{name}")
            else:
                attr = f"_bt_constraint_{i}"
                setattr(self, attr, constraint)
                exprs.append(f"self.{attr}(self.{name})")

        if exprs:
            ret = f"({', '.join(exprs)},)"
        else:
            ret = "()"

        src = f"def _bt_constrained(self):\n    return {ret}\n"

        ns: dict[str, Any] = {}
        exec(src, ns)

        self._bt_constrained_fn = ns["_bt_constrained"]

    # ------------------------------------------------------------------
    # Spec helpers
    # ------------------------------------------------------------------

    def _resolve_default(
        self,
        value: float | Callable[[nn.Module], float],
    ) -> float:
        if callable(value):
            return float(value(self))
        return float(value)

    def _spec_shape(self, spec: Spec, x: Tensor) -> tuple[int, ...]:
        shape = getattr(spec, "shape", "input")

        if shape is None or shape == "input":
            return tuple(x.shape)

        assert isinstance(shape, tuple)
        return shape

    def _explicit_state_dtype(
        self,
        dtype: Optional[torch.dtype],
    ) -> Optional[torch.dtype]:
        if dtype is None:
            return None
        return _floating_dtype(dtype)

    # ------------------------------------------------------------------
    # Hidden-mode allocation / step
    # ------------------------------------------------------------------

    def _alloc_hidden(self, x: Tensor) -> None:
        dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()

        for name, spec in self._bt_spec_entries:
            self.register_buffer(
                name,
                torch.full(
                    self._spec_shape(spec, x),
                    self._resolve_default(spec.default),
                    device=x.device,
                    dtype=dtype,
                ),
                persistent=False,
            )

        self._bt_allocated = True

    def allocate_like(self, x: Tensor) -> BlowtorchModule:
        """
        Materialize hidden buffers outside of ``torch.compile``.

        Lazy allocation on the first forward call happens inside the compiled
        region and breaks the scan graph (and, under CUDA graphs, registers
        buffers that alias the compiler's recycled memory pool). Call this once
        eagerly before compiling an ``init_hidden=True`` module:

            module.allocate_like(x)
            module.fast_sequence_()
        """
        if self.init_hidden and not self._bt_allocated:
            self._alloc_hidden(x)

        return self

    def _check_step_output(self, out: StepOutput) -> None:
        if not isinstance(out, tuple):
            raise TypeError(
                f"{type(self).__name__}._step must return a tuple of tensors"
            )

        expected = len(self._bt_spec_entries)

        if len(out) != expected:
            raise ValueError(
                f"{type(self).__name__}._step returned {len(out)} tensors, "
                f"expected {expected} according to Specs"
            )

    def _store_hidden_outputs(self, out: StepOutput) -> None:
        for (name, spec), t in zip(self._bt_spec_entries, out):
            if not spec.differentiable:
                t = t.detach()

            # Buffers already exist after _alloc_hidden.
            self._buffers[name] = t

    def _hidden_step(self, x: Tensor) -> StepOutput:
        state = tuple(getattr(self, name) for name in self._bt_state_names)

        out = self._step(x, *state)

        if self.validate:
            self._check_step_output(out)

        if isinstance(out, tuple):
            spk = out[0]
            pre_state = out[1:]
            next_state = self._bt_apply_resets(pre_state, spk)
            final_out = (spk,) + tuple(next_state)
        else:
            final_out = out

        self._store_hidden_outputs(final_out)
        return final_out

    def _forward_hidden(self, x: Tensor) -> Tensor | StepOutput:
        if not self._bt_allocated:
            self._alloc_hidden(x)

        out = self._hidden_step(x)

        n_outputs = len(self._bt_output_names)

        if n_outputs == 1:
            return out[0]

        return out[:n_outputs]

    # ------------------------------------------------------------------
    # Explicit-mode step
    # ------------------------------------------------------------------

    def _forward_explicit(self, x: Tensor, *state: Tensor) -> StepOutput:
        if self.validate:
            expected = len(self._bt_state_names)
            if len(state) != expected:
                raise ValueError(
                    f"{type(self).__name__} expects {expected} state tensors, "
                    f"got {len(state)}"
                )

        out = self._step(x, *state)

        if self.validate:
            self._check_step_output(out)

        if isinstance(out, tuple):
            spk = out[0]
            pre_state = out[1:]
            next_state = self._bt_apply_resets(pre_state, spk)
            return (spk,) + tuple(next_state)

        return out

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, *state: Tensor) -> Tensor | StepOutput:
        if self.init_hidden:
            if state:
                raise ValueError(
                    f"{type(self).__name__} is in init_hidden=True mode; "
                    f"do not pass state explicitly"
                )

            return self._forward_hidden(x)

        return self._forward_explicit(x, *state)

    # ------------------------------------------------------------------
    # Trainer / loop convenience
    # ------------------------------------------------------------------

    def step_state(
        self,
        x: Tensor,
        state: tuple[Tensor, ...],
    ) -> tuple[Tensor | StepOutput, tuple[Tensor, ...]]:
        """
        Explicit step with tuple state.

        Returns:
            (output(s), next_state)
        """
        if self.init_hidden:
            raise ValueError(
                f"{type(self).__name__}.step_state requires init_hidden=False"
            )

        out = self.forward(x, *state)
        assert isinstance(out, tuple)

        n_outputs = len(self._bt_output_names)

        if n_outputs == 1:
            return out[0], out[n_outputs:]

        return out[:n_outputs], out[n_outputs:]

    def step(
        self,
        x: Tensor,
        state: tuple[Tensor, ...],
    ) -> tuple[Tensor | StepOutput, tuple[Tensor, ...]]:
        """
        Alias of step_state.
        """
        return self.step_state(x, state)

    # ------------------------------------------------------------------
    # State factories
    # ------------------------------------------------------------------

    def initial_state(
        self,
        batch_shape: tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, ...]:
        """
        Create canonical initial explicit state using Spec defaults.
        """
        dtype = self._explicit_state_dtype(dtype)

        state: list[Tensor] = []

        for spec in self._bt_state_specs:
            state.append(
                torch.full(
                    batch_shape,
                    self._resolve_default(spec.default),
                    device=device,
                    dtype=dtype,
                )
            )

        return tuple(state)

    def zero_state(
        self,
        batch_shape: tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, ...]:
        """
        Create zeroed explicit state.
        """
        dtype = self._explicit_state_dtype(dtype)

        state: list[Tensor] = []

        for _ in self._bt_state_specs:
            state.append(
                torch.zeros(
                    batch_shape,
                    device=device,
                    dtype=dtype,
                )
            )

        return tuple(state)

    def initial_state_like(
        self,
        x: Tensor,
        batch_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[Tensor, ...]:
        shape = tuple(x.shape) if batch_shape is None else tuple(batch_shape)

        return self.initial_state(
            shape,
            device=x.device,
            dtype=x.dtype,
        )

    def initial_state_for_sequence(
        self,
        x_seq: Tensor,
    ) -> tuple[Tensor, ...]:
        if x_seq.dim() < 3:
            raise ValueError(
                f"{type(self).__name__}.initial_state_for_sequence expects "
                f"(time, batch, features), got {x_seq.dim()} dims"
            )

        return self.initial_state(
            tuple(x_seq.shape[1:]),
            device=x_seq.device,
            dtype=x_seq.dtype,
        )

    # ------------------------------------------------------------------
    # Reset / detach
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset hidden buffers to their Spec defaults.
        """
        if not self.init_hidden:
            return

        for name, spec in self._bt_spec_entries:
            t = self._buffers.get(name, None)

            if isinstance(t, Tensor):
                self._buffers[name] = torch.full_like(
                    t,
                    self._resolve_default(spec.default),
                )

    def detach(self) -> None:
        """
        Detach hidden buffers from autograd.
        """
        if not self.init_hidden:
            return

        for name, _ in self._bt_spec_entries:
            t = self._buffers.get(name, None)

            if isinstance(t, Tensor):
                self._buffers[name] = t.detach()

    # ------------------------------------------------------------------
    # Checkpointing hidden state
    # ------------------------------------------------------------------

    def get_extra_state(self) -> Optional[dict[str, Tensor]]:
        """
        Include hidden buffers in state_dict even though they are non-persistent.

        Only applies in ``init_hidden=True`` mode, where the module owns its
        state. In explicit mode the caller owns the state tensors, so they are
        not serialized here; persist them yourself alongside the state_dict.
        """
        if not self.init_hidden:
            return None

        out: dict[str, Tensor] = {}
        for name, _ in self._bt_spec_entries:
            t = self._buffers.get(name)
            if t is not None:
                out[name] = t.detach()
                
        return out

    def set_extra_state(self, state: Any) -> None:
        if not self.init_hidden or state is None:
            return

        for name, t in state.items():
            if name in self._buffers:
                self._buffers[name] = t
            else:
                self.register_buffer(name, t, persistent=False)

        self._bt_allocated = True

    # ------------------------------------------------------------------
    # Sequence scan
    # ------------------------------------------------------------------

    def forward_sequence(
        self,
        x_seq: Tensor,
        state: Optional[tuple[Tensor, ...]] = None,
    ) -> Tensor | StepOutput:
        """
        Time-major sequence scan.

        Input shape:
            (time, batch, features)

        Hidden mode:
            single output -> output sequence:
                (time, batch, features)
            multiple outputs -> tuple of output sequences:
                ((time, batch, features), ...)

        Explicit mode:
            returns:
                (output_sequence(s), *final_state)

        After ``fast_sequence_()`` this routes through the compiled scan and
        clones returned tensors (safe across ``torch.compile`` / CUDA graphs).

        Note: the eager path uses ``torch.compiler.is_compiling()`` to branch
        between the chunked and the flat scan. Under ``torch.compile`` each
        wrapper sees its own trace; recompiling or nesting multiple compiled
        wrappers is fine, but a manually written wrapper that calls
        ``forward_sequence`` outside a proper compile context may take the
        eager branch even while surrounding code is compiled. Prefer
        ``fast_sequence_()`` over wrapping this method yourself.
        """
        compiled = self._bt_compiled_sequence

        if compiled is not None:
            return compiled(x_seq, state)

        return self._reference_sequence_scan(x_seq, state)

    def _reference_sequence_scan(
        self,
        x_seq: Tensor,
        state: Optional[tuple[Tensor, ...]] = None,
    ) -> Tensor | StepOutput:
        """
        The reference per-step scan. This is the compile unit for
        ``compile_sequence_scan``; keep it free of state-allocation side
        effects that break tracing.
        """
        if x_seq.dim() < 3:
            raise ValueError(
                f"{type(self).__name__}.forward_sequence expects "
                f"(time, batch, features), got {x_seq.dim()} dims"
            )

        if x_seq.shape[0] == 0:
            raise ValueError(
                f"{type(self).__name__}.forward_sequence expects at least one timestep"
            )

        if self.init_hidden:
            if state is not None:
                raise ValueError(
                    f"{type(self).__name__} is in init_hidden=True mode; "
                    f"forward_sequence does not accept explicit state"
                )

            return self._hidden_sequence_scan(x_seq)

        return self._explicit_sequence_scan(x_seq, state)

    def _hidden_sequence_scan(self, x_seq: Tensor) -> Tensor | StepOutput:
        if not self._bt_allocated:
            self._alloc_hidden(x_seq[0])

        T = x_seq.shape[0]
        n_outputs = len(self._bt_output_names)

        out0 = self._hidden_step(x_seq[0])

        # Preallocate one (T, B, F) buffer per output stream.
        ys = tuple(
            torch.empty((T, *o.shape), dtype=o.dtype, device=o.device)
            for o in out0[:n_outputs]
        )

        # Write each step's outputs into the preallocated buffers.
        # Under torch.compile, constant-index index_copy_ lowers to a plain
        # contiguous store fused into that step's kernel; in eager, batch K
        # steps into one scatter so we keep eager speed without a (T, B, F)
        # transient list + torch.stack copy.
        if torch.compiler.is_compiling():
            for k, y in enumerate(ys):
                y.index_copy_(0, torch.tensor([0], device=y.device), out0[k].unsqueeze(0))

            for t in range(1, T):
                out = self._hidden_step(x_seq[t])

                for k, y in enumerate(ys):
                    y.index_copy_(
                        0,
                        torch.tensor([t], device=y.device),
                        out[k].unsqueeze(0),
                    )
        else:
            for k, y in enumerate(ys):
                y[0] = out0[k]

            idx = torch.arange(T, device=x_seq.device)

            for lo in range(1, T, _SEQUENCE_SCAN_CHUNK):
                hi = min(lo + _SEQUENCE_SCAN_CHUNK, T)

                # Run each step once, collecting one list per output stream.
                chunks: list[list[Tensor]] = [[] for _ in range(n_outputs)]

                for t in range(lo, hi):
                    out = self._hidden_step(x_seq[t])

                    for k in range(n_outputs):
                        chunks[k].append(out[k])

                for k, y in enumerate(ys):
                    y.index_copy_(0, idx[lo:hi], torch.stack(chunks[k]))

        if n_outputs == 1:
            return ys[0]

        return ys

    def _explicit_sequence_scan(
        self,
        x_seq: Tensor,
        state: Optional[tuple[Tensor, ...]],
    ) -> StepOutput:
        if state is None:
            state = self.initial_state_for_sequence(x_seq)

        out = self.forward(x_seq[0], *state)
        assert isinstance(out, tuple)

        n_outputs = len(self._bt_output_names)
        y0 = out[:n_outputs]

        ys = tuple(
            torch.empty(
                (x_seq.shape[0], *o.shape),
                dtype=o.dtype,
                device=o.device,
            )
            for o in y0
        )

        cur = out[n_outputs:]

        # Same preallocated-output collection as _hidden_sequence_scan.
        if torch.compiler.is_compiling():
            for k, y in enumerate(ys):
                y.index_copy_(0, torch.tensor([0], device=y.device), y0[k].unsqueeze(0))

            for t in range(1, x_seq.shape[0]):
                out = self.forward(x_seq[t], *cur)
                assert isinstance(out, tuple)

                for k, y in enumerate(ys):
                    y.index_copy_(
                        0,
                        torch.tensor([t], device=y.device),
                        out[k].unsqueeze(0),
                    )

                cur = out[n_outputs:]
        else:
            for k, y in enumerate(ys):
                y[0] = y0[k]

            idx = torch.arange(x_seq.shape[0], device=x_seq.device)

            for lo in range(1, x_seq.shape[0], _SEQUENCE_SCAN_CHUNK):
                hi = min(lo + _SEQUENCE_SCAN_CHUNK, x_seq.shape[0])

                chunks: list[list[Tensor]] = [[] for _ in range(n_outputs)]

                for t in range(lo, hi):
                    out = self.forward(x_seq[t], *cur)
                    assert isinstance(out, tuple)

                    for k in range(n_outputs):
                        chunks[k].append(out[k])

                    cur = out[n_outputs:]

                for k, y in enumerate(ys):
                    y.index_copy_(0, idx[lo:hi], torch.stack(chunks[k]))

        if n_outputs == 1:
            return (ys[0], *cur)

        return (*ys, *cur)

    def compile_sequence_scan(self, **kwargs: Any) -> BlowtorchModule:
        """
        Compile the reference sequence scan and route forward_sequence through it.

        Output tensors are cloned before returning only when CUDA graphs are in
        play (``mode="reduce-overhead"`` or ``mode="max-autotune"``): a subsequent
        graph run would otherwise overwrite the previously returned tensor. With
        any other mode the outputs are returned as-is - cloning would cost a full
        extra copy of the spike tensor on every call.

        Note on sequence length: the compiled unit is the fully unrolled T-step
        scan, so compilation cost and peak memory grow with T. It is fast up to
        roughly T~1000 and impractical (``RecursionError`` in inductor, or
        minutes-long compiles) beyond T~3000. This is not specific to blowtorch:
        norse's compiled sequence and any fully unrolled scan hit the same wall.
        For very long sequences, split the input into chunks and call
        ``forward_sequence`` per chunk.
        """
        needs_clone = kwargs.get("mode") in ("reduce-overhead", "max-autotune")
        compiled = torch.compile(self._reference_sequence_scan, **kwargs)

        def _compiled(
            x_seq: Tensor,
            state: Optional[tuple[Tensor, ...]] = None,
        ) -> Tensor | StepOutput:
            if self.init_hidden and x_seq.shape[0] > 0:
                # Allocate hidden buffers *before* the compiled call. If the
                # initial trace ran the alloc path, the buffer registration
                # side effects break the scan into separate graphs (and under
                # CUDA graphs they alias the compiler's memory pool).
                self.allocate_like(x_seq[0])

            out = compiled(x_seq, state)

            if isinstance(out, Tensor):
                return out.clone() if needs_clone else out

            return tuple(t.clone() for t in out) if needs_clone else out

        self._bt_compiled_sequence = _compiled

        return self

    def fast_sequence_(self, compile_scan: bool = True, **compile_kwargs: Any) -> BlowtorchModule:
        """
        Enable a fast research path: validation off + optional compiled scan.

        ``mode="default"`` is always used. ``reduce-overhead`` (CUDA graphs)
        is avoided: it is slower to compile and incompatible with hidden-mode
        buffer registration.
        """
        self.validate = False

        if compile_scan:
            compile_kwargs.setdefault("mode", "default")
            self.compile_sequence_scan(**compile_kwargs)

        return self

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        parts: list[str] = []

        if self.size is not None:
            parts.append(f"size={self.size}")

        parts.append(f"init_hidden={self.init_hidden}")

        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Abstract step
    # ------------------------------------------------------------------

    def _step(self, x: Tensor, *state: Tensor) -> StepOutput:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _step()"
        )


_BTModuleT = TypeVar("_BTModuleT", bound="type[BlowtorchModule]")


def extend_specs(**extensions: Callable[..., Any]):
    """
    Decorate a BlowtorchModule subclass with StateSpec extra handlers.

    Each extra key declared on a StateSpec (e.g. ``StateSpec(reset=...)``) is
    dispatched at construction time to the matching handler callable:

        @extend_specs(reset=ResetHandler)
        class SnnModule(BlowtorchModule):
            ...

    Handlers are called as ``handler(module, state_index, spec, value)``.
    """

    def decorator(cls: _BTModuleT) -> _BTModuleT:
        existing = dict(getattr(cls, "_bt_spec_extensions", {}))
        existing.update(extensions)
        cls._bt_spec_extensions = existing
        return cls

    return decorator