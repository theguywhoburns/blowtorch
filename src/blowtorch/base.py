from __future__ import annotations

import inspect
import keyword
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional, TypeVar, Union, overload

import torch
import torch.nn as nn

__all__ = [
    "BlowtorchModule",
    "Param",
    "ParamSpec",
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

    Constraints apply only to learnable parameters: a fixed (non-learnable)
    param is used raw in ``constrained()`` / resets, while a learnable one has
    its constraint applied on the hot path.
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
class InputSpec:
    """
    Declares a named step input on a module.

    Inputs are declared positionally in a nested ``Inputs`` class:

        class Inputs:
            x: Tensor
            inh: Tensor

    The first declared input (or the one marked ``primary=True``) drives
    default state shapes (``StateSpec(shape="input")``) and hidden-mode
    device/dtype. ``dtype`` is stored as validation metadata only; it is not
    used to cast inputs.
    """
    primary: bool = False
    dtype: Any = None


@overload
def Input(*, primary: bool = False, dtype: None = None) -> Any: ...


@overload
def Input(*, primary: bool = False, dtype: type[T]) -> T: ...


def Input(*, primary: bool = False, dtype: Any = None) -> Any:
    """
    Declarative named-input field.

    With ``dtype=`` the call is typed as that Python type, so writing

        x = BlowtorchModule.Input(primary=True, dtype=float)

    makes the assigned attribute a ``float`` for static type checkers. Returns
    ``Any`` otherwise.
    """
    return InputSpec(primary=primary, dtype=dtype)


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

    `shape` controls the state buffer's shape:
      - ``"input"`` (default) / ``None``: same shape as the primary input, so a
        (B, F) primary input yields a (B, F) state.
      - a string matching an input name: that input's shape. Useful for
        multi-input modules where a state should follow a non-primary input:

            v = BlowtorchModule.StateSpec(shape="inh")

      - an explicit tuple: decouples state shape from input shape:

            mem = BlowtorchModule.StateSpec(shape=(F,))   # per-feature state

    `None` behaves identically to "input" (the state follows the primary input
    shape); there is no scalar-shaped state convention.
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

# Time-major scan over a pure step function.


def sequence_scan(
    step: Callable[[tuple[Tensor, ...], tuple[Tensor, ...]], tuple[Tensor, ...]],
    inputs_seq: tuple[Tensor, ...],
    state0: tuple[Tensor, ...],
    n_outputs: int,
) -> tuple[Tensor, ...]:
    """
    Scan a pure step function over a time-major input.

    ``step`` is ``(inputs, state) -> (*outputs, *next_state)`` where ``inputs``
    is the canonical per-timestep input tuple (one tensor per declared input):
    the first ``n_outputs`` tensors are outputs, the rest become the state for
    the next step. Returns ``(*ys, *final_state)`` where each ``ys[k]`` is a
    preallocated ``(T, *output_shape)`` buffer.

    ``inputs_seq`` is the canonical tuple of time-major sequences (one per
    declared input, all sharing time length ``T``).

    In eager this batches ``_SEQUENCE_SCAN_CHUNK`` steps into one
    ``index_copy_`` scatter so peak memory stays at input + output. Under
    ``torch.compile`` it lowers to a flat fused loop; the whole scan becomes a
    single graph.
    """
    T = inputs_seq[0].shape[0]

    inputs0 = tuple(seq[0] for seq in inputs_seq)
    out0 = step(inputs0, state0)
    assert isinstance(out0, tuple)

    ys = tuple(
        torch.empty((T, *o.shape), dtype=o.dtype, device=o.device)
        for o in out0[:n_outputs]
    )

    if torch.compiler.is_compiling():
        for k, y in enumerate(ys):
            y.index_copy_(0, torch.tensor([0], device=y.device), out0[k].unsqueeze(0))

        cur = out0[n_outputs:]

        for t in range(1, T):
            inputs_t = tuple(seq[t] for seq in inputs_seq)
            out = step(inputs_t, cur)

            for k, y in enumerate(ys):
                y.index_copy_(
                    0,
                    torch.tensor([t], device=y.device),
                    out[k].unsqueeze(0),
                )

            cur = out[n_outputs:]
    else:
        for k, y in enumerate(ys):
            y[0] = out0[k]

        cur = out0[n_outputs:]
        idx = torch.arange(T, device=inputs_seq[0].device)

        for lo in range(1, T, _SEQUENCE_SCAN_CHUNK):
            hi = min(lo + _SEQUENCE_SCAN_CHUNK, T)

            chunks: list[list[Tensor]] = [[] for _ in range(n_outputs)]

            for t in range(lo, hi):
                inputs_t = tuple(seq[t] for seq in inputs_seq)
                out = step(inputs_t, cur)

                for k in range(n_outputs):
                    chunks[k].append(out[k])

                cur = out[n_outputs:]

            for k, y in enumerate(ys):
                y.index_copy_(0, idx[lo:hi], torch.stack(chunks[k]))

    return (*ys, *cur)


# Generic Blowtorch module

class BlowtorchModule(nn.Module):
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
      - reset / detach
      - basic sequence scan

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

    # Metadata collected at class creation time.
    _bt_param_specs: ClassVar[dict[str, ParamSpec]] = {}
    _bt_param_annotations: ClassVar[dict[str, Any]] = {}
    _bt_constant_specs: ClassVar[dict[str, ConstantSpec]] = {}
    _bt_constant_annotations: ClassVar[dict[str, Any]] = {}

    _bt_input_entries: ClassVar[tuple[tuple[str, InputSpec], ...]] = (
        ("x", InputSpec(primary=True)),
    )
    _bt_input_names: ClassVar[tuple[str, ...]] = ("x",)
    _bt_input_specs: ClassVar[tuple[InputSpec, ...]] = (InputSpec(primary=True),)
    _bt_primary_input_index: ClassVar[int] = 0

    _bt_spec_entries: ClassVar[tuple[tuple[str, Spec], ...]] = ()

    _bt_output_names: ClassVar[tuple[str, ...]] = ()
    _bt_state_names: ClassVar[tuple[str, ...]] = ()

    _bt_output_specs: ClassVar[tuple[OutputSpec, ...]] = ()
    _bt_state_specs: ClassVar[tuple[StateSpec, ...]] = ()

    _bt_spec_extensions: ClassVar[dict[str, Callable[..., Any]]] = {}

    @staticmethod
    def safe_exp(t: Tensor) -> Tensor:
        """
        Exponential that stays finite for any input.

        The argument is clamped below ``log(finfo(dtype).max)`` (with a small
        margin so the result can't round up to ``inf``), so neither the output
        nor its gradient ever overflows for any dtype. Matches ``exp`` wherever
        the plain exponential is safely below the dtype maximum.
        """
        max_arg = torch.log(torch.tensor(torch.finfo(t.dtype).max, dtype=t.dtype)) - 1
        return torch.clamp(t, max=max_arg).exp()

    # Class construction / metadata

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        cls._bt_param_specs, cls._bt_param_annotations, cls._bt_constant_specs, cls._bt_constant_annotations = cls._collect_params()
        cls._bt_spec_entries = cls._collect_specs()

        cls._bt_input_entries = cls._collect_inputs()
        cls._bt_input_names = tuple(
            name for name, _ in cls._bt_input_entries
        )
        cls._bt_input_specs = tuple(
            spec for _, spec in cls._bt_input_entries
        )

        primary_indices = [
            i
            for i, (_, spec) in enumerate(cls._bt_input_entries)
            if spec.primary
        ]

        if len(primary_indices) > 1:
            raise TypeError(
                f"{cls.__name__} declares multiple primary inputs: "
                f"{[cls._bt_input_names[i] for i in primary_indices]}. "
                f"At most one Input may set primary=True."
            )

        cls._bt_primary_input_index = primary_indices[0] if primary_indices else 0

        cls._check_input_namespace_collisions()

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

    @classmethod
    def _collect_inputs(cls) -> tuple[tuple[str, InputSpec], ...]:
        """
        Collect named step inputs from nested `Inputs` classes across the MRO.

        Supports annotation-only syntax (``x: Tensor``), ``InputSpec`` values,
        and mixed forms. Declaration order is preserved. If no ``Inputs``
        class exists, a single implicit primary input ``x`` is assumed.
        """
        inputs: dict[str, InputSpec] = {}

        for klass in reversed(cls.__mro__):
            inputs_cls = vars(klass).get("Inputs", None)
            if inputs_cls is None:
                continue

            scope_annotations = getattr(inputs_cls, "__annotations__", {})
            vars_ = vars(inputs_cls)

            for name in scope_annotations:
                if name.startswith("_"):
                    continue

                value = vars_.get(name)
                if isinstance(value, InputSpec):
                    inputs[name] = value
                else:
                    inputs[name] = InputSpec()

            for name, value in vars_.items():
                if name.startswith("_") or name in scope_annotations:
                    continue

                if isinstance(value, InputSpec):
                    inputs[name] = value

        if not inputs:
            return (("x", InputSpec(primary=True)),)

        for name in inputs:
            if not name.isidentifier() or keyword.iskeyword(name):
                raise TypeError(
                    f"{cls.__name__} input name {name!r} must be a valid "
                    f"non-keyword Python identifier"
                )

        return tuple(inputs.items())

    @classmethod
    def _check_input_namespace_collisions(cls) -> None:
        """
        Input names share the module namespace with params, constants, states,
        and outputs; reject collisions with a clear message.
        """
        param_names = set(cls._bt_param_specs)
        constant_names = set(cls._bt_constant_specs)
        input_names = set(cls._bt_input_names)
        spec_names = set(name for name, _ in cls._bt_spec_entries)

        for other, label in (
            (param_names, "parameter"),
            (constant_names, "constant"),
            (spec_names, "output/state"),
        ):
            clash = input_names & other

            if clash:
                raise TypeError(
                    f"{cls.__name__} input name(s) {sorted(clash)} collide "
                    f"with {label} name(s)"
                )

    # Runtime typed hints / signature generation

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
            value_orig = value

            if spec.validate is not None:
                try:
                    spec.validate(value)
                except ValueError as exc:
                    raise ValueError(
                        f"{type(self).__name__} {name}: {exc}"
                    ) from None

            if isinstance(spec.dtype, type):
                value = spec.dtype(value)
                if value != value_orig:
                    raise ValueError(
                        f"{type(self).__name__} {name}: value {value_orig!r} "
                        f"cannot be stored as {spec.dtype.__name__}"
                    )

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

    # Validation flag

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

    # Hot-path constrained parameter accessor

    def constrained(self) -> tuple[Tensor, ...]:
        """
        Return constrained parameters in Params declaration order.

        Hot path:
          - no strings
          - no dict lookups
          - no metadata resolution

        The returned expression is frozen at init time.
        """
        return self._bt_constrained_fn()

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

        self._bt_constrained_fn = types.MethodType(ns["_bt_constrained"], self)

    # Spec helpers

    def _resolve_default(
        self,
        value: float | Callable[[nn.Module], float],
    ) -> float:
        if callable(value):
            return float(value(self))
        return float(value)

    def _spec_shape(self, spec: Spec, inputs: tuple[Tensor, ...]) -> tuple[int, ...]:
        shape = getattr(spec, "shape", "input")

        if shape is None or shape == "input":
            return tuple(inputs[self._bt_primary_input_index].shape)

        if isinstance(shape, str):
            if shape not in self._bt_input_names:
                raise ValueError(
                    f"{type(self).__name__} StateSpec(shape={shape!r}) refers "
                    f"to an unknown input; declared inputs are "
                    f"{self._bt_input_names}"
                )

            return tuple(inputs[self._bt_input_names.index(shape)].shape)

        assert isinstance(shape, tuple)
        return shape

    def _explicit_state_dtype(
        self,
        dtype: Optional[torch.dtype],
    ) -> Optional[torch.dtype]:
        if dtype is None:
            return None
        return _floating_dtype(dtype)

    # Hidden-mode allocation / step

    def _alloc_hidden(self, inputs: tuple[Tensor, ...]) -> None:
        primary = inputs[self._bt_primary_input_index]
        dtype = (
            primary.dtype
            if primary.is_floating_point()
            else torch.get_default_dtype()
        )

        for name, spec in self._bt_spec_entries:
            self.register_buffer(
                name,
                torch.full(
                    self._spec_shape(spec, inputs),
                    self._resolve_default(spec.default),
                    device=primary.device,
                    dtype=dtype,
                ),
                persistent=False,
            )

        self._bt_allocated = True

    def allocate_like(
        self,
        *inputs: Tensor | tuple[Tensor, ...] | dict[str, Tensor],
    ) -> BlowtorchModule:
        """
        Materialize hidden buffers outside of ``torch.compile``.

        Lazy allocation on the first forward call happens inside the compiled
        region and breaks the scan graph (and, under CUDA graphs, registers
        buffers that alias the compiler's recycled memory pool). Call this once
        eagerly before compiling an ``init_hidden=True`` module:

            module.allocate_like(x)
            module.allocate_like((x, inh))
            module.fast_sequence_()

        The call is a no-op once buffers exist. Hidden buffers keep the shape
        of the first inputs; later inputs with different batch/feature dims
        raise ``ValueError`` from the forward paths when validation is on.
        """
        if self.init_hidden and not self._bt_allocated:
            if len(inputs) == 1:
                canonical = self._canonicalize_inputs(inputs[0])
            else:
                expanded: list[Tensor] = []

                for t in inputs:
                    if not isinstance(t, Tensor):
                        raise TypeError(
                            f"{type(self).__name__}.allocate_like expanded "
                            f"inputs must be tensors, got {type(t).__name__}"
                        )

                    expanded.append(t)

                canonical = tuple(expanded)

            self._alloc_hidden(canonical)

        return self

    def _check_hidden_input_shape(self, inputs: tuple[Tensor, ...]) -> None:
        for name, spec in zip(self._bt_state_names, self._bt_state_specs):
            ref = self._buffers.get(name)
            expected = self._spec_shape(spec, inputs)

            if ref is not None and ref.shape != expected:
                raise ValueError(
                    f"{type(self).__name__} hidden buffers were allocated for "
                    f"shape {tuple(ref.shape)}, got input shape {tuple(expected)}; "
                    f"the batch/feature dims must stay fixed in hidden mode"
                )

    def _check_input_dtypes(self, inputs: tuple[Tensor, ...]) -> None:
        for name, spec, x in zip(
            self._bt_input_names, self._bt_input_specs, inputs
        ):
            if spec.dtype is None:
                continue

            expected: torch.dtype

            if isinstance(spec.dtype, torch.dtype):
                expected = spec.dtype
            elif spec.dtype is float:
                expected = _floating_dtype(x.dtype)
            elif spec.dtype is int:
                if x.dtype.is_floating_point:
                    raise TypeError(
                        f"{type(self).__name__} input {name!r} declared with "
                        f"dtype=int but got floating-point tensor {x.dtype}"
                    )
                continue
            else:
                continue

            if x.dtype != expected:
                raise TypeError(
                    f"{type(self).__name__} input {name!r} declared with "
                    f"dtype={spec.dtype} but got tensor of dtype {x.dtype}"
                )

    def _check_step_output(self, out: StepOutput) -> None:
        if not isinstance(out, tuple):
            # Runtime contract check: a user-authored _step may return a
            # non-tuple despite the annotation, so this branch is reachable.
            raise TypeError(  # pyright: ignore[reportUnreachable]
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

    # Input canonicalization

    def _canonicalize_inputs(
        self,
        inputs: Tensor | tuple[Tensor, ...] | list[Tensor] | dict[str, Tensor],
    ) -> tuple[Tensor, ...]:
        """
        Normalize a public input value into a canonical ordered tuple matching
        ``_bt_input_names``.

        Accepts a single tensor (single-input modules), a tuple/list of tensors
        in declaration order, or a dict keyed by input name (ordered by
        declaration). Structural errors raise regardless of the ``validate``
        flag; they indicate a programming mistake, not a shape mismatch.
        """
        if isinstance(inputs, dict):
            missing = [
                name for name in self._bt_input_names if name not in inputs
            ]

            if missing:
                raise ValueError(
                    f"{type(self).__name__} input dict is missing keys {missing}"
                )

            return tuple(inputs[name] for name in self._bt_input_names)

        if isinstance(inputs, (tuple, list)):
            if len(inputs) != len(self._bt_input_names):
                raise ValueError(
                    f"{type(self).__name__} expects {len(self._bt_input_names)} "
                    f"inputs, got {len(inputs)}"
                )

            for t in inputs:
                if not isinstance(t, Tensor):
                    raise TypeError(
                        f"{type(self).__name__} inputs must be tensors, "
                        f"got {type(t).__name__}"
                    )

            return tuple(inputs)

        if isinstance(inputs, Tensor):
            if len(self._bt_input_names) != 1:
                raise ValueError(
                    f"{type(self).__name__} expects "
                    f"{len(self._bt_input_names)} inputs, got a single tensor"
                )

            return (inputs,)

        raise TypeError(
            f"{type(self).__name__} inputs must be a Tensor, a tuple/list of "
            f"tensors, or a dict keyed by input name; got "
            f"{type(inputs).__name__}"
        )

    def _canonicalize_input_sequence(
        self,
        x_seq: Tensor | tuple[Tensor, ...] | list[Tensor] | dict[str, Tensor],
    ) -> tuple[Tensor, ...]:
        """
        Canonicalize a time-major sequence input (or a tuple/dict of
        sequences) into an ordered tuple of sequences. Every sequence must be
        at least ``(time, ...)`` and all sequences must share the time length.
        """
        inputs_seq = self._canonicalize_inputs(x_seq)

        for seq in inputs_seq:
            if seq.dim() < 3:
                raise ValueError(
                    f"{type(self).__name__} expects (time, batch, features) "
                    f"sequence inputs, got {seq.dim()} dims"
                )

        time = inputs_seq[0].shape[0]

        if any(seq.shape[0] != time for seq in inputs_seq):
            raise ValueError(
                f"{type(self).__name__} requires all input sequences to share "
                f"the same time length"
            )

        return inputs_seq

    @staticmethod
    def _first_inputs(inputs_seq: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Per-timestep inputs for the first timestep of a sequence scan.
        """
        return tuple(seq[0] for seq in inputs_seq)

    # Hidden-mode step

    def _hidden_step(self, inputs: tuple[Tensor, ...]) -> StepOutput:
        """
        One hidden-mode step: run the pure explicit step on the current hidden
        buffers, then store the results back into the buffers.
        """
        state = tuple(getattr(self, name) for name in self._bt_state_names)

        out = self._forward_explicit(inputs, state)

        self._store_hidden_outputs(out)
        return out

    def _forward_hidden(self, inputs: tuple[Tensor, ...]) -> Tensor | StepOutput:
        if not self._bt_allocated:
            self._alloc_hidden(inputs)
        elif self.validate:
            self._check_hidden_input_shape(inputs)

        out = self._hidden_step(inputs)

        n_outputs = len(self._bt_output_names)

        if n_outputs == 1:
            return out[0]

        return out[:n_outputs]

    # Explicit-mode step

    def _forward_explicit(
        self,
        inputs: tuple[Tensor, ...],
        state: tuple[Tensor, ...],
    ) -> StepOutput:
        if self.validate:
            expected_inputs = len(self._bt_input_names)

            if len(inputs) != expected_inputs:
                raise ValueError(
                    f"{type(self).__name__} expects {expected_inputs} input "
                    f"tensors, got {len(inputs)}"
                )

            expected = len(self._bt_state_names)
            if len(state) != expected:
                raise ValueError(
                    f"{type(self).__name__} expects {expected} state tensors, "
                    f"got {len(state)}"
                )

            self._check_input_dtypes(inputs)

        out = self._step(*inputs, *state)

        if self.validate:
            self._check_step_output(out)

        if isinstance(out, tuple):
            spk = out[0]
            pre_state = out[1:]
            next_state = self._bt_apply_resets(pre_state, spk)
            return (spk,) + tuple(next_state)

        return out

    # Public forward

    def forward(
        self,
        inputs: Tensor | tuple[Tensor, ...] | list[Tensor] | dict[str, Tensor],
        *state: Tensor,
    ) -> Tensor | StepOutput:
        inputs = self._canonicalize_inputs(inputs)

        if self.init_hidden:
            if state:
                raise ValueError(
                    f"{type(self).__name__} is in init_hidden=True mode; "
                    f"do not pass state explicitly"
                )

            return self._forward_hidden(inputs)

        return self._forward_explicit(inputs, tuple(state))

    # Trainer / loop convenience

    def step_state(
        self,
        inputs: Tensor | tuple[Tensor, ...] | list[Tensor] | dict[str, Tensor],
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

        out = self.forward(inputs, *state)
        assert isinstance(out, tuple)

        n_outputs = len(self._bt_output_names)

        if n_outputs == 1:
            return out[0], out[n_outputs:]

        return out[:n_outputs], out[n_outputs:]

    def step(
        self,
        inputs: Tensor | tuple[Tensor, ...] | list[Tensor] | dict[str, Tensor],
        state: tuple[Tensor, ...],
    ) -> tuple[Tensor | StepOutput, tuple[Tensor, ...]]:
        """
        Alias of step_state.
        """
        return self.step_state(inputs, state)

    # State factories

    def initial_state(
        self,
        batch_shape: tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, ...]:
        """
        Create canonical initial explicit state using Spec defaults.

        ``batch_shape`` is the shape of the primary step input. States whose
        ``StateSpec.shape`` is an explicit tuple use that tuple instead, so
        the explicit state factories agree with hidden-mode allocation. For
        multi-input modules, prefer ``initial_state_like(inputs)`` so states
        shaped by a named input resolve from the real input shapes.
        """
        dtype = self._explicit_state_dtype(dtype)

        state: list[Tensor] = []

        for spec in self._bt_state_specs:
            shape = spec.shape if isinstance(spec.shape, tuple) else batch_shape

            state.append(
                torch.full(
                    shape,
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

        ``batch_shape`` is the shape of the primary step input. States whose
        ``StateSpec.shape`` is an explicit tuple use that tuple instead.
        """
        dtype = self._explicit_state_dtype(dtype)

        state: list[Tensor] = []

        for spec in self._bt_state_specs:
            shape = spec.shape if isinstance(spec.shape, tuple) else batch_shape

            state.append(
                torch.zeros(
                    shape,
                    device=device,
                    dtype=dtype,
                )
            )

        return tuple(state)

    def initial_state_like(
        self,
        inputs: Tensor | tuple[Tensor, ...] | list[Tensor] | dict[str, Tensor],
        batch_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[Tensor, ...]:
        """
        Create initial state from example inputs, resolving each state's shape
        from the input it references (``StateSpec.shape="input"`` follows the
        primary input; a named string follows that input). Uses the primary
        input for device and dtype.
        """
        inputs = self._canonicalize_inputs(inputs)
        primary = inputs[self._bt_primary_input_index]
        dtype = self._explicit_state_dtype(primary.dtype)

        state: list[Tensor] = []

        for spec in self._bt_state_specs:
            shape = (
                tuple(batch_shape)
                if batch_shape is not None
                else self._spec_shape(spec, inputs)
            )

            state.append(
                torch.full(
                    shape,
                    self._resolve_default(spec.default),
                    device=primary.device,
                    dtype=dtype,
                )
            )

        return tuple(state)

    def zero_state_like(
        self,
        inputs: Tensor | tuple[Tensor, ...] | list[Tensor] | dict[str, Tensor],
        batch_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[Tensor, ...]:
        """
        Create zeroed state from example inputs (see ``initial_state_like``).
        """
        inputs = self._canonicalize_inputs(inputs)
        primary = inputs[self._bt_primary_input_index]
        dtype = self._explicit_state_dtype(primary.dtype)

        state: list[Tensor] = []

        for spec in self._bt_state_specs:
            shape = (
                tuple(batch_shape)
                if batch_shape is not None
                else self._spec_shape(spec, inputs)
            )

            state.append(
                torch.zeros(
                    shape,
                    device=primary.device,
                    dtype=dtype,
                )
            )

        return tuple(state)

    def initial_state_for_sequence(
        self,
        x_seq: Tensor | tuple[Tensor, ...] | list[Tensor] | dict[str, Tensor],
    ) -> tuple[Tensor, ...]:
        inputs_seq = self._canonicalize_input_sequence(x_seq)

        return self.initial_state_like(self._first_inputs(inputs_seq))

    # Reset / detach

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

    # Checkpointing hidden state

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

    # Sequence scan

    def forward_sequence(
        self,
        x_seq: Tensor | tuple[Tensor, ...] | list[Tensor] | dict[str, Tensor],
        state: Optional[tuple[Tensor, ...]] = None,
    ) -> Tensor | StepOutput:
        """
        Time-major sequence scan.

        Input shape:
            (time, batch, features)

        Multi-input modules pass a tuple/dict of sequences, one per declared
        input (all sharing the time length):

            (x_seq, inh_seq)

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

        return self._reference_sequence_scan(
            self._canonicalize_input_sequence(x_seq),
            state,
        )

    def _reference_sequence_scan(
        self,
        inputs_seq: tuple[Tensor, ...],
        state: Optional[tuple[Tensor, ...]] = None,
    ) -> Tensor | StepOutput:
        """
        The reference per-step scan. This is the compile unit for
        ``compile_sequence_scan``; keep it free of state-allocation side
        effects that break tracing. ``inputs_seq`` must already be canonical.
        """
        if inputs_seq[0].shape[0] == 0:
            raise ValueError(
                f"{type(self).__name__}.forward_sequence expects at least one timestep"
            )

        if self.init_hidden:
            if state is not None:
                raise ValueError(
                    f"{type(self).__name__} is in init_hidden=True mode; "
                    f"forward_sequence does not accept explicit state"
                )

            return self._hidden_sequence_scan(inputs_seq)

        return self._explicit_sequence_scan(inputs_seq, state)

    def _hidden_sequence_scan(
        self,
        inputs_seq: tuple[Tensor, ...],
    ) -> Tensor | StepOutput:
        first_inputs = self._first_inputs(inputs_seq)

        if not self._bt_allocated:
            self._alloc_hidden(first_inputs)
        elif self.validate:
            self._check_hidden_input_shape(first_inputs)

        n_outputs = len(self._bt_output_names)
        state0 = tuple(getattr(self, name) for name in self._bt_state_names)

        # Hidden mode is an explicit scan plus buffer bookkeeping at the
        # edges: the scan itself is pure, so it can share sequence_scan with
        # explicit mode (and with multi-module containers).
        result = sequence_scan(
            lambda inputs, s: self._forward_explicit(inputs, s),
            inputs_seq,
            state0,
            n_outputs,
        )

        ys = result[:n_outputs]

        for name, t in zip(self._bt_state_names, result[n_outputs:]):
            self._buffers[name] = t

        for name, spec, t in zip(self._bt_output_names, self._bt_output_specs, ys):
            last = t[-1] if t.dim() > 0 else t
            if not spec.differentiable:
                last = last.detach()
            self._buffers[name] = last

        if n_outputs == 1:
            return ys[0]

        return ys

    def _explicit_sequence_scan(
        self,
        inputs_seq: tuple[Tensor, ...],
        state: Optional[tuple[Tensor, ...]],
    ) -> StepOutput:
        if state is None:
            state = self.initial_state_for_sequence(inputs_seq)

        n_outputs = len(self._bt_output_names)

        result = sequence_scan(
            lambda inputs, s: self._forward_explicit(inputs, s),
            inputs_seq,
            state,
            n_outputs,
        )

        if n_outputs == 1:
            return (result[0], *result[1:])

        return result

    def compile_sequence_scan(self, **kwargs: Any) -> BlowtorchModule:
        """
        Compile the reference sequence scan and route forward_sequence through it.

        Output tensors are cloned before returning only when CUDA graphs are in
        play (``mode="reduce-overhead"`` or ``mode="max-autotune"``): a subsequent
        graph run would otherwise overwrite the previously returned tensor. With
        any other mode the outputs are returned as-is - cloning would cost a full
        extra copy of the spike tensor on every call.

        Explicit mode and ``state=None``: the initial state is allocated inside
        the compiled call on every invocation. That is functionally correct in
        all modes (the returned state is freshly allocated, and graph modes
        clone it), but it is a per-call allocation the compiler must plan for.
        If you call the compiled scan repeatedly on chunks of a long sequence,
        allocate the state once with ``initial_state`` and pass it explicitly to
        skip that allocation and keep the output state un-cloned in default mode.

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
            x_seq: Tensor | tuple[Tensor, ...] | list[Tensor] | dict[str, Tensor],
            state: Optional[tuple[Tensor, ...]] = None,
        ) -> Tensor | StepOutput:
            inputs_seq = self._canonicalize_input_sequence(x_seq)

            if self.init_hidden and inputs_seq[0].shape[0] > 0:
                # Allocate hidden buffers *before* the compiled call. If the
                # initial trace ran the alloc path, the buffer registration
                # side effects break the scan into separate graphs (and under
                # CUDA graphs they alias the compiler's memory pool).
                self.allocate_like(self._first_inputs(inputs_seq))

            out = compiled(inputs_seq, state)

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

    # Repr

    def extra_repr(self) -> str:
        parts: list[str] = []

        if self.size is not None:
            parts.append(f"size={self.size}")

        if len(self._bt_input_names) != 1 or self._bt_input_names[0] != "x":
            parts.append(f"inputs={self._bt_input_names}")

        parts.append(f"init_hidden={self.init_hidden}")

        return ", ".join(parts)

    # Abstract step

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