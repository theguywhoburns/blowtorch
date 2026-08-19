from __future__ import annotations

import inspect
import keyword
from typing import TYPE_CHECKING, Any, Optional

from .specs import (
    ConstantSpec,
    Constraint,
    InputSpec,
    OutputSpec,
    ParamSpec,
    Spec,
    StateSpec,
)

if TYPE_CHECKING:
    from . import BlowtorchModule


def collect_params(
    cls: type[BlowtorchModule],
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


def collect_specs(cls: type[BlowtorchModule]) -> tuple[tuple[str, Spec], ...]:
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


def collect_inputs(cls: type[BlowtorchModule]) -> tuple[tuple[str, InputSpec], ...]:
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


def check_input_namespace_collisions(cls: type[BlowtorchModule]) -> None:
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


def collect_metadata(cls: type[BlowtorchModule]) -> None:
    """
    Gather all ``_bt_*`` metadata on a freshly created subclass.

    Runs from ``__init_subclass__`` after the base's own hook; every derived
    module keeps its own copy of the collected specs/names so later
    subclasses in the MRO cannot leak state into it.
    """
    (
        cls._bt_param_specs,
        cls._bt_param_annotations,
        cls._bt_constant_specs,
        cls._bt_constant_annotations,
    ) = collect_params(cls)
    cls._bt_spec_entries = collect_specs(cls)

    cls._bt_input_entries = collect_inputs(cls)
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

    check_input_namespace_collisions(cls)

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


def generate_signature(cls: type[BlowtorchModule]) -> None:
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

    setattr(cls, "__signature__", inspect.Signature(sig_params))