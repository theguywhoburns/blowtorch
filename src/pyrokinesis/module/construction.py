from __future__ import annotations

import keyword
import types
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn as nn

from .specs import Constraint, identity

_PK_CONSTRAINED_CACHE: dict[tuple, Callable[..., Any]] = {}

if TYPE_CHECKING:
    from . import PyroModule


def build_params(
    module: PyroModule,
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], list[Constraint]]:
    """
    Pop declared param/constant kwargs and register parameters.

    ``kwargs`` is consumed in place: every key matched to a ``ParamSpec`` or
    ``ConstantSpec`` is removed. Returns the leftover kwargs (unmatched keys)
    and the list of per-parameter constraint functions in declaration order.
    """
    constraint_fns: list[Constraint] = []

    _UNSET = object()

    for name, spec in module._pk_param_specs.items():
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
                f"{type(module).__name__} parameter {name!r} has no value or default"
            )

        tensor = torch.as_tensor(
            value,
            dtype=spec.dtype if isinstance(spec.dtype, torch.dtype) else None,
        )

        if not tensor.is_floating_point():
            tensor = tensor.to(torch.get_default_dtype())

        if isinstance(value, torch.Tensor):
            # Never alias caller-owned storage: later mutation of the
            # source tensor must not move the parameter.
            tensor = tensor.detach().clone()

        module.register_parameter(
            name,
            nn.Parameter(
                tensor,
                requires_grad=learnable,
            ),
        )

        # Fixed parameters skip constraint work entirely.
        constraint_fns.append(constraint if learnable else identity)

    for name, spec in module._pk_constant_specs.items():
        value = kwargs.pop(name, spec.default)
        value_orig = value

        if spec.validate is not None:
            try:
                spec.validate(value)
            except ValueError as exc:
                raise ValueError(
                    f"{type(module).__name__} {name}: {exc}"
                ) from None

        if isinstance(spec.dtype, type):
            try:
                value = spec.dtype(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{type(module).__name__} {name}: value {value_orig!r} "
                    f"cannot be stored as {spec.dtype.__name__}"
                ) from None
            if value != value_orig:
                raise ValueError(
                    f"{type(module).__name__} {name}: value {value_orig!r} "
                    f"cannot be stored as {spec.dtype.__name__}"
                )

        setattr(module, name, value)

    return kwargs, constraint_fns


def _pk_install_constrained(module: PyroModule) -> None:
    """
    Freeze the constrained-parameter return expression on the module.

    Example generated function for LIF:

        def _pk_constrained(self):
            return (self._pk_constraint_0(self.beta), self.threshold)
    """
    exprs: list[str] = []

    # Safety: the generated body below is exec'd, so every interpolated
    # name must come from module-controlled sources. Param names are
    # restricted to non-keyword Python identifiers (validated here and at
    # class creation); constraint fns are attributes set via setattr from
    # the declarative ParamSpecs, never from user strings. Nothing from
    # user input can reach the exec namespace.
    param_names = tuple(module._pk_param_specs.keys())

    for i, (name, constraint) in enumerate(
        zip(param_names, module._pk_constraints, strict=True)
    ):
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(
                f"Param name {name!r} must be a valid non-keyword Python identifier"
            )

        if constraint is identity:
            exprs.append(f"self.{name}")
        else:
            attr = f"_pk_constraint_{i}"
            setattr(module, attr, constraint)
            exprs.append(f"self.{attr}(self.{name})")

    if exprs:
        ret = f"({', '.join(exprs)},)"
    else:
        ret = "()"

    src = f"def _pk_constrained(self):\n    return {ret}\n"
    key = (param_names, tuple(id(c) for c in module._pk_constraints), src)
    cached = _PK_CONSTRAINED_CACHE.get(key)
    if cached is not None:
        module._pk_constrained_fn = types.MethodType(cached, module)
        return

    ns: dict[str, Any] = {}
    exec(src, ns)

    _PK_CONSTRAINED_CACHE[key] = ns["_pk_constrained"]
    module._pk_constrained_fn = types.MethodType(ns["_pk_constrained"], module)


def remaining_kwargs_error(
    module: PyroModule,
    kwargs: dict[str, Any],
) -> TypeError:
    """
    Build the leftover-kwarg TypeError with the exact message users expect.
    """
    return TypeError(
        f"{type(module).__name__} got unexpected keyword arguments: "
        f"{sorted(kwargs)}"
    )