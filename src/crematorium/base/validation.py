from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar, Optional, Protocol

import torch

from .specs import (
    InputSpec,
    Spec,
    StateSpec,
    StepOutput,
    Tensor,
    _floating_dtype,
)

if TYPE_CHECKING:
    pass

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


# The checks below are free functions, so mixin and host code both call them
# with `self` no matter how `self` is typed. They operate on the minimal host
# surface declared by _ValidationHost; crematoriumModule and SequenceScanMixin
# both satisfy it structurally.


class _ValidationHost(Protocol):
    _validate_override: Optional[bool]
    _bt_state_names: ClassVar[tuple[str, ...]]
    _bt_state_specs: ClassVar[tuple[StateSpec, ...]]
    _bt_input_names: ClassVar[tuple[str, ...]]
    _bt_input_specs: ClassVar[tuple[InputSpec, ...]]
    _bt_spec_entries: ClassVar[tuple[tuple[str, Spec], ...]]
    _buffers: dict[str, Optional[Tensor]]

    def _spec_shape(
        self,
        spec: Spec,
        inputs: tuple[Tensor, ...],
    ) -> tuple[int, ...]: ...


def is_validating(module: _ValidationHost) -> bool:
    """
    Effective validation flag for a module instance.

    ``validate=None`` (the default at construction) follows the global toggle;
    an explicit bool wins.
    """
    override = module._validate_override

    return get_validation() if override is None else override


def set_validating(module: _ValidationHost, value: bool) -> None:
    """Set the per-instance validation override."""
    module._validate_override = bool(value)


def check_hidden_input_shape(
    module: _ValidationHost,
    inputs: tuple[Tensor, ...],
) -> None:
    for name, spec in zip(module._bt_state_names, module._bt_state_specs):
        ref = module._buffers.get(name)
        expected = module._spec_shape(spec, inputs)

        if ref is not None and ref.shape != expected:
            raise ValueError(
                f"{type(module).__name__} hidden buffers were allocated for "
                f"shape {tuple(ref.shape)}, got input shape {tuple(expected)}; "
                f"the batch/feature dims must stay fixed in hidden mode"
            )


def check_input_dtypes(
    module: _ValidationHost,
    inputs: tuple[Tensor, ...],
) -> None:
    for name, spec, x in zip(
        module._bt_input_names, module._bt_input_specs, inputs
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
                    f"{type(module).__name__} input {name!r} declared with "
                    f"dtype=int but got floating-point tensor {x.dtype}"
                )
            continue
        else:
            continue

        if x.dtype != expected:
            raise TypeError(
                f"{type(module).__name__} input {name!r} declared with "
                f"dtype={spec.dtype} but got tensor of dtype {x.dtype}"
            )


def check_step_output(module: _ValidationHost, out: StepOutput) -> None:
    if not isinstance(out, tuple):
        # Runtime contract check: a user-authored _step may return a
        # non-tuple despite the annotation, so this branch is reachable.
        raise TypeError(  # pyright: ignore[reportUnreachable]
            f"{type(module).__name__}._step must return a tuple of tensors"
        )

    expected = len(module._bt_spec_entries)

    if len(out) != expected:
        raise ValueError(
            f"{type(module).__name__}._step returned {len(out)} tensors, "
            f"expected {expected} according to Specs"
        )


class ValidationMixin:
    """The ``validate`` toggle. The actual checks are free functions."""

    _validate_override: Optional[bool]

    @property
    def validate(self) -> bool:
        """
        Runtime validation flag.
        If constructed with validate=None, follows the global toggle.
        If an explicit bool was passed, that value wins.
        """
        override = self._validate_override

        return get_validation() if override is None else override

    @validate.setter
    def validate(self, value: bool) -> None:
        """Allow dynamic toggling: `module.validate = False`."""
        self._validate_override = bool(value)