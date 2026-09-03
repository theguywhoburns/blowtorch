from __future__ import annotations

from typing import Any, ClassVar

from ..specs import ConstantSpec


class ConstantMixin:
    """Declared construction-time constants (non-tensor hyperparameters)."""

    _cr_constant_specs: ClassVar[dict[str, ConstantSpec]] = {}
    _cr_constant_annotations: ClassVar[dict[str, Any]] = {}
