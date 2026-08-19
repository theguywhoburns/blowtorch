from __future__ import annotations

from typing import Any, ClassVar

from .specs import ConstantSpec


class ConstantMixin:
    """Declared construction-time constants (non-tensor hyperparameters)."""

    _bt_constant_specs: ClassVar[dict[str, ConstantSpec]] = {}
    _bt_constant_annotations: ClassVar[dict[str, Any]] = {}