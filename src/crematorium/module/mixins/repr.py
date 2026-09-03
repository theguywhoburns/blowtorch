from __future__ import annotations

from typing import ClassVar, Optional

from .inputs import InputMixin


class ReprMixin(InputMixin):
    """Module repr."""

    size: Optional[int]
    init_hidden: bool
    _cr_input_names: ClassVar[tuple[str, ...]]

    def extra_repr(self) -> str:
        parts: list[str] = []

        if self.size is not None:
            parts.append(f"size={self.size}")

        if len(self._cr_input_names) != 1 or self._cr_input_names[0] != "x":
            parts.append(f"inputs={self._cr_input_names}")

        parts.append(f"init_hidden={self.init_hidden}")

        return ", ".join(parts)