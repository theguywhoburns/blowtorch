from __future__ import annotations

from typing import Union

__all__ = ["positive"]


def positive(value: Union[int, float]) -> None:
    """Validate that ``value`` is a positive int or float (not bool)."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise ValueError(f"must be a positive number, got {value!r}")