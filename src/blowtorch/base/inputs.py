from __future__ import annotations

from typing import ClassVar

from .specs import InputSpec, InputTensor, Tensor


class InputMixin:
    """Declared named step inputs and public-input canonicalization."""

    _bt_input_entries: ClassVar[tuple[tuple[str, InputSpec], ...]] = (
        ("x", InputSpec(primary=True)),
    )
    _bt_input_names: ClassVar[tuple[str, ...]] = ("x",)
    _bt_input_specs: ClassVar[tuple[InputSpec, ...]] = (InputSpec(primary=True),)
    _bt_primary_input_index: ClassVar[int] = 0

    def _canonicalize_inputs(
        self,
        inputs: InputTensor,
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