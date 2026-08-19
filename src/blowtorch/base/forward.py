from __future__ import annotations

from typing import ClassVar, Optional

import torch

from .specs import (
    InputSpec,
    InputTensor,
    Spec,
    StateSpec,
    StepOutput,
    Tensor,
)
from .validation import (
    check_hidden_input_shape,
    check_input_dtypes,
    check_step_output,
    is_validating,
)

# The per-timestep forward path and loop conveniences. Host members used below
# but owned by earlier mixins are declared as type-only stubs; BlowtorchModule's
# MRO (this mixin sits after their owners) resolves the real implementations.


class ForwardMixin:
    init_hidden: bool
    size: Optional[int]
    _bt_allocated: bool
    _bt_input_names: ClassVar[tuple[str, ...]]
    _bt_state_names: ClassVar[tuple[str, ...]]
    _bt_output_names: ClassVar[tuple[str, ...]]
    _validate_override: Optional[bool]
    _bt_state_specs: ClassVar[tuple[StateSpec, ...]]
    _bt_input_specs: ClassVar[tuple[InputSpec, ...]]
    _bt_spec_entries: ClassVar[tuple[tuple[str, Spec], ...]]
    _buffers: dict[str, Optional[Tensor]]

    def _spec_shape(self, spec: Spec, inputs: tuple[Tensor, ...]) -> tuple[int, ...]: ...

    def _canonicalize_inputs(
        self,
        inputs: InputTensor,
    ) -> tuple[Tensor, ...]: ...

    def _alloc_hidden(self, inputs: tuple[Tensor, ...]) -> None: ...

    def _store_hidden_outputs(self, out: StepOutput) -> None: ...

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
        elif is_validating(self):
            check_hidden_input_shape(self, inputs)

        out = self._hidden_step(inputs)

        n_outputs = len(self._bt_output_names)

        if n_outputs == 1:
            return out[0]

        return out[:n_outputs]

    def _post_step(self, out: StepOutput) -> StepOutput:
        """
        Hook applied to the raw output of ``_step`` before it is returned or
        stored.

        Base modules return the output unchanged. Subclasses (e.g.
        ``SnnModule``) override this to apply domain-specific transformations
        such as per-state resets.
        """
        return out

    def _forward_explicit(
        self,
        inputs: tuple[Tensor, ...],
        state: tuple[Tensor, ...],
    ) -> StepOutput:
        if is_validating(self):
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

            check_input_dtypes(self, inputs)

        out = self._step(*inputs, *state)

        if is_validating(self):
            check_step_output(self, out)

        return self._post_step(out)

    def forward(
        self,
        inputs: InputTensor,
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

    def step_state(
        self,
        inputs: InputTensor,
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
        inputs: InputTensor,
        state: tuple[Tensor, ...],
    ) -> tuple[Tensor | StepOutput, tuple[Tensor, ...]]:
        """
        Alias of step_state.
        """
        return self.step_state(inputs, state)

    def _step(self, x: Tensor, *state: Tensor) -> StepOutput:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _step()"
        )