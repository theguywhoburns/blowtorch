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
# but owned by earlier mixins are declared as type-only stubs; PyroModule's
# MRO (this mixin sits after their owners) resolves the real implementations.


class ForwardMixin:
    init_hidden: bool
    size: Optional[int]
    _pk_allocated: bool
    _pk_input_names: ClassVar[tuple[str, ...]]
    _pk_state_names: ClassVar[tuple[str, ...]]
    _pk_output_names: ClassVar[tuple[str, ...]]
    _validate_override: Optional[bool]
    _pk_state_specs: ClassVar[tuple[StateSpec, ...]]
    _pk_input_specs: ClassVar[tuple[InputSpec, ...]]
    _pk_spec_entries: ClassVar[tuple[tuple[str, Spec], ...]]
    _buffers: dict[str, Optional[Tensor]]

    def _pk_spec_shape(self, spec: Spec, inputs: tuple[Tensor, ...]) -> tuple[int, ...]: ...

    def _pk_canonicalize_inputs(
        self,
        inputs: InputTensor,
    ) -> tuple[Tensor, ...]: ...

    def _pk_alloc_hidden(self, inputs: tuple[Tensor, ...]) -> None: ...

    def _pk_store_hidden_outputs(self, out: StepOutput) -> None: ...

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

    def _pk_hidden_step(self, inputs: tuple[Tensor, ...]) -> StepOutput:
        """
        One hidden-mode step: run the pure explicit step on the current hidden
        buffers, then store the results back into the buffers.
        """
        state = tuple(getattr(self, name) for name in self._pk_state_names)

        out = self._pk_forward_explicit(inputs, state)

        self._pk_store_hidden_outputs(out)
        return out

    def _pk_forward_hidden(self, inputs: tuple[Tensor, ...]) -> Tensor | StepOutput:
        if not self._pk_allocated:
            self._pk_alloc_hidden(inputs)
        elif is_validating(self):
            check_hidden_input_shape(self, inputs)

        out = self._pk_hidden_step(inputs)

        n_outputs = len(self._pk_output_names)

        if n_outputs == 1:
            return out[0]

        return out[:n_outputs]

    def _pk_post_step(self, out: StepOutput) -> StepOutput:
        """
        Hook applied to the raw output of ``_step`` before it is returned or
        stored.

        Base modules return the output unchanged. Subclasses (e.g.
        ``SnnModule``) override this to apply domain-specific transformations
        such as per-state resets.
        """
        return out

    def _pk_forward_explicit(
        self,
        inputs: tuple[Tensor, ...],
        state: tuple[Tensor, ...],
    ) -> StepOutput:
        if is_validating(self):
            expected_inputs = len(self._pk_input_names)

            if len(inputs) != expected_inputs:
                raise ValueError(
                    f"{type(self).__name__} expects {expected_inputs} input "
                    f"tensors, got {len(inputs)}"
                )

            expected = len(self._pk_state_names)
            if len(state) != expected:
                raise ValueError(
                    f"{type(self).__name__} expects {expected} state tensors, "
                    f"got {len(state)}"
                )

            check_input_dtypes(self, inputs)

        out = self._step(*inputs, *state)

        if is_validating(self):
            check_step_output(self, out)

        return self._pk_post_step(out)

    def forward(
        self,
        inputs: InputTensor,
        *state: Tensor,
    ) -> Tensor | StepOutput:
        inputs = self._pk_canonicalize_inputs(inputs)

        if self.init_hidden:
            if state:
                raise ValueError(
                    f"{type(self).__name__} is in init_hidden=True mode; "
                    f"do not pass state explicitly"
                )

            return self._pk_forward_hidden(inputs)

        return self._pk_forward_explicit(inputs, tuple(state))

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

        n_outputs = len(self._pk_output_names)

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