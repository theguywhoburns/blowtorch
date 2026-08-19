from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional, Self

import torch

from .specs import (
    InputTensor,
    OutputSpec,
    Spec,
    StateSpec,
    StepOutput,
    Tensor,
    _floating_dtype,
)

if TYPE_CHECKING:
    from . import BlowtorchModule

# State/output declarations, hidden-mode allocation, and state factories.
# Host members used below but owned by earlier mixins are declared as type-only
# stubs; BlowtorchModule's MRO (this mixin sits after their owners) resolves
# the real implementations at runtime.


class StateMixin:
    _bt_spec_entries: ClassVar[tuple[tuple[str, Spec], ...]] = ()
    _bt_output_names: ClassVar[tuple[str, ...]] = ()
    _bt_state_names: ClassVar[tuple[str, ...]] = ()
    _bt_output_specs: ClassVar[tuple[OutputSpec, ...]] = ()
    _bt_state_specs: ClassVar[tuple[StateSpec, ...]] = ()
    _bt_spec_extensions: ClassVar[dict[str, Callable[..., Any]]] = {}

    _bt_input_names: ClassVar[tuple[str, ...]]
    _bt_primary_input_index: ClassVar[int]
    init_hidden: bool
    _bt_allocated: bool
    _buffers: dict[str, Optional[Tensor]]
    _non_persistent_buffers_set: set[str]

    def _canonicalize_inputs(
        self,
        inputs: InputTensor,
    ) -> tuple[Tensor, ...]: ...

    def _process_spec_extensions(self) -> None:
        """
        Dispatch each StateSpec extra through the handler registry.

        Handlers run after specs are resolved so they can rely on class
        metadata (``_bt_state_specs``, ``_bt_param_specs``) and on instance
        parameters already being registered.
        """
        for i, spec in enumerate(self._bt_state_specs):
            for key, value in spec.extras.items():
                handler = self._bt_spec_extensions.get(key)
                if handler is not None:
                    handler(self, i, spec, value)

    def _resolve_default(
        self,
        value: float | Callable[[Any], float],
    ) -> float:
        if callable(value):
            return float(value(self))
        return float(value)

    def _spec_shape(self, spec: Spec, inputs: tuple[Tensor, ...]) -> tuple[int, ...]:
        shape = getattr(spec, "shape", "input")

        if shape is None or shape == "input":
            return tuple(inputs[self._bt_primary_input_index].shape)

        if isinstance(shape, str):
            if shape not in self._bt_input_names:
                raise ValueError(
                    f"{type(self).__name__} StateSpec(shape={shape!r}) refers "
                    f"to an unknown input; declared inputs are "
                    f"{self._bt_input_names}"
                )

            return tuple(inputs[self._bt_input_names.index(shape)].shape)

        assert isinstance(shape, tuple)
        return shape

    def _explicit_state_dtype(
        self,
        dtype: Optional[torch.dtype],
    ) -> Optional[torch.dtype]:
        if dtype is None:
            return None
        return _floating_dtype(dtype)

    def _alloc_hidden(self, inputs: tuple[Tensor, ...]) -> None:
        primary = inputs[self._bt_primary_input_index]
        dtype = (
            primary.dtype
            if primary.is_floating_point()
            else torch.get_default_dtype()
        )

        for name, spec in self._bt_spec_entries:
            self._buffers[name] = torch.full(
                self._spec_shape(spec, inputs),
                self._resolve_default(spec.default),
                device=primary.device,
                dtype=dtype,
            )
            self._non_persistent_buffers_set.add(name)

        self._bt_allocated = True

    def allocate_like(
        self,
        *inputs: InputTensor,
    ) -> Self:
        """
        Materialize hidden buffers outside of ``torch.compile``.

        Lazy allocation on the first forward call happens inside the compiled
        region and breaks the scan graph (and, under CUDA graphs, registers
        buffers that alias the compiler's recycled memory pool). Call this once
        eagerly before compiling an ``init_hidden=True`` module:

            module.allocate_like(x)
            module.allocate_like((x, inh))
            module.fast_sequence_()

        The call is a no-op once buffers exist. Hidden buffers keep the shape
        of the first inputs; later inputs with different batch/feature dims
        raise ``ValueError`` from the forward paths when validation is on.
        """
        if self.init_hidden and not self._bt_allocated:
            if len(inputs) == 1:
                canonical = self._canonicalize_inputs(inputs[0])
            else:
                expanded: list[Tensor] = []

                for t in inputs:
                    if not isinstance(t, Tensor):
                        raise TypeError(
                            f"{type(self).__name__}.allocate_like expanded "
                            f"inputs must be tensors, got {type(t).__name__}"
                        )

                    expanded.append(t)

                canonical = tuple(expanded)

            self._alloc_hidden(canonical)

        return self

    def _store_hidden_outputs(self, out: StepOutput) -> None:
        for (name, spec), t in zip(self._bt_spec_entries, out):
            if not spec.differentiable:
                t = t.detach()

            # Buffers already exist after _alloc_hidden.
            self._buffers[name] = t

    def initial_state(
        self,
        batch_shape: tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, ...]:
        """
        Create canonical initial explicit state using Spec defaults.

        ``batch_shape`` is the shape of the primary step input. States whose
        ``StateSpec.shape`` is an explicit tuple use that tuple instead, so
        the explicit state factories agree with hidden-mode allocation. For
        multi-input modules, prefer ``initial_state_like(inputs)`` so states
        shaped by a named input resolve from the real input shapes.
        """
        dtype = self._explicit_state_dtype(dtype)

        state: list[Tensor] = []

        for spec in self._bt_state_specs:
            shape = spec.shape if isinstance(spec.shape, tuple) else batch_shape

            state.append(
                torch.full(
                    shape,
                    self._resolve_default(spec.default),
                    device=device,
                    dtype=dtype,
                )
            )

        return tuple(state)

    def zero_state(
        self,
        batch_shape: tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, ...]:
        """
        Create zeroed explicit state.

        ``batch_shape`` is the shape of the primary step input. States whose
        ``StateSpec.shape`` is an explicit tuple use that tuple instead.
        """
        dtype = self._explicit_state_dtype(dtype)

        state: list[Tensor] = []

        for spec in self._bt_state_specs:
            shape = spec.shape if isinstance(spec.shape, tuple) else batch_shape

            state.append(
                torch.zeros(
                    shape,
                    device=device,
                    dtype=dtype,
                )
            )

        return tuple(state)

    def initial_state_like(
        self,
        inputs: InputTensor,
        batch_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[Tensor, ...]:
        """
        Create initial state from example inputs, resolving each state's shape
        from the input it references (``StateSpec.shape="input"`` follows the
        primary input; a named string follows that input). Uses the primary
        input for device and dtype.
        """
        inputs = self._canonicalize_inputs(inputs)
        primary = inputs[self._bt_primary_input_index]
        dtype = self._explicit_state_dtype(primary.dtype)

        state: list[Tensor] = []

        for spec in self._bt_state_specs:
            shape = (
                tuple(batch_shape)
                if batch_shape is not None
                else self._spec_shape(spec, inputs)
            )

            state.append(
                torch.full(
                    shape,
                    self._resolve_default(spec.default),
                    device=primary.device,
                    dtype=dtype,
                )
            )

        return tuple(state)

    def zero_state_like(
        self,
        inputs: InputTensor,
        batch_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[Tensor, ...]:
        """
        Create zeroed state from example inputs (see ``initial_state_like``).
        """
        inputs = self._canonicalize_inputs(inputs)
        primary = inputs[self._bt_primary_input_index]
        dtype = self._explicit_state_dtype(primary.dtype)

        state: list[Tensor] = []

        for spec in self._bt_state_specs:
            shape = (
                tuple(batch_shape)
                if batch_shape is not None
                else self._spec_shape(spec, inputs)
            )

            state.append(
                torch.zeros(
                    shape,
                    device=primary.device,
                    dtype=dtype,
                )
            )

        return tuple(state)

    def reset(self) -> None:
        """
        Reset hidden buffers to their Spec defaults.
        """
        if not self.init_hidden:
            return

        for name, spec in self._bt_spec_entries:
            t = self._buffers.get(name, None)

            if isinstance(t, Tensor):
                self._buffers[name] = torch.full_like(
                    t,
                    self._resolve_default(spec.default),
                )

    def detach(self) -> None:
        """
        Detach hidden buffers from autograd.
        """
        if not self.init_hidden:
            return

        for name, _ in self._bt_spec_entries:
            t = self._buffers.get(name, None)

            if isinstance(t, Tensor):
                self._buffers[name] = t.detach()