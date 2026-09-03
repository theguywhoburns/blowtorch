from __future__ import annotations

from typing import Any, Callable, ClassVar, Optional, Self

import torch

from ..specs import (
    InputTensor,
    OutputSpec,
    Spec,
    StateSpec,
    StepOutput,
    Tensor,
    _pk_floating_dtype,
)
from .inputs import InputMixin

# State/output declarations, hidden-mode allocation, and state factories.
# Depends on InputMixin (named-input shapes); pure callers below depend on
# this mixin instead of redeclaring its members as stubs.


class StateMixin(InputMixin):
    _pk_spec_entries: ClassVar[tuple[tuple[str, Spec], ...]] = ()
    _pk_output_names: ClassVar[tuple[str, ...]] = ()
    _pk_state_names: ClassVar[tuple[str, ...]] = ()
    _pk_output_specs: ClassVar[tuple[OutputSpec, ...]] = ()
    _pk_state_specs: ClassVar[tuple[StateSpec, ...]] = ()
    _pk_spec_extensions: ClassVar[dict[str, Callable[..., Any]]] = {}

    _pk_input_names: ClassVar[tuple[str, ...]]
    _pk_primary_input_index: ClassVar[int]
    init_hidden: bool
    _pk_allocated: bool
    _buffers: dict[str, Optional[Tensor]]
    _non_persistent_buffers_set: set[str]
    _pk_hook_specs_steps: ClassVar[tuple[Callable[..., Any], ...]]

    def _pk_process_spec_extensions(self) -> None:
        """
        Dispatch each StateSpec extra through the handler registry.

        Handlers run after specs are resolved so they can rely on class
        metadata (``_pk_state_specs``, ``_pk_param_specs``) and on instance
        parameters already being registered. Then runs the frozen
        ``_pk_hook_specs__*`` chain (e.g. reset-fn install); hooks are
        collected once per class, never resolved per call.
        """
        for i, spec in enumerate(self._pk_state_specs):
            for key, value in spec.extras.items():
                handler = self._pk_spec_extensions.get(key)
                if handler is not None:
                    handler(self, i, spec, value)
        for fn in self._pk_hook_specs_steps:
            fn(self)

    def _pk_resolve_default(
        self,
        value: float | Callable[[Any], float],
    ) -> float:
        if callable(value):
            return float(value(self))
        return float(value)

    def _pk_spec_shape(self, spec: Spec, inputs: tuple[Tensor, ...]) -> tuple[int, ...]:
        shape = getattr(spec, "shape", "input")

        if shape is None or shape == "input":
            return tuple(inputs[self._pk_primary_input_index].shape)

        if isinstance(shape, str):
            if shape not in self._pk_input_names:
                raise ValueError(
                    f"{type(self).__name__} StateSpec(shape={shape!r}) refers "
                    f"to an unknown input; declared inputs are "
                    f"{self._pk_input_names}"
                )

            return tuple(inputs[self._pk_input_names.index(shape)].shape)

        assert isinstance(shape, tuple)
        return shape

    def _pk_explicit_state_dtype(
        self,
        dtype: Optional[torch.dtype],
    ) -> Optional[torch.dtype]:
        if dtype is None:
            return None
        return _pk_floating_dtype(dtype)

    def _pk_alloc_hidden(self, inputs: tuple[Tensor, ...]) -> None:
        primary = inputs[self._pk_primary_input_index]
        dtype = (
            primary.dtype
            if primary.is_floating_point()
            else torch.get_default_dtype()
        )

        # Allocate hidden STATE buffers only. Output buffers are written by
        # the first store (_pk_store_hidden_outputs / the sequence-buffer
        # helper), so pre-allocating them here was dead work: for an output
        # whose shape differs from the primary input, the allocation was
        # discarded on the first step and its shape lied until then.
        for name, spec in self._pk_spec_entries:
            if not isinstance(spec, StateSpec):
                continue

            self._buffers[name] = torch.full(
                self._pk_spec_shape(spec, inputs),
                self._pk_resolve_default(spec.default),
                device=primary.device,
                dtype=dtype,
            )
            self._non_persistent_buffers_set.add(name)

        self._pk_allocated = True

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
        if not inputs:
            raise TypeError(
                f"{type(self).__name__}.allocate_like requires example "
                f"inputs, e.g. allocate_like(x) or allocate_like((x, inh))"
            )
        if self.init_hidden and not self._pk_allocated:
            if len(inputs) == 1:
                canonical = self._pk_canonicalize_inputs(inputs[0])
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

            self._pk_alloc_hidden(canonical)

        return self

    def _pk_store_hidden_outputs(self, out: StepOutput) -> None:
        for (name, spec), t in zip(self._pk_spec_entries, out, strict=True):
            if not spec.differentiable:
                t = t.detach()

            # Buffers already exist after _pk_alloc_hidden (states) or are
            # created here on first store (outputs). Either way they must be
            # non-persistent: hidden-mode contents travel via get_extra_state,
            # not as plain state_dict keys.
            self._non_persistent_buffers_set.add(name)
            self._buffers[name] = t

    def _pk_shape_for_batch(self, spec: StateSpec, batch_shape: tuple[int, ...]) -> tuple[int, ...]:
        if isinstance(spec.shape, str) and spec.shape != "input":
            raise ValueError(
                f"{type(self).__name__}.initial_state cannot resolve "
                f"StateSpec(shape={spec.shape!r}) without example inputs; "
                f"use initial_state_like(inputs)"
            )
        return spec.shape if isinstance(spec.shape, tuple) else batch_shape

    def _pk_build_state(
        self,
        shapes: list[tuple[int, ...]],
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        fill: str,
    ) -> tuple[Tensor, ...]:
        out: list[Tensor] = []
        for shape, spec in zip(shapes, self._pk_state_specs, strict=True):
            if fill == "full":
                out.append(torch.full(shape, self._pk_resolve_default(spec.default), device=device, dtype=dtype))
            else:
                out.append(torch.zeros(shape, device=device, dtype=dtype))
        return tuple(out)

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
        dtype = self._pk_explicit_state_dtype(dtype)
        shapes = [self._pk_shape_for_batch(s, batch_shape) for s in self._pk_state_specs]
        return self._pk_build_state(shapes, device, dtype, "full")

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
        dtype = self._pk_explicit_state_dtype(dtype)
        shapes = [self._pk_shape_for_batch(s, batch_shape) for s in self._pk_state_specs]
        return self._pk_build_state(shapes, device, dtype, "zero")

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
        inputs = self._pk_canonicalize_inputs(inputs)
        primary = inputs[self._pk_primary_input_index]
        dtype = self._pk_explicit_state_dtype(primary.dtype)
        shapes = [
            tuple(batch_shape) if batch_shape is not None else self._pk_spec_shape(s, inputs)
            for s in self._pk_state_specs
        ]
        return self._pk_build_state(shapes, primary.device, dtype, "full")

    def zero_state_like(
        self,
        inputs: InputTensor,
        batch_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[Tensor, ...]:
        """
        Create zeroed state from example inputs (see ``initial_state_like``).
        """
        inputs = self._pk_canonicalize_inputs(inputs)
        primary = inputs[self._pk_primary_input_index]
        dtype = self._pk_explicit_state_dtype(primary.dtype)
        shapes = [
            tuple(batch_shape) if batch_shape is not None else self._pk_spec_shape(s, inputs)
            for s in self._pk_state_specs
        ]
        return self._pk_build_state(shapes, primary.device, dtype, "zero")

    def reset(self) -> None:
        """
        Reset hidden buffers to their Spec defaults.
        """
        if not self.init_hidden:
            return

        for name, spec in self._pk_spec_entries:
            t = self._buffers.get(name, None)

            if isinstance(t, Tensor):
                self._buffers[name] = torch.full_like(
                    t,
                    self._pk_resolve_default(spec.default),
                )

    def detach(self) -> None:
        """
        Detach hidden buffers from autograd.
        """
        if not self.init_hidden:
            return

        for name, _ in self._pk_spec_entries:
            t = self._buffers.get(name, None)

            if isinstance(t, Tensor):
                self._buffers[name] = t.detach()