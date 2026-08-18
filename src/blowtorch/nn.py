from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from blowtorch.base import (
    BlowtorchModule,
    StateSpec,
    StepOutput,
    Tensor,
    get_validation,
    sequence_scan,
)

__all__ = ["Sequential"]


class Sequential(nn.Module):
    """
    Stack layers into a network.

    A plain ``nn.Module`` topology manager: it moves data between layers and
    threads a flat state tuple through the time-major scan. Stateful
    ``BlowtorchModule`` layers stay pure functions (``init_hidden=False``);
    stateless ``nn.Module`` layers are applied once per timestep. The whole
    stack runs as one step function, so ``fast_sequence_()`` compiles a single
    fused scan over every layer.

    ``init_hidden=True`` stateful layers are rejected: the container owns the
    state bundle, so children must run in explicit mode.
    """

    def __init__(
        self,
        *layers: nn.Module,
        init_hidden: bool = False,
        validate: Optional[bool] = None,
    ) -> None:
        super().__init__()

        if not layers:
            raise ValueError("Sequential requires at least one layer")

        for layer in layers:
            if not isinstance(layer, nn.Module):
                raise TypeError(
                    f"Sequential layers must be nn.Module instances, "
                    f"got {type(layer).__name__}"
                )

            if isinstance(layer, BlowtorchModule) and layer.init_hidden:
                raise ValueError(
                    f"stateful layer {type(layer).__name__} is in "
                    f"init_hidden=True mode; Sequential owns the state, pass "
                    f"init_hidden=False"
                )

        self._layers: list[nn.Module] = list(layers)

        for i, layer in enumerate(self._layers):
            setattr(self, f"layer{i}", layer)

        self.init_hidden = init_hidden
        self._validate_override = validate

        self._bt_allocated = False
        self._bt_compiled_sequence: Optional[Callable[..., Any]] = None
        self._bt_states: Optional[list[Tensor]] = None
        self._bt_alloc_input_shape: Optional[tuple[int, ...]] = None

        # Flat state registry: one entry per stateful-layer state, in layer
        # order. Defaults are resolved against the owning layer, so hidden-mode
        # allocation and the state factories fill the same values the layer
        # itself would use.
        state_entries: list[tuple[str, float]] = []

        for i, layer in enumerate(self._layers):
            if not isinstance(layer, BlowtorchModule):
                continue

            if len(layer._bt_output_specs) != 1:
                raise ValueError(
                    f"Sequential requires single-output layers; "
                    f"{type(layer).__name__} declares "
                    f"{len(layer._bt_output_specs)} outputs"
                )

            for name, spec in layer._bt_spec_entries:
                if isinstance(spec, StateSpec):
                    state_entries.append(
                        (f"l{i}_{name}", layer._resolve_default(spec.default))
                    )

        self._bt_state_names = tuple(name for name, _ in state_entries)
        self._state_defaults = tuple(default for _, default in state_entries)
        self._n_outputs = 1

    # Validation flag (mirrors BlowtorchModule semantics).

    @property
    def validate(self) -> bool:
        return (
            get_validation()
            if self._validate_override is None
            else self._validate_override
        )

    @validate.setter
    def validate(self, value: bool) -> None:
        self._validate_override = bool(value)

    # The per-step dispatcher: the compiled unit for the whole network.

    def _step(self, x: Tensor, state: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        offset = 0
        next_states: list[Tensor] = []

        for layer in self._layers:
            if isinstance(layer, BlowtorchModule):
                n = len(layer._bt_state_names)
                out = layer.forward(x, *state[offset : offset + n])
                assert isinstance(out, tuple)
                x = out[0]
                next_states.extend(out[1:])
                offset += n
            else:
                x = layer(x)
                # Runtime contract check: a user stateless layer may return a
                # non-tensor despite nn.Module.__call__'s annotation.
                if not isinstance(x, Tensor):
                    raise TypeError(  # pyright: ignore[reportUnreachable]
                        f"stateless layer {type(layer).__name__} in "
                        f"Sequential must return a single tensor"
                    )

        return (x, *next_states)

    # Shape pass: meta-device walk, no math, no side effects.

    def _probe(
        self,
        batch_shape: tuple[int, ...],
        dtype: Optional[torch.dtype],
    ) -> tuple[tuple[int, ...], list[tuple[int, ...]]]:
        # Stateless layers like nn.Linear have matrix parameters on the real
        # device; a meta input rejects them. Temporarily move every param and
        # buffer to meta for the shape walk, then restore them untouched.
        saved: list[tuple[nn.Module, str, str, Any]] = []

        try:
            for module in self.modules():
                for name, param in list(module._parameters.items()):
                    if param is not None:
                        saved.append((module, "_parameters", name, param))
                        module._parameters[name] = nn.Parameter(
                            param.to(device="meta"),
                            requires_grad=param.requires_grad,
                        )
                for name, buf in list(module._buffers.items()):
                    if buf is not None:
                        saved.append((module, "_buffers", name, buf))
                        module._buffers[name] = buf.to(device="meta")

            x = torch.empty(batch_shape, device="meta", dtype=dtype)
            shapes: list[tuple[int, ...]] = []

            for layer in self._layers:
                if isinstance(layer, BlowtorchModule):
                    state = layer.initial_state(
                        tuple(x.shape),
                        device=torch.device("meta"),
                        dtype=dtype,
                    )
                    out = layer.forward(x, *state)
                    assert isinstance(out, tuple)
                    x = out[0]
                    shapes.extend(tuple(t.shape) for t in out[1:])
                else:
                    x = layer(x)
                    if not isinstance(x, Tensor):
                        raise TypeError(
                            f"stateless layer {type(layer).__name__} in "
                            f"Sequential must return a single tensor"
                        )
        finally:
            for module, kind, name, tensor in saved:
                getattr(module, kind)[name] = tensor

        return tuple(x.shape), shapes

    # State factories.

    def initial_state(
        self,
        batch_shape: tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, ...]:
        state_dtype = dtype
        if state_dtype is not None and not state_dtype.is_floating_point:
            state_dtype = torch.get_default_dtype()

        _, shapes = self._probe(tuple(batch_shape), dtype)

        return tuple(
            torch.full(shape, default, device=device, dtype=state_dtype)
            for shape, default in zip(shapes, self._state_defaults)
        )

    def zero_state(
        self,
        batch_shape: tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, ...]:
        state_dtype = dtype
        if state_dtype is not None and not state_dtype.is_floating_point:
            state_dtype = torch.get_default_dtype()

        _, shapes = self._probe(tuple(batch_shape), dtype)

        return tuple(
            torch.zeros(shape, device=device, dtype=state_dtype)
            for shape in shapes
        )

    def initial_state_like(
        self,
        x: Tensor,
        batch_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[Tensor, ...]:
        shape = tuple(x.shape) if batch_shape is None else tuple(batch_shape)
        return self.initial_state(shape, device=x.device, dtype=x.dtype)

    def initial_state_for_sequence(
        self,
        x_seq: Tensor,
    ) -> tuple[Tensor, ...]:
        if x_seq.dim() < 3:
            raise ValueError(
                f"{type(self).__name__}.initial_state_for_sequence expects "
                f"(time, batch, features), got {x_seq.dim()} dims"
            )

        return self.initial_state(
            tuple(x_seq.shape[1:]),
            device=x_seq.device,
            dtype=x_seq.dtype,
        )

    # Hidden-mode state ownership.

    def _alloc_hidden(self, x: Tensor) -> None:
        dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
        out_shape, shapes = self._probe(tuple(x.shape), dtype)

        self._bt_states = [
            torch.full(shape, default, device=x.device, dtype=dtype)
            for shape, default in zip(shapes, self._state_defaults)
        ]
        self._bt_out_shape = out_shape
        self._bt_alloc_input_shape = tuple(x.shape)
        self._bt_allocated = True

    def allocate_like(self, x: Tensor) -> "Sequential":
        if self.init_hidden and not self._bt_allocated:
            self._alloc_hidden(x)
        return self

    def _states(self) -> tuple[Tensor, ...]:
        assert self._bt_states is not None
        return tuple(self._bt_states)

    def _check_hidden_input_shape(self, x: Tensor) -> None:
        if self._bt_alloc_input_shape is None:
            return
        if tuple(x.shape) != self._bt_alloc_input_shape:
            raise ValueError(
                f"{type(self).__name__} hidden buffers were allocated for "
                f"input shape {tuple(self._bt_alloc_input_shape)}, got "
                f"{tuple(x.shape)}; the input shape must stay fixed in hidden "
                f"mode"
            )

    def _forward_hidden(self, x: Tensor) -> Tensor:
        if not self._bt_allocated:
            self._alloc_hidden(x)
        elif self.validate:
            self._check_hidden_input_shape(x)

        out = self._step(x, self._states())
        assert self._bt_states is not None
        self._bt_states = list(out[1:])
        return out[0]

    # Public forward.

    def forward(self, x: Tensor, *state: Tensor) -> Tensor | StepOutput:
        if self.init_hidden:
            if state:
                raise ValueError(
                    f"{type(self).__name__} is in init_hidden=True mode; "
                    f"do not pass state explicitly"
                )
            return self._forward_hidden(x)

        if self.validate:
            if len(state) != len(self._bt_state_names):
                raise ValueError(
                    f"{type(self).__name__} expects {len(self._bt_state_names)} "
                    f"state tensors, got {len(state)}"
                )

        return self._step(x, state)

    def step_state(
        self,
        x: Tensor,
        state: tuple[Tensor, ...],
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        if self.init_hidden:
            raise ValueError(
                f"{type(self).__name__}.step_state requires init_hidden=False"
            )

        out = self._step(x, tuple(state))
        return out[0], out[1:]

    def step(
        self,
        x: Tensor,
        state: tuple[Tensor, ...],
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        return self.step_state(x, state)

    # Sequence scan.

    def forward_sequence(
        self,
        x_seq: Tensor,
        state: Optional[tuple[Tensor, ...]] = None,
    ) -> Tensor | StepOutput:
        if x_seq.dim() < 3:
            raise ValueError(
                f"{type(self).__name__}.forward_sequence expects "
                f"(time, batch, features), got {x_seq.dim()} dims"
            )
        if x_seq.shape[0] == 0:
            raise ValueError(
                f"{type(self).__name__}.forward_sequence expects at least one "
                f"timestep"
            )

        compiled = self._bt_compiled_sequence
        if compiled is not None:
            return compiled(x_seq, state)

        if self.init_hidden:
            if state is not None:
                raise ValueError(
                    f"{type(self).__name__} is in init_hidden=True mode; "
                    f"forward_sequence does not accept explicit state"
                )
            return self._hidden_sequence_scan(x_seq)

        if state is None:
            state = self.initial_state_for_sequence(x_seq)

        result = sequence_scan(
            lambda inputs, s: self._step(inputs[0], s),
            (x_seq,),
            state,
            self._n_outputs,
        )
        return (result[0], *result[1:])

    def _hidden_sequence_scan(self, x_seq: Tensor) -> Tensor:
        if not self._bt_allocated:
            self._alloc_hidden(x_seq[0])
        elif self.validate:
            self._check_hidden_input_shape(x_seq[0])

        result = sequence_scan(
            lambda inputs, s: self._step(inputs[0], s),
            (x_seq,),
            self._states(),
            self._n_outputs,
        )
        self._bt_states = list(result[self._n_outputs:])
        return result[0]

    # Compile path.

    def compile_sequence_scan(self, **kwargs: Any) -> "Sequential":
        needs_clone = kwargs.get("mode") in ("reduce-overhead", "max-autotune")
        compiled = torch.compile(
            lambda inputs_seq, state: sequence_scan(
                lambda inputs, s: self._step(inputs[0], s),
                inputs_seq,
                state,
                self._n_outputs,
            ),
            **kwargs,
        )

        def _compiled(
            x_seq: Tensor,
            state: Optional[tuple[Tensor, ...]] = None,
        ) -> Tensor | StepOutput:
            if self.init_hidden:
                if x_seq.shape[0] > 0:
                    self.allocate_like(x_seq[0])
                state = self._states()
            elif state is None:
                state = self.initial_state_for_sequence(x_seq)

            out = compiled((x_seq,), state)

            if self.init_hidden:
                self._bt_states = list(out[self._n_outputs :])

            if isinstance(out, Tensor):
                return out.clone() if needs_clone else out

            return tuple(t.clone() if needs_clone else t for t in out)

        self._bt_compiled_sequence = _compiled
        return self

    def fast_sequence_(self, compile_scan: bool = True, **compile_kwargs: Any) -> "Sequential":
        self.validate = False

        for layer in self._layers:
            if isinstance(layer, BlowtorchModule):
                layer.validate = False

        if compile_scan:
            compile_kwargs.setdefault("mode", "default")
            self.compile_sequence_scan(**compile_kwargs)

        return self

    # Hidden-state bookkeeping.

    def reset(self) -> None:
        if not self.init_hidden or not self._bt_allocated:
            return

        self._bt_states = [
            torch.full_like(t, default)
            for t, default in zip(self._states(), self._state_defaults)
        ]

    def detach(self) -> None:
        if not self.init_hidden or not self._bt_allocated:
            return

        self._bt_states = [t.detach() for t in self._states()]

    def get_extra_state(self) -> Optional[dict[str, Tensor]]:
        if not self.init_hidden or not self._bt_allocated:
            return None

        return {
            name: t.detach()
            for name, t in zip(self._bt_state_names, self._states())
        }

    def set_extra_state(self, state: Any) -> None:
        if not self.init_hidden or state is None:
            return

        self._bt_states = [state[name] for name in self._bt_state_names]
        self._bt_allocated = True

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(layer) for layer in self._layers)
        return f"{type(self).__name__}(\n  {inner}\n)"
