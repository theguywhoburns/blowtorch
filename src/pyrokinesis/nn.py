from __future__ import annotations

import inspect
import threading
from typing import Any, Optional

import torch
import torch.nn as nn

from pyrokinesis.module import (
    PyroModule,
    InputTensor,
    OutputSpec,
    StateSpec,
    StepOutput,
    Tensor,
    sequence_scan,
)
from pyrokinesis.module.scan import _store_hidden_seq_buffers

__all__ = ["Sequential"]

_PK_PROBE_LOCK = threading.Lock()


class Sequential(PyroModule):
    """
    Stack layers into a network.

    A ``PyroModule`` topology manager: it moves data between layers and
    threads a flat state tuple through the time-major scan. Stateful
    ``PyroModule`` layers stay pure functions (``init_hidden=False``);
    stateless ``nn.Module`` layers are applied once per timestep. The whole
    stack runs as one step function, so ``fast_sequence_()`` compiles a single
    fused scan over every layer.

    ``init_hidden=True`` stateful layers are rejected: the container owns the
    state bundle, so children must run in explicit mode.

    The container reuses the ``PyroModule`` machinery. Hidden state lives
    in non-persistent buffers named ``l{layer_index}_{state_name}`` (e.g.
    ``l0_mem``), and scans, compiled scans, validation, and serialization all
    come from the base mixins. State shapes are not declared: they are
    resolved from the layer stack by a meta shape walk (``_pk_spec_shape``).
    """

    # Container metadata is built per instance in __init__; the class-level
    # ClassVar copies from the base mixins stay empty.
    _pk_output_names: tuple[str, ...]  # pyright: ignore[reportIncompatibleVariableOverride]
    _pk_output_specs: tuple[OutputSpec, ...]  # pyright: ignore[reportIncompatibleVariableOverride]
    _pk_state_names: tuple[str, ...]  # pyright: ignore[reportIncompatibleVariableOverride]
    _pk_state_specs: tuple[StateSpec, ...]  # pyright: ignore[reportIncompatibleVariableOverride]
    _pk_spec_entries: tuple[tuple[str, Any], ...]  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *layers: nn.Module,
        init_hidden: bool = False,
        validate: Optional[bool] = None,
    ) -> None:
        super().__init__(init_hidden=init_hidden, validate=validate)

        if not layers:
            raise ValueError("Sequential requires at least one layer")

        for layer in layers:
            if not isinstance(layer, nn.Module):
                raise TypeError(
                    f"Sequential layers must be nn.Module instances, "
                    f"got {type(layer).__name__}"
                )

            if isinstance(layer, PyroModule) and layer.init_hidden:
                raise ValueError(
                    f"stateful layer {type(layer).__name__} is in "
                    f"init_hidden=True mode; Sequential owns the state, pass "
                    f"init_hidden=False"
                )

        self._layers: list[nn.Module] = list(layers)

        for i, layer in enumerate(self._layers):
            setattr(self, f"layer{i}", layer)

        # Flat state registry: one StateSpec per stateful-layer state, in
        # layer order. Defaults are resolved against the owning layer, so
        # hidden-mode allocation and the state factories fill the same values
        # the layer itself would use. Shapes stay unset ("input"): they are
        # resolved from the layer stack via _pk_spec_shape.
        state_entries: list[tuple[str, StateSpec]] = []

        for i, layer in enumerate(self._layers):
            if not isinstance(layer, PyroModule):
                continue

            if len(layer._pk_input_names) != 1:
                raise ValueError(
                    f"Sequential requires single-input layers; "
                    f"{type(layer).__name__} declares "
                    f"{len(layer._pk_input_names)} inputs"
                )
            if len(layer._pk_output_specs) != 1:
                raise ValueError(
                    f"Sequential requires single-output layers; "
                    f"{type(layer).__name__} declares "
                    f"{len(layer._pk_output_specs)} outputs"
                )

            for name, spec in layer._pk_spec_entries:
                if isinstance(spec, StateSpec):
                    state_entries.append(
                        (
                            f"l{i}_{name}",
                            StateSpec(default=layer._pk_resolve_default(spec.default)),
                        )
                    )

        output_spec = OutputSpec()
        # Base mixins keep these as ClassVar per class; Sequential needs per-instance
        # values, so they are redeclared above with pyright: ignore and set here
        # via object.__setattr__ to avoid a second ignore on the assignment.
        object.__setattr__(self, "_pk_output_names", ("y",))
        object.__setattr__(self, "_pk_output_specs", (output_spec,))
        object.__setattr__(self, "_pk_state_names", tuple(name for name, _ in state_entries))
        object.__setattr__(self, "_pk_state_specs", tuple(spec for _, spec in state_entries))
        object.__setattr__(self, "_pk_spec_entries", (("y", output_spec), *state_entries))

    def _pk_rebuild_state_registry(self) -> None:
        state_entries: list[tuple[str, StateSpec]] = []
        for i, layer in enumerate(self._layers):
            if isinstance(layer, PyroModule):
                for name, spec in layer._pk_spec_entries:
                    if isinstance(spec, StateSpec):
                        state_entries.append(
                            (f"l{i}_{name}", StateSpec(default=layer._pk_resolve_default(spec.default)))
                        )
        out_spec = self._pk_output_specs[0] if self._pk_output_specs else OutputSpec()
        object.__setattr__(self, "_pk_output_names", ("y",))
        object.__setattr__(self, "_pk_output_specs", (out_spec,))
        object.__setattr__(self, "_pk_state_names", tuple(n for n, _ in state_entries))
        object.__setattr__(self, "_pk_state_specs", tuple(s for _, s in state_entries))
        object.__setattr__(self, "_pk_spec_entries", (("y", out_spec), *state_entries))
        self.__dict__.pop("_pk_probe_cache", None)
        self._pk_compiled_sequence = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("layer") and name[5:].isdigit() and "_layers" in self.__dict__:
            idx = int(name[5:])
            layers = self.__dict__["_layers"]
            if 0 <= idx < len(layers) and layers[idx] is not value:
                if not isinstance(value, nn.Module):
                    raise TypeError(f"Sequential layer {idx} must be nn.Module, got {type(value).__name__}")
                if isinstance(value, PyroModule) and value.init_hidden:
                    raise ValueError(f"stateful layer {type(value).__name__} is in init_hidden=True mode; Sequential owns state")
                layers[idx] = value
                self._pk_rebuild_state_registry()
            super().__setattr__(name, value)
            return
        if name == "_layers" and "_layers" in self.__dict__:
            super().__setattr__(name, value)
            # sync layer{i} attributes without re-entering the layer branch
            for key in list(self.__dict__.keys()):
                if key.startswith("layer") and key[5:].isdigit() and int(key[5:]) >= len(value):
                    try:
                        super().__delattr__(key)
                    except AttributeError:
                        pass
            for i, layer in enumerate(value):
                nn.Module.__setattr__(self, f"layer{i}", layer)  # type: ignore[attr-defined]
            self._pk_rebuild_state_registry()
            return
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name.startswith("layer") and name[5:].isdigit() and "_layers" in self.__dict__:
            idx = int(name[5:])
            layers = self.__dict__["_layers"]
            if 0 <= idx < len(layers):
                raise AttributeError(f"Sequential layer {idx} cannot be deleted via delattr; use del net[{idx}]")
        super().__delattr__(name)

    def __setitem__(self, idx: int, value: nn.Module) -> None:
        if not isinstance(value, nn.Module):
            raise TypeError(f"Sequential layer {idx} must be nn.Module")
        if isinstance(value, PyroModule) and value.init_hidden:
            raise ValueError(f"stateful layer {type(value).__name__} is init_hidden=True")
        self._layers[idx] = value
        nn.Module.__setattr__(self, f"layer{idx}", value)  # type: ignore[attr-defined]
        self._pk_rebuild_state_registry()

    def __delitem__(self, idx: int) -> None:
        n = len(self._layers)
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(idx)
        old_len = n
        del self._layers[idx]
        # re-register remaining layers under correct indices
        for key in list(self.__dict__.keys()):
            if key.startswith("layer") and key[5:].isdigit():
                try:
                    super().__delattr__(key)
                except AttributeError:
                    pass
        for i, layer in enumerate(self._layers):
            nn.Module.__setattr__(self, f"layer{i}", layer)  # type: ignore[attr-defined]
        # _modules already updated via __delattr__/__setattr__; rebuild registry
        if len(self._layers) == 0:
            raise ValueError("Sequential requires at least one layer")
        self._pk_rebuild_state_registry()
        # remove stale layer{old_len-1} if any remains in _modules
        try:
            super().__delattr__(f"layer{old_len-1}")
        except AttributeError:
            pass

    def __getitem__(self, idx: int) -> nn.Module:
        return self._layers[idx]

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):  # type: ignore[override]
        return iter(self._layers)

    # The per-step dispatcher: the compiled unit for the whole network.

    def _step(self, x: Tensor, *state: Tensor) -> StepOutput:
        offset = 0
        next_states: list[Tensor] = []

        for layer in self._layers:
            if isinstance(layer, PyroModule):
                n = len(layer._pk_state_names)
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

    # Shape pass: meta-device walk.

    def _pk_probe(
        self,
        batch_shape: tuple[int, ...],
        dtype: Optional[torch.dtype],
    ) -> tuple[tuple[int, ...], list[tuple[int, ...]]]:
        # Stateless layers like nn.Linear have matrix parameters on the real
        # device; a meta input rejects them. Temporarily move every param and
        # buffer to meta for the shape walk, then restore them untouched.
        # RNG is forked so stochastic _step doesn't bleed into the global stream.
        saved: list[tuple[nn.Module, str, str, Any]] = []

        with _PK_PROBE_LOCK:
            with torch.random.fork_rng():
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
                        try:
                            if isinstance(layer, PyroModule):
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
                                    raise TypeError(  # pyright: ignore[reportUnreachable]
                                        f"stateless layer {type(layer).__name__} in "
                                        f"Sequential must return a single tensor"
                                    )
                        except RuntimeError as exc:
                            msg = str(exc)
                            if "meta" in msg or "Tensor.item" in msg:
                                raise RuntimeError(
                                    f"{type(self).__name__} shape probe failed on layer {type(layer).__name__}: "
                                    f"_step branched on tensor data or used Tensor.item() with meta tensors; "
                                    f"keep _step shape-safe (no data-dependent Python control flow)"
                                ) from exc
                            raise
                finally:
                    for module, kind, name, tensor in saved:
                        getattr(module, kind)[name] = tensor

        return tuple(x.shape), shapes

    def _pk_probe_shapes(
        self,
        x: Tensor,
    ) -> tuple[tuple[int, ...], list[tuple[int, ...]]]:
        """
        Shape walk with caching, so the per-spec ``_pk_spec_shape`` lookups in
        hidden allocation and validation do not re-walk the stack each time.
        """
        # Fingerprint topology + param shapes so stale cache can't survive
        # a layer replacement or weight resize without an explicit setattr.
        topo = tuple(id(layer) for layer in self._layers)
        param_shapes = tuple(
            tuple(tuple(p.shape) for p in layer.parameters())
            for layer in self._layers
        )
        key = (tuple(x.shape), x.dtype, topo, param_shapes)
        cache = self.__dict__.get("_pk_probe_cache")

        if cache is None:
            cache = {}
            self.__dict__["_pk_probe_cache"] = cache

        hit = cache.get(key)

        if hit is not None:
            return hit

        result = self._pk_probe(tuple(x.shape), x.dtype)
        cache[key] = result
        return result

    # Container state shapes come from the layer stack, not from a declarative
    # spec, so resolve them with the meta walk and map back by spec identity.

    def _pk_spec_shape(self, spec: Any, inputs: tuple[Tensor, ...]) -> tuple[int, ...]:
        out_shape, state_shapes = self._pk_probe_shapes(
            inputs[self._pk_primary_input_index]
        )

        if spec is self._pk_output_specs[0]:
            return out_shape

        for i, s in enumerate(self._pk_state_specs):
            if s is spec:
                return state_shapes[i]

        raise AssertionError(
            f"{type(self).__name__}._pk_spec_shape called with a spec it does not own"
        )

    # State factories. These override StateMixin's: container state shapes are
    # resolved by the shape walk, not from ``StateSpec.shape``/``batch_shape``.

    def _pk_seq_shapes_for_batch(self, batch_shape: tuple[int, ...], dtype: Optional[torch.dtype]) -> list[tuple[int, ...]]:
        _, shapes = self._pk_probe(tuple(batch_shape), dtype)
        return shapes

    def _pk_seq_state(
        self,
        shapes: list[tuple[int, ...]],
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        fill: str,
    ) -> tuple[Tensor, ...]:
        dtype = dtype if dtype is None or dtype.is_floating_point else torch.get_default_dtype()
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
        return self._pk_seq_state(self._pk_seq_shapes_for_batch(batch_shape, dtype), device, dtype, "full")

    def zero_state(
        self,
        batch_shape: tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, ...]:
        return self._pk_seq_state(self._pk_seq_shapes_for_batch(batch_shape, dtype), device, dtype, "zero")

    def initial_state_like(
        self,
        inputs: InputTensor,
        batch_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[Tensor, ...]:
        inputs = self._pk_canonicalize_inputs(inputs)
        primary = inputs[self._pk_primary_input_index]
        dtype = primary.dtype if primary.is_floating_point() else torch.get_default_dtype()
        _, shapes = self._pk_probe_shapes(primary)
        shapes = [tuple(batch_shape) if batch_shape is not None else s for s in shapes]
        return self._pk_seq_state(shapes, primary.device, dtype, "full")

    def zero_state_like(
        self,
        inputs: InputTensor,
        batch_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[Tensor, ...]:
        inputs = self._pk_canonicalize_inputs(inputs)
        primary = inputs[self._pk_primary_input_index]
        dtype = primary.dtype if primary.is_floating_point() else torch.get_default_dtype()
        _, shapes = self._pk_probe_shapes(primary)
        shapes = [tuple(batch_shape) if batch_shape is not None else s for s in shapes]
        return self._pk_seq_state(shapes, primary.device, dtype, "zero")

    # Compile path. State resolution stays outside the compiled region: the
    # shape walk temporarily swaps submodule params to meta, which would break
    # inside a ``torch.compile`` trace.

    def compile_sequence_scan(self, **kwargs: Any) -> "Sequential":
        needs_clone = kwargs.get("mode") in ("reduce-overhead", "max-autotune")
        n_outputs = len(self._pk_output_names)
        compiled = torch.compile(
            lambda inputs_seq, state: sequence_scan(
                lambda inputs, s: self._pk_forward_explicit(inputs, s),
                inputs_seq,
                state,
                n_outputs,
            ),
            **kwargs,
        )

        def _pk_compiled(
            x_seq: Tensor,
            state: Optional[tuple[Tensor, ...]] = None,
        ) -> Tensor | StepOutput:
            inputs_seq = self._pk_canonicalize_input_sequence(x_seq)

            if self.init_hidden:
                if inputs_seq[0].shape[0] > 0:
                    self.allocate_like(self._pk_first_inputs(inputs_seq))
                state = tuple(getattr(self, name) for name in self._pk_state_names)
            elif state is None:
                state = self.initial_state_for_sequence(inputs_seq)

            out = compiled(inputs_seq, state)

            if self.init_hidden:
                y = out[0]
                _store_hidden_seq_buffers(self, (y,), out[1:])  # type: ignore[arg-type]
                return y.clone() if needs_clone else y

            if isinstance(out, Tensor):
                return out.clone() if needs_clone else out

            return tuple(t.clone() if needs_clone else t for t in out)

        self._pk_compiled_sequence = _pk_compiled
        return self

    def fast_sequence_(
        self,
        compile_scan: bool = True,
        **compile_kwargs: Any,
    ) -> "Sequential":
        for layer in self._layers:
            if isinstance(layer, PyroModule):
                layer.validate = False

        super().fast_sequence_(compile_scan, **compile_kwargs)
        return self

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(layer) for layer in self._layers)
        return f"{type(self).__name__}(\n  {inner}\n)"


# PyroModule's __init_subclass__ generates a parameter-only signature that
# hides the positional *layers. Restore the real constructor signature for
# help()/inspect.
setattr(  # noqa: B010
    Sequential,
    "__signature__",
    inspect.Signature(
        [
            inspect.Parameter("layers", inspect.Parameter.VAR_POSITIONAL),
            inspect.Parameter(
                "init_hidden", inspect.Parameter.KEYWORD_ONLY, default=False
            ),
            inspect.Parameter(
                "validate", inspect.Parameter.KEYWORD_ONLY, default=None
            ),
        ]
    ),
)