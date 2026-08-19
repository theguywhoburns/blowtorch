from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional, Self

import torch

from .specs import (
    InputSpec,
    InputTensor,
    OutputSpec,
    Spec,
    StateSpec,
    StepOutput,
    Tensor,
)
from .validation import (
    check_hidden_input_shape,
    is_validating,
    set_validating,
)

if TYPE_CHECKING:
    from . import BlowtorchModule

# In eager scans, batch this many steps into a single index_copy_ scatter so
# peak memory stays at input + output (no per-step (B, F) list held for stack).
_SEQUENCE_SCAN_CHUNK = 8


def set_sequence_scan_chunk(chunk: int) -> None:
    """
    Set the eager-scan chunk size used by ``forward_sequence``.

    Chunking batches several steps into one ``index_copy_`` scatter. Larger
    values reduce dispatch overhead but increase peak memory; the optimal size
    depends on hardware, dtype, and batch/feature dims. Pass ``1`` to disable
    chunking.
    """
    global _SEQUENCE_SCAN_CHUNK

    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
        raise ValueError(f"scan chunk must be a positive int, got {chunk!r}")

    _SEQUENCE_SCAN_CHUNK = chunk


# Time-major scan over a pure step function.


def sequence_scan(
    step: Callable[[tuple[Tensor, ...], tuple[Tensor, ...]], tuple[Tensor, ...]],
    inputs_seq: tuple[Tensor, ...],
    state0: tuple[Tensor, ...],
    n_outputs: int,
) -> tuple[Tensor, ...]:
    """
    Scan a pure step function over a time-major input.

    ``step`` is ``(inputs, state) -> (*outputs, *next_state)`` where ``inputs``
    is the canonical per-timestep input tuple (one tensor per declared input):
    the first ``n_outputs`` tensors are outputs, the rest become the state for
    the next step. Returns ``(*ys, *final_state)`` where each ``ys[k]`` is a
    preallocated ``(T, *output_shape)`` buffer.

    ``inputs_seq`` is the canonical tuple of time-major sequences (one per
    declared input, all sharing time length ``T``).

    In eager this batches ``_SEQUENCE_SCAN_CHUNK`` steps into one
    ``index_copy_`` scatter so peak memory stays at input + output. Under
    ``torch.compile`` it lowers to a flat fused loop; the whole scan becomes a
    single graph.
    """
    T = inputs_seq[0].shape[0]

    inputs0 = tuple(seq[0] for seq in inputs_seq)
    out0 = step(inputs0, state0)
    assert isinstance(out0, tuple)

    ys = tuple(
        torch.empty((T, *o.shape), dtype=o.dtype, device=o.device)
        for o in out0[:n_outputs]
    )

    if torch.compiler.is_compiling():
        for k, y in enumerate(ys):
            y.index_copy_(0, torch.tensor([0], device=y.device), out0[k].unsqueeze(0))

        cur = out0[n_outputs:]

        for t in range(1, T):
            inputs_t = tuple(seq[t] for seq in inputs_seq)
            out = step(inputs_t, cur)

            for k, y in enumerate(ys):
                y.index_copy_(
                    0,
                    torch.tensor([t], device=y.device),
                    out[k].unsqueeze(0),
                )

            cur = out[n_outputs:]
    else:
        for k, y in enumerate(ys):
            y[0] = out0[k]

        cur = out0[n_outputs:]
        idx = torch.arange(T, device=inputs_seq[0].device)

        for lo in range(1, T, _SEQUENCE_SCAN_CHUNK):
            hi = min(lo + _SEQUENCE_SCAN_CHUNK, T)

            chunks: list[list[Tensor]] = [[] for _ in range(n_outputs)]

            for t in range(lo, hi):
                inputs_t = tuple(seq[t] for seq in inputs_seq)
                out = step(inputs_t, cur)

                for k in range(n_outputs):
                    chunks[k].append(out[k])

                cur = out[n_outputs:]

            for k, y in enumerate(ys):
                y.index_copy_(0, idx[lo:hi], torch.stack(chunks[k]))

    return (*ys, *cur)


class SequenceScanMixin:
    """Time-major sequence scans over a BlowtorchModule.

    Members the mixin needs from the host are declared as type-only stubs
    below (annotations set no runtime attribute, so BlowtorchModule's real
    implementations always win the MRO). The stub surface also structurally
    satisfies ``_ValidationHost`` so the free validation helpers accept
    ``self``.
    """

    # Host attributes and methods referenced by the scan methods.

    _validate_override: Optional[bool]
    init_hidden: bool
    _bt_allocated: bool
    _bt_compiled_sequence: Optional[Callable[..., Any]]
    _bt_output_names: ClassVar[tuple[str, ...]]
    _bt_state_names: ClassVar[tuple[str, ...]]
    _bt_state_specs: ClassVar[tuple[StateSpec, ...]]
    _bt_input_names: ClassVar[tuple[str, ...]]
    _bt_input_specs: ClassVar[tuple[InputSpec, ...]]
    _bt_spec_entries: ClassVar[tuple[tuple[str, Spec], ...]]
    _bt_output_specs: ClassVar[tuple[OutputSpec, ...]]
    _buffers: dict[str, Optional[Tensor]]

    def _spec_shape(
        self,
        spec: Spec,
        inputs: tuple[Tensor, ...],
    ) -> tuple[int, ...]: ...

    def _canonicalize_inputs(
        self,
        inputs: InputTensor,
    ) -> tuple[Tensor, ...]: ...

    def _alloc_hidden(self, inputs: tuple[Tensor, ...]) -> None: ...

    def _forward_explicit(
        self,
        inputs: tuple[Tensor, ...],
        state: tuple[Tensor, ...],
    ) -> StepOutput: ...

    def allocate_like(self, *inputs: InputTensor) -> Self: ...

    def initial_state_like(
        self,
        inputs: InputTensor,
        batch_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[Tensor, ...]: ...

    # Sequence scan implementation.

    def _canonicalize_input_sequence(
        self,
        x_seq: InputTensor,
    ) -> tuple[Tensor, ...]:
        """
        Canonicalize a time-major sequence input (or a tuple/dict of
        sequences) into an ordered tuple of sequences. Every sequence must be
        at least ``(time, ...)`` and all sequences must share the time length.
        """
        inputs_seq = self._canonicalize_inputs(x_seq)

        for seq in inputs_seq:
            if seq.dim() < 3:
                raise ValueError(
                    f"{type(self).__name__} expects (time, batch, features) "
                    f"sequence inputs, got {seq.dim()} dims"
                )

        time = inputs_seq[0].shape[0]

        if any(seq.shape[0] != time for seq in inputs_seq):
            raise ValueError(
                f"{type(self).__name__} requires all input sequences to share "
                f"the same time length"
            )

        return inputs_seq

    @staticmethod
    def _first_inputs(inputs_seq: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Per-timestep inputs for the first timestep of a sequence scan.
        """
        return tuple(seq[0] for seq in inputs_seq)

    def forward_sequence(
        self,
        x_seq: InputTensor,
        state: Optional[tuple[Tensor, ...]] = None,
    ) -> Tensor | StepOutput:
        """
        Time-major sequence scan.

        Input shape:
            (time, batch, features)

        Multi-input modules pass a tuple/dict of sequences, one per declared
        input (all sharing the time length):

            (x_seq, inh_seq)

        Hidden mode:
            single output -> output sequence:
                (time, batch, features)
            multiple outputs -> tuple of output sequences:
                ((time, batch, features), ...)

        Explicit mode:
            returns:
                (output_sequence(s), *final_state)

        After ``fast_sequence_()`` this routes through the compiled scan and
        clones returned tensors (safe across ``torch.compile`` / CUDA graphs).

        Note: the eager path uses ``torch.compiler.is_compiling()`` to branch
        between the chunked and the flat scan. Under ``torch.compile`` each
        wrapper sees its own trace; recompiling or nesting multiple compiled
        wrappers is fine, but a manually written wrapper that calls
        ``forward_sequence`` outside a proper compile context may take the
        eager branch even while surrounding code is compiled. Prefer
        ``fast_sequence_()`` over wrapping this method yourself.
        """
        compiled = self._bt_compiled_sequence

        if compiled is not None:
            return compiled(x_seq, state)

        return self._reference_sequence_scan(
            self._canonicalize_input_sequence(x_seq),
            state,
        )

    def _reference_sequence_scan(
        self,
        inputs_seq: tuple[Tensor, ...],
        state: Optional[tuple[Tensor, ...]] = None,
    ) -> Tensor | StepOutput:
        """
        The reference per-step scan. This is the compile unit for
        ``compile_sequence_scan``; keep it free of state-allocation side
        effects that break tracing. ``inputs_seq`` must already be canonical.
        """
        if inputs_seq[0].shape[0] == 0:
            raise ValueError(
                f"{type(self).__name__}.forward_sequence expects at least one timestep"
            )

        if self.init_hidden:
            if state is not None:
                raise ValueError(
                    f"{type(self).__name__} is in init_hidden=True mode; "
                    f"forward_sequence does not accept explicit state"
                )

            return self._hidden_sequence_scan(inputs_seq)

        return self._explicit_sequence_scan(inputs_seq, state)

    def _hidden_sequence_scan(
        self,
        inputs_seq: tuple[Tensor, ...],
    ) -> Tensor | StepOutput:
        first_inputs = self._first_inputs(inputs_seq)

        if not self._bt_allocated:
            self._alloc_hidden(first_inputs)
        elif is_validating(self):
            check_hidden_input_shape(self, first_inputs)

        n_outputs = len(self._bt_output_names)
        state0 = tuple(getattr(self, name) for name in self._bt_state_names)

        # Hidden mode is an explicit scan plus buffer bookkeeping at the
        # edges: the scan itself is pure, so it can share sequence_scan with
        # explicit mode (and with multi-module containers).
        result = sequence_scan(
            lambda inputs, s: self._forward_explicit(inputs, s),
            inputs_seq,
            state0,
            n_outputs,
        )

        ys = result[:n_outputs]

        for name, t in zip(self._bt_state_names, result[n_outputs:]):
            self._buffers[name] = t

        for name, spec, t in zip(self._bt_output_names, self._bt_output_specs, ys):
            last = t[-1] if t.dim() > 0 else t
            if not spec.differentiable:
                last = last.detach()
            self._buffers[name] = last

        if n_outputs == 1:
            return ys[0]

        return ys

    def _explicit_sequence_scan(
        self,
        inputs_seq: tuple[Tensor, ...],
        state: Optional[tuple[Tensor, ...]],
    ) -> StepOutput:
        if state is None:
            state = self.initial_state_for_sequence(inputs_seq)

        n_outputs = len(self._bt_output_names)

        result = sequence_scan(
            lambda inputs, s: self._forward_explicit(inputs, s),
            inputs_seq,
            state,
            n_outputs,
        )

        if n_outputs == 1:
            return (result[0], *result[1:])

        return result

    def compile_sequence_scan(
        self,
        **kwargs: Any,
    ) -> Self:
        """
        Compile the reference sequence scan and route forward_sequence through it.

        Output tensors are cloned before returning only when CUDA graphs are in
        play (``mode="reduce-overhead"`` or ``mode="max-autotune"``): a subsequent
        graph run would otherwise overwrite the previously returned tensor. With
        any other mode the outputs are returned as-is - cloning would cost a full
        extra copy of the spike tensor on every call.

        Explicit mode and ``state=None``: the initial state is allocated inside
        the compiled call on every invocation. That is functionally correct in
        all modes (the returned state is freshly allocated, and graph modes
        clone it), but it is a per-call allocation the compiler must plan for.
        If you call the compiled scan repeatedly on chunks of a long sequence,
        allocate the state once with ``initial_state`` and pass it explicitly to
        skip that allocation and keep the output state un-cloned in default mode.

        Note on sequence length: the compiled unit is the fully unrolled T-step
        scan, so compilation cost and peak memory grow with T. It is fast up to
        roughly T~1000 and impractical (``RecursionError`` in inductor, or
        minutes-long compiles) beyond T~3000. This is not specific to blowtorch:
        norse's compiled sequence and any fully unrolled scan hit the same wall.
        For very long sequences, split the input into chunks and call
        ``forward_sequence`` per chunk.
        """
        needs_clone = kwargs.get("mode") in ("reduce-overhead", "max-autotune")
        compiled = torch.compile(self._reference_sequence_scan, **kwargs)

        def _compiled(
            x_seq: InputTensor,
            state: Optional[tuple[Tensor, ...]] = None,
        ) -> Tensor | StepOutput:
            inputs_seq = self._canonicalize_input_sequence(x_seq)

            if self.init_hidden and inputs_seq[0].shape[0] > 0:
                # Allocate hidden buffers *before* the compiled call. If the
                # initial trace ran the alloc path, the buffer registration
                # side effects break the scan into separate graphs (and under
                # CUDA graphs they alias the compiler's memory pool).
                self.allocate_like(self._first_inputs(inputs_seq))

            out = compiled(inputs_seq, state)

            if isinstance(out, Tensor):
                return out.clone() if needs_clone else out

            return tuple(t.clone() for t in out) if needs_clone else out

        self._bt_compiled_sequence = _compiled

        return self

    def fast_sequence_(
        self,
        compile_scan: bool = True,
        **compile_kwargs: Any,
    ) -> Self:
        """
        Enable a fast research path: validation off + optional compiled scan.

        ``mode="default"`` is always used. ``reduce-overhead`` (CUDA graphs)
        is avoided: it is slower to compile and incompatible with hidden-mode
        buffer registration.
        """
        set_validating(self, False)

        if compile_scan:
            compile_kwargs.setdefault("mode", "default")
            self.compile_sequence_scan(**compile_kwargs)

        return self

    def initial_state_for_sequence(
        self,
        x_seq: InputTensor,
    ) -> tuple[Tensor, ...]:
        inputs_seq = self._canonicalize_input_sequence(x_seq)

        return self.initial_state_like(self._first_inputs(inputs_seq))