# Sequence scans

`forward_sequence` scans a time-major input `(time, batch, features)` through
the module, returning the per-step output(s) stacked along time.

```python
spikes, mem = lif.forward_sequence(x_seq, state)  # explicit
spikes = lif.forward_sequence(x_seq)  # hidden
```

## Eager scan

The eager reference scan is **chunked**: it runs `_SEQUENCE_SCAN_CHUNK` steps
(8 by default) per Python iteration and scatters the results into
preallocated `(T, B, F)` buffers, avoiding a per-step list + `torch.stack`
copy while keeping dispatch overhead low. Tune the chunk with
`set_sequence_scan_chunk(n)`.

## Compiled scan

`compile_sequence_scan()` compiles the fully unrolled T-step scan with
`torch.compile` and routes `forward_sequence` through it. `fast_sequence_()`
is the convenience entry point: it disables validation and installs a
compiled scan (`mode="default"`).

```python
lif.fast_sequence_()  # validation off + compiled scan
spikes = lif.forward_sequence(x_seq)
```

Important properties:

- **Output cloning is mode-dependent.** Only `mode="reduce-overhead"` and
  `mode="max-autotune"` (CUDA graphs) clone returned tensors, so held
  references survive the next graph run. In `mode="default"` outputs are
  returned as-is - cloning would add a full copy of the spike tensor per call.
- **`state=None` allocates the initial state inside the compiled call** every
  time. It is correct in all modes (and graph modes clone the returned
  state), but it is a per-call allocation. For repeated calls (e.g. feeding
  chunks of a long sequence), allocate once with `initial_state` and pass it
  explicitly.
- **Provided state is never mutated.** The scan reads the state you pass and
  returns fresh final-state tensors.
- **Hidden-mode buffers must exist before the compiled call.** The wrapper
  allocates them for you (`allocate_like`), but under CUDA graphs call
  `allocate_like(x)` yourself before capturing if you control capture timing.

## Length limits

The compiled unit is the fully unrolled T-step graph, so compile time and
peak memory grow with T: fast up to roughly T~1000, impractical
(`RecursionError` in inductor or multi-minute compiles) past ~T=3000. This is
inherent to fully unrolled scans - norse's compiled sequence hits the same
wall. For long sequences, split the input into chunks and call
`forward_sequence` per chunk with an explicit, rolling state.

## Edge cases

- `T=1` and `T=2` are handled by both the eager and compiled scans (the
  chunked path starts at index 1 and the compiled path unrolls whatever T is).
- Sequence inputs must be 3D: `(time, batch, features)`. Other ranks raise
  `ValueError` from `initial_state_for_sequence`.
- The eager path branches on `torch.compiler.is_compiling()` internally.
  Prefer `fast_sequence_()` over wrapping `forward_sequence` in your own
  `torch.compile` wrapper.
