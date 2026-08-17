# Execution modes: hidden vs explicit

Every `SnnModule` runs in one of two modes, set at construction with
`init_hidden=`.

## Explicit mode (`init_hidden=False`, default)

The **caller owns the state**. The module is a pure function of
`(x, *state)` - no hidden buffers, nothing mutated between calls.

```python
from blowtorch.snn import LIF

lif = LIF()

state = lif.zero_state((32, 64))          # zeroed initial state
state = lif.initial_state((32, 64))       # Spec.default initial state

spk, (mem,) = lif.step_state(x_step, state)   # (output, next_state)

x_seq = torch.randn(1000, 32, 64)         # (time, batch, features)
spikes, mem = lif.forward_sequence(x_seq, state)  # (output_seq, *final_state)
```

Per-step:

- `forward(x, *state)` returns `(spk, *next_state)`.
- `step_state(x, state)` / `step(x, state)` take the state as a tuple and
  return `(output, next_state)`.
- `forward_sequence(x_seq, state)` returns
  `(output_sequence(s), *final_state)`. With `state=None` the initial state
  is derived from the input shape (a per-call allocation; pass your own state
  when calling repeatedly).

Use explicit mode for pure, replayable stepping: e.g. BPTT, gradient
checkpointing, or sweeping initial states. The state tensors are returned
fresh each call; nothing is written in place.

## Hidden mode (`init_hidden=True`)

The **module owns the state** as non-persistent buffers, allocated lazily on
the first forward call from the input shape.

```python
lif = LIF(init_hidden=True)
spikes = lif.forward_sequence(x_seq)      # state lives in module buffers
spikes, mem = lif(x_step)                 # per-step hidden forward
```

Key rules:

- Passing explicit state while `init_hidden=True` raises `ValueError`.
- `forward_sequence` returns only the output sequence(s), not a final state.
- The buffers keep the **first** input's shape; a later input with different
  batch/feature dims raises `ValueError` (when validation is on).
- `reset()` refills the buffers with Spec defaults (same shape);
  `detach()` detaches them from autograd.
- Single-output modules return a plain tensor; multi-output modules return a
  tuple of output tensors.

Use hidden mode for self-contained modules you feed and read back - the
ergonomic default for sequence training.

## Which to pick

|                         | explicit                   | hidden                     |
| ----------------------- | -------------------------- | -------------------------- |
| owns state              | caller                     | module buffers             |
| forward returns         | `(spk, *next_state)`       | output tensor(s)           |
| forward_sequence returns| `(outputs, *final_state)`  | output sequence(s)         |
| state factories         | `initial_state`/`zero_state` | `reset()`, lazy alloc     |
| best for                | BPTT, replayable stepping  | self-contained sequences   |

Both modes apply the same declarative resets after `_step`, and both route
through `compile_sequence_scan()` when a compiled scan is installed.
