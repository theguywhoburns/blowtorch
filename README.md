# Blowtorch

A declarative spiking neural network library for PyTorch research. Neurons
are described by plain nested classes (`Params`, `Specs`) and a single
`_step`; everything else - parameter creation, constraints, resets, hidden
state, and sequence scans - is handled automatically.

Plain PyTorch only. No custom CUDA kernels; the speed comes from
`torch.compile` over pure Python math.

## Features

- **Declarative neurons** - parameters and state are plain class attributes;
  write `_step(x, *state)` and get `forward`, `step_state`, state factories,
  and `forward_sequence` for free.
- **Learnable params & constraints** - every `Param` can be toggled with
  `learnable_<name>=...` / `force_learn_<name>=...` and wrapped in a
  constraint (e.g. `clamp_unit_interval`, `clamp_positive`) applied on the
  hot path. Constraints apply **only to learnable params**: a fixed param is
  used raw, a learnable one is clamped every step.
- **Two execution modes** - hidden (`init_hidden=True`, module owns its
  state buffers) and explicit (caller owns state; a pure function of
  `(x, *state)`).
- **Declarative resets** - `Reset.subtract` / `zero` / `hard_zero` / `set` /
  `add` / `custom`, applied per-state after `_step` in both modes.
- **Surrogate gradients** - per-module `spike_grad=` callable; defaults to a
  straight-through estimator, with sigmoid / atan / triangular / fast-sigmoid
  options (see `blowtorch.util.surrogate_gradients`).
- **Sequence scans** - `forward_sequence` on `(time, batch, features)` with a
  chunked eager scan plus `fast_sequence_()` / `compile_sequence_scan()` to
  compile the whole scan through `torch.compile`.
- **Included neurons** - `LIF`, `AdEx`, `ALIF`, `HH`, `Izhikevich`, `SRM0`,
  `TwoCompartment`.

## Docs

- [Execution modes: hidden vs explicit](docs/execution-modes.md)
- [Resets](docs/resets.md)
- [Constraints](docs/constraints.md)
- [Sequence scans](docs/sequence-scan.md)

## Install

`blowtorch` uses `torch>=2.13.0`, selected via an extra matching your
backend:

```bash
pip install blowtorch[cu130]   # NVIDIA / CUDA 13.0
pip install blowtorch[cpu]     # CPU only
pip install blowtorch[rocm]    # AMD / ROCm 7.2
```

The extras are mutually exclusive: pick exactly one. The `rocm` extra also
pulls the ROCm triton builds (`pytorch-triton-rocm`, `triton-rocm`) needed for
compiled scans on AMD.

## Quick start

```python
import torch
from blowtorch.snn import LIF

x = torch.randn(1000, 32, 64)  # (time, batch, features)

lif = LIF()
state = lif.initial_state_for_sequence(x)
spikes, mem = lif.forward_sequence(x, state)
```

Per-step (explicit state):

```python
spk, (mem,) = lif.step_state(x[0], lif.zero_state((32, 64)))
```

Hidden mode (module owns state):

```python
lif = LIF(init_hidden=True)
spikes = lif.forward_sequence(x)   # state lives in module buffers
```

Compiled scan:

```python
lif.fast_sequence_()               # validation off + compiled scan
spikes = lif.forward_sequence(x)
```

Training (explicit mode, backprop through time over `forward_sequence`):

```python
lif = LIF(init_hidden=False, learnable_beta=True, learnable_threshold=True)
opt = torch.optim.Adam(lif.parameters(), lr=1e-2)

for epoch in range(100):
    opt.zero_grad()

    x = torch.randn(T, B, F)                    # (time, batch, features)
    spikes, _ = lif.forward_sequence(x, lif.zero_state((B, F)))
    rate = spikes.mean(dim=(0, 2))              # mean firing rate per sample

    target = torch.where(x.mean(dim=(0, 2)) > 0, 1.0, 0.0)
    loss = torch.nn.functional.mse_loss(rate, target)
    loss.backward()
    opt.step()
```

See [examples/sequence_training.ipynb](examples/sequence_training.ipynb) for a
complete worked example.

## Defining a neuron

Subclass `SnnModule`, declare `Params` and `Specs`, and implement `_step`:

```python
from blowtorch.base import clamp_positive, clamp_unit_interval
from blowtorch.snn import Reset, SnnModule

class MyLIF(SnnModule):
    class Params:
        beta = SnnModule.Param(0.9, constraint=clamp_unit_interval)
        threshold = SnnModule.Param(1.0, constraint=clamp_positive)

    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        mem = SnnModule.StateSpec(reset=Reset.subtract("threshold"))

    def _step(self, x, mem):
        beta, threshold = self.constrained()   # hot path, constraints applied
        mem = beta * mem + x
        spk = self.spike_grad(mem - threshold)
        return spk, mem
```

Key pieces:

- `Params` entries become constructor kwargs and `nn.Parameter`s, each with
  `learnable_<name>`, `force_learn_<name>`, and `<name>_constraint`
  overrides. `self.constrained()` returns the constrained values in
  declaration order.
- `Specs` entries are `OutputSpec` (returned, not recurrent) or `StateSpec`
  (recurrent state, threaded through `_step`). `StateSpec(shape=...)` can
  decouple the state shape from the input shape.
- `_step(x, *state)` returns a tuple whose first element is the spike output;
  resets declared in `Specs` are applied to the state afterwards.
- `Constant` declares a non-learnable hyperparameter (e.g. `dt`), with an
  optional `validate=` callable.

## Behavior & tuning

- **Surrogate gradient** - pass `spike_grad=...` at construction to swap the
  default straight-through estimator for a smooth surrogate. Available:
  `sigmoid_surrogate(beta)`, `atan_surrogate(beta)`,
  `triangular_surrogate(beta)`, `fast_sigmoid_surrogate(beta)`, and
  `straight_through_surrogate` (the default).
- **Resets** - `Reset.subtract("threshold")` subtracts a target param,
  `zero()` / `hard_zero()` reset to zero, `set("v_reset")` sets a value,
  `add("b")` injects an amount (AdEx adaptation), `custom(fn)` calls a method.
- **Validation** - `validate=True` (default) checks step arity and state
  shapes on each call; override per module, or globally with
  `set_validation()` / `no_validation()`.
- **Long sequences** - the compiled scan is a fully unrolled T-step graph, so
  compilation cost and memory grow with T (fast up to ~T=1000, impractical
  past ~T=3000). Chunk the input for very long sequences, or tune the eager
  scan with `set_sequence_scan_chunk(n)`.
- **CUDA graphs** - hidden-mode buffers are allocated lazily; call
  `allocate_like(x)` before capturing a compiled graph.

## Bench vs snnTorch / Norse (LIF)

**NOTE: Benchmarks ran on** `NVIDIA GeForce RTX 3050 Laptop GPU 4G`,
T=1000, B=32, F=1024.

| library       | mode          | compiled | ms     | steps/s | peak MiB | vs blowtorch seq eager |
| ------------- | ------------- | -------- | ------ | ------- | -------- | ---------------------- |
| blowtorch LIF | seq hidden    | eager    | 41.41  | 24,151  | 254.4    | 1.00x                  |
| blowtorch LIF | seq hidden    | compile  | 3.87   | 258,415 | 252.6    | 0.09x                  |
| blowtorch LIF | seq explicit  | eager    | 44.69  | 22,377  | 254.6    | 1.08x                  |
| blowtorch LIF | seq explicit  | compile  | 3.67   | 272,464 | 252.4    | 0.09x                  |
| blowtorch LIF | step hidden   | eager    | 40.05  | 24,966  | 126.8    | 0.97x                  |
| blowtorch LIF | step hidden   | compile  | 35.45  | 28,212  | 126.5    | 0.86x                  |
| blowtorch LIF | step explicit | eager    | 37.63  | 26,574  | 127.0    | 0.91x                  |
| blowtorch LIF | step explicit | compile  | 31.08  | 32,175  | 126.8    | 0.75x                  |
| snntorch      | seq           | eager    | 119.75 | 8,351   | 377.5    | 2.89x                  |
| snntorch      | seq           | compile  | 45.98  | 21,750  | 378.0    | 1.11x                  |
| norse         | seq           | eager    | 112.29 | 8,906   | 378.3    | 2.71x                  |
| norse         | seq           | compile  | 5.31   | 188,418 | 376.6    | 0.13x                  |
| norse         | step          | eager    | 111.96 | 8,932   | 128.0    | 2.70x                  |
| norse         | step          | compile  | 41.83  | 23,907  | 127.3    | 1.01x                  |

Even **non-compiled** blowtorch (41.4 ms) beats **eager** snnTorch
(119.8 ms, ~2.9x) and Norse (112.3 ms, ~2.7x). The compiled scan (3.7-3.9 ms)
is ~11x faster than blowtorch's own eager scan, ~12x faster than compiled
snnTorch, and ~1.4x faster than compiled Norse (5.3 ms) - all with zero
custom kernels, pure Python math through `torch.compile`.

Compiling the **per-step loop** barely helps (35.5 vs 40.1 ms): Python
dispatch per step dominates. Compiling the whole **sequence scan** is where
the win is, because the scan fuses the full unrolled graph.

Memory is comparable across frameworks in the same mode. Sequence mode holds
the full `(T, B, F)` spike stack (blowtorch ~252 MiB vs snnTorch/Norse
~377 MiB - blowtorch also holds the least while returning the same output);
step mode only holds the state (~127 MiB across all three).

> **Ratio convention**: `bench_all_vs.py` prints `framework_time / blowtorch_seq_eager_time` as the trailing `(N.Nx)` factor. A value **below
> 1.0 means the framework is faster** than blowtorch LIF seq eager. The base
> is the first measured row, `blowtorch LIF seq hidden eager`.

Run the benchmark yourself:

```bash
uv run --group bench python benchmarks/bench_all_vs.py --steps 1000
```

Results are printed to the console and exported to a CSV under
`benchmarks/results/`.

## Development

```bash
uv run pytest            # run the test suite
uv run pyright src       # type-check
```
