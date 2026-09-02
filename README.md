# pyrokinesis

[![CI](https://github.com/theguywhoburns/pyrokinesis/actions/workflows/test.yml/badge.svg)](https://github.com/theguywhoburns/pyrokinesis/actions/workflows/test.yml)

A declarative spiking neural network library for PyTorch research. Neurons
are described by plain nested classes (`Params`, `Specs`) and a single
`_step`; the rest - parameter creation, constraints, resets, hidden state,
and sequence scans - is automatic.

Plain PyTorch only. No custom CUDA kernels; the speed comes from
`torch.compile` over pure Python math.

## Features

- **Declarative neurons** - parameters and state are plain class attributes;
  write `_step(x, *state)`; `forward`, `step_state`, the state factories, and
  `forward_sequence` follow.
- **Learnable params & constraints** - every `Param` can be toggled with
  `learnable_<name>=...` / `force_learn_<name>=...` and wrapped in a
  constraint (e.g. `clamp_unit_interval`, `clamp_positive`) applied on the
  hot path. Constraints apply **only to learnable params**: a fixed param is
  used raw, a learnable one is clamped every step.
- **Two execution modes** - hidden (`init_hidden=True`, module owns its
  state buffers) and explicit (caller owns state; a pure function of
  `(x, *state)`).
- **Multi-input modules** - declare a nested `Inputs` class and `_step`
  receives the inputs positionally (e.g. an MCN's basal and apical inputs).
  Call sites accept a single tensor, a tuple/list, or a dict keyed by input
  name, and `StateSpec(shape=...)` can follow any named input (added in
  e4cab2c).
- **Declarative resets** - `Reset.subtract` / `zero` / `hard_zero` / `set` /
  `add` / `custom`, applied per-state after `_step` in both modes.
- **Surrogate gradients** - per-module `spike_grad=` callable; defaults to a
  straight-through estimator, with sigmoid / atan / triangular / fast-sigmoid
  options (see `pyrokinesis.util.surrogate_gradients`).
- **Sequence scans** - `forward_sequence` on `(time, batch, features)` with a
  chunked eager scan plus `fast_sequence_()` / `compile_sequence_scan()` to
  compile the whole scan through `torch.compile`.
- **Spike trains** - `SpikeTrain` turns quantile fractions into Poisson
  population spike trains via Gaussian receptive fields, and also ships
  Poisson and latency encoders plus `.custom(...)`; dense or event-packed GPU
  form (`pyrokinesis.util.SpikeTrain`). See
  [examples/spike_train.py](examples/spike_train.py).
- **Included neurons** - `LIF`, `AdEx`, `ALIF`, `HH`, `Izhikevich`, `SRM0`,
  `TwoCompartment`, `MCN` (three-compartment basal/apical/soma).

## Docs

- [Execution modes: hidden vs explicit](docs/execution-modes.md)
- [Resets](docs/resets.md)
- [Constraints](docs/constraints.md)
- [Sequence scans](docs/sequence-scan.md)
- [Sequential networks](docs/sequential.md)
- [Publishing to PyPI](docs/publishing.md)

## Install

`pyrokinesis` uses `torch>=2.13.0`, selected via an extra matching your
backend:

```bash
pip install pyrokinesis[cu130]   # NVIDIA / CUDA 13.0
pip install pyrokinesis[cpu]     # CPU only
pip install pyrokinesis[rocm]    # AMD / ROCm 7.2
```

The extras are mutually exclusive: pick exactly one. The `rocm` extra also
pulls the ROCm triton builds (`pytorch-triton-rocm`, `triton-rocm`) needed for
compiled scans on AMD.

> **Note:** the `[tool.uv.sources]` index selection for `torch` is `uv`-only;
> plain `pip` will resolve `torch` from the default PyPI index regardless of
> the extra. Use `uv` for the backend-pinned installs above.

## Quick start

```python
import torch
from pyrokinesis.snn import LIF

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
from pyrokinesis import clamp_positive, clamp_unit_interval
from pyrokinesis.snn import Reset, SnnModule

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
T=1000, B=32, F=1024. Pyrokinesis runs with `validate=False` (snnTorch/Norse
have no equivalent toggle) — steady-state `torch.compile` numbers are comparable
either way.

| library         | mode          | compiled | ms      | steps/s | peak MiB | vs pyrokinesis seq eager |
| --------------- | ------------- | -------- | ------- | ------- | -------- | ------------------------ |
| pyrokinesis LIF | seq hidden    | eager    | 39.000  | 25,641  | 254.6    | 1.00x                    |
| pyrokinesis LIF | seq hidden    | compile  | 3.393   | 294,696 | 252.5    | 0.09x                    |
| pyrokinesis LIF | seq explicit  | eager    | 46.600  | 21,459  | 254.8    | 1.19x                    |
| pyrokinesis LIF | seq explicit  | compile  | 3.382   | 295,673 | 252.4    | 0.09x                    |
| pyrokinesis LIF | step hidden   | eager    | 42.658  | 23,443  | 126.8    | 1.09x                    |
| pyrokinesis LIF | step hidden   | compile  | 35.960  | 27,809  | 126.5    | 0.92x                    |
| pyrokinesis LIF | step explicit | eager    | 39.207  | 25,505  | 127.0    | 1.01x                    |
| pyrokinesis LIF | step explicit | compile  | 34.105  | 29,321  | 126.8    | 0.87x                    |
| snntorch        | seq           | eager    | 100.060 | 9,994   | 377.5    | 2.57x                    |
| snntorch        | seq           | compile  | 37.414  | 26,728  | 378.0    | 0.96x                    |
| norse           | seq           | eager    | 94.028  | 10,635  | 378.3    | 2.41x                    |
| norse           | seq           | compile  | 4.298   | 232,693 | 376.6    | 0.11x                    |
| norse           | step          | eager    | 97.280  | 10,280  | 128.0    | 2.49x                    |
| norse           | step          | compile  | 35.817  | 27,920  | 127.3    | 0.92x                    |

Even uncompiled, pyrokinesis (39.0 ms) beats eager snnTorch (100.1 ms, ~2.6x)
and eager Norse (94.0 ms, ~2.4x). The compiled scan (3.39 ms) is ~11.5x faster
than pyrokinesis's own eager scan, ~11x faster than compiled snnTorch
(37.4 ms), and ~1.27x faster than compiled Norse (4.30 ms). No custom
kernels; the speed is pure Python math through `torch.compile`.

Compiled timings vary run-to-run on this laptop GPU (typical single runs read
~3.3-3.8 ms; best-of runs drop to ~3.38 ms); the compiled rows above are the
latest steady-state values.

Compiling the per-step loop barely helps (35.9 vs 42.6 ms): per-step Python
dispatch dominates. Compiling the whole sequence scan wins because it fuses
the unrolled graph into one call.

Memory is comparable in step mode (~127 MiB across all three). Sequence mode
holds the full `(T, B, F)` spike stack: snnTorch/Norse ~378 MiB;
pyrokinesis ~254 MiB for both hidden and explicit (eager and compiled) —
~1.5x less — because hidden buffers no longer hold a view into the `(T,B,F)`
output (see `_store_hidden_seq_buffers` in `src/pyrokinesis/module/scan.py`).

> **Ratio convention**: `bench_all_vs.py` prints `framework_time / pyrokinesis_seq_eager_time` as the trailing `(N.Nx)` factor. A value **below
> 1.0 means the framework is faster** than pyrokinesis LIF seq eager. The base
> is the first measured row, `pyrokinesis LIF seq hidden eager`.

Run the benchmark yourself:

```bash
uv run --group bench python benchmarks/bench_all_vs.py --steps 1000
```

Results are printed to the console and exported to a CSV under
`benchmarks/results/`.

## Bench vs snnTorch / Norse (multilayer LIF network)

**NOTE:** Benchmarks ran on `NVIDIA GeForce RTX 3050 Laptop GPU 4G`,
T=1000, B=32, F=512, network `Linear(512,512) -> LIF x4 -> Linear(512,10) ->
LIF` (Linear-light, LIF-heavy).

| library                | mode      | compiled | ms      | steps/s | peak MiB | vs pyrokinesis seq eager |
| ---------------------- | --------- | -------- | ------- | ------- | -------- | ------------------------ |
| pyrokinesis Sequential | seq       | eager    | 213.395 | 4,686   | 74.4     | 1.00x                    |
| pyrokinesis Sequential | seq       | compile  | 22.611  | 44,225  | 79.5     | 0.11x                    |
| pyrokinesis Sequential | step      | eager    | 221.243 | 4,520   | 73.7     | 1.04x                    |
| snntorch               | step loop | eager    | 565.490 | 1,768   | 76.1     | 2.65x                    |
| snntorch               | step loop | compile  | 101.953 | 9,808   | 75.6     | 0.48x                    |
| norse                  | seq       | eager    | 513.739 | 1,947   | 260.7    | 2.41x                    |
| norse                  | seq       | compile  | 25.647  | 38,991  | 260.1    | 0.12x                    |

The whole network compiles as one fused scan (22.6 ms): ~10x faster than
pyrokinesis's own eager sequence, ~1.13x faster than compiled Norse (25.6 ms),
~2.4x faster than eager Norse, and ~2.65x faster than eager snnTorch. The
compiled stack also uses ~3.3x less GPU memory than Norse (79.5 vs 260.1 MiB).

snnTorch has no sequence module, so it runs the canonical per-step loop with
explicit membrane state (the `step loop` rows). Its compiled row (101.9 ms) is
not representative: torch.compile stalls on the Python state-threading loop.
The eager loop is the fair comparison.

Run the benchmark yourself:

```bash
uv run --group bench python benchmarks/bench_multilayer_sequential_vs_all.py --steps 1000
```

Results are printed to the console and exported to a CSV under
`benchmarks/results/`.

## Development

```bash
uv run pytest            # run the test suite
uv run pyright src       # type-check (src/ only; tests/benchmarks excluded)
uv run ruff check        # lint (rules: E, F, B, RUF; see pyproject.toml)
```

- Tests: pytest. Type-checking: pyright covers `src/` only (the declarative
  `Param()` is `Any`-typed by design, so the tests/benchmarks surface
  produces ~300 noise diagnostics; see the `exclude` in `[tool.pyright]`).
- Formatting: intentionally not auto-formatted; don't send format-only PRs.
