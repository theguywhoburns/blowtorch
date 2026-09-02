# Sequential networks

`pyrokinesis.nn.Sequential` stacks layers into a network: stateful neurons
(`PyroModule` subclasses, e.g. `LIF`) and plain stateless `nn.Module`
layers (e.g. `nn.Linear`) in any order. It is a thin topology manager - it
moves data between layers and threads a flat state tuple through the
time-major scan, mirroring `torch.nn.Sequential`.

```python
import torch
from pyrokinesis.nn import Sequential
from pyrokinesis.snn import LIF

net = Sequential(nn.Linear(4, 8), LIF(), LIF())
```

## Shape rules

Each layer maps an input shape to an output shape. Stateful layers own their
state shapes; the container walks the whole stack on a meta-device pass
(no math, no side effects) to learn the output shape and the shape of every
state tensor.

- Stateful layers must be **single-input** and **single-output** (one tensor
  per step). A layer declaring multiple inputs (e.g. `MCN`) or multiple outputs
  raises at construction.
- Stateless layers must return a **single tensor** per step.
- Inputs are `(batch, features)` per step, `(time, batch, features)` for
  sequences.
- The feature dimension must match between consecutive layers: `nn.Linear`
  output features must equal the next neuron's input features, and so on.

## Running a network

### Hidden mode (the container owns the state)

```python
net = Sequential(nn.Linear(4, 8), LIF(), LIF(), init_hidden=True)

out = net(x)                 # (batch, 8), buffers auto-allocated
spikes = net.forward_sequence(x_seq)  # (time, batch, 8)
net.reset()                  # re-initialize buffers
```

The input shape must stay fixed in hidden mode; a different shape raises
`ValueError` (disable with `net.validate = False` or `fast_sequence_()`).

### Explicit mode (you own the state)

```python
net = Sequential(nn.Linear(4, 8), LIF(), LIF())

state = net.initial_state((batch, 4))
out, next_state = net.step(x, state)          # single step
spikes, final_state = net.forward_sequence(x_seq, state)  # time-major
```

State is a flat tuple, one tensor per stateful layer's state, in layer order.
`initial_state` / `zero_state` / `initial_state_like` / `initial_state_for_sequence`
fill the same defaults the layers themselves use.

## Compiling the whole stack

`fast_sequence_()` compiles the entire network as one fused scan and disables
validation on the container and every stateful child.

```python
net.fast_sequence_()
spikes = net.forward_sequence(x_seq)
```

For more control, `compile_sequence_scan(**kwargs)` compiles the scan without
touching validation; `fast_sequence_(compile_scan=False)` only disables
validation. The same rules as [Sequence scans](sequence-scan.md) apply:
graph modes clone outputs, `state=None` allocates per call, provided state is
never mutated.

## Training

Parameters train exactly as with bare modules - the scan is differentiable.

```python
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
ys, *_ = net.forward_sequence(x_seq, net.initial_state((batch, 4)))
loss = torch.nn.functional.mse_loss(ys, target)
loss.backward()
opt.step()
```

`state_dict()` / `load_state_dict()` work as usual; in hidden mode the live
buffers are kept out of the state dict via `get_extra_state` /
`set_extra_state`.

## `__repr__`

`repr(net)` lists each child layer with its own repr:

```
Sequential(
  Linear(in_features=4, out_features=8, bias=True),
  LIF(...),
  LIF(...)
)
```
