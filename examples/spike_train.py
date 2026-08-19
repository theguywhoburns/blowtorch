"""How to use ``SpikeTrain``: encoders, packing, and GPU iteration.

``SpikeTrain`` is a time-major spike container that stays compact on the GPU:
the dense view is ``(T, B, ...)`` integer spike counts, while the packed view
keeps only the events (``spk_ind`` + ``time_pointer``). Generators
(``population``, ``poisson``, ``latency``, ``custom``) build trains directly on
whatever device their inputs live on.

Run with::

    uv run python examples/spike_train.py
"""

from __future__ import annotations

import torch

from blowtorch.nn import Sequential
from blowtorch.snn import LIF
from blowtorch.util import SpikeTrain

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def section(title: str) -> None:
    print(f"\n== {title}")


section("population encoding (Gaussian receptive fields, Poisson counts)")
tau = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]], device=DEVICE)
pop = SpikeTrain.population(tau, M=32, T=16, seed=0)
print(f"shape={pop.shape}  device={pop.device}  spikes={pop.num_spikes}")
print("dense counts per step:", [int(pop[t].sum()) for t in range(pop.T)])

section("poisson (constant rate)")
rate = torch.full((4, 8), 0.1, device=DEVICE)
pois = SpikeTrain.poisson(rate, T=1000, dt=1.0, seed=0)
print(f"shape={pois.shape}  spikes={pois.num_spikes}  "
      f"mean rate={pois.num_spikes / (1000 * 4 * 8):.3f} (expect ~0.1)")

section("latency-to-first-spike")
values = torch.tensor([[0.0, 0.5, 1.0]], device=DEVICE)
lat = SpikeTrain.latency(values, T=16)
print(f"shape={lat.shape}  first-spike times={[int(t) for t in lat.tensor.argmax(0).flatten()]}")
print("(v=0 never fires, v=1 fires at t=0, v=0.5 fires at t=round(7.5)=8)")

section("custom encoder (packing handled for you)")
def delta_encoder(values: torch.Tensor, T: int) -> torch.Tensor:
    """One spike at the last step for every unit."""
    dense = torch.zeros(T, *values.shape, dtype=torch.int64)
    dense[-1] = 1
    return dense

custom = SpikeTrain.custom(delta_encoder, torch.rand(4, 4, device=DEVICE), T=16)
print(f"shape={custom.shape}  spikes={custom.num_spikes}  all at t=15: "
      f"{bool((custom[15] == 1).all())}")

section("packed view: only the events are stored")
spk_ind, time_pointer = pop.packed
print(f"spk_ind={tuple(spk_ind.shape)}  time_pointer={tuple(time_pointer.shape)}  "
      f"vs dense cells={pop.shape[0] * pop.shape[1] * pop.shape[2] * pop.shape[3]}")
assert torch.equal(pop.to_dense(), SpikeTrain.from_dense(pop.to_dense()).to_dense())

section("iteration feeds a Sequential step loop (no dense tensor kept)")
net = Sequential(LIF(), init_hidden=True).to(DEVICE)
out = net(pop[0].float())
for step in pop:
    out = net(step.float())
print(f"scanned {pop.T} steps -> output {tuple(out.shape)}")

section("moving a train between devices")
if DEVICE == "cuda":
    moved = lat.to("cpu")
    print(f"lat cpu: {moved.device}, back on gpu: {moved.to('cuda').device}")

print("\ndone")