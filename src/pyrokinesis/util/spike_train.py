from __future__ import annotations

import math
from typing import Any, Callable, Iterator, Optional

import torch

from pyrokinesis import Tensor

__all__ = ["SpikeTrain"]


class SpikeTrain:
    """
    A time-major spike train in dense or event-packed form.

    The dense view is a ``(T, B, ...)`` integer tensor of spike counts
    (Poisson encoders may fire several spikes per step). The packed view
    stores only the spike events, GPU-friendly:

      - ``spk_ind``: ``(S,)`` int64 flat indices into the ``T * B * D`` grid
        (row-major, ``idx = ((t * B + b) * D + d)``). Events are sorted by
        time; multi-spike cells appear as repeated indices.
      - ``time_pointer``: ``(T + 1,)`` int64 cumulative event counts, so
        ``spk_ind[time_pointer[t] : time_pointer[t + 1]]`` are the events of
        timestep ``t``.

    Construction and ``from_dense``/``to_dense`` are vectorized on the packed
    tensors' device (``argwhere`` / ``repeat_interleave`` / ``index_add_``) with
    no Python loop over timesteps, so large trains stay on the GPU.
    Per-timestep access (``__getitem__``/``__iter__``) iterates in Python and
    syncs ``time_pointer`` per step.

    ``SpikeTrain.custom(fn, *args, **kwargs)`` packs the output of any
    user-provided generator: ``fn`` returns a dense ``(T, B, ...)`` tensor and
    the class handles the rest.
    """

    def __init__(
        self,
        spk_ind: Tensor,
        time_pointer: Tensor,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.int64,
    ) -> None:
        if time_pointer.dim() != 1 or time_pointer.shape[0] != shape[0] + 1:
            raise ValueError(
                f"time_pointer must have shape (T+1,) = ({shape[0] + 1},), "
                f"got {tuple(time_pointer.shape)}"
            )
        self.spk_ind = spk_ind
        self.time_pointer = time_pointer
        self.shape = tuple(shape)
        self.dtype = dtype

    # Generators

    @classmethod
    def population(
        cls,
        tau: Tensor,  # (B, N) fractions in [0, 1]
        M: int = 64,
        T: int = 8,
        sigma: float = 0.15,
        phi: float = 1.0,
        dt: float = 1.0,
        seed: Optional[int] = None,
    ) -> "SpikeTrain":
        """
        Population encoding: scalar quantile fractions become Poisson
        population spike trains via Gaussian receptive fields (paper sec. 3.2,
        eq. 13-17).

        ``M`` neurons with Gaussian receptive fields tile the fraction axis
        ``[0, 1]``: neuron ``j`` has center ``mu_j = (j - 1)/(M - 1)`` and
        firing rate ``r_j = phi * exp(-(tau - mu_j)**2 / (2 * sigma**2))``.
        Each neuron fires as a Poisson process: per step, ``Poisson(r_j * dt)``
        spike counts. Returns a ``(T, B, N, M)`` train. Non-differentiable
        sampling: use this as an input encoder, not a differentiable layer.
        """
        if tau.ndim != 2:
            raise ValueError(f"tau must be (B, N), got shape {tuple(tau.shape)}")
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if phi < 0:
            raise ValueError(f"phi must be non-negative, got {phi}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        if M == 1:
            mu = torch.tensor([0.5], device=tau.device, dtype=tau.dtype)
        else:
            mu = torch.linspace(0.0, 1.0, M, device=tau.device, dtype=tau.dtype)

        # r: (B, N, M) firing rates.
        rates = phi * torch.exp(-((tau[..., None] - mu) ** 2) / (2 * sigma**2))

        # Per-step Poisson spike counts, shape (T, B, N, M).
        mean = (rates * dt).unsqueeze(0).expand(T, *rates.shape)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=tau.device)
            generator.manual_seed(seed)

        counts = torch.poisson(mean, generator=generator)
        return cls.from_dense(counts.long())

    @classmethod
    def poisson(
        cls,
        rate: Tensor,  # (B, ...) per-step firing rates
        T: int,
        dt: float = 1.0,
        seed: Optional[int] = None,
    ) -> "SpikeTrain":
        """
        Poisson spike trains from constant per-step rates.

        Each step draws ``Poisson(rate * dt)`` spike counts, so the expected
        total count over ``T`` steps is ``rate * dt * T``.
        """
        if rate.dim() < 1:
            raise ValueError(f"rate must be at least (B,), got shape {tuple(rate.shape)}")

        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")

        mean = (rate * dt).unsqueeze(0).expand(T, *rate.shape)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=rate.device)
            generator.manual_seed(seed)

        counts = torch.poisson(mean, generator=generator).long()
        return cls.from_dense(counts)

    @classmethod
    def latency(
        cls,
        value: Tensor,  # (B, ...) in [0, 1]
        T: int,
    ) -> "SpikeTrain":
        """
        Latency-to-first-spike encoding: each unit fires exactly once.

        A unit with value ``v`` fires at ``t = round((1 - v) * (T - 1))``
        (``v = 1`` fires at ``t = 0``, small values fire near ``T - 1``) and
        never fires when ``v = 0``.
        """
        if value.dim() < 1:
            raise ValueError(f"value must be at least (B,), got shape {tuple(value.shape)}")

        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")

        value = torch.clamp(value, 0.0, 1.0)
        B = value.shape[0]
        spatial = value.shape[1:]
        D = math.prod(spatial) if spatial else 1

        times = torch.round((1 - value) * (T - 1)).long()
        valid = value > 0
        positions = torch.arange(B * D, device=value.device).reshape(value.shape)
        idx = times * (B * D) + positions
        idx = idx[valid]

        dense = torch.zeros(T * B * D, dtype=torch.int64, device=value.device)

        if idx.numel():
            dense[idx] = 1

        dense = dense.reshape(T, B, *spatial)
        return cls.from_dense(dense)

    @classmethod
    def custom(
        cls,
        fn: Callable[..., Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> "SpikeTrain":
        """
        Pack the output of a user-provided generator.

        ``fn(*args, **kwargs)`` must return a dense ``(T, B, ...)`` tensor of
        spike counts; packing, metadata, and slicing are handled here:

            SpikeTrain.custom(
                lambda tau, T: my_encoder(tau, T),
                tau, T=8,
            )
        """
        dense = fn(*args, **kwargs)

        if not isinstance(dense, Tensor):
            raise TypeError(
                f"custom encoder must return a Tensor, got {type(dense).__name__}"
            )

        return cls.from_dense(dense)

    # Construction from dense

    @classmethod
    def from_dense(cls, dense: Tensor) -> "SpikeTrain":
        """
        Build a packed train from a dense ``(T, B, ...)`` integer tensor.

        Nonzero entries are spike counts: a cell with count ``k`` becomes
        ``k`` events (repeated indices), so Poisson multi-spike trains
        round-trip exactly. Float/bool inputs are treated as counts (rounded
        down) / presence.
        """
        if dense.dim() < 2:
            raise ValueError(
                f"dense spike train must be (T, B, ...), got shape {tuple(dense.shape)}"
            )

        T, B = dense.shape[0], dense.shape[1]
        flat = dense.reshape(T, B, -1).reshape(-1)

        mask = flat != 0
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        spk_ind = torch.repeat_interleave(idx, flat[mask].long())

        step_counts = dense.reshape(T, B, -1).sum(dim=(1, 2)).long()
        time_pointer = torch.zeros(T + 1, dtype=torch.int64, device=dense.device)
        time_pointer[1:] = step_counts.cumsum(dim=0)

        return cls(spk_ind, time_pointer, tuple(dense.shape))

    # Views

    @property
    def T(self) -> int:
        return self.shape[0]

    @property
    def batch(self) -> int:
        return self.shape[1]

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self.shape[2:]

    @property
    def device(self) -> torch.device:
        return self.spk_ind.device

    @property
    def num_spikes(self) -> int:
        return int(self.spk_ind.numel())

    @property
    def packed(self) -> tuple[Tensor, Tensor]:
        """``(spk_ind, time_pointer)`` event-packed form."""
        return self.spk_ind, self.time_pointer

    def to_dense(self) -> Tensor:
        """
        Reconstruct the dense ``(T, B, ...)`` integer count tensor.

        One vectorized ``index_add_`` scatter; the dense tensor is not kept in
        memory, so packed trains stay compact until unpacked.
        """
        D = math.prod(self.spatial_shape) if self.spatial_shape else 1
        grid = torch.zeros(
            self.T * self.batch * D,
            dtype=self.dtype,
            device=self.device,
        )

        if self.num_spikes:
            grid.index_add_(0, self.spk_ind, torch.ones_like(self.spk_ind))

        return grid.reshape(self.shape)

    @property
    def tensor(self) -> Tensor:
        """Dense ``(T, B, ...)`` count view."""
        return self.to_dense()

    def to(self, device: torch.device | str) -> "SpikeTrain":
        """Return a copy of this train on ``device``."""
        return SpikeTrain(
            self.spk_ind.to(device),
            self.time_pointer.to(device),
            self.shape,
            self.dtype,
        )

    # Per-timestep access

    def __getitem__(self, t: int) -> Tensor:
        """Dense ``(B, ...)`` count slice for timestep ``t``."""
        if t < 0:
            t += self.T

        if t < 0 or t >= self.T:
            raise IndexError(f"timestep {t} out of range for T={self.T}")

        D = math.prod(self.spatial_shape) if self.spatial_shape else 1
        lo = int(self.time_pointer[t])
        hi = int(self.time_pointer[t + 1])

        grid = torch.zeros(
            self.batch * D,
            dtype=self.dtype,
            device=self.device,
        )

        if hi > lo:
            local = self.spk_ind[lo:hi] - t * self.batch * D
            grid.index_add_(0, local, torch.ones_like(local))

        return grid.reshape(self.shape[1:])

    def __iter__(self) -> Iterator[Tensor]:
        for t in range(self.T):
            yield self[t]

    def __len__(self) -> int:
        return self.T

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(shape={tuple(self.shape)}, "
            f"spikes={self.num_spikes}, device={self.device})"
        )