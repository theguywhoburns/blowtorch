from __future__ import annotations

from typing import Optional

import torch

from blowtorch.base import Tensor

__all__ = ["population_encode"]


def population_encode(
    tau: Tensor,  # (B, N) fractions in [0, 1]
    M: int = 64,
    T: int = 8,
    sigma: float = 0.15,
    phi: float = 1.0,
    dt: float = 1.0,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Encode scalar quantile fractions into population spike trains (paper
    sec. 3.2, eq. 13-17).

    ``M`` neurons with Gaussian receptive fields tile the fraction axis
    ``[0, 1]``: neuron ``j`` has center ``mu_j = (j - 1)/(M - 1)`` (evenly
    spaced, endpoints included) and firing rate

        r_j = phi * exp(-(tau - mu_j)**2 / (2 * sigma**2))

    (the paper writes ``mu_j = j/N``, which is a subscript typo: with
    ``M`` neurons over fractions in ``[0, 1]`` the centers must tile the unit
    interval, not depend on the quantile count ``N``). Each neuron fires as a
    Poisson process: per time step, the spike count is drawn as
    ``Poisson(r_j * dt)``, matching eq. (16) with ``Delta t = dt``.

    Non-differentiable sampling: use this as an input encoder, not as a
    differentiable layer.

    Args:
        tau: quantile fractions in ``[0, 1]``, shape ``(B, N)``.
        M: number of encoding neurons.
        T: number of time steps.
        sigma: receptive-field width ``C``.
        phi: peak firing rate.
        dt: Poisson time-window per step; expected per-step count is
            ``r_j * dt``, so the expected total count over ``T`` steps is
            ``r_j * dt * T``.
        seed: optional RNG seed (uses a local generator; does not disturb the
            global RNG state).

    Returns:
        ``(T, B, N, M)`` integer spike counts.
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
    return counts.long()