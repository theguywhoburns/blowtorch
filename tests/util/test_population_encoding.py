from __future__ import annotations

import pytest
import torch

from blowtorch.util.population_encoding import population_encode


def test_population_encode_shape_and_dtype():
    tau = torch.tensor([[0.1, 0.5, 0.9]])
    out = population_encode(tau, M=32, T=8, seed=0)
    assert out.shape == (8, 1, 3, 32)
    assert out.dtype == torch.int64
    assert (out >= 0).all()


def test_population_encode_mean_matches_receptive_field():
    # tau: (B=1, N=5); average over T steps approximates the firing rate r.
    tau = torch.linspace(0.0, 1.0, 5).view(1, 5)
    M, T = 8, 2000
    out = population_encode(tau, M=M, T=T, sigma=0.15, seed=0)

    mean = out.float().mean(dim=0)  # (1, 5, M) mean per step
    mu = torch.linspace(0.0, 1.0, M)
    rates = torch.exp(-(tau[..., None] - mu) ** 2 / (2 * 0.15**2))
    assert torch.allclose(mean, rates, atol=0.1)


def test_population_encode_mean_count_approx_rate_times_T():
    # Center neuron (rate 1.0) fires Poisson(r*dt) per step; over T steps the
    # mean count is r*dt*T.
    tau = torch.full((32, 1), 0.5)
    T = 100
    out = population_encode(tau, M=3, T=T, sigma=0.15, seed=0)  # centers 0, .5, 1

    center = out[:, :, 0, 1].float()  # the neuron centered at 0.5
    assert abs(center.mean().item() - 1.0) < 0.05
    assert abs(center.sum(dim=0).mean().item() - T) < 3.0


def test_population_encode_dt_scales_mean_count():
    tau = torch.full((64, 1), 0.5)
    dt1 = population_encode(tau, M=3, T=8, dt=1.0, seed=0).float()
    dt2 = population_encode(tau, M=3, T=8, dt=2.0, seed=0).float()

    assert dt2.mean() > 1.5 * dt1.mean()


def test_population_encode_seed_deterministic():
    tau = torch.rand(4, 3)
    a = population_encode(tau, seed=42)
    b = population_encode(tau, seed=42)
    c = population_encode(tau, seed=43)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_population_encode_centers_span_unit_interval():
    # M=3 gives centers {0, 0.5, 1}; tau at an endpoint fires the matching
    # end neuron (peak rate phi = 1.0) and (almost) nothing elsewhere.
    M = 3
    T = 2000
    tau0 = torch.zeros(1, 1)
    tau1 = torch.ones(1, 1)

    out0 = population_encode(tau0, M=M, T=T, seed=0).float().mean(dim=0).flatten()
    out1 = population_encode(tau1, M=M, T=T, seed=0).float().mean(dim=0).flatten()

    assert out0.argmax().item() == 0
    assert out1.argmax().item() == M - 1
    assert out0[0].item() == pytest.approx(1.0, abs=0.05)
    assert out1[-1].item() == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"M": 0},
        {"T": 0},
        {"sigma": 0.0},
        {"sigma": -1.0},
        {"phi": -1.0},
        {"dt": 0.0},
    ],
)
def test_population_encode_validation(kwargs):
    tau = torch.tensor([[0.5]])
    with pytest.raises(ValueError):
        population_encode(tau, **kwargs)


def test_population_encode_bad_rank():
    with pytest.raises(ValueError, match=r"\(B, N\)"):
        population_encode(torch.tensor([0.5, 0.5]))