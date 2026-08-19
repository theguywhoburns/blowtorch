from __future__ import annotations

import pytest
import torch

from blowtorch.util.spike_train import SpikeTrain


def test_from_dense_roundtrip_counts():
    # Poisson-style multi-spike counts must survive the pack/unpack round trip.
    dense = torch.zeros(6, 3, 2, 2, dtype=torch.int64)
    dense[0, 0, 0, 0] = 3
    dense[2, 1, 1, 0] = 1
    dense[5, 2, 0, 1] = 2

    train = SpikeTrain.from_dense(dense)

    assert train.shape == (6, 3, 2, 2)
    assert train.T == 6
    assert train.batch == 3
    assert train.spatial_shape == (2, 2)
    assert train.num_spikes == 6
    assert torch.equal(train.to_dense(), dense)
    assert torch.equal(train.tensor, dense)


def test_from_dense_binary_and_bool():
    dense = torch.zeros(4, 2, 3, dtype=torch.int64)
    dense[0, 1, 2] = 1
    dense[3, 0, 0] = 1
    train = SpikeTrain.from_dense(dense)
    assert torch.equal(train.to_dense(), dense)

    bool_dense = dense.bool()
    bool_train = SpikeTrain.from_dense(bool_dense)
    assert torch.equal(bool_train.to_dense(), dense)


def test_time_pointer_slices_events_per_step():
    dense = torch.zeros(4, 2, 3, dtype=torch.int64)
    dense[0, 0, 1] = 1
    dense[1, 1, 0] = 1
    dense[1, 1, 2] = 1
    dense[3, 0, 0] = 1

    train = SpikeTrain.from_dense(dense)
    ptr = train.time_pointer

    for t in range(4):
        events = train.spk_ind[ptr[t] : ptr[t + 1]]
        expected = torch.nonzero(dense[t], as_tuple=True)[0].numel()
        assert events.numel() == expected

    # events sorted by time: all indices of step t sit before step t+1
    assert torch.equal(ptr, torch.tensor([0, 1, 3, 3, 4]))


def test_empty_spike_train():
    dense = torch.zeros(5, 2, 4, dtype=torch.int64)
    train = SpikeTrain.from_dense(dense)
    assert train.num_spikes == 0
    assert torch.equal(train.to_dense(), dense)
    assert torch.equal(train.time_pointer, torch.zeros(6, dtype=torch.int64))


def test_iteration_and_indexing():
    dense = torch.zeros(3, 2, 4, dtype=torch.int64)
    dense[0, 1, 2] = 2
    dense[2, 0, 0] = 1
    train = SpikeTrain.from_dense(dense)

    for t, step in enumerate(train):
        assert torch.equal(step, dense[t])

    assert torch.equal(train[0], dense[0])
    assert torch.equal(train[-1], dense[2])

    with pytest.raises(IndexError):
        train[3]


def test_packed_and_unpacked_agree():
    tau = torch.rand(4, 5)
    train = SpikeTrain.population(tau, M=16, T=8, seed=0)

    spk_ind, time_pointer = train.packed
    assert spk_ind.shape[0] == time_pointer[-1]

    # pack/unpack is lossless: rebuilding from the dense view gives the same events
    rebuilt = SpikeTrain.from_dense(train.to_dense())
    assert torch.equal(rebuilt.spk_ind, spk_ind)
    assert torch.equal(rebuilt.time_pointer, time_pointer)


def test_population_shape_and_dtype():
    tau = torch.tensor([[0.1, 0.5, 0.9]])
    train = SpikeTrain.population(tau, M=32, T=8, seed=0)
    dense = train.to_dense()
    assert train.shape == (8, 1, 3, 32)
    assert dense.dtype == torch.int64
    assert (dense >= 0).all()


def test_population_mean_matches_receptive_field():
    tau = torch.linspace(0.0, 1.0, 5).view(1, 5)
    train = SpikeTrain.population(tau, M=8, T=2000, sigma=0.15, seed=0)
    mean = train.to_dense().float().mean(dim=0)
    mu = torch.linspace(0.0, 1.0, 8)
    rates = torch.exp(-(tau[..., None] - mu) ** 2 / (2 * 0.15**2))
    assert torch.allclose(mean, rates, atol=0.1)


def test_population_mean_count_approx_rate_times_T():
    tau = torch.full((32, 1), 0.5)
    train = SpikeTrain.population(tau, M=3, T=100, sigma=0.15, seed=0)
    center = train.to_dense()[:, :, 0, 1].float()
    assert abs(center.mean().item() - 1.0) < 0.05
    assert abs(center.sum(dim=0).mean().item() - 100) < 3.0


def test_population_dt_scales_mean_count():
    tau = torch.full((64, 1), 0.5)
    d1 = SpikeTrain.population(tau, M=3, T=8, dt=1.0, seed=0).to_dense().float()
    d2 = SpikeTrain.population(tau, M=3, T=8, dt=2.0, seed=0).to_dense().float()
    assert d2.mean() > 1.5 * d1.mean()


def test_population_seed_deterministic():
    tau = torch.rand(4, 3)
    a = SpikeTrain.population(tau, seed=42)
    b = SpikeTrain.population(tau, seed=42)
    c = SpikeTrain.population(tau, seed=43)
    assert torch.equal(a.spk_ind, b.spk_ind)
    assert not torch.equal(a.spk_ind, c.spk_ind)


def test_population_centers_span_unit_interval():
    T = 2000
    tau0 = torch.zeros(1, 1)
    tau1 = torch.ones(1, 1)
    out0 = (
        SpikeTrain.population(tau0, M=3, T=T, seed=0).to_dense().float().mean(dim=0).flatten()
    )
    out1 = (
        SpikeTrain.population(tau1, M=3, T=T, seed=0).to_dense().float().mean(dim=0).flatten()
    )
    assert out0.argmax().item() == 0
    assert out1.argmax().item() == 2
    assert out0[0].item() == pytest.approx(1.0, abs=0.05)
    assert out1[-1].item() == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize(
    "kwargs",
    [{"M": 0}, {"T": 0}, {"sigma": 0.0}, {"sigma": -1.0}, {"phi": -1.0}, {"dt": 0.0}],
)
def test_population_validation(kwargs):
    with pytest.raises(ValueError):
        SpikeTrain.population(torch.tensor([[0.5]]), **kwargs)


def test_population_bad_rank():
    with pytest.raises(ValueError, match=r"\(B, N\)"):
        SpikeTrain.population(torch.tensor([0.5, 0.5]))


def test_poisson_mean_approx_rate_dt():
    rate = torch.full((32, 1), 0.25)
    train = SpikeTrain.poisson(rate, T=200, dt=1.0, seed=0)
    dense = train.to_dense()
    assert dense.shape == (200, 32, 1)
    assert (dense >= 0).all()
    mean = dense.float().mean().item()
    assert mean == pytest.approx(0.25, abs=0.05)


def test_poisson_dt_scales():
    r1 = SpikeTrain.poisson(torch.full((16, 1), 0.5), T=100, dt=1.0, seed=0)
    r2 = SpikeTrain.poisson(torch.full((16, 1), 0.5), T=100, dt=2.0, seed=0)
    assert r2.num_spikes > 1.5 * r1.num_spikes


def test_poisson_seed_deterministic():
    rate = torch.rand(4, 3)
    a = SpikeTrain.poisson(rate, T=16, seed=42)
    b = SpikeTrain.poisson(rate, T=16, seed=42)
    c = SpikeTrain.poisson(rate, T=16, seed=43)
    assert torch.equal(a.spk_ind, b.spk_ind)
    assert not torch.equal(a.spk_ind, c.spk_ind)


def test_latency_encoding():
    value = torch.tensor([[1.0, 0.0, 0.5]])
    T = 8
    train = SpikeTrain.latency(value, T)
    dense = train.to_dense()  # (8, 1, 3)

    # v=1 fires at t=0
    assert dense[0, 0, 0] == 1
    assert dense[:, 0, 0].sum() == 1

    # v=0 never fires
    assert dense[:, 0, 1].sum() == 0

    # v=0.5 fires once at t=round(0.5*7)=4
    assert dense[4, 0, 2] == 1
    assert dense[:, 0, 2].sum() == 1


def test_latency_monotonic():
    T = 32
    values = torch.linspace(1.0, 0.1, 8).view(1, 8)  # descending
    train = SpikeTrain.latency(values, T)
    dense = train.to_dense()
    first_spike = torch.argmax(dense, dim=0).squeeze(0)

    # all units fire exactly once, later units (lower value) fire later
    assert torch.all(dense.sum(dim=0) == 1)
    assert torch.all(first_spike.diff() >= 0)
    assert first_spike[0] == 0  # value=1 fires immediately


def test_custom_packs_any_generator():
    def delta_encoder(values: torch.Tensor, T: int) -> torch.Tensor:
        # fire exactly once, at the last timestep, for every unit
        dense = torch.zeros(T, *values.shape, dtype=torch.int64)
        dense[-1] = 1
        return dense

    values = torch.rand(2, 3)
    train = SpikeTrain.custom(delta_encoder, values, T=6)

    assert train.shape == (6, 2, 3)
    assert train.num_spikes == 2 * 3
    assert torch.equal(train[5], torch.ones(2, 3, dtype=torch.int64))
    assert torch.equal(train[0], torch.zeros(2, 3, dtype=torch.int64))


def test_custom_rejects_non_tensor():
    with pytest.raises(TypeError, match="must return a Tensor"):
        SpikeTrain.custom(lambda: 42)


def test_to_device_moves_packed_tensors():
    if not torch.cuda.is_available():
        pytest.skip("cuda not available")

    dense = torch.zeros(4, 2, 3, dtype=torch.int64)
    dense[0, 0, 1] = 1
    train = SpikeTrain.from_dense(dense)

    moved = train.to("cuda")
    assert moved.device.type == "cuda"
    assert moved.spk_ind.device.type == "cuda"
    assert moved.time_pointer.device.type == "cuda"
    assert torch.equal(moved.to_dense().cpu(), dense)

    back = moved.to("cpu")
    assert torch.equal(back.to_dense(), dense)


def test_from_dense_validation():
    with pytest.raises(ValueError, match=r"\(T, B"):
        SpikeTrain.from_dense(torch.zeros(4))


def test_poisson_latency_validation():
    with pytest.raises(ValueError):
        SpikeTrain.poisson(torch.rand(2, 3), T=0)
    with pytest.raises(ValueError):
        SpikeTrain.latency(torch.rand(2, 3), T=0)


def test_repr_and_len():
    train = SpikeTrain.latency(torch.tensor([[0.5, 0.5]]), T=4)
    assert len(train) == 4
    assert "SpikeTrain" in repr(train)
    assert "spikes=" in repr(train)