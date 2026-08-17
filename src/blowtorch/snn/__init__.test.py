import blowtorch.snn
from blowtorch.snn.neurons.AdEx import AdEx
from blowtorch.snn.neurons.HH import HH
from blowtorch.snn.neurons.LIF import LIF


def test_snn_public_exports_match_all():
    expected = [
        "SnnModule",
        "Reset",
        "ResetSpec",
        "subtract_reset",
        "zero_reset",
        "hard_zero_reset",
        "no_reset",
        "default_spike_grad",
        "AdEx",
        "HH",
        "LIF",
    ]
    assert blowtorch.snn.__all__ == expected

    for name in expected:
        assert callable(getattr(blowtorch.snn, name))

    assert blowtorch.snn.LIF is LIF
    assert blowtorch.snn.AdEx is AdEx
    assert blowtorch.snn.HH is HH


def test_snn_star_import_exposes_lif():
    ns = {}
    exec("from blowtorch.snn import *", ns)

    assert ns["LIF"] is LIF
    assert ns["AdEx"] is AdEx
    assert ns["HH"] is HH
    assert ns["SnnModule"] is blowtorch.snn.SnnModule