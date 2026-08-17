import blowtorch.snn
from blowtorch.snn.neurons.AdEx import AdEx
from blowtorch.snn.neurons.ALIF import ALIF
from blowtorch.snn.neurons.HH import HH
from blowtorch.snn.neurons.Izhikevich import Izhikevich
from blowtorch.snn.neurons.LIF import LIF
from blowtorch.snn.neurons.SRM0 import SRM0
from blowtorch.snn.neurons.TwoCompartment import TwoCompartment


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
        "straight_through_surrogate",
        "sigmoid_surrogate",
        "atan_surrogate",
        "triangular_surrogate",
        "fast_sigmoid_surrogate",
        "AdEx",
        "ALIF",
        "HH",
        "Izhikevich",
        "LIF",
        "SRM0",
        "TwoCompartment",
    ]
    assert blowtorch.snn.__all__ == expected

    for name in expected:
        assert callable(getattr(blowtorch.snn, name))

    assert blowtorch.snn.LIF is LIF
    assert blowtorch.snn.AdEx is AdEx
    assert blowtorch.snn.HH is HH
    assert blowtorch.snn.ALIF is ALIF
    assert blowtorch.snn.Izhikevich is Izhikevich
    assert blowtorch.snn.SRM0 is SRM0
    assert blowtorch.snn.TwoCompartment is TwoCompartment


def test_snn_star_import_exposes_lif():
    ns = {}
    exec("from blowtorch.snn import *", ns)

    assert ns["LIF"] is LIF
    assert ns["AdEx"] is AdEx
    assert ns["ALIF"] is ALIF
    assert ns["HH"] is HH
    assert ns["Izhikevich"] is Izhikevich
    assert ns["SRM0"] is SRM0
    assert ns["TwoCompartment"] is TwoCompartment
    assert ns["SnnModule"] is blowtorch.snn.SnnModule