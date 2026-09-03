import crematorium.snn
from crematorium.snn.neurons.AdEx import AdEx
from crematorium.snn.neurons.ALIF import ALIF
from crematorium.snn.neurons.HH import HH
from crematorium.snn.neurons.Izhikevich import Izhikevich
from crematorium.snn.neurons.LIF import LIF
from crematorium.snn.neurons.MCN import MCN
from crematorium.snn.neurons.SRM0 import SRM0
from crematorium.snn.neurons.TwoCompartment import TwoCompartment


def test_snn_public_exports_match_all():
    expected = [
        "SnnModule",
        "Reset",
        "ResetSpec",
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
        "MCN",
        "SRM0",
        "TwoCompartment",
    ]
    assert sorted(crematorium.snn.__all__) == sorted(expected)

    for name in expected:
        assert callable(getattr(crematorium.snn, name))

    assert crematorium.snn.LIF is LIF
    assert crematorium.snn.AdEx is AdEx
    assert crematorium.snn.HH is HH
    assert crematorium.snn.ALIF is ALIF
    assert crematorium.snn.Izhikevich is Izhikevich
    assert crematorium.snn.SRM0 is SRM0
    assert crematorium.snn.MCN is MCN
    assert crematorium.snn.TwoCompartment is TwoCompartment


def test_snn_star_import_exposes_lif():
    ns = {}
    exec("from crematorium.snn import *", ns)

    assert ns["LIF"] is LIF
    assert ns["AdEx"] is AdEx
    assert ns["ALIF"] is ALIF
    assert ns["HH"] is HH
    assert ns["Izhikevich"] is Izhikevich
    assert ns["SRM0"] is SRM0
    assert ns["MCN"] is MCN
    assert ns["TwoCompartment"] is TwoCompartment
    assert ns["SnnModule"] is crematorium.snn.SnnModule


def test_reset_module_is_exportable():
    from crematorium.snn import reset as reset_module

    assert reset_module.Reset is crematorium.snn.Reset
    assert reset_module.ResetSpec is crematorium.snn.ResetSpec
    assert callable(reset_module.ResetHandler.apply)
