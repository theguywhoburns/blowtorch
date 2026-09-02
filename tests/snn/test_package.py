import pyrokinesis.snn
from pyrokinesis.snn.neurons.AdEx import AdEx
from pyrokinesis.snn.neurons.ALIF import ALIF
from pyrokinesis.snn.neurons.HH import HH
from pyrokinesis.snn.neurons.Izhikevich import Izhikevich
from pyrokinesis.snn.neurons.LIF import LIF
from pyrokinesis.snn.neurons.MCN import MCN
from pyrokinesis.snn.neurons.SRM0 import SRM0
from pyrokinesis.snn.neurons.TwoCompartment import TwoCompartment


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
    assert sorted(pyrokinesis.snn.__all__) == sorted(expected)

    for name in expected:
        assert callable(getattr(pyrokinesis.snn, name))

    assert pyrokinesis.snn.LIF is LIF
    assert pyrokinesis.snn.AdEx is AdEx
    assert pyrokinesis.snn.HH is HH
    assert pyrokinesis.snn.ALIF is ALIF
    assert pyrokinesis.snn.Izhikevich is Izhikevich
    assert pyrokinesis.snn.SRM0 is SRM0
    assert pyrokinesis.snn.MCN is MCN
    assert pyrokinesis.snn.TwoCompartment is TwoCompartment


def test_snn_star_import_exposes_lif():
    ns = {}
    exec("from pyrokinesis.snn import *", ns)

    assert ns["LIF"] is LIF
    assert ns["AdEx"] is AdEx
    assert ns["ALIF"] is ALIF
    assert ns["HH"] is HH
    assert ns["Izhikevich"] is Izhikevich
    assert ns["SRM0"] is SRM0
    assert ns["MCN"] is MCN
    assert ns["TwoCompartment"] is TwoCompartment
    assert ns["SnnModule"] is pyrokinesis.snn.SnnModule


def test_reset_module_is_exportable():
    from pyrokinesis.snn import reset as reset_module

    assert reset_module.Reset is pyrokinesis.snn.Reset
    assert reset_module.ResetSpec is pyrokinesis.snn.ResetSpec
    assert callable(reset_module.ResetHandler.apply)