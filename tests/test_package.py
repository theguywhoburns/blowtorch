import crematorium


def test_public_exports_match_all():
    expected = [
        "CrModule",
        "Tensor",
        "StepOutput",
        "Param",
        "ParamSpec",
        "Constant",
        "ConstantSpec",
        "Input",
        "InputSpec",
        "OutputSpec",
        "StateSpec",
        "extend_specs",
        "identity",
        "clamp_unit_interval",
        "clamp_positive",
        "set_sequence_scan_chunk",
        "set_validation",
        "get_validation",
        "no_validation",
    ]
    assert sorted(crematorium.__all__) == sorted(expected)

    for name in expected:
        assert callable(getattr(crematorium, name))

    for name in ("set_validation", "get_validation", "no_validation"):
        assert callable(getattr(crematorium, name))


def test_star_import_exposes_core_names():
    ns = {}
    exec("from crematorium import *", ns)

    for name in (
        "CrModule",
        "Tensor",
        "StepOutput",
        "Param",
        "ParamSpec",
        "Constant",
        "ConstantSpec",
        "Input",
        "InputSpec",
        "OutputSpec",
        "StateSpec",
        "extend_specs",
        "identity",
        "clamp_unit_interval",
        "clamp_positive",
        "set_sequence_scan_chunk",
        "set_validation",
        "get_validation",
        "no_validation",
    ):
        assert name in ns
