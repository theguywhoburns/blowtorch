import crematorium


def test_public_exports_match_all():
    expected = [
        "crematoriumModule",
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
    assert crematorium.__all__ == expected

    for name in expected:
        assert callable(getattr(crematorium, name))

    for name in ("set_validation", "get_validation", "no_validation"):
        assert callable(getattr(crematorium, name))


def test_star_import_exposes_core_names():
    ns = {}
    exec("from crematorium import *", ns)

    for name in (
        "crematoriumModule",
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