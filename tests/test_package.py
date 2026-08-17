import blowtorch


def test_public_exports_match_all():
    expected = [
        "BlowtorchModule",
        "Param",
        "ParamSpec",
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
    assert blowtorch.__all__ == expected

    for name in expected:
        assert callable(getattr(blowtorch, name))

    for name in ("set_validation", "get_validation", "no_validation"):
        assert callable(getattr(blowtorch, name))


def test_star_import_exposes_core_names():
    ns = {}
    exec("from blowtorch import *", ns)

    for name in (
        "BlowtorchModule",
        "Param",
        "ParamSpec",
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