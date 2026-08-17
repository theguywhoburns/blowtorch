import pytest

from blowtorch.base import get_validation, set_validation


@pytest.fixture(autouse=True)
def _restore_global_validation():
    """Snapshot the global validation flag and restore it after each test.

    `set_validation(False)` in one test would otherwise poison every later
    test: `_GLOBAL_VALIDATE` is module state that leaks across tests.
    """
    snapshot = get_validation()
    yield
    set_validation(snapshot)


@pytest.fixture()
def device():
    """A device for state-factory tests: cuda when available, else cpu."""
    import torch

    return torch.device("cuda") if torch.cuda.is_available() else "cpu"
