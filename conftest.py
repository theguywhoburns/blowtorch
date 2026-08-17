import pytest

import blowtorch.base
import blowtorch.snn.base
from blowtorch.base import get_validation, set_validation
from blowtorch.snn.neurons import LIF

# pytest's `--import-mode=importlib` treats `X.test.py` as the submodule
# `X.test` and, because the real module `X` lacks `__path__`, re-imports the
# parent from the wrong file (`src/.../__init__.py`), poisoning sys.modules
# (e.g. `blowtorch.base` becomes the package init). Giving each real module an
# empty `__path__` marks it as a package so pytest keeps the already-imported
# module as the parent.
for _module in (blowtorch.base, blowtorch.snn.base, LIF):
    _module.__path__ = []


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