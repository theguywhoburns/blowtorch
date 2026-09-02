import pytest

from pyrokinesis import get_validation, set_validation


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
    """A device for state-factory tests: the available accelerator, else cpu.

    Detects backends in order. ``torch.cuda`` is the backend on CUDA *and*
    ROCm/HIP builds, so the cuda branch covers both NVIDIA and AMD.
    """
    import torch

    for kind in ("cuda", "xpu", "mps", "npu", "mtia"):
        backend = getattr(torch, kind, None)
        if backend is not None and backend.is_available():
            return torch.device(kind)

    return torch.device("cpu")
