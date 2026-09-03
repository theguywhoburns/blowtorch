"""Tests for init-signature generation, hook-chain semantics, and Sequential signatures."""

from __future__ import annotations

import inspect
from typing import Callable, ClassVar, Optional

import pytest

from pyrokinesis import StepOutput, Tensor
from pyrokinesis.nn import Sequential
from pyrokinesis.snn import LIF, SnnModule


# ---------------------------------------------------------------------------
# _pk_extra_init_params aggregation dedups by name
# ---------------------------------------------------------------------------


class TestExtraInitParamsDedup:
    def test_subclass_redeclaring_spike_grad_does_not_crash(self):
        class MyNeuron(SnnModule):
            class Specs:
                out = SnnModule.OutputSpec()
                mem = SnnModule.StateSpec()

            @classmethod
            def _pk_extra_init_params(cls) -> list[inspect.Parameter]:
                # Mirrors SnnModule's own entry — base-first dedup keeps the
                # base one and skips this duplicate.
                return [
                    inspect.Parameter(
                        "spike_grad",
                        inspect.Parameter.KEYWORD_ONLY,
                        default=None,
                        annotation=Optional[Callable[[Tensor], Tensor]],
                    ),
                ]

            def _step(self, x: Tensor, mem: Tensor) -> StepOutput:
                return (x, mem)

        MyNeuron()

        sig = inspect.signature(MyNeuron)
        param_names = [p.name for p in sig.parameters.values()]
        assert param_names.count("spike_grad") == 1

    @pytest.mark.parametrize("cls", [SnnModule, LIF])
    def test_spike_grad_appears_once(self, cls):
        sig = inspect.signature(cls)
        param_names = [p.name for p in sig.parameters.values()]
        assert param_names.count("spike_grad") == 1


# ---------------------------------------------------------------------------
# Hook chain replace semantics
# ---------------------------------------------------------------------------


class TestHookReplaceSemantics:
    def test_lif_has_reset_mixin_hook(self):
        hook_names = [fn.__qualname__ for fn in LIF._pk_hook_post_steps]
        assert any("ResetMixin._pk_hook_post__rst" in h for h in hook_names), (
            f"Expected ResetMixin._pk_hook_post__rst in chain, got: {hook_names}"
        )

    def test_subclass_override_replaces_parent_hook(self):
        class ChildLIF(LIF):
            _hook_fired: ClassVar[list[str]] = []

            def _pk_hook_post__rst(self, out: StepOutput) -> StepOutput:
                ChildLIF._hook_fired.append("child")
                return out

        hook_names = [fn.__qualname__ for fn in ChildLIF._pk_hook_post_steps]
        assert any("ChildLIF._pk_hook_post__rst" in h for h in hook_names), (
            f"Expected ChildLIF hook, got: {hook_names}"
        )
        assert not any("ResetMixin._pk_hook_post__rst" in h for h in hook_names), (
            f"Parent ResetMixin hook should NOT be in chain: {hook_names}"
        )

    def test_subclass_adding_new_hook_tag(self):
        class ExtendedLIF(LIF):
            def _pk_hook_post__custom(self, out: StepOutput) -> StepOutput:
                return out

        hook_names = [fn.__qualname__ for fn in ExtendedLIF._pk_hook_post_steps]
        assert any("ResetMixin._pk_hook_post__rst" in h for h in hook_names), (
            f"Expected base _rst hook, got: {hook_names}"
        )
        assert any("ExtendedLIF._pk_hook_post__custom" in h for h in hook_names), (
            f"Expected new _custom hook, got: {hook_names}"
        )


# ---------------------------------------------------------------------------
# Sequential signatures
# ---------------------------------------------------------------------------


def _assert_layers_signature(cls):
    sig = inspect.signature(cls)
    params = list(sig.parameters.values())
    assert params[0].name == "layers"
    assert params[0].kind is inspect.Parameter.VAR_POSITIONAL
    kw_names = [p.name for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY]
    assert "init_hidden" in kw_names
    assert "validate" in kw_names


class TestSequentialSignature:
    def test_sequential_base_signature(self):
        _assert_layers_signature(Sequential)

    def test_subclass_preserves_signature(self):
        class MySeq(Sequential):
            pass

        _assert_layers_signature(MySeq)

    def test_subclass_custom_init_keeps_own_signature(self):
        class MyNet(Sequential):
            def __init__(self, foo=1, *layers, **kw):
                self.foo = foo
                super().__init__(*layers, **kw)

        sig = inspect.signature(MyNet)
        assert "foo" in sig.parameters
        seq = MyNet(2, LIF())
        assert seq.foo == 2
        assert len(seq._layers) == 1

    def test_sequential_construction_works(self):
        seq = Sequential(LIF(), LIF())
        assert len(seq._layers) == 2

    def test_subclass_construction_works(self):
        class MySeq(Sequential):
            pass

        seq = MySeq(LIF())
        assert len(seq._layers) == 1
