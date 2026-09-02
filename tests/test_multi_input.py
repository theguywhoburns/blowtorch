from __future__ import annotations

import pytest
import torch

from pyrokinesis import (
    PyroModule,
    InputSpec,
    no_validation,
)
from pyrokinesis import Tensor

B, F = 4, 8
G = 5


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------


class _TwoInput(PyroModule):
    """Basal + apical inputs; state follows the primary (basal) input."""

    class Inputs:
        x: Tensor
        inh: Tensor

    class Specs:
        out = PyroModule.OutputSpec()
        v = PyroModule.StateSpec(shape="x")

    def _step(self, x, inh, v):
        return torch.cat([x, inh], dim=-1), v + 1


class _TwoInputListLike(PyroModule):
    """Same dynamics; accepts list inputs (also exercises shape="input")."""

    class Inputs:
        x: Tensor
        inh: Tensor

    class Specs:
        out = PyroModule.OutputSpec()
        v = PyroModule.StateSpec(shape="input")

    def _step(self, x, inh, v):
        return torch.cat([x, inh], dim=-1), v + 1


class _ExplicitPrimary(PyroModule):
    """Marks the second input primary via the mixed declaration syntax."""

    class Inputs:
        x: Tensor
        inh: Tensor = PyroModule.Input(primary=True)

    class Specs:
        out = PyroModule.OutputSpec()
        v = PyroModule.StateSpec(shape="input")

    def _step(self, x, inh, v):
        return torch.cat([x, inh], dim=-1), v + 1


class _StateFromInh(PyroModule):
    """State shape follows the non-primary input by name."""

    class Inputs:
        x: Tensor
        inh: Tensor

    class Specs:
        out = PyroModule.OutputSpec()
        v = PyroModule.StateSpec(shape="inh")

    def _step(self, x, inh, v):
        return x.sum(dim=-1, keepdim=True) + v, v


class _FixedShapeState(PyroModule):
    """Explicit tuple state shape with two inputs."""

    class Inputs:
        x: Tensor
        inh: Tensor

    class Specs:
        out = PyroModule.OutputSpec()
        v = PyroModule.StateSpec(shape=(F,))

    def _step(self, x, inh, v):
        return x + inh, v + 1


class _BaseInputs(PyroModule):
    class Inputs:
        x: Tensor
        base_in: Tensor

    class Specs:
        out = PyroModule.OutputSpec()
        v = PyroModule.StateSpec()

    def _step(self, x, base_in, v):
        return x + base_in, v


class _ChildInputs(_BaseInputs):
    class Inputs:
        child_in: Tensor

    def _step(self, x, base_in, child_in, v):
        return x + base_in + child_in, v


class _OverrideInputs(_BaseInputs):
    class Inputs:
        base_in: Tensor = PyroModule.Input(primary=True)

    def _step(self, x, base_in, v):
        return x + base_in, v


class _DefaultState(PyroModule):
    class Inputs:
        x: Tensor
        inh: Tensor

    class Specs:
        out = PyroModule.OutputSpec()
        v = PyroModule.StateSpec(shape=None)

    def _step(self, x, inh, v):
        return torch.cat([x, inh], dim=-1), v + 1


class _UnknownShape(PyroModule):
    class Inputs:
        x: Tensor
        inh: Tensor

    class Specs:
        out = PyroModule.OutputSpec()
        v = PyroModule.StateSpec(shape="nope")

    def _step(self, x, inh, v):
        return x + inh, v


def _inputs():
    return torch.randn(B, F), torch.randn(B, G)


def _seqs():
    return torch.randn(T, B, F), torch.randn(T, B, G)


T = 5


# ----------------------------------------------------------------------
# 17.1 Default single-input behavior
# ----------------------------------------------------------------------


def test_default_implicit_input_metadata():
    class _Leaky(PyroModule):
        class Specs:
            out = PyroModule.OutputSpec()
            mem = PyroModule.StateSpec()

        def _step(self, x, mem):
            return x, mem

    assert _Leaky._pk_input_names == ("x",)
    assert _Leaky._pk_primary_input_index == 0
    assert _Leaky._pk_input_specs[0].primary

    m = _Leaky()
    assert "inputs" not in repr(m)


def test_single_input_public_api_unchanged():
    class _Leaky(PyroModule):
        class Specs:
            out = PyroModule.OutputSpec()
            mem = PyroModule.StateSpec()

        def _step(self, x, mem):
            return x, mem

    m = _Leaky()
    x = torch.randn(B, F)
    state = m.initial_state((B, F))
    assert m(x, *state)[0].shape == (B, F)
    assert m.step_state(x, state)[1][0].shape == (B, F)
    assert m.forward_sequence(torch.randn(T, B, F))[0].shape == (T, B, F)
    assert m.initial_state_like(x)[0].shape == (B, F)
    m.allocate_like(x)


# ----------------------------------------------------------------------
# 17.2 Explicit multi-input behavior
# ----------------------------------------------------------------------


def test_explicit_tuple_input_canonicalizes():
    m = _TwoInput()
    x, inh = _inputs()
    state = m.initial_state((B, F))

    out, next_state = m.step_state((x, inh), state)

    assert out.shape == (B, F + G)
    assert torch.allclose(out[..., :F], x)
    assert torch.allclose(out[..., F:], inh)
    assert torch.allclose(next_state[0], torch.full((B, F), 1.0))


def test_explicit_list_input_canonicalizes():
    m = _TwoInputListLike()
    x, inh = _inputs()
    state = m.initial_state((B, F))

    out, _ = m.step_state([x, inh], state)

    assert out.shape == (B, F + G)


def test_dict_input_ordered_by_declaration():
    m = _TwoInput()
    x, inh = _inputs()
    state = m.initial_state((B, F))

    out, _ = m.step_state({"inh": inh, "x": x}, state)

    assert torch.allclose(out[..., :F], x)
    assert torch.allclose(out[..., F:], inh)


def test_wrong_tuple_length_raises():
    m = _TwoInput()
    x, inh = _inputs()
    state = m.initial_state((B, F))

    with pytest.raises(ValueError, match="expects 2 inputs, got 1"):
        m.step_state((x,), state)

    with pytest.raises(ValueError, match="expects 2 inputs, got 3"):
        m.step_state((x, inh, x), state)


def test_single_tensor_to_multi_input_raises():
    m = _TwoInput()
    x, _ = _inputs()
    state = m.initial_state((B, F))

    with pytest.raises(ValueError, match="got a single tensor"):
        m.step_state(x, state)


def test_missing_dict_keys_raise():
    m = _TwoInput()
    x, _ = _inputs()
    state = m.initial_state((B, F))

    with pytest.raises(ValueError, match="missing keys \\['inh'\\]"):
        m.step_state({"x": x}, state)


def test_non_tensor_input_raises():
    m = _TwoInput()
    x, _ = _inputs()
    state = m.initial_state((B, F))

    with pytest.raises(TypeError, match="must be tensors"):
        m.step_state((x, "nope"), state)


def test_unexpected_input_type_raises():
    m = _TwoInput()
    state = m.initial_state((B, F))

    with pytest.raises(TypeError, match="Tensor, a tuple/list of tensors"):
        m.step_state(3.14, state)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# 17.3 Hidden multi-input behavior
# ----------------------------------------------------------------------


def test_hidden_alloc_uses_first_inputs():
    m = _TwoInput(init_hidden=True)
    x, inh = _inputs()

    out = m((x, inh))

    assert out.shape == (B, F + G)
    assert m._buffers["v"].shape == (B, F)


def test_hidden_primary_determines_dtype():
    m = _TwoInput(init_hidden=True)
    x = torch.randint(0, 3, (B, F), dtype=torch.int64)
    inh = torch.randn(B, G)

    m((x, inh))

    assert m._buffers["v"].dtype == torch.get_default_dtype()


def test_hidden_rejects_explicit_state():
    m = _TwoInput(init_hidden=True)
    x, inh = _inputs()

    with pytest.raises(ValueError, match="do not pass state explicitly"):
        m((x, inh), torch.randn(B, F))


def test_hidden_forward_sequence():
    m = _TwoInput(init_hidden=True)
    xs, is_ = _seqs()

    out = m.forward_sequence((xs, is_))

    assert out.shape == (T, B, F + G)
    assert torch.allclose(m._buffers["v"], torch.full((B, F), float(T)))


def test_allocate_like_tuple_and_expanded():
    for alloc in ((torch.randn(B, F), torch.randn(B, G)),):
        m = _TwoInput(init_hidden=True)
        m.allocate_like(alloc)
        assert m._pk_allocated

    m = _TwoInput(init_hidden=True)
    m.allocate_like(torch.randn(B, F), torch.randn(B, G))
    assert m._pk_allocated


def test_hidden_shape_mismatch_raises():
    m = _TwoInput(init_hidden=True)
    m((torch.randn(B, F), torch.randn(B, G)))

    with pytest.raises(ValueError, match="batch/feature dims must stay fixed"):
        m((torch.randn(7, F), torch.randn(B, G)))


# ----------------------------------------------------------------------
# 17.4 State shape resolution
# ----------------------------------------------------------------------


def test_shape_input_and_none_follow_primary():
    m = _TwoInputListLike()  # shape="input"
    n = _DefaultState()  # shape=None
    x, inh = _inputs()

    assert m.initial_state_like((x, inh))[0].shape == (B, F)
    assert n.initial_state_like((x, inh))[0].shape == (B, F)


def test_shape_named_input_follows_that_input():
    m = _StateFromInh()  # shape="inh"
    x, inh = _inputs()

    state = m.initial_state_like((x, inh))

    assert state[0].shape == (B, G)


def test_shape_explicit_tuple():
    m = _FixedShapeState()
    x, inh = _inputs()

    state = m.initial_state_like((x, inh))

    assert state[0].shape == (F,)


def test_shape_unknown_name_raises():
    m = _UnknownShape()
    x, inh = _inputs()

    with pytest.raises(ValueError, match="refers to an unknown input"):
        m.initial_state_like((x, inh))


# ----------------------------------------------------------------------
# 17.5 Primary input behavior
# ----------------------------------------------------------------------


def test_first_input_primary_by_default():
    assert _TwoInput._pk_primary_input_index == 0


def test_explicit_primary_works():
    m = _ExplicitPrimary()
    assert m._pk_primary_input_index == 1

    # shape="input" now follows the inh input.
    x, inh = _inputs()
    state = m.initial_state_like((x, inh))
    assert state[0].shape == (B, G)

    # Hidden allocation follows the primary input.
    h = _ExplicitPrimary(init_hidden=True)
    h((x, inh))
    assert h._buffers["v"].shape == (B, G)


def test_multiple_primary_inputs_raise():
    with pytest.raises(TypeError, match="multiple primary inputs"):

        class _Bad(PyroModule):
            class Inputs:
                x = PyroModule.Input(primary=True)
                inh = PyroModule.Input(primary=True)

            class Specs:
                out = PyroModule.OutputSpec()
                v = PyroModule.StateSpec()

            def _step(self, x, inh, v):
                return x + inh, v


def test_input_spec_dtype_stored():
    class _D(PyroModule):
        class Inputs:
            x: Tensor = PyroModule.Input(dtype=float)

        class Specs:
            out = PyroModule.OutputSpec()
            v = PyroModule.StateSpec()

        def _step(self, x, v):
            return x, v

    assert _D._pk_input_specs[0].dtype is float


def test_input_dtype_exact_match_enforced():
    class _D(PyroModule):
        class Inputs:
            x: Tensor = PyroModule.Input(dtype=torch.float32)

        class Specs:
            out = PyroModule.OutputSpec()
            v = PyroModule.StateSpec()

        def _step(self, x, v):
            return x, v

    state = _D().initial_state((B, F))
    m = _D()
    assert m(torch.randn(B, F, dtype=torch.float32), *state)[0].dtype == torch.float32

    n = _D()
    with pytest.raises(TypeError, match=r"declared with dtype=torch.float32"):
        n(torch.randn(B, F, dtype=torch.float64), *state)


def test_input_dtype_python_float_enforced():
    class _D(PyroModule):
        class Inputs:
            x: Tensor = PyroModule.Input(dtype=float)

        class Specs:
            out = PyroModule.OutputSpec()
            v = PyroModule.StateSpec()

        def _step(self, x, v):
            return x, v

    state = _D().initial_state((B, F))
    m = _D()
    assert m(torch.randn(B, F), *state)[0].dtype == torch.float32

    n = _D()
    with pytest.raises(TypeError, match="declared with dtype=<class 'float'>"):
        n(torch.randint(0, 3, (B, F)), *state)


def test_input_dtype_python_int_enforced():
    class _D(PyroModule):
        class Inputs:
            x: Tensor = PyroModule.Input(dtype=int)

        class Specs:
            out = PyroModule.OutputSpec()
            v = PyroModule.StateSpec()

        def _step(self, x, v):
            return x, v

    state = _D().initial_state((B, F))
    m = _D()
    assert m(torch.randint(0, 3, (B, F)), *state)[0].dtype == torch.int64

    n = _D()
    with pytest.raises(TypeError, match="dtype=int but got floating-point"):
        n(torch.randn(B, F), *state)


# ----------------------------------------------------------------------
# 17.6 Sequence scanning
# ----------------------------------------------------------------------


def test_multi_input_eager_scan_without_state():
    m = _TwoInput()
    xs, is_ = _seqs()

    out, *final = m.forward_sequence((xs, is_))

    assert out.shape == (T, B, F + G)
    assert len(final) == 1
    assert torch.allclose(final[0], torch.full((B, F), float(T)))


def test_multi_input_eager_scan_with_explicit_state():
    m = _TwoInput()
    xs, is_ = _seqs()
    state = m.initial_state((B, F))

    _out, *final = m.forward_sequence((xs, is_), state)

    assert torch.allclose(final[0], state[0] + float(T))


def test_multi_input_scan_dict_sequences():
    m = _TwoInput()
    xs, is_ = _seqs()

    out, *_ = m.forward_sequence({"inh": is_, "x": xs})

    assert torch.allclose(out[..., :F], xs)
    assert torch.allclose(out[..., F:], is_)


def test_mismatched_time_lengths_raise():
    m = _TwoInput()
    xs, is_ = _seqs()

    with pytest.raises(ValueError, match="same time length"):
        m.forward_sequence((xs, is_[:-1]))


def test_too_few_dims_sequence_raises():
    m = _TwoInput()

    with pytest.raises(ValueError, match="\\(time, batch, features\\)"):
        m.forward_sequence((torch.randn(B, F), torch.randn(B, G)))


def test_hidden_sequence_scan_updates_buffers():
    m = _TwoInput(init_hidden=True)
    xs, is_ = _seqs()

    m.forward_sequence((xs, is_))

    assert torch.allclose(m._buffers["v"], torch.full((B, F), float(T)))


# ----------------------------------------------------------------------
# 17.7 Compiled sequence scanning
# ----------------------------------------------------------------------


def test_compiled_multi_input_matches_eager():
    m = _TwoInput().compile_sequence_scan(mode="default")
    eager = _TwoInput()
    xs, is_ = _seqs()

    out = m.forward_sequence((xs, is_))
    ref = eager.forward_sequence((xs, is_))

    assert torch.allclose(out[0], ref[0])
    assert torch.allclose(out[1], ref[1])


def test_fast_sequence_multi_input_matches_eager():
    m = _TwoInput().fast_sequence_()
    eager = _TwoInput()
    xs, is_ = _seqs()

    out = m.forward_sequence((xs, is_))
    ref = eager.forward_sequence((xs, is_))

    assert torch.allclose(out[0], ref[0])
    assert torch.allclose(out[1], ref[1])


def test_compiled_hidden_multi_input():
    m = _TwoInput(init_hidden=True).compile_sequence_scan(mode="default")
    xs, is_ = _seqs()
    x, inh = xs[0], is_[0]

    m.allocate_like((x, inh))
    out = m.forward_sequence((xs, is_))

    assert out.shape == (T, B, F + G)
    assert torch.allclose(m._buffers["v"], torch.full((B, F), float(T)))


# ----------------------------------------------------------------------
# 17.8 Validation
# ----------------------------------------------------------------------


def test_wrong_input_count_raises_when_validate_on():
    m = _TwoInput()
    x, _inh = _inputs()
    state = m.initial_state((B, F))

    with pytest.raises(ValueError, match="expects 2 input tensors"):
        m._pk_forward_explicit((x,), state)


def test_wrong_state_count_raises():
    m = _TwoInput()
    x, inh = _inputs()

    with pytest.raises(ValueError, match="expects 1 state tensors"):
        m((x, inh), torch.randn(B, F), torch.randn(B, F))


def test_validation_disabled_skips_input_count_check():
    class _AnyArity(PyroModule):
        class Inputs:
            x: Tensor
            inh: Tensor

        class Specs:
            out = PyroModule.OutputSpec()
            v = PyroModule.StateSpec()

        def _step(self, x, inh, *extra):
            return x + inh, (extra[0] if extra else torch.zeros_like(x))

    x, _inh = _inputs()
    state = _AnyArity().initial_state((B, F))

    m = _AnyArity(validate=False)
    out, *_ = m._pk_forward_explicit((x,), state)
    assert out.shape == (B, F)

    n = _AnyArity(validate=True)
    with pytest.raises(ValueError, match="expects 2 input tensors"):
        n._pk_forward_explicit((x,), state)


def test_no_validation_context_disables():
    m = _TwoInput()
    x, inh = _inputs()

    with no_validation():
        out, *_ = m.forward((x, inh), torch.randn(B, F))
    assert out.shape == (B, F + G)


def test_fast_sequence_disables_validation():
    m = _TwoInput().fast_sequence_(compile_scan=False)
    x, inh = _inputs()

    out, *_ = m.forward((x, inh), torch.randn(B, F))
    assert out.shape == (B, F + G)


# ----------------------------------------------------------------------
# 17.9 Name collision tests
# ----------------------------------------------------------------------


def test_input_collides_with_param_raises():
    with pytest.raises(TypeError, match="collide with parameter"):

        class _Bad(PyroModule):
            class Params:
                inh = PyroModule.Param(0.5)

            class Inputs:
                x: Tensor
                inh: Tensor

            class Specs:
                out = PyroModule.OutputSpec()
                v = PyroModule.StateSpec()

            def _step(self, x, inh, v):
                return x + inh, v


def test_input_collides_with_constant_raises():
    with pytest.raises(TypeError, match="collide with constant"):

        class _Bad(PyroModule):
            dt = PyroModule.Constant(0.01, dtype=float)

            class Inputs:
                x: Tensor
                dt: Tensor

            class Specs:
                out = PyroModule.OutputSpec()
                v = PyroModule.StateSpec()

            def _step(self, x, dt, v):
                return x + dt, v


def test_input_collides_with_output_raises():
    with pytest.raises(TypeError, match="collide with output/state"):

        class _Bad(PyroModule):
            class Inputs:
                x: Tensor
                out: Tensor

            class Specs:
                out = PyroModule.OutputSpec()
                v = PyroModule.StateSpec()

            def _step(self, x, out, v):
                return x + out, v


def test_input_collides_with_state_raises():
    with pytest.raises(TypeError, match="collide with output/state"):

        class _Bad(PyroModule):
            class Inputs:
                x: Tensor
                v: Tensor

            class Specs:
                out = PyroModule.OutputSpec()
                v = PyroModule.StateSpec()

            def _step(self, x, v, s):
                return x + v, s


def _module_with_inputs(entries):
    inputs_cls = type("Inputs", (), dict(entries))
    specs_cls = type(
        "Specs",
        (),
        {
            "out": PyroModule.OutputSpec(),
            "v": PyroModule.StateSpec(),
        },
    )

    def _step(self, *args):
        return args[0], args[-1]

    return type(
        "_Dynamic",
        (PyroModule,),
        {"Inputs": inputs_cls, "Specs": specs_cls, "_step": _step},
    )


def test_keyword_input_name_raises():
    with pytest.raises(TypeError, match="non-keyword Python identifier"):
        _module_with_inputs({"class": InputSpec()})


def test_non_identifier_input_name_raises():
    with pytest.raises(TypeError, match="non-keyword Python identifier"):
        _module_with_inputs({"bad-name": InputSpec()})


# ----------------------------------------------------------------------
# 17.10 Inheritance tests
# ----------------------------------------------------------------------


def test_subclass_inherits_and_extends_inputs():
    c = _ChildInputs()

    assert c._pk_input_names == ("x", "base_in", "child_in")
    assert c._pk_primary_input_index == 0


def test_subclass_override_primary():
    c = _OverrideInputs()

    assert c._pk_input_names == ("x", "base_in")
    assert c._pk_primary_input_index == 1


def test_inherited_forward_and_scan():
    c = _ChildInputs()
    x = torch.randn(B, F)
    base_in = torch.randn(B, F)
    child_in = torch.randn(B, F)
    state = c.initial_state((B, F))

    out, *_ = c.forward((x, base_in, child_in), *state)
    assert out.shape == (B, F)

    xs = torch.randn(T, B, F)
    base_seq = torch.randn(T, B, F)
    child_seq = torch.randn(T, B, F)
    out_seq, *_ = c.forward_sequence((xs, base_seq, child_seq))
    assert out_seq.shape == (T, B, F)