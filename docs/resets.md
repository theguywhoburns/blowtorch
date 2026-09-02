# Resets

Resets are a strictly **SNN** feature: only `SnnModule` subclasses may declare
them. The base `PyroModule` is a pure state-threading engine and makes no
assumptions about spikes or resets; `SnnModule` overrides the generic
`_post_step` hook to apply resets to the pre-reset state returned by `_step`.

Resets are declared per-state in `Specs` via the `Reset` factory. They run
**after** `_step` in both execution modes, using the step's spike output:

```python
class MyLIF(SnnModule):
    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        mem = SnnModule.StateSpec(reset=Reset.subtract("threshold"))
```

A reset's target can be the **name of a `Params` entry** (validated against
the module at init) or a `ParamSpec` object directly (e.g. a module-level
sentinel). Targets are resolved through `constrained()`, so the constrained
value (not the raw parameter) is used.

## Kinds

| factory               | effect                                                      |
| --------------------- | ----------------------------------------------------------- |
| `Reset.none()`        | nothing (default)                                           |
| `Reset.subtract("p")` | `state = state - spk * p` (LIF soft reset)                  |
| `Reset.zero()`        | `state = state * (1 - spk)` (multiplicative hard reset)     |
| `Reset.hard_zero()`   | `state = state.masked_fill(spk > 0, 0)` (masked hard reset) |
| `Reset.set("p")`      | `state = (1 - spk) * state + spk * p` (reset to a value)    |
| `Reset.add("p")`      | `state = state + spk * p` (inject, e.g. AdEx adaptation)    |
| `Reset.custom(fn)`    | `state = self.fn(state, spk)` (per-spike method)            |

## Custom resets

`Reset.custom` takes the **method name as a string** or the **bound method**
itself:

```python
class WithHomeostasis(SnnModule):
    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        mem = SnnModule.StateSpec(reset=Reset.custom("_homeostatic_decay"))

    def _homeostatic_decay(self, mem, spk):
        return mem * (1 - 0.01 * spk)
```

The custom method must take `(state, spk)` and return the new state. A lambda
or a name that is not a callable method on the module raises `ValueError` at
construction time.

## Notes

- The spike used by resets is the (possibly non-differentiable) output of
  `_step`, before any output handling.
- `zero()` differs from `hard_zero()`: `zero()` multiplies by `(1 - spk)` and
  is exact for binary spikes but leaves float spikes partially intact;
  `hard_zero()` masks unconditionally and works for any spike values.
- The reset expression is code-generated once at init by `SnnModule`
  (`_pk_apply_resets`), so applying resets costs nothing at step time.
