# Constraints

Every `Param` can be wrapped in a constraint applied on the hot path.
`self.constrained()` returns the constrained parameter values in `Params`
declaration order - this is the documented way to read parameters inside
`_step`.

## Policy: constraints apply to learnable params only

A **fixed** param is returned raw, without applying its constraint. A
**learnable** param is clamped (or otherwise projected) every step. This is
deliberate: fixed params are set by the user to exact values and should be
used as given; learnable params are updated by an optimizer and need the
constraint enforced to stay in the valid region.

```python
from pyrokinesis import clamp_positive, clamp_unit_interval
from pyrokinesis.snn import SnnModule

class LIF(SnnModule):
    class Params:
        beta = SnnModule.Param(0.9, constraint=clamp_unit_interval)
        threshold = SnnModule.Param(1.0, constraint=clamp_positive)
```

- `LIF(beta=2.0)` -> fixed, `constrained()` returns `2.0` (no clamping).
- `LIF(beta=2.0, learnable_beta=True)` -> learnable, clamped to `[0, 1]`
  every step.
- Resets that target a param also use its constrained value.

## Overriding learnability and constraints

Every `Param` gets three constructor kwargs:

| kwarg                 | effect                                   |
| --------------------- | ---------------------------------------- |
| `learnable_<name>=`   | force learnable `True`/`False`           |
| `force_learn_<name>=` | spec-level override; an explicit `True` wins even against `learnable_<name>=False` |
| `<name>_constraint=`  | replace the constraint (e.g. `identity`) |

```python
lif = LIF(
    beta=0.5,
    learnable_beta=True,                 # train beta
    beta_constraint=clamp_unit_interval, # keep it in [0, 1]
)
```

The constraint override applies only when the param is learnable; a fixed
param still bypasses it.

## Provided constraints

| constraint            | effect                            |
| --------------------- | --------------------------------- |
| `clamp_unit_interval` | clamp to `[0, 1]`                 |
| `clamp_positive`      | clamp to `[0, inf)`               |
| `identity`            | no-op (use to strip a constraint) |

`identity` is also the default when no constraint is declared.

## Hot path

Constraints are resolved into a single frozen expression at init
(`_pk_constrained_fn`): no string lookups or metadata resolution on the step
path. The same frozen expression backs reset targets, so constrained values
stay consistent between the step math and the reset math.
