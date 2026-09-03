# Mixin contract — `crematorium.module.mixins`

Order-less mixins. Base order in `CrModule` must not matter.

## 1. Tags

Every private member belongs to exactly one mixin and is prefixed
`_cr_<tag>_`. Public API (`forward`, `initial_state`, `constrain`, …)
is exempt.

| tag | mixin (file) | base | owns |
| --- | --- | --- | --- |
| `in` | `InputMixin` (`inputs.py`) | — | input entries/names/specs, `_cr_in_canonical` |
| `par` | `ParamMixin` (`params.py`) | — | param specs/constraints, `_cr_param_constraint_map` |
| `cst` | `ConstantMixin` (`constants.py`) | — | constant specs |
| `st` | `StateMixin` (`states.py`) | `in` | output/state entries, shapes, alloc, state factories |
| `fwd` | `ForwardMixin` (`forward.py`) | `st` | `_cr_fwd_explicit`, `_cr_fwd_hidden`, `forward`, `step` |
| `scn` | `SequenceScanMixin` (`scan.py`) | `fwd` | scans, `_cr_scn_compiled`, chunk ctx |
| `val` | `ValidationMixin` (`validation.py`) | — | `_cr_val_override`, `validate`, global ctx |
| `ser` | `SerializationMixin` (`serialization.py`) | `st` | `get/set_extra_state` (public, no `_cr_`) |
| `rpr` | `ReprMixin` (`repr.py`) | `in` | `extra_repr` (public, no `_cr_`) |
| `rst` | `ResetMixin` (`snn/reset.py`) | `st` | `_cr_rst_exprs`, `_cr_rst_install`, `_cr_rst_apply` |
| `seq` | `Sequential` (`nn.py`, not a mixin) | — | container registry/probe/compile |

Dependency DAG (edges point to prerequisites): `scn→fwd→st→in`,
`ser→st`, `rpr→in`, `rst→st`. `CrModule` lists bases dependents-first
(`scn, ser, rpr, fwd, st, val, cst, par, in`) — C3 requires base order
compatible with the DAG; anything else raises `TypeError` at class
creation instead of silently shadowing. `rst` extends `st` only:
adding `par` would contradict `CrModule`'s order (C3 conflict), so
param access stays duck-typed.

Free helpers (`collect_metadata`, `build_params`, `sequence_scan`) keep
plain names — they are functions, not mixin state.

## 2. Ownership (single definition) + two frozen hook chains

A `_cr_<tag>_*` name may appear in exactly one mixin `__dict__` across
the whole MRO. Call owners by full tagged name
(`self._cr_st_alloc(...)`); there are no `...` stubs — a missing owner
is an honest `AttributeError`.

Two points are frozen hook chains, not single owners. Contributors define
bare `_cr_hook_<point>__<tag>` methods (no `super()` calls); `CrModule`
freezes ordered tuples once per class in `__init_subclass__` (child-first;
a subclass redefining the same hook name *replaces* the parent entry)
and drivers iterate them. Zero per-call `super()`/`getattr` — this keeps
Dynamo from graph-breaking on the hot path:

* `_cr_hook_specs__*` — run by `StateMixin._cr_process_spec_extensions`
  after spec-extra dispatch. `rst` installs its reset fn here.
* `_cr_hook_post__*` — `(self, out) -> out`, run by
  `ForwardMixin._cr_post_step`. Empty for plain modules (one tuple
  iteration, no-op).

## 3. Hooks (multiple definitions, base knows no tags)

Only names starting `_cr_hook_<point>__<tag>` may repeat. The base
freezes ordered tuples in `__init_subclass__` by scanning
`cls.__mro__` (child-first) for that prefix: a subclass redefining the
same hook name replaces the parent entry (replace, not additive) — no
`super()` chain, no `hasattr(<tag>)` in base.

Points:

* `_cr_hook_specs__*` — run after `StateMixin` spec-extension dispatch.
  `rst` installs its reset fn here. Replaces
  `ResetMixin._cr_process_spec_extensions` override.
* `_cr_hook_post__*` — `(self, out) -> out`, run at the end of
  `ForwardMixin._cr_fwd_explicit`. `fwd` identity is the default when
  no hooks registered; `rst` applies resets here. Replaces
  `_cr_post_step` overrides.

Init params (`_cr_extra_init_params`) are not a hook chain: every mixin
in the MRO may contribute, aggregated base-first over `reversed(mro)`
with deduplication by parameter name (first contributor wins).

Adding a mixin = new tag + hook definitions. Zero base edits.

## 4. Forbidden

* Non-frozen `_cr_*` overrides (a link that neither contributes a
  `_cr_hook_*` entry nor aggregates over the MRO) — except
  `nn.Module.__init__` / `__init_subclass__`, which stay cooperative
  per Python convention. In particular: no `super()._cr_*()` and no
  `getattr(super(), ...)` on any hot path (Dynamo graph-breaks on it).
* `hasattr(self, "_cr_rst_*")` / `getattr(self, "_cr_rst_*")` outside
  `rst` — that is the leak this contract removes.
* Untagged `_cr_*` state on mixins.
* Type-only `...` stubs duplicating another mixin's owner.
