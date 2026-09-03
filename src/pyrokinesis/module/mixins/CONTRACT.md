# Mixin contract — `pyrokinesis.module.mixins`

Order-less mixins. Base order in `PyroModule` must not matter.

## 1. Tags

Every private member belongs to exactly one mixin and is prefixed
`_pk_<tag>_`. Public API (`forward`, `initial_state`, `constrained`, …)
is exempt.

| tag | mixin (file) | base | owns |
| --- | --- | --- | --- |
| `in` | `InputMixin` (`inputs.py`) | — | input entries/names/specs, `_pk_in_canonical` |
| `par` | `ParamMixin` (`params.py`) | — | param specs/constraints, `_pk_par_constrained*` |
| `cst` | `ConstantMixin` (`constants.py`) | — | constant specs |
| `st` | `StateMixin` (`states.py`) | `in` | output/state entries, shapes, alloc, state factories |
| `fwd` | `ForwardMixin` (`forward.py`) | `st` | `_pk_fwd_explicit`, `_pk_fwd_hidden`, `forward`, `step` |
| `scn` | `SequenceScanMixin` (`scan.py`) | `fwd` | scans, `_pk_scn_compiled`, chunk ctx |
| `val` | `ValidationMixin` (`validation.py`) | — | `_pk_val_override`, `validate`, global ctx |
| `ser` | `SerializationMixin` (`serialization.py`) | `st` | `get/set_extra_state` (public, no `_pk_`) |
| `rpr` | `ReprMixin` (`repr.py`) | `in` | `extra_repr` (public, no `_pk_`) |
| `rst` | `ResetMixin` (`snn/reset.py`) | `st` | `_pk_rst_exprs`, `_pk_rst_install`, `_pk_rst_apply` |
| `seq` | `Sequential` (`nn.py`, not a mixin) | — | container registry/probe/compile |

Dependency DAG (edges point to prerequisites): `scn→fwd→st→in`,
`ser→st`, `rpr→in`, `rst→st`. `PyroModule` lists bases dependents-first
(`scn, ser, rpr, fwd, st, val, cst, par, in`) — C3 requires base order
compatible with the DAG; anything else raises `TypeError` at class
creation instead of silently shadowing. `rst` extends `st` only:
adding `par` would contradict `PyroModule`'s order (C3 conflict), so
param access stays duck-typed.

Free helpers (`collect_metadata`, `build_params`, `sequence_scan`) keep
plain names — they are functions, not mixin state.

## 2. Ownership (single definition) + two cooperative chains

A `_pk_<tag>_*` name may appear in exactly one mixin `__dict__` across
the whole MRO. Call owners by full tagged name
(`self._pk_st_alloc(...)`); there are no `...` stubs — a missing owner
is an honest `AttributeError`.

Two points are frozen hook chains, not single owners. Contributors define
bare `_pk_hook_<point>__<tag>` methods (no `super()` calls); `PyroModule`
freezes ordered tuples once per class in `__init_subclass__` (base-first)
and drivers iterate them. Zero per-call `super()`/`getattr` — this keeps
Dynamo from graph-breaking on the hot path:

* `_pk_hook_specs__*` — run by `StateMixin._pk_process_spec_extensions`
  after spec-extra dispatch. `rst` installs its reset fn here.
* `_pk_hook_post__*` — `(self, out) -> out`, run by
  `ForwardMixin._pk_post_step`. Empty for plain modules (one tuple
  iteration, no-op).

## 3. Hooks (multiple definitions, base knows no tags)

Only names starting `_pk_hook_<point>__<tag>` may repeat. The base
freezes ordered tuples in `__init_subclass__` by scanning
`reversed(cls.__mro__)` for that prefix — deterministic base-first
order, no `super()` chain, no `hasattr(<tag>)` in base.

Points:

* `_pk_hook_specs__*` — run after `StateMixin` spec-extension dispatch.
  `rst` installs its reset fn here. Replaces
  `ResetMixin._pk_process_spec_extensions` override.
* `_pk_hook_post__*` — `(self, out) -> out`, run at the end of
  `ForwardMixin._pk_fwd_explicit`. `fwd` identity is the default when
  no hooks registered; `rst` applies resets here. Replaces
  `_pk_post_step` overrides.
* `_pk_hook_params__*` — each returns `list[inspect.Parameter]`;
  `collection.generate_signature` concatenates them. Replaces
  `_pk_extra_init_params` overrides (which dropped `super()`).

Adding a mixin = new tag + hook definitions. Zero base edits.

## 4. Forbidden

* Non-frozen `_pk_*` overrides (a link that neither contributes a
  `_pk_hook_*` entry nor aggregates over the MRO) — except
  `nn.Module.__init__` / `__init_subclass__`, which stay cooperative
  per Python convention. In particular: no `super()._pk_*()` and no
  `getattr(super(), ...)` on any hot path (Dynamo graph-breaks on it).
* `hasattr(self, "_pk_rst_*")` / `getattr(self, "_pk_rst_*")` outside
  `rst` — that is the leak this contract removes.
* Untagged `_pk_*` state on mixins.
* Type-only `...` stubs duplicating another mixin's owner.
