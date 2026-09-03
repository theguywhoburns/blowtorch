from __future__ import annotations

from typing import Any, ClassVar, Optional

from ..specs import Spec, Tensor
from .states import StateMixin


class SerializationMixin(StateMixin):
    """state_dict integration for hidden buffers."""

    init_hidden: bool
    _pk_allocated: bool
    _pk_spec_entries: ClassVar[tuple[tuple[str, Spec], ...]]
    _buffers: dict[str, Optional[Tensor]]
    _non_persistent_buffers_set: set[str]

    def get_extra_state(self) -> Optional[dict[str, Tensor]]:
        """
        Include hidden buffers in state_dict even though they are non-persistent.

        Only applies in ``init_hidden=True`` mode, where the module owns its
        state. In explicit mode the caller owns the state tensors, so they are
        not serialized here; persist them yourself alongside the state_dict.

        Returns ``None`` when nothing has been allocated, so a round-trip
        through ``state_dict``/``load_state_dict`` does not mark an unallocated
        module as allocated.
        """
        if not self.init_hidden or not self._pk_allocated:
            return None

        out: dict[str, Tensor] = {}
        for name, _ in self._pk_spec_entries:
            t = self._buffers.get(name)
            if t is not None:
                out[name] = t.detach()

        return out

    def set_extra_state(self, state: Any) -> None:
        if not self.init_hidden or not state:
            return

        for name, t in state.items():
            if name not in self._buffers:
                self._non_persistent_buffers_set.add(name)
            self._buffers[name] = t

        self._pk_allocated = True