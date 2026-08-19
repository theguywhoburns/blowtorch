from __future__ import annotations

from typing import Any, ClassVar, Optional

from .specs import Spec, Tensor


class SerializationMixin:
    """state_dict integration for hidden buffers."""

    init_hidden: bool
    _bt_allocated: bool
    _bt_spec_entries: ClassVar[tuple[tuple[str, Spec], ...]]
    _buffers: dict[str, Optional[Tensor]]
    _non_persistent_buffers_set: set[str]

    def get_extra_state(self) -> Optional[dict[str, Tensor]]:
        """
        Include hidden buffers in state_dict even though they are non-persistent.

        Only applies in ``init_hidden=True`` mode, where the module owns its
        state. In explicit mode the caller owns the state tensors, so they are
        not serialized here; persist them yourself alongside the state_dict.
        """
        if not self.init_hidden:
            return None

        out: dict[str, Tensor] = {}
        for name, _ in self._bt_spec_entries:
            t = self._buffers.get(name)
            if t is not None:
                out[name] = t.detach()

        return out

    def set_extra_state(self, state: Any) -> None:
        if not self.init_hidden or state is None:
            return

        for name, t in state.items():
            if name in self._buffers:
                self._buffers[name] = t
            else:
                self._buffers[name] = t
                self._non_persistent_buffers_set.add(name)

        self._bt_allocated = True