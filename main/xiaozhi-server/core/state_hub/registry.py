from __future__ import annotations

import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .hub import StateHub

_lock = threading.RLock()
_hub: Optional["StateHub"] = None


def set_state_hub(hub: "StateHub") -> None:
    global _hub
    with _lock:
        _hub = hub


def get_state_hub() -> Optional["StateHub"]:
    with _lock:
        return _hub

