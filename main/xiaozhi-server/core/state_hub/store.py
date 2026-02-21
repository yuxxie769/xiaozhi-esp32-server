from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple


class StateHubStore:
    """Latest-state store with optional persistence to data/."""

    def __init__(self, persist_path: str):
        self._lock = threading.RLock()
        self.persist_path = str(persist_path or "").strip()

        self.entities: Dict[str, Dict[str, Any]] = {}
        self.allowlist: list[str] = []
        self.target: Dict[str, Any] = {}

        self.conn_state: str = "DISCONNECTED"
        self.last_error: str = ""
        self.last_event_at_ms: int = 0
        self.rev: int = 0

        self._loaded = False

    def load_from_disk_once(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
        if not self.persist_path:
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                return
            entities = obj.get("entities")
            allowlist = obj.get("allowlist")
            target = obj.get("target")
            last_event_at_ms = obj.get("last_event_at_ms")
            rev = obj.get("rev")
            with self._lock:
                if isinstance(entities, dict):
                    self.entities = {
                        str(k): v for k, v in entities.items() if k and isinstance(v, dict)
                    }
                if isinstance(allowlist, list):
                    self.allowlist = [str(x) for x in allowlist if x]
                if isinstance(target, dict):
                    self.target = target
                if isinstance(last_event_at_ms, (int, float)):
                    self.last_event_at_ms = int(last_event_at_ms)
                if isinstance(rev, (int, float)):
                    self.rev = int(rev)
                # Disk snapshot is always outdated until a new live event arrives.
                self.conn_state = "DISCONNECTED"
        except FileNotFoundError:
            return
        except Exception:
            return

    def mark_connected(self, state: str) -> None:
        with self._lock:
            self.conn_state = str(state or "").strip() or "DISCONNECTED"

    def reset_last_event_time(self) -> None:
        """Reset last_event_at_ms so callers can treat any existing disk snapshot as outdated
        until a new live event is received on the current connection.
        """
        with self._lock:
            self.last_event_at_ms = 0

    def set_error(self, err: str) -> None:
        with self._lock:
            self.last_error = str(err or "")

    def set_target_and_allowlist(self, target: Dict[str, Any], allowlist: list[str]) -> None:
        with self._lock:
            self.target = target if isinstance(target, dict) else {}
            self.allowlist = list(allowlist or [])

    def apply_snapshot_added(self, added: Dict[str, Any]) -> bool:
        changed = False
        with self._lock:
            for eid, raw in (added or {}).items():
                if not eid or not isinstance(raw, dict):
                    continue
                self.entities[str(eid)] = raw
                changed = True
            if changed:
                self.rev += 1
                self.last_event_at_ms = int(time.time() * 1000)
        return changed

    def apply_changed(self, changed_payload: Dict[str, Any]) -> bool:
        changed_any = False
        with self._lock:
            for eid, diff in (changed_payload or {}).items():
                if not eid or not isinstance(diff, dict):
                    continue
                eid = str(eid)
                cur = self.entities.get(eid)
                if not isinstance(cur, dict):
                    cur = {}
                # Support {"+": {...}} diff or direct field updates.
                if "+" in diff and isinstance(diff.get("+"), dict):
                    plus = diff.get("+") or {}
                    for k, v in plus.items():
                        cur[k] = v
                else:
                    for k, v in diff.items():
                        cur[k] = v
                self.entities[eid] = cur
                changed_any = True
            if changed_any:
                self.rev += 1
                self.last_event_at_ms = int(time.time() * 1000)
        return changed_any

    def apply_removed(self, removed: Any) -> bool:
        removed_any = False
        with self._lock:
            # removed can be list[str] or dict or single str depending on HA version.
            if isinstance(removed, list):
                eids = [str(x) for x in removed if x]
            elif isinstance(removed, dict):
                eids = [str(k) for k in removed.keys() if k]
            elif isinstance(removed, str):
                eids = [removed]
            else:
                eids = []
            for eid in eids:
                if eid in self.entities:
                    self.entities.pop(eid, None)
                    removed_any = True
            if removed_any:
                self.rev += 1
                self.last_event_at_ms = int(time.time() * 1000)
        return removed_any

    def snapshot_for_disk(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "v": 1,
                "saved_at_ms": int(time.time() * 1000),
                "rev": self.rev,
                "last_event_at_ms": self.last_event_at_ms,
                "target": self.target,
                "allowlist": list(self.allowlist),
                "entities": dict(self.entities),
            }

    def compute_freshness(self, *, outdated_after_seconds: int | None = None) -> Tuple[bool, bool, Optional[int]]:
        with self._lock:
            conn_state = self.conn_state
            last_event_at_ms = int(self.last_event_at_ms or 0)
        connected = conn_state == "READY"
        if last_event_at_ms <= 0:
            return connected, True, None
        now_ms = int(time.time() * 1000)
        age_s = int(max(0, (now_ms - last_event_at_ms) / 1000))
        # "outdated" should mean "offline/possibly stale", not "no state changes recently".
        # Even if no entity changes arrive for a long time, the cached state is still current.
        # We only treat it as outdated when not connected (or when we have no data at all).
        outdated = not connected
        return connected, outdated, age_s

    def save_atomic(self) -> None:
        if not self.persist_path:
            return
        parent = os.path.dirname(self.persist_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = self.persist_path + ".tmp"
        payload = self.snapshot_for_disk()
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, self.persist_path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
