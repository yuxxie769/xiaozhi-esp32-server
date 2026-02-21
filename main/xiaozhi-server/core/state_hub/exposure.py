from __future__ import annotations

import json
import os
import threading
from typing import Dict


class ExposureStore:
    """Persist expose_to_llm switches to a json file.

    File format:
      {"switch.xxx": true, "sensor.yyy": false, ...}
    """

    def __init__(self, path: str):
        self._path = str(path or "").strip()
        self._lock = threading.RLock()
        self._data: Dict[str, bool] = {}
        self._loaded = False

    @property
    def path(self) -> str:
        return self._path

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            if not self._path:
                self._data = {}
                return
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    self._data = {str(k): bool(v) for k, v in obj.items() if k}
                else:
                    self._data = {}
            except FileNotFoundError:
                self._data = {}
            except Exception:
                self._data = {}

    def get(self, entity_id: str) -> bool | None:
        entity_id = str(entity_id or "").strip()
        if not entity_id:
            return None
        self.load()
        with self._lock:
            if entity_id not in self._data:
                return None
            return bool(self._data.get(entity_id))

    def set(self, entity_id: str, expose: bool) -> None:
        entity_id = str(entity_id or "").strip()
        if not entity_id:
            return
        self.load()
        with self._lock:
            self._data[entity_id] = bool(expose)
        self._save_atomic()

    def snapshot(self) -> Dict[str, bool]:
        self.load()
        with self._lock:
            return dict(self._data)

    def _save_atomic(self) -> None:
        if not self._path:
            return
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = self._path + ".tmp"
        with self._lock:
            payload = dict(self._data)
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

