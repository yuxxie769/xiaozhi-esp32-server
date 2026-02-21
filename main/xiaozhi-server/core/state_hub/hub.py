from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any, Dict, Optional

from config.logger import setup_logging
from config.config_loader import get_project_dir

from .client import HaWsClient
from .exposure import ExposureStore
from .store import StateHubStore
from .summary import ClipRules, build_entity_rows, build_highlight, build_local_summary_text

TAG = __name__


class StateHub:
    def __init__(self, server):
        self.server = server
        self.logger = setup_logging()
        self._last_conn_state: str | None = None
        self._last_error_logged: str | None = None

        self._force_reconnect = asyncio.Event()
        self._force_refresh = asyncio.Event()

        self._save_task: Optional[asyncio.Task] = None
        self._dirty = False
        self._dirty_event = asyncio.Event()

        # Persist under project_root/data regardless of runtime CWD.
        # (Do not rely on config.log.data_dir; keep State Hub storage deterministic.)
        data_dir = os.path.join(get_project_dir(), "data")
        persist_dir = os.path.join(data_dir, "state_hub")
        self.state_path = os.path.join(persist_dir, "state.json")
        self.exposure_path = os.path.join(persist_dir, "exposure.json")

        self.store = StateHubStore(self.state_path)
        self.store.load_from_disk_once()

        self.exposure = ExposureStore(self.exposure_path)
        self.exposure.load()

    def request_reconnect(self) -> None:
        self._force_reconnect.set()

    def request_refresh_target(self) -> None:
        self._force_refresh.set()

    def set_exposure(self, entity_id: str, expose: bool) -> None:
        self.exposure.set(entity_id, expose)

    def _cfg(self) -> Dict[str, Any]:
        plugins = (getattr(self.server, "config", None) or {}).get("plugins", {})
        return plugins.get("state_hub", {}) if isinstance(plugins, dict) else {}

    def _set_conn_state(self, state: str) -> None:
        state = str(state or "").strip() or "DISCONNECTED"
        self.store.mark_connected(state)

    def _log_error_once(self, err: str) -> None:
        err = str(err or "").strip()
        if not err:
            self._last_error_logged = None
            return
        if self._last_error_logged == err:
            return
        self._last_error_logged = err
        self.logger.bind(tag=TAG).error(f"state_hub error: {err}")

    def _rules(self, cfg: Dict[str, Any]) -> ClipRules:
        return ClipRules(cfg)

    def view_highlight(self) -> Dict[str, Any]:
        cfg = self._cfg()
        outdated_after = int(cfg.get("outdated_after_seconds", 60) or 60)
        connected, outdated, age_s = self.store.compute_freshness(outdated_after_seconds=outdated_after)
        rules = self._rules(cfg)
        exposure_map = self.exposure.snapshot()
        with self.store._lock:  # internal lock reuse (store is thread-safe)
            entities = dict(self.store.entities)
        hl = build_highlight(
            entities,
            exposure=exposure_map,
            rules=rules,
            max_items=rules.max_highlight,
        )
        return {
            "connected": bool(connected),
            "outdated": bool(outdated),
            "age_s": age_s,
            "highlight": hl,
            "rev": self.store.rev,
            "conn_state": self.store.conn_state,
            "last_error": self.store.last_error,
        }

    def view_entities(self) -> Dict[str, Any]:
        cfg = self._cfg()
        outdated_after = int(cfg.get("outdated_after_seconds", 60) or 60)
        connected, outdated, age_s = self.store.compute_freshness(outdated_after_seconds=outdated_after)
        rules = self._rules(cfg)
        exposure_map = self.exposure.snapshot()
        with self.store._lock:
            entities = dict(self.store.entities)
        rows = build_entity_rows(entities, exposure=exposure_map, rules=rules)
        return {
            "connected": bool(connected),
            "outdated": bool(outdated),
            "age_s": age_s,
            "items": rows,
            "rev": self.store.rev,
            "conn_state": self.store.conn_state,
            "last_error": self.store.last_error,
        }

    def local_summary_text(self) -> str:
        v = self.view_highlight()
        return build_local_summary_text(v["connected"], v["outdated"], v.get("age_s"), v.get("highlight") or [])

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._dirty_event.set()

    async def _save_loop(self) -> None:
        # Debounced save: at most once per 2 seconds while dirty.
        while True:
            await self._dirty_event.wait()
            self._dirty_event.clear()
            await asyncio.sleep(2.0)
            if not self._dirty:
                continue
            self._dirty = False
            await asyncio.to_thread(self.store.save_atomic)

    async def run_forever(self) -> None:
        if not self._save_task:
            self._save_task = asyncio.create_task(self._save_loop(), name="state_hub:save")

        backoff = 1.0
        while True:
            cfg = self._cfg()
            enabled = bool(cfg.get("enabled", False))
            if not enabled:
                self._set_conn_state("DISCONNECTED")
                await asyncio.sleep(1.0)
                continue

            ha_url = str(cfg.get("ha_ws_url", "") or "").strip()
            token = str(cfg.get("access_token", "") or "").strip()
            target = cfg.get("target", {}) if isinstance(cfg.get("target", {}), dict) else {}
            refresh_target_seconds = float(cfg.get("refresh_target_seconds", 600) or 600)
            if not ha_url or not token or not target:
                self.store.set_error("missing_state_hub_config")
                self._set_conn_state("ERROR")
                self._log_error_once("missing_state_hub_config")
                await asyncio.sleep(2.0)
                continue

            client = HaWsClient(ha_url, token)
            try:
                self.store.set_error("")
                self._set_conn_state("CONNECTING")
                await client.connect()
                self._set_conn_state("AUTHING")
                await client.auth()
                client.start_reader()
                self._set_conn_state("READY")
                # Ensure any disk-loaded snapshot is treated as outdated until we receive a new
                # subscription event (initial snapshot) on this connection.
                self.store.reset_last_event_time()
                backoff = 1.0
                self._log_error_once("")

                # Extract allowlist
                self._force_refresh.clear()
                allowlist = await self._extract_allowlist(client, target)
                self.store.set_target_and_allowlist(target, allowlist)
                self._mark_dirty()

                # Subscribe
                sub_id = await self._subscribe_entities(client, allowlist)

                # Main read loop
                last_refresh_at = time.time()
                while True:
                    if self._force_reconnect.is_set():
                        self._force_reconnect.clear()
                        raise RuntimeError("forced_reconnect")

                    now = time.time()
                    if self._force_refresh.is_set() or (now - last_refresh_at) >= refresh_target_seconds:
                        self._force_refresh.clear()
                        last_refresh_at = now
                        new_allowlist = await self._extract_allowlist(client, target)
                        if set(new_allowlist) != set(allowlist):
                            raise RuntimeError("allowlist_changed")

                    # Receive with timeout so we can detect idle and check flags.
                    msg = await client.recv_message(timeout=1.0)
                    if msg is None:
                        continue

                    if msg.get("type") == "_closed":
                        err = str(msg.get("error") or "")
                        if err:
                            raise RuntimeError(f"ws_closed:{err}")
                        raise RuntimeError("ws_closed")

                    # Handle subscription events
                    if msg.get("type") == "event" and msg.get("id") == sub_id:
                        event = msg.get("event") if isinstance(msg.get("event"), dict) else {}
                        dirty = False
                        if isinstance(event.get("a"), dict):
                            dirty = self.store.apply_snapshot_added(event.get("a")) or dirty
                        if isinstance(event.get("c"), dict):
                            dirty = self.store.apply_changed(event.get("c")) or dirty
                        if "r" in event:
                            dirty = self.store.apply_removed(event.get("r")) or dirty
                        if dirty:
                            self._mark_dirty()

            except Exception as e:
                err = str(e or "")
                self.store.set_error(err)
                self._set_conn_state("BACKOFF")
                self._log_error_once(err)
                try:
                    await client.close()
                except Exception:
                    pass
                # backoff with jitter
                sleep_s = min(30.0, backoff) + random.random() * 0.5
                await asyncio.sleep(sleep_s)
                backoff = min(30.0, backoff * 2.0)

    async def _extract_allowlist(self, client: HaWsClient, target: Dict[str, Any]) -> list[str]:
        msg = await client.request("extract_from_target", {"target": target}, timeout=10.0)
        if not isinstance(msg, dict) or not msg.get("success"):
            raise RuntimeError("extract_from_target_failed")
        result = msg.get("result") if isinstance(msg.get("result"), dict) else {}
        ref = result.get("referenced_entities", [])
        if not isinstance(ref, list):
            ref = []
        allowlist = [str(x) for x in ref if x]
        return allowlist

    async def _subscribe_entities(self, client: HaWsClient, entity_ids: list[str]) -> int:
        msg = await client.request("subscribe_entities", {"entity_ids": list(entity_ids or [])}, timeout=10.0)
        if not isinstance(msg, dict) or not msg.get("success"):
            raise RuntimeError("subscribe_entities_failed")
        sub_id = msg.get("id")
        if not isinstance(sub_id, int):
            raise RuntimeError("subscribe_entities_missing_id")
        return int(sub_id)
