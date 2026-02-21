from __future__ import annotations

import json
from aiohttp import web

from core.api.base_handler import BaseHandler
from core.state_hub.registry import get_state_hub


class StateHubApiHandler(BaseHandler):
    async def handle_view(self, request: web.Request) -> web.StreamResponse:
        hub = get_state_hub()
        if not hub:
            resp = web.json_response({"success": True, "data": {"connected": False, "outdated": True, "age_s": None, "highlight": [], "rev": 0, "conn_state": "DISCONNECTED"}})
            self._add_cors_headers(resp)
            return resp

        data = hub.view_highlight()
        resp = web.json_response({"success": True, "data": data}, dumps=lambda o: json.dumps(o, ensure_ascii=False))
        self._add_cors_headers(resp)
        return resp

    async def handle_entities(self, request: web.Request) -> web.StreamResponse:
        hub = get_state_hub()
        if not hub:
            resp = web.json_response(
                {
                    "success": True,
                    "data": {
                        "connected": False,
                        "outdated": True,
                        "age_s": None,
                        "items": [],
                        "rev": 0,
                        "conn_state": "DISCONNECTED",
                    },
                }
            )
            self._add_cors_headers(resp)
            return resp

        data = hub.view_entities()
        resp = web.json_response({"success": True, "data": data}, dumps=lambda o: json.dumps(o, ensure_ascii=False))
        self._add_cors_headers(resp)
        return resp

    async def handle_status(self, request: web.Request) -> web.StreamResponse:
        hub = get_state_hub()
        if not hub:
            data = {"connected": False, "conn_state": "DISCONNECTED", "outdated": True, "age_s": None, "rev": 0, "allowlist_size": 0, "last_error": ""}
        else:
            v = hub.view_highlight()
            with hub.store._lock:
                allowlist_size = len(hub.store.allowlist or [])
            data = {
                "connected": v.get("connected", False),
                "conn_state": v.get("conn_state", "DISCONNECTED"),
                "outdated": v.get("outdated", True),
                "age_s": v.get("age_s"),
                "rev": v.get("rev", 0),
                "allowlist_size": allowlist_size,
                "last_error": v.get("last_error", ""),
            }
        resp = web.json_response({"success": True, "data": data}, dumps=lambda o: json.dumps(o, ensure_ascii=False))
        self._add_cors_headers(resp)
        return resp

    async def handle_reconnect(self, request: web.Request) -> web.StreamResponse:
        hub = get_state_hub()
        if hub:
            hub.request_reconnect()
        resp = web.json_response({"success": True, "data": {"ok": True}})
        self._add_cors_headers(resp)
        return resp

    async def handle_refresh_target(self, request: web.Request) -> web.StreamResponse:
        hub = get_state_hub()
        if hub:
            hub.request_refresh_target()
        resp = web.json_response({"success": True, "data": {"ok": True}})
        self._add_cors_headers(resp)
        return resp

    async def handle_exposure(self, request: web.Request) -> web.StreamResponse:
        hub = get_state_hub()
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        entity_id = str((payload or {}).get("entity_id") or "").strip()
        expose = bool((payload or {}).get("expose_to_llm", True))
        if hub and entity_id:
            hub.set_exposure(entity_id, expose)
        resp = web.json_response({"success": True, "data": {"ok": True}})
        self._add_cors_headers(resp)
        return resp
