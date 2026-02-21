from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

import websockets


class HaWsClient:
    def __init__(self, url: str, access_token: str):
        self.url = str(url or "").strip()
        self.access_token = str(access_token or "").strip()
        self.ws = None
        self._next_id = 1
        self._pending: Dict[int, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._messages: asyncio.Queue = asyncio.Queue()
        self._reader_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        if not self.url:
            raise ValueError("ha_ws_url required")
        self.ws = await websockets.connect(self.url)

    async def close(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except BaseException:
                pass
            self._reader_task = None
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        self.ws = None

    async def _send_json(self, obj: Dict[str, Any]) -> None:
        if not self.ws:
            raise RuntimeError("ws not connected")
        await self.ws.send(json.dumps(obj, ensure_ascii=False))

    async def auth(self) -> None:
        if not self.ws:
            raise RuntimeError("ws not connected")
        # Read until auth_required then auth_ok
        first = json.loads(await self.ws.recv())
        if first.get("type") != "auth_required":
            raise RuntimeError(f"unexpected first message: {first.get('type')}")
        await self._send_json({"type": "auth", "access_token": self.access_token})
        second = json.loads(await self.ws.recv())
        if second.get("type") != "auth_ok":
            raise RuntimeError(f"auth failed: {second}")

    def start_reader(self) -> None:
        if self._reader_task is not None and not self._reader_task.done():
            return
        self._reader_task = asyncio.create_task(self._reader_loop(), name="state_hub:ha_reader")

    async def request(self, msg_type: str, payload: Dict[str, Any], *, timeout: float = 10.0) -> Any:
        async with self._lock:
            req_id = self._next_id
            self._next_id += 1
            fut = asyncio.get_running_loop().create_future()
            self._pending[req_id] = fut
            obj = {"id": req_id, "type": msg_type}
            obj.update(payload or {})
            await self._send_json(obj)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(req_id, None)

    def dispatch_result(self, msg: Dict[str, Any]) -> bool:
        if not isinstance(msg, dict):
            return False
        if msg.get("type") != "result":
            return False
        req_id = msg.get("id")
        if not isinstance(req_id, int):
            return False
        fut = self._pending.get(req_id)
        if fut and not fut.done():
            fut.set_result(msg)
            return True
        return False

    async def recv_message(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        try:
            return await asyncio.wait_for(self._messages.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def _reader_loop(self) -> None:
        err = ""
        try:
            while self.ws is not None:
                msg_text = await self.ws.recv()
                try:
                    msg = json.loads(msg_text)
                except Exception:
                    continue
                if self.dispatch_result(msg):
                    continue
                await self._messages.put(msg)
        except Exception as e:
            err = str(e or "")
        finally:
            # Wake consumers so they can reconnect promptly.
            try:
                await self._messages.put({"type": "_closed", "error": err})
            except Exception:
                pass
