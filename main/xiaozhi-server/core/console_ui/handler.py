from __future__ import annotations

from pathlib import Path

from aiohttp import web


class ConsoleUiHandler:
    def __init__(self):
        self._root_dir = Path(__file__).resolve().parent
        self._index_path = self._root_dir / "index.html"

    async def handle_redirect(self, request: web.Request) -> web.StreamResponse:
        raise web.HTTPFound("/console/")

    async def handle_index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(path=self._index_path)

