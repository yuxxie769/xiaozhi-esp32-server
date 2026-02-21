from __future__ import annotations

import asyncio
from typing import Any

from config.logger import setup_logging
from .registry import register_service
from core.state_hub.hub import StateHub
from core.state_hub.registry import set_state_hub

TAG = __name__
logger = setup_logging()


@register_service("state_hub")
async def state_hub_service(server: Any) -> None:
    hub = StateHub(server)
    set_state_hub(hub)
    logger.bind(tag=TAG).debug("state_hub service started")
    await hub.run_forever()
