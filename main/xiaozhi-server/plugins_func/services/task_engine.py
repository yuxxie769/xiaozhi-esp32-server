from __future__ import annotations

from typing import Any

from config.logger import setup_logging
from core.task_engine.engine import task_engine_service_loop

from .registry import register_service

TAG = __name__
logger = setup_logging()


@register_service("task_engine")
async def task_engine_service(server: Any) -> None:
    try:
        await task_engine_service_loop(server)
    except Exception as e:
        logger.bind(tag=TAG).opt(exception=True).error(f"task_engine crashed: {e}")
        raise

