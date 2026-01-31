from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

ServiceFactory = Callable[[Any], Awaitable[None]]

_service_factories: dict[str, ServiceFactory] = {}


def register_service(name: str) -> Callable[[ServiceFactory], ServiceFactory]:
    """Register a background service factory."""

    def decorator(factory: ServiceFactory) -> ServiceFactory:
        _service_factories[name] = factory
        logger.bind(tag=TAG).debug(f"service registered: {name}")
        return factory

    return decorator


def start_all_services(server) -> list[Any]:
    """Start all registered services on current loop."""
    import asyncio

    tasks: list[Any] = []
    for name, factory in _service_factories.items():
        try:
            coro = factory(server)
            tasks.append(asyncio.create_task(coro, name=f"service:{name}"))
            logger.bind(tag=TAG).info(f"service started: {name}")
        except Exception as e:
            logger.bind(tag=TAG).opt(exception=True).error(
                f"failed to start service {name}: {e}"
            )
    return tasks
