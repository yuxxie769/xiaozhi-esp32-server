from __future__ import annotations

from typing import Any, Protocol

from .store import TaskStore


class TaskHandler(Protocol):
    task_type: str

    async def kickoff(
        self,
        server: Any,
        store: TaskStore,
        *,
        account_id: str,
        trigger: dict[str, Any],
    ) -> int | None:
        """Create/refresh today's instance for this task. Returns instance_id if ok."""

    async def run_attempt(
        self,
        server: Any,
        *,
        instance: dict[str, Any],
        task: dict[str, Any],
        now_ms: int,
    ) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
        """
        Returns (result_code, result_json, decision_code, decision_json).
        """


_handlers: dict[str, TaskHandler] = {}


def register_handler(handler: TaskHandler) -> None:
    key = str(getattr(handler, "task_type", "") or "").strip()
    if not key:
        raise ValueError("handler.task_type is required")
    _handlers[key] = handler


def get_handler(task_type: str) -> TaskHandler | None:
    return _handlers.get(str(task_type or "").strip())


def list_handlers() -> list[str]:
    return sorted(_handlers.keys())
