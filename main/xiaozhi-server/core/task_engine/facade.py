from __future__ import annotations

from datetime import datetime
from typing import Any

from config.logger import setup_logging

from .registry import get_handler
from .store import TaskStore

TAG = __name__
logger = setup_logging()


def _engine_enabled(server) -> tuple[bool, str]:
    plugins = (getattr(server, "config", None) or {}).get("plugins", {})
    cfg = plugins.get("task_engine", {}) if isinstance(plugins, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    db_path = str(cfg.get("db_path", "data/tasks.db") or "data/tasks.db")
    return enabled, db_path


def _account_key(conn) -> str:
    return (
        getattr(conn, "client_id", None)
        or (getattr(conn, "headers", {}) or {}).get("client-id")
        or getattr(conn, "device_id", None)
        or ""
    )


def _now_ms() -> int:
    return int(datetime.now().timestamp() * 1000)


async def kickoff_wake_up_from_greeting(server: Any, conn: Any, *, planned_at_ms: int) -> bool:
    enabled, db_path = _engine_enabled(server)
    if not enabled:
        return False
    handler = get_handler("wake_up")
    if not handler:
        return False

    account_id = _account_key(conn)
    if not account_id:
        return False

    device_id = getattr(conn, "device_id", None) or ""
    store = TaskStore(db_path)
    store.init_schema()

    trigger = {
        "source": "scheduled_greeting",
        "device_id": str(device_id or ""),
        "planned_at_ms": int(planned_at_ms),
        "instance_key": datetime.now().strftime("%Y-%m-%d"),
        "now_ms": _now_ms(),
    }
    try:
        instance_id = await handler.kickoff(server, store, account_id=account_id, trigger=trigger)
        if not instance_id:
            return False
        return True
    except Exception as e:
        logger.bind(tag=TAG).warning(f"kickoff_wake_up_from_greeting failed: {e}")
        return False
