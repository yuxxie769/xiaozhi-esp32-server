from __future__ import annotations

from datetime import datetime
from typing import Any

from config.logger import setup_logging

from .registry import get_handler
from .store import TaskStore
from .handlers import wake_up as _wake_up_handler  # noqa: F401
from .handlers import ha_event as _ha_event_handler  # noqa: F401

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


def _pick_fallback_device_id(server: Any) -> str:
    conns = getattr(server, "active_connections_by_device", {}) or {}
    if not isinstance(conns, dict) or not conns:
        return ""
    # Current single-device-first behavior: pick the first online connection.
    for device_id, conn in conns.items():
        if not device_id:
            continue
        if not conn:
            continue
        return str(device_id)
    return ""


async def kickoff_wake_up_from_greeting(server: Any, conn: Any, *, planned_at_ms: int) -> bool:
    enabled, db_path = _engine_enabled(server)
    account_id =  "user"  #_account_key(conn)
    device_id = str(getattr(conn, "device_id", None) or "")
    planned_at_ms = int(planned_at_ms)
    instance_key = datetime.now().strftime("%Y-%m-%d")

    if not enabled:
        logger.bind(tag=TAG).info(
            f"kickoff_wake_up_from_greeting skipped: reason=task_engine_disabled, "
            f"account={account_id or '-'}, device={device_id or '-'}"
        )
        return False
 
    handler = get_handler("wake_up")
    if not handler:
        logger.bind(tag=TAG).warning(
            "kickoff_wake_up_from_greeting skipped: reason=handler_not_registered, "
            "task_type=wake_up"
        )
        return False

    if not account_id:
        logger.bind(tag=TAG).warning(
            f"kickoff_wake_up_from_greeting skipped: reason=empty_account_id, device={device_id or '-'}"
        )
        return False

    store = TaskStore(db_path)
    store.init_schema()

    trigger = {
        "source": "scheduled_greeting",
        "device_id": str(device_id or ""),
        "planned_at_ms": planned_at_ms,
        "instance_key": instance_key,
        "now_ms": _now_ms(),
    }

    logger.bind(tag=TAG).info(
        f"kickoff_wake_up_from_greeting start: account={account_id}, device={device_id or '-'}, "
        f"instance_key={instance_key}, planned_at_ms={planned_at_ms}"
    )
    try:
        instance_id = await handler.kickoff(server, store, account_id=account_id, trigger=trigger)
        if not instance_id:
            logger.bind(tag=TAG).warning(
                f"kickoff_wake_up_from_greeting failed: reason=empty_instance_id, "
                f"account={account_id}, device={device_id or '-'}, instance_key={instance_key}"
            )
            return False
        logger.bind(tag=TAG).info(
            f"kickoff_wake_up_from_greeting success: account={account_id}, device={device_id or '-'}, "
            f"instance_id={instance_id}, instance_key={instance_key}"
        )
        return True
    except Exception as e:
        logger.bind(tag=TAG).opt(exception=True).warning(
            f"kickoff_wake_up_from_greeting exception: account={account_id}, "
            f"device={device_id or '-'}, instance_key={instance_key}, error={e}"
        )
        return False


async def kickoff_ha_event_from_state_hub(
    server: Any,
    *,
    event_id: str,
    title: str,
    instruction: str,
    data: Any,
    now_ms: int,
) -> int | None:
    enabled, db_path = _engine_enabled(server)
    if not enabled:
        return None

    handler = get_handler("ha_event")
    if not handler:
        return None

    store = TaskStore(db_path)
    store.init_schema()

    trigger = {
        "source": "state_hub",
        "event_id": str(event_id or "").strip(),
        "title": str(title or "").strip(),
        "instruction": str(instruction or "").strip(),
        "data": data,
        "device_id": _pick_fallback_device_id(server),
        "now_ms": int(now_ms),
    }
    if not trigger["event_id"] or not trigger["instruction"]:
        return None

    account_id = "user"
    return await handler.kickoff(server, store, account_id=account_id, trigger=trigger)
