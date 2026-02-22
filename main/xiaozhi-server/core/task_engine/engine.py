from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from config.logger import setup_logging

from .registry import get_handler
from .store import TaskStore

# Ensure handlers are registered.
from .handlers import wake_up as _wake_up_handler  # noqa: F401
from .handlers import ha_event as _ha_event_handler  # noqa: F401

TAG = __name__
logger = setup_logging()

try:
    from plugins_func.services.proactive_speech_gate import proactive_chat as _proactive_chat
except Exception:
    _proactive_chat = None

_ATTEMPTS_CONTEXT_LIMIT = 20


@dataclass(frozen=True)
class TaskEngineConfig:
    enabled: bool
    db_path: str
    tick_seconds: float
    batch_size: int


def _load_engine_config(server) -> TaskEngineConfig:
    plugins = (getattr(server, "config", None) or {}).get("plugins", {})
    cfg = plugins.get("task_engine", {}) if isinstance(plugins, dict) else {}

    enabled = bool(cfg.get("enabled", False))
    db_path = str(cfg.get("db_path", "data/tasks.db") or "data/tasks.db")
    try:
        tick_seconds = float(cfg.get("tick_seconds", 5.0))
    except Exception:
        tick_seconds = 5.0
    try:
        batch_size = int(cfg.get("batch_size", 10))
    except Exception:
        batch_size = 10

    if tick_seconds < 0.2:
        tick_seconds = 0.2
    if batch_size < 1:
        batch_size = 1

    return TaskEngineConfig(
        enabled=enabled,
        db_path=db_path,
        tick_seconds=tick_seconds,
        batch_size=batch_size,
    )


def _now_ms() -> int:
    return int(datetime.now().timestamp() * 1000)


def _parse_policy(policy_json: str | None) -> dict[str, Any]:
    if not policy_json:
        return {}
    try:
        obj = json.loads(policy_json)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _can_chat(conn) -> bool:
    if not conn:
        return False
    if getattr(conn, "stop_event", None) and conn.stop_event.is_set():
        return False
    if getattr(conn, "close_after_chat", False):
        return False
    if not getattr(conn, "tts", None) or not getattr(conn, "llm", None):
        return False
    if getattr(conn, "client_is_speaking", False):
        return False
    if not getattr(conn, "llm_finish_task", True):
        return False
    return True


def _try_json_load_or_keep(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return raw


def _build_attempts_context(store: TaskStore, *, instance_id: int, limit: int) -> tuple[list[dict[str, Any]], bool]:
    rows = store.list_attempts(instance_id=instance_id, limit=max(1, int(limit)) + 1)
    truncated = len(rows) > limit
    rows = rows[:limit]

    attempts: list[dict[str, Any]] = []
    for row in rows:
        attempts.append(
            {
                "attempt_id": int(row.get("attempt_id") or 0),
                "at_ms": int(row.get("at_ms") or 0),
                "result_code": str(row.get("result_code") or ""),
                "decision_code": str(row.get("decision_code") or ""),
                "result_json": _try_json_load_or_keep(row.get("result_json")),
                "decision_json": _try_json_load_or_keep(row.get("decision_json")),
            }
        )
    return attempts, truncated


def _inject_attempts_context_into_prompt(prompt: str, attempts: list[dict[str, Any]]) -> str:
    try:
        attempts_json = json.dumps(attempts, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        attempts_json = "[]"
    return f"[Task Engine Attempts]\n{attempts_json}\n[/Task Engine Attempts]\n\n{prompt}"


async def run_due_once(server: Any) -> None:
    cfg = _load_engine_config(server)
    if not cfg.enabled:
        return
    store = TaskStore(cfg.db_path)
    store.init_schema()
    await _run_due_instances(server, store, now_ms=_now_ms(), limit=cfg.batch_size)


async def _run_due_instances(server: Any, store: TaskStore, *, now_ms: int, limit: int) -> None:
    rows = store.list_due_instances(now_ms=now_ms, limit=limit)
    if not rows:
        return
    for row in rows:
        try:
            await run_instance_row(server, store, row, now_ms=now_ms)
        except Exception as e:
            logger.bind(tag=TAG).opt(exception=True).error(f"run_instance_row failed: {e}")


async def run_instance_row(server: Any, store: TaskStore, row: dict[str, Any], *, now_ms: int) -> None:
    task_type = str(row.get("task_type") or "").strip()
    handler = get_handler(task_type)
    if not handler:
        instance_id = int(row.get("instance_id") or 0)
        if instance_id:
            store.append_attempt(
                instance_id=instance_id,
                at_ms=now_ms,
                result_code="unsupported_task",
                result_json={"task_type": task_type},
                decision_code="abandon",
                decision_json={"consume_attempt": False},
            )
            store.set_instance_status(instance_id=instance_id, status="ABANDONED", now_ms=now_ms)
        return

    # Build task/instance views.
    instance = {k: row.get(k) for k in row.keys() if not k.startswith("task_")}
    task = {
        "account_id": row.get("account_id"),
        "task_type": task_type,
        "enabled": int(row.get("task_enabled") or 0),
        "policy_json": row.get("task_policy_json") or "{}",
    }

    instance_id = int(instance.get("instance_id") or 0)
    if not instance_id:
        return

    max_runs = int(instance.get("max_runs") or 0)
    run_count = int(instance.get("run_count") or 0)
    if max_runs > 0 and run_count >= max_runs:
        store.append_attempt(
            instance_id=instance_id,
            at_ms=now_ms,
            result_code="max_runs",
            result_json={"reason": "max_runs", "run_count": run_count, "max_runs": max_runs},
            decision_code="abandon",
            decision_json={"consume_attempt": False},
        )
        store.set_instance_status(instance_id=instance_id, status="ABANDONED", now_ms=now_ms)
        return

    store.increment_run_count(instance_id=instance_id, now_ms=now_ms)

    result_code, result_json, decision_code, decision_json = await handler.run_attempt(
        server, instance=instance, task=task, now_ms=now_ms
    )

    # Keep large prompt out of DB; it is only used to trigger a chat.
    nudge_prompt = ""
    decision_json_to_store: dict[str, Any] = decision_json if isinstance(decision_json, dict) else {}
    if isinstance(decision_json_to_store.get("nudge_prompt"), str):
        nudge_prompt = str(decision_json_to_store.get("nudge_prompt") or "").strip()
        if nudge_prompt:
            decision_json_to_store = dict(decision_json_to_store)
            decision_json_to_store.pop("nudge_prompt", None)

    consume_attempt = bool((decision_json or {}).get("consume_attempt", True))
    if consume_attempt:
        store.increment_attempt_count(instance_id=instance_id, now_ms=now_ms)

    store.append_attempt(
        instance_id=instance_id,
        at_ms=now_ms,
        result_code=result_code,
        result_json=result_json,
        decision_code=decision_code,
        decision_json=decision_json_to_store,
    )

    # Update instance status/next_action.
    next_action_at_ms = (decision_json or {}).get("next_action_at_ms")
    try:
        next_action_at_ms = int(next_action_at_ms) if next_action_at_ms is not None else None
    except Exception:
        next_action_at_ms = None

    if decision_code == "complete":
        store.set_instance_status(
            instance_id=instance_id,
            status="COMPLETED",
            now_ms=now_ms,
            next_action_at_ms=next_action_at_ms,
        )
    elif decision_code == "abandon":
        store.set_instance_status(
            instance_id=instance_id,
            status="ABANDONED",
            now_ms=now_ms,
            next_action_at_ms=next_action_at_ms,
        )
        return
    elif decision_code == "pause":
        store.set_instance_status(
            instance_id=instance_id,
            status="PAUSED",
            now_ms=now_ms,
            next_action_at_ms=next_action_at_ms,
        )
        return
    else:
        # retry / skip
        current_status = str(instance.get("status") or "").strip() or "PENDING"
        new_status = "IN_PROGRESS" if decision_code == "retry" else current_status
        if next_action_at_ms is None:
            next_action_at_ms = int(instance.get("next_action_at_ms") or now_ms)
        store.set_instance_status(
            instance_id=instance_id,
            status=new_status,
            now_ms=now_ms,
            next_action_at_ms=next_action_at_ms,
        )

    if not nudge_prompt or decision_code not in ("retry", "complete"):
        return

    chat_device_id = str((decision_json or {}).get("chat_device_id") or "").strip()
    if chat_device_id:
        device_id = chat_device_id
    else:
        policy = _parse_policy(task.get("policy_json"))
        device_id = str(policy.get("device_id") or "").strip()
    conn = (getattr(server, "active_connections_by_device", {}) or {}).get(device_id)
    if not _can_chat(conn):
        return

    nudge_meta: dict[str, Any] = {
        "task_type": task_type,
        "instance_id": instance_id,
        "result_code": result_code,
        "decision_code": decision_code,
    }
    nudge_prompt_to_send = nudge_prompt
    try:
        attempts, truncated = _build_attempts_context(
            store, instance_id=instance_id, limit=_ATTEMPTS_CONTEXT_LIMIT
        )
        nudge_meta["attempts"] = attempts
        nudge_prompt_to_send = _inject_attempts_context_into_prompt(nudge_prompt, attempts)
        logger.bind(tag=TAG).info(
            f"task_engine attempts context: instance_id={instance_id}, "
            f"attempts_count={len(attempts)}, truncated={truncated}"
        )
    except Exception as e:
        logger.bind(tag=TAG).warning(
            f"task_engine attempts context build failed: instance_id={instance_id}, error={e}"
        )

    try:
        scenario = f"task_engine/{task_type or 'unknown'}"
        if _proactive_chat is None:
            await asyncio.to_thread(conn.chat, nudge_prompt_to_send)
        else:
            try:
                await _proactive_chat(
                    server,
                    conn,
                    nudge_prompt_to_send,
                    scenario=scenario,
                    meta=nudge_meta,
                    gate_tool_call_mode=("off" if task_type == "wake_up" else "allow"),
                )
            except Exception as gate_err:
                logger.bind(tag=TAG).warning(
                    f"proactive_chat unavailable during task_engine nudge, fallback chat: {gate_err}"
                )
                await asyncio.to_thread(conn.chat, nudge_prompt_to_send)
    except Exception as e:
        logger.bind(tag=TAG).warning(f"nudge chat failed: {e}")


async def run_instance_id(server: Any, *, instance_id: int) -> None:
    cfg = _load_engine_config(server)
    if not cfg.enabled:
        return
    store = TaskStore(cfg.db_path)
    store.init_schema()
    inst = store.get_instance_by_id(int(instance_id))
    if not inst:
        return
    task = store.get_task(account_id=str(inst.get("account_id") or ""), task_type=str(inst.get("task_type") or ""))
    if not task or int(task.get("enabled") or 0) != 1:
        return

    row = dict(inst)
    row["task_enabled"] = int(task.get("enabled") or 0)
    row["task_policy_json"] = task.get("policy_json") or "{}"
    await run_instance_row(server, store, row, now_ms=_now_ms())


async def task_engine_service_loop(server: Any) -> None:
    logger.bind(tag=TAG).info(
        f"proactive_speech_gate bridge: {'ready' if _proactive_chat is not None else 'unavailable'}"
    )
    store: TaskStore | None = None
    last_cfg_repr: str | None = None
    while True:
        cfg = _load_engine_config(server)
        cfg_repr = (
            f"enabled={cfg.enabled}, db={cfg.db_path}, tick={cfg.tick_seconds}s, batch={cfg.batch_size}"
        )
        if cfg_repr != last_cfg_repr:
            logger.bind(tag=TAG).info(f"task_engine config: {cfg_repr}")
            last_cfg_repr = cfg_repr

        if not cfg.enabled:
            store = None
            await asyncio.sleep(1.0)
            continue

        if store is None or store.db_path != cfg.db_path:
            store = TaskStore(cfg.db_path)
            store.init_schema()

        try:
            await _run_due_instances(server, store, now_ms=_now_ms(), limit=cfg.batch_size)
        except Exception as loop_err:
            logger.bind(tag=TAG).opt(exception=True).error(f"task_engine loop error: {loop_err}")

        await asyncio.sleep(cfg.tick_seconds)
