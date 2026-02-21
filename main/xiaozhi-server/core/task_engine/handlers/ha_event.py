from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .wake_up import _json_loads, _load_engine_task_type_cfg, _safe_int
from ..registry import register_handler
from ..store import TaskStore


def _now_ms() -> int:
    return int(datetime.now().timestamp() * 1000)


def _can_chat(conn: Any) -> bool:
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


def _json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return "{}"


@dataclass(frozen=True)
class _Policy:
    device_id: str
    offline_retry_sec: int
    window_minutes: int
    max_attempts: int


class HaEventTaskHandler:
    task_type = "ha_event"

    def default_policy(self, server) -> dict[str, Any]:
        cfg = _load_engine_task_type_cfg(server, "ha_event")
        device_id = str(cfg.get("device_id") or "").strip()
        offline_retry_sec = _safe_int(cfg.get("offline_retry_sec", 15), 15)
        window_minutes = _safe_int(cfg.get("window_minutes", 5), 5)
        max_attempts = _safe_int(cfg.get("max_attempts", 10), 10)

        if offline_retry_sec < 1:
            offline_retry_sec = 1
        if window_minutes < 1:
            window_minutes = 1
        if max_attempts < 0:
            max_attempts = 0

        return {
            "device_id": device_id,
            "offline_retry_sec": int(offline_retry_sec),
            "window_minutes": int(window_minutes),
            "max_attempts": int(max_attempts),
        }

    async def kickoff(
        self,
        server: Any,
        store: TaskStore,
        *,
        account_id: str,
        trigger: dict[str, Any],
    ) -> int | None:
        now_ms = _safe_int(trigger.get("now_ms", _now_ms()), _now_ms())
        event_id = str(trigger.get("event_id") or "").strip()
        if not event_id:
            return None

        title = str(trigger.get("title") or "").strip()
        instruction = str(trigger.get("instruction") or "").strip()
        data = trigger.get("data", None)
        fallback_device_id = str(trigger.get("device_id") or "").strip()

        task = store.get_task(account_id=account_id, task_type=self.task_type)
        if not task:
            base_policy = self.default_policy(server)
            if fallback_device_id and not str(base_policy.get("device_id") or "").strip():
                base_policy["device_id"] = fallback_device_id
            store.upsert_task(
                account_id=account_id,
                task_type=self.task_type,
                enabled=True,
                policy=base_policy,
                now_ms=now_ms,
            )
        else:
            defaults = self.default_policy(server)
            existing_policy = _json_loads(task.get("policy_json"))
            patch: dict[str, Any] = {}
            for k, v in defaults.items():
                if k not in existing_policy:
                    patch[k] = v
            if fallback_device_id and not str(existing_policy.get("device_id") or "").strip():
                patch["device_id"] = fallback_device_id
            if patch:
                store.upsert_task(
                    account_id=account_id,
                    task_type=self.task_type,
                    enabled=bool(int(task.get("enabled", 1)) == 1),
                    policy=patch,
                    now_ms=now_ms,
                )

        # Build retry window.
        task = store.get_task(account_id=account_id, task_type=self.task_type) or {}
        policy = _json_loads(task.get("policy_json"))
        defaults = self.default_policy(server)
        offline_retry_sec = _safe_int(
            policy.get("offline_retry_sec", defaults["offline_retry_sec"]), defaults["offline_retry_sec"]
        )
        window_minutes = _safe_int(
            policy.get("window_minutes", defaults["window_minutes"]), defaults["window_minutes"]
        )
        if offline_retry_sec < 1:
            offline_retry_sec = 1
        if window_minutes < 1:
            window_minutes = 1

        interval_sec = max(1, int(offline_retry_sec))
        window_sec = int(window_minutes) * 60
        max_runs = max(1, (window_sec + interval_sec - 1) // interval_sec)

        window_start_at_ms = now_ms
        window_end_at_ms = now_ms + int(window_minutes) * 60 * 1000

        instance = store.ensure_instance(
            account_id=account_id,
            task_type=self.task_type,
            instance_key=event_id,
            status="PENDING",
            planned_at_ms=now_ms,
            window_start_at_ms=window_start_at_ms,
            window_end_at_ms=window_end_at_ms,
            next_action_at_ms=now_ms,
            max_runs=int(max_runs),
            now_ms=now_ms,
        )
        instance_id = int(instance.get("instance_id") or 0) or 0
        if not instance_id:
            return None

        ctx = {
            "v": 1,
            "event_id": event_id,
            "title": title,
            "instruction": instruction,
            "data": data,
            "device_id": fallback_device_id,
        }
        store.set_instance_context(instance_id=instance_id, context=ctx, now_ms=now_ms)
        return instance_id

    async def run_attempt(
        self,
        server: Any,
        *,
        instance: dict[str, Any],
        task: dict[str, Any],
        now_ms: int,
    ) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
        policy = _json_loads(task.get("policy_json"))
        defaults = self.default_policy(server)

        offline_retry_sec = _safe_int(
            policy.get("offline_retry_sec", defaults["offline_retry_sec"]), defaults["offline_retry_sec"]
        )
        max_attempts = _safe_int(policy.get("max_attempts", defaults["max_attempts"]), defaults["max_attempts"])
        if offline_retry_sec < 1:
            offline_retry_sec = 1
        if max_attempts < 0:
            max_attempts = 0

        window_end_at_ms = _safe_int(instance.get("window_end_at_ms", 0), 0)
        attempt_count = _safe_int(instance.get("attempt_count", 0), 0)
        attempt_no = attempt_count + 1

        if window_end_at_ms > 0 and now_ms >= window_end_at_ms:
            return (
                "window_end",
                {"reason": "window_end", "now_ms": now_ms, "window_end_at_ms": window_end_at_ms},
                "abandon",
                {"consume_attempt": False},
            )
        if max_attempts >= 0 and attempt_no > max_attempts:
            return (
                "max_attempts",
                {"reason": "max_attempts", "attempt_no": attempt_no, "max_attempts": max_attempts},
                "abandon",
                {"consume_attempt": False},
            )

        ctx = {}
        try:
            ctx = json.loads(instance.get("context_json") or "{}")
            if not isinstance(ctx, dict):
                ctx = {}
        except Exception:
            ctx = {}

        device_id = str(policy.get("device_id") or "").strip()
        if not device_id:
            device_id = str(ctx.get("device_id") or "").strip()
        if not device_id:
            conns = getattr(server, "active_connections_by_device", {}) or {}
            if isinstance(conns, dict):
                for k, v in conns.items():
                    if k and v:
                        device_id = str(k)
                        break
        if not device_id:
            next_action_at_ms = now_ms + int(offline_retry_sec) * 1000
            if window_end_at_ms > 0:
                next_action_at_ms = min(next_action_at_ms, window_end_at_ms)
            return (
                "not_ready",
                {"reason": "missing_device_id"},
                "skip",
                {"consume_attempt": False, "next_action_at_ms": next_action_at_ms},
            )

        conns = getattr(server, "active_connections_by_device", {}) or {}
        conn = conns.get(device_id) if device_id else None
        if not _can_chat(conn):
            next_action_at_ms = now_ms + int(offline_retry_sec) * 1000
            if window_end_at_ms > 0:
                next_action_at_ms = min(next_action_at_ms, window_end_at_ms)
            return (
                "not_ready",
                {"reason": "device_not_ready", "device_id": device_id},
                "skip",
                {"consume_attempt": False, "next_action_at_ms": next_action_at_ms},
            )

        instruction = str(ctx.get("instruction") or "").strip()
        data = ctx.get("data", None)

        if not instruction:
            return (
                "invalid_context",
                {"reason": "missing_instruction"},
                "abandon",
                {"consume_attempt": False},
            )

        prompt = f"instruction: {instruction}"
        if data is not None and data != {}:
            prompt += f"\ndata: {_json_dumps(data)}"

        return (
            "ok",
            {"reason": "ok"},
            "complete",
            {"nudge_prompt": prompt, "chat_device_id": device_id},
        )


register_handler(HaEventTaskHandler())
