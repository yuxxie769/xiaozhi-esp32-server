from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from jinja2 import Template

from plugins_func.functions.confirm_event import DEFAULT_VISION_TOOL_NAME, run_confirm_event
from core.task_engine.prompts.greeting_style import build_greeting_style_prompt

from ..registry import register_handler
from ..store import TaskStore


_NUDGE_TEMPLATE = Template(
    """\
Task: Wake-up Attempt

You are running a wake-up task attempt for the user.

Context:
- Planned time: {{ PLANNED_TIME }}
- Attempt: {{ ATTEMPT_NO }}
- Wake check: {{ WAKE_CHECK }}

Rules:
- If wake check is available, you MUST use it and must not fabricate anything beyond it.
- If wake check is unavailable, you MUST explicitly mention you couldn't get visual status this time.
- If wake check indicates the user is already awake, respond with a short acknowledgement / light greeting.
- If wake check indicates the user is not awake, check evidence first, WAKE UP the user based on your personality.
- Plain text only.
- 1–3 sentences. No questions. No waiting/dependency tone.
- Before final output, verify that your response is consistent with the evidence to ensure reliable and non-contradictory conclusions.
- Output language should remain consistent with the setting defined in [Role].
""".strip()
)

_WAKE_UP_FOLLOWUP_TEMPLATE = _NUDGE_TEMPLATE


def _now_date_key(now: datetime) -> str:
    return now.strftime("%Y-%m-%d")


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _json_loads(s: str | None) -> dict[str, Any]:
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _format_ms_hhmm(ms: int) -> str:
    try:
        return datetime.fromtimestamp(ms / 1000.0).strftime("%H:%M")
    except Exception:
        return ""


def _format_wake_up_check_result_text(
    *,
    result_code: str,
    result_json: dict[str, Any],
) -> str:
    code = str(result_code or "").strip() or "unknown"
    if code in ("ok", "unwake", "unknown", "nobody", "unavailable", "low_confidence"):
        flag = code
    else:
        flag = "unknown"

    awake = "unknown"
    confidence = 0.0
    evidence = ""

    if flag == "unavailable":
        evidence = str(result_json.get("error") or "unavailable").strip()
    else:
        awake = str(result_json.get("awake") or "unknown").strip().lower() or "unknown"
        confidence = _safe_float(result_json.get("confidence", 0.0), 0.0)
        evidence = str(result_json.get("evidence") or "").strip()

    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0
    return (
        f"wake_up_check_result({flag}): awake={awake}, confidence={confidence:.2f}, "
        f"evidence={evidence}"
    )


def _build_first_attempt_prompt(*, now: datetime, planned_time: str, wake_check_text: str) -> str:
    return build_greeting_style_prompt(
        slot="morning",
        now=now,
        planned_time=planned_time,
        extra_context=wake_check_text,
    )


def _build_followup_attempt_prompt(
    *,
    planned_time: str,
    attempt_no: int,
    wake_check_text: str,
) -> str:
    # NOTE: 这里预留“第二次及以后 attempt 使用另一套提示词”的入口。
    # 目前先复用旧的 wake_up nudge 模板，后续你可以直接替换 _WAKE_UP_FOLLOWUP_TEMPLATE 或本函数实现。
    return _WAKE_UP_FOLLOWUP_TEMPLATE.render(
        PLANNED_TIME=planned_time or "",
        ATTEMPT_NO=str(int(attempt_no)),
        WAKE_CHECK=wake_check_text,
    )


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out.get(k) or {}, v)
        else:
            out[k] = v
    return out


def _load_engine_task_type_cfg(server, task_type: str) -> dict[str, Any]:
    plugins = getattr(server, "config", {}) or {}
    plugins = plugins.get("plugins", {}) if isinstance(plugins, dict) else {}
    te = plugins.get("task_engine", {}) if isinstance(plugins, dict) else {}
    task_types = te.get("task_types", {}) if isinstance(te, dict) else {}
    if not isinstance(task_types, dict):
        task_types = {}
    base = task_types.get("default") if isinstance(task_types.get("default"), dict) else {}
    specific = task_types.get(task_type) if isinstance(task_types.get(task_type), dict) else {}
    merged = _deep_merge_dict(base, specific)

    # Legacy support: plugins.task_engine.<task_type> or plugins.task_engine.wake_up
    legacy = te.get(task_type, {}) if isinstance(te.get(task_type), dict) else {}
    if task_type == "wake_up" and not legacy:
        legacy = te.get("wake_up", {}) if isinstance(te.get("wake_up"), dict) else {}
    if legacy:
        merged = _deep_merge_dict(merged, legacy)
    return merged if isinstance(merged, dict) else {}


def _load_confirm_event_cfg(server) -> dict[str, Any]:
    plugins = getattr(server, "config", {}) or {}
    plugins = plugins.get("plugins", {}) if isinstance(plugins, dict) else {}
    ce = plugins.get("confirm_event", {}) if isinstance(plugins, dict) else {}
    return ce if isinstance(ce, dict) else {}


@dataclass(frozen=True)
class _CheckCfg:
    tool_name: str
    timeout_seconds: float
    retries: int


def _get_check_cfg(server) -> _CheckCfg:
    ce_cfg = _load_confirm_event_cfg(server)

    tool_name = str(
        ce_cfg.get("tool_name", DEFAULT_VISION_TOOL_NAME)
        or DEFAULT_VISION_TOOL_NAME
    )
    timeout_seconds = _safe_float(
        ce_cfg.get("timeout_seconds", 10.0), 10.0
    )
    retries = _safe_int(ce_cfg.get("retries", 2), 2)
    if retries < 0:
        retries = 0
    if timeout_seconds < 1.0:
        timeout_seconds = 1.0
    return _CheckCfg(tool_name=tool_name, timeout_seconds=timeout_seconds, retries=retries)


class WakeUpTaskHandler:
    task_type = "wake_up"

    def default_policy(self, server) -> dict[str, Any]:
        wake_cfg = _load_engine_task_type_cfg(server, "wake_up")
        return {
            "cooldown_sec": _safe_int(wake_cfg.get("cooldown_sec", 300), 300),
            "offline_retry_sec": _safe_int(wake_cfg.get("offline_retry_sec", 300), 300),
            "max_attempts": _safe_int(wake_cfg.get("max_attempts", 6), 6),
            "window_minutes": _safe_int(wake_cfg.get("window_minutes", 30), 30),
            "nudge_enabled": bool(wake_cfg.get("nudge_enabled", True)),
        }

    async def kickoff(
        self,
        server: Any,
        store: TaskStore,
        *,
        account_id: str,
        trigger: dict[str, Any],
    ) -> int | None:
        now = datetime.now()
        now_ms = int(trigger.get("now_ms") or int(now.timestamp() * 1000))

        device_id = str(trigger.get("device_id") or "").strip()
        planned_at_ms = _safe_int(trigger.get("planned_at_ms", now_ms), now_ms)

        task = store.get_task(account_id=account_id, task_type=self.task_type)
        if not task:
            base_policy = self.default_policy(server)
            if device_id:
                base_policy["device_id"] = device_id
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
            policy_patch: dict[str, Any] = {}
            if device_id:
                policy_patch["device_id"] = device_id
            for k, v in (defaults or {}).items():
                if k not in existing_policy:
                    policy_patch[k] = v
            if policy_patch:
                store.upsert_task(
                    account_id=account_id,
                    task_type=self.task_type,
                    enabled=bool(int(task.get("enabled", 1)) == 1),
                    policy=policy_patch,
                    now_ms=now_ms,
                )

        # Create/refresh today's instance (instance_key: YYYY-MM-DD).
        instance_key = str(trigger.get("instance_key") or _now_date_key(now))

        # Compute window from planned time.
        task = store.get_task(account_id=account_id, task_type=self.task_type) or {}
        policy = _json_loads(task.get("policy_json"))
        defaults = self.default_policy(server)
        cooldown_sec = _safe_int(policy.get("cooldown_sec", defaults["cooldown_sec"]), defaults["cooldown_sec"])
        window_minutes = _safe_int(policy.get("window_minutes", defaults["window_minutes"]), defaults["window_minutes"])
        offline_retry_sec = _safe_int(
            policy.get("offline_retry_sec", defaults.get("offline_retry_sec", cooldown_sec)), cooldown_sec
        )

        if cooldown_sec < 1:
            cooldown_sec = 1
        if offline_retry_sec < 1:
            offline_retry_sec = 1
        if window_minutes < 1:
            window_minutes = 1

        interval_sec = min(int(cooldown_sec), int(offline_retry_sec))
        if interval_sec < 1:
            interval_sec = 1
        window_sec = int(window_minutes) * 60
        max_runs = max(1, (window_sec + interval_sec - 1) // interval_sec)

        window_start_at_ms = planned_at_ms
        window_end_at_ms = planned_at_ms + int(window_minutes) * 60 * 1000

        instance = store.ensure_instance(
            account_id=account_id,
            task_type=self.task_type,
            instance_key=instance_key,
            status="PENDING",
            planned_at_ms=planned_at_ms,
            window_start_at_ms=window_start_at_ms,
            window_end_at_ms=window_end_at_ms,
            next_action_at_ms=now_ms,  # run now
            max_runs=int(max_runs),
            now_ms=now_ms,
        )
        return int(instance.get("instance_id") or 0) or None

    async def run_attempt(
        self,
        server: Any,
        *,
        instance: dict[str, Any],
        task: dict[str, Any],
        now_ms: int,
    ) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
        now_dt = datetime.fromtimestamp(int(now_ms) / 1000.0)
        policy = _json_loads(task.get("policy_json"))
        defaults = self.default_policy(server)

        cooldown_sec = _safe_int(policy.get("cooldown_sec", defaults["cooldown_sec"]), defaults["cooldown_sec"])
        offline_retry_sec = _safe_int(
            policy.get("offline_retry_sec", defaults.get("offline_retry_sec", cooldown_sec)), cooldown_sec
        )
        max_attempts = _safe_int(policy.get("max_attempts", defaults["max_attempts"]), defaults["max_attempts"])
        nudge_enabled = bool(policy.get("nudge_enabled", defaults["nudge_enabled"]))

        if cooldown_sec < 1:
            cooldown_sec = 1
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

        device_id = str(policy.get("device_id") or "").strip()
        if not device_id:
            return (
                "invalid_policy",
                {"reason": "missing_device_id"},
                "abandon",
                {"consume_attempt": False},
            )
        conns = getattr(server, "active_connections_by_device", {}) or {}
        conn = conns.get(device_id) if device_id else None
        if not conn:
            # Device offline: don't consume an attempt; try again soon (bounded by window end).
            next_action_at_ms = now_ms + int(offline_retry_sec) * 1000
            if window_end_at_ms > 0:
                next_action_at_ms = min(next_action_at_ms, window_end_at_ms)
            return (
                "offline",
                {"reason": "device_offline", "device_id": device_id},
                "skip",
                {"consume_attempt": False, "next_action_at_ms": next_action_at_ms},
            )

        check_cfg = _get_check_cfg(server)
        confirm = await run_confirm_event(
            conn,
            "wake_up",
            tool_name=check_cfg.tool_name,
            timeout_seconds=check_cfg.timeout_seconds,
            retries=check_cfg.retries,
        )

        result_json: dict[str, Any] = {}
        result_code = "unknown"
        ok = bool(isinstance(confirm, dict) and confirm.get("ok"))
        if not ok:
            result_code = "unavailable"
            result_json = {
                "error": str((confirm or {}).get("error") or "unavailable"),
                "raw": (confirm or {}).get("raw", ""),
            }
        else:
            data = confirm.get("data") if isinstance(confirm.get("data"), dict) else {}
            awake_val = data.get("awake")
            confidence = _safe_float(data.get("confidence", 0.0), 0.0)
            evidence = str(data.get("evidence", "") or "").strip()
            awake_str = (
                "true"
                if awake_val is True
                else "false"
                if awake_val is False
                else str(awake_val or "unknown").strip().lower()
            )
            result_json = {
                "awake": awake_str,
                "confidence": confidence,
                "evidence": evidence,
            }

            if awake_str == "false":
                result_code = "unwake"
            elif awake_str == "nobody":
                result_code = "nobody"
            elif awake_str == "unknown":
                result_code = "unknown"
            else:
                result_code = "ok"

        if result_code == "ok":
            decision_json: dict[str, Any] = {"consume_attempt": True}
            should_chat = bool(nudge_enabled) and (attempt_no == 1 or result_code in ("ok", "unwake"))
            if should_chat:
                planned_at_ms = _safe_int(instance.get("planned_at_ms", now_ms), now_ms)
                planned_hhmm = _format_ms_hhmm(planned_at_ms)
                wake_check_text = _format_wake_up_check_result_text(
                    result_code=result_code,
                    result_json=result_json,
                )
                if attempt_no == 1:
                    decision_json["nudge_prompt"] = _build_first_attempt_prompt(
                        now=now_dt,
                        planned_time=planned_hhmm or "",
                        wake_check_text=wake_check_text,
                    )
                else:
                    decision_json["nudge_prompt"] = _build_followup_attempt_prompt(
                        planned_time=planned_hhmm or "",
                        attempt_no=attempt_no,
                        wake_check_text=wake_check_text,
                    )
            return (result_code, result_json, "complete", decision_json)

        next_action_at_ms = now_ms + int(cooldown_sec) * 1000
        if window_end_at_ms > 0:
            next_action_at_ms = min(next_action_at_ms, window_end_at_ms)

        decision_json: dict[str, Any] = {
            "consume_attempt": True,
            "next_action_at_ms": next_action_at_ms,
            "cooldown_sec": int(cooldown_sec),
        }

        # Speak policy:
        # - attempt #1: always speak (even if wake_check is unavailable/uncertain)
        # - attempt #2+: speak only when vision confirms awake/unwake (ok/unwake)
        should_chat = bool(nudge_enabled) and (attempt_no == 1 or result_code in ("ok", "unwake"))
        if should_chat:
            planned_at_ms = _safe_int(instance.get("planned_at_ms", now_ms), now_ms)
            planned_hhmm = _format_ms_hhmm(planned_at_ms)
            wake_check_text = _format_wake_up_check_result_text(
                result_code=result_code,
                result_json=result_json,
            )
            if attempt_no == 1:
                decision_json["nudge_prompt"] = _build_first_attempt_prompt(
                    now=now_dt,
                    planned_time=planned_hhmm or "",
                    wake_check_text=wake_check_text,
                )
            else:
                decision_json["nudge_prompt"] = _build_followup_attempt_prompt(
                    planned_time=planned_hhmm or "",
                    attempt_no=attempt_no,
                    wake_check_text=wake_check_text,
                )

        return (
            result_code,
            result_json,
            "retry",
            decision_json,
        )


register_handler(WakeUpTaskHandler())
