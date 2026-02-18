from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from config.logger import setup_logging
from .registry import register_service
from plugins_func.functions.confirm_event import wake_check
from core.task_engine.facade import kickoff_wake_up_from_greeting
from core.task_engine.prompts.greeting_style import build_greeting_style_prompt

TAG = __name__
logger = setup_logging()

# 提示词生成逻辑已抽离到 core.task_engine.prompts.greeting_style


@dataclass(frozen=True)
class ScheduleConfig:
    enabled: bool
    target_devices: list[str]
    tick_seconds: float
    times: dict[str, tuple[int, int]]  # slot -> (hour, minute)
    quiet_start: tuple[int, int]
    quiet_end: tuple[int, int]
    wake_check: "WakeCheckConfig"


@dataclass(frozen=True)
class WakeCheckConfig:
    enabled: bool
    tool_name: str
    min_confidence: float
    timeout_seconds: float
    retries: int


def _parse_hhmm(raw: str) -> tuple[int, int]:
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty time")
    hh, mm = s.split(":")
    h = int(hh)
    m = int(mm)
    if h == 24 and m == 0:
        return 0, 0
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"invalid time: {raw}")
    return h, m


def _time_to_minutes(h: int, m: int) -> int:
    return h * 60 + m


def _is_in_quiet_hours(now: datetime, start: tuple[int, int], end: tuple[int, int]) -> bool:
    now_min = _time_to_minutes(now.hour, now.minute)
    start_min = _time_to_minutes(*start)
    end_min = _time_to_minutes(*end)
    if start_min <= end_min:
        return start_min <= now_min <= end_min
    return now_min >= start_min or now_min <= end_min


def _load_config(server) -> ScheduleConfig:
    plugins = (server.config or {}).get("plugins", {})
    cfg = plugins.get("scheduled_greeting", {}) if isinstance(plugins, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    target_devices = cfg.get("target_devices", []) or []
    tick_seconds = float(cfg.get("tick_seconds", 10.0))

    raw_schedule = cfg.get("schedule", {}) or {}
    times: dict[str, tuple[int, int]] = {}
    if isinstance(raw_schedule, dict) and len(raw_schedule) > 0:
        for slot, hhmm in raw_schedule.items():
            try:
                times[str(slot)] = _parse_hhmm(str(hhmm))
            except Exception:
                continue
    else:
        # Defaults (compatible with older configs)
        times = {
            "morning": _parse_hhmm("09:30"),
            "noon": _parse_hhmm("12:00"),
            "night": _parse_hhmm("24:00"),
        }

    quiet = cfg.get("quiet_hours", {}) or {}
    quiet_start = _parse_hhmm(quiet.get("start", "01:00"))
    quiet_end = _parse_hhmm(quiet.get("end", "09:29"))

    # Prefer the new, scenario-specific key `morning_wake_check` to avoid confusion with the
    # wrapper tool name `wake_check`. Keep legacy `wake_check` for backward compatibility.
    wake_cfg = (
        (cfg.get("morning_wake_check", None) if isinstance(cfg, dict) else None)
        or (cfg.get("wake_check", {}) if isinstance(cfg, dict) else {})
    )
    enabled_wake = bool((wake_cfg or {}).get("enabled", False))
    tool_name = str((wake_cfg or {}).get("tool_name", "vision_assistant") or "vision_assistant")
    try:
        min_confidence = float((wake_cfg or {}).get("min_confidence", 0.6))
    except Exception:
        min_confidence = 0.6
    try:
        timeout_seconds = float((wake_cfg or {}).get("timeout_seconds", 10.0))
    except Exception:
        timeout_seconds = 10.0
    try:
        retries = int((wake_cfg or {}).get("retries", 1))
    except Exception:
        retries = 1
    if min_confidence < 0.0:
        min_confidence = 0.0
    if min_confidence > 1.0:
        min_confidence = 1.0
    if timeout_seconds < 1.0:
        timeout_seconds = 1.0
    if retries < 0:
        retries = 0

    return ScheduleConfig(
        enabled=enabled,
        target_devices=list(target_devices),
        tick_seconds=tick_seconds,
        times=times,
        quiet_start=quiet_start,
        quiet_end=quiet_end,
        wake_check=WakeCheckConfig(
            enabled=enabled_wake,
            tool_name=tool_name,
            min_confidence=min_confidence,
            timeout_seconds=timeout_seconds,
            retries=retries,
        ),
    )


def _account_key(conn) -> str:
    return (
        getattr(conn, "client_id", None)
        or (getattr(conn, "headers", {}) or {}).get("client-id")
        or getattr(conn, "device_id", None)
        or ""
    )


async def _maybe_wake_check(conn, cfg: WakeCheckConfig) -> str | None:
    if not cfg.enabled:
        return None
    try:
        resp = await wake_check(conn)
        text = getattr(resp, "result", None)
        if not isinstance(text, str) or not text.strip():
            return None
        text = text.strip()
        if not text.startswith("wake_up_check_result("):
            logger.bind(tag=TAG).info("wake_check skipped: unexpected format")
            return None
        if text.startswith("wake_up_check_result(unavailable)"):
            logger.bind(tag=TAG).info("wake_check skipped: unavailable")
            return None
        return text
    except Exception as e:
        logger.bind(tag=TAG).warning(f"wake_check failed: {e}")
        return None


@register_service("scheduled_greeting")
async def scheduled_greeting_service(server: Any) -> None:
    sent_today: dict[tuple[str, str, str], bool] = {}
    last_cfg_repr: str | None = None
    last_cfg_warn_repr: str | None = None

    while True:
        try:
            cfg = _load_config(server)
            cfg_repr = (
                f"enabled={cfg.enabled}, targets={cfg.target_devices}, "
                f"schedule={cfg.times}, quiet={cfg.quiet_start}-{cfg.quiet_end}, "
                f"tick={cfg.tick_seconds}, "
                f"wake_check={cfg.wake_check.enabled}/{cfg.wake_check.tool_name}/"
                f"{cfg.wake_check.min_confidence}/{cfg.wake_check.timeout_seconds}s/"
                f"retries={cfg.wake_check.retries}"
            )
            if cfg_repr != last_cfg_repr:
                logger.bind(tag=TAG).info(f"scheduled_greeting config: {cfg_repr}")
                last_cfg_repr = cfg_repr
                last_cfg_warn_repr = None

            # Warn once per config snapshot when any scheduled slot is inside quiet hours.
            if cfg.enabled and cfg.target_devices and last_cfg_warn_repr != cfg_repr:
                warned = False
                for slot, (h, m) in (cfg.times or {}).items():
                    try:
                        slot_min = _time_to_minutes(int(h), int(m))
                        start_min = _time_to_minutes(*cfg.quiet_start)
                        end_min = _time_to_minutes(*cfg.quiet_end)
                        in_quiet = (
                            start_min <= end_min and start_min <= slot_min <= end_min
                        ) or (
                            start_min > end_min and (slot_min >= start_min or slot_min <= end_min)
                        )
                        if in_quiet:
                            warned = True
                            logger.bind(tag=TAG).warning(
                                f"scheduled_greeting slot inside quiet_hours: slot={slot}, "
                                f"time={int(h):02d}:{int(m):02d}, quiet={cfg.quiet_start}-{cfg.quiet_end} (will be skipped)"
                            )
                    except Exception:
                        continue
                if warned:
                    last_cfg_warn_repr = cfg_repr
            if not cfg.enabled or not cfg.target_devices:
                await asyncio.sleep(1.0)
                continue

            conns = getattr(server, "active_connections_by_device", {}) or {}
            if not conns:
                await asyncio.sleep(cfg.tick_seconds)
                continue

            now = datetime.now()
            today_key = now.date().isoformat()
            if sent_today:
                sent_today = {
                    k: v for k, v in sent_today.items() if k[1] == today_key
                }

            # Determine which slots (if any) should fire at this minute
            slots_to_fire = [
                slot
                for slot, (h, m) in cfg.times.items()
                if now.hour == h and now.minute == m
            ]

            if _is_in_quiet_hours(now, cfg.quiet_start, cfg.quiet_end):
                if slots_to_fire:
                    logger.bind(tag=TAG).info(
                        f"定点报时跳过(静默时段): slot={','.join(slots_to_fire)}, time={now:%H:%M}, quiet={cfg.quiet_start}-{cfg.quiet_end}"
                    )
                await asyncio.sleep(cfg.tick_seconds)
                continue
            if not slots_to_fire:
                await asyncio.sleep(cfg.tick_seconds)
                continue

            logger.bind(tag=TAG).info(
                f"定点报时检查: slot={','.join(slots_to_fire)}, time={now:%H:%M}, targets={len(cfg.target_devices)}"
            )

            for slot_to_fire in slots_to_fire:
                for device_id in list(cfg.target_devices):
                    conn = conns.get(device_id)
                    if not conn:
                        logger.bind(tag=TAG).info(
                            f"定点报时跳过(离线): device={device_id}"
                        )
                        continue

                    account_id = _account_key(conn) or device_id
                    sent_key = (account_id, today_key, slot_to_fire)
                    if sent_today.get(sent_key):
                        logger.bind(tag=TAG).info(
                            f"定点报时跳过(已播): device={device_id}, account={account_id}, slot={slot_to_fire}"
                        )
                        continue

                    if getattr(conn, "stop_event", None) and conn.stop_event.is_set():
                        logger.bind(tag=TAG).info(
                            f"定点报时跳过(关闭中): device={device_id}"
                        )
                        continue
                    if not getattr(conn, "tts", None) or not getattr(conn, "llm", None):
                        logger.bind(tag=TAG).info(
                            f"定点报时跳过(未就绪): device={device_id}"
                        )
                        continue
                    if getattr(conn, "client_is_speaking", False):
                        logger.bind(tag=TAG).info(
                            f"定点报时跳过(正在播报): device={device_id}"
                        )
                        continue
                    if not getattr(conn, "llm_finish_task", True):
                        logger.bind(tag=TAG).info(
                            f"定点报时跳过(LLM忙): device={device_id}"
                        )
                        continue

                    h, m = cfg.times.get(slot_to_fire, (now.hour, now.minute))
                    planned_time = f"{int(h):02d}:{int(m):02d}"
                    extra_context = None
                    if slot_to_fire == "morning":
                        planned_at_ms = int(
                            now.replace(hour=int(h), minute=int(m), second=0, microsecond=0).timestamp()
                            * 1000
                        )
                        kicked = await kickoff_wake_up_from_greeting(
                            server, conn, planned_at_ms=planned_at_ms
                        )
                        if kicked:
                            sent_today[sent_key] = True
                            logger.bind(tag=TAG).info(
                                f"定点报时触发(task_engine): slot={slot_to_fire}, device={device_id}, account={account_id}"
                            )
                            continue
                        logger.bind(tag=TAG).warning(
                            f"定点报时回退到保底问候: reason=task_engine_kickoff_failed, "
                            f"slot={slot_to_fire}, device={device_id}, account={account_id}, "
                            f"planned_time={planned_time}, planned_at_ms={planned_at_ms}"
                        )
                        extra_context = await _maybe_wake_check(conn, cfg.wake_check)
                    prompt = build_greeting_style_prompt(
                        slot=slot_to_fire,
                        now=now,
                        planned_time=planned_time,
                        extra_context=extra_context,
                    )
                    try:
                        # Mark as sent before calling chat to prevent same-minute re-entry.
                        sent_today[sent_key] = True
                        await asyncio.to_thread(conn.chat, prompt)
                        logger.bind(tag=TAG).info(
                            f"定点报时触发: slot={slot_to_fire}, device={device_id}, account={account_id}"
                        )
                    except Exception as e:
                        sent_today.pop(sent_key, None)
                        logger.bind(tag=TAG).opt(exception=True).error(
                            f"定点报时触发失败: {e}"
                        )

        except Exception as loop_err:
            logger.bind(tag=TAG).opt(exception=True).error(
                f"scheduled_greeting loop error: {loop_err}"
            )

        await asyncio.sleep(max(1.0, cfg.tick_seconds if "cfg" in locals() else 10.0))
