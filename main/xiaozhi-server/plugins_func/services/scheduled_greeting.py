from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from config.logger import setup_logging
from jinja2 import Template
from .registry import register_service

TAG = __name__
logger = setup_logging()

# 单一母提示词模板：你后续只需要替换这段即可，不用改代码逻辑。
# 变量约定（Jinja2）：
# - {{ NOW_TIME }}: 本次任务的“计划时刻”(HH:MM)，来自 schedule 配置
# - {{ MAIN_HINT }}: 该 slot 的主意图（写死在代码里）
# - {{ OPTIONAL_HINTS }}: 从候选池随机抽取的风格提示（可能为空）
# - {{ CONTEXT_VARS }}: 额外上下文信息（例如：工作日/周末、天气等；目前只注入工作日/周末）
SCHEDULED_TASK_BASE_PROMPT_TEMPLATE = """\
Task: Natural Engagement Generation

Based on the predefined [Role], generate a proactive, non-coercive engagement message.

Input integration (weave into one coherent passage; you may integrate the elements in any order; do not mechanically stitch blocks):
	1) Time anchor:
	- MUST include the time anchor (this run: {{ NOW_TIME }}) at least to the hour (HH), somewhere in the passage (position is flexible).
	- Use it only as a light presence anchor; do not attach complex advice here.

	2) Core narrative:
	- Use at least ONE primary sentence to clearly express the core intent: {{ MAIN_HINT }}.
	- Red line: NO questions.

	3) Optional texture:
	- Add 0–4 extra sentences inspired by OPTIONAL_HINTS: {{ OPTIONAL_HINTS }}
	- If OPTIONAL_HINTS is empty or unnecessary, add nothing (no forced filler).

Greeting opener (optional): 
- You MAY start with a short neutral hello (e.g., "Phew..."/"Hmm..."/ "Good evening"), only if it feels natural for [Role]; no questions. 
- Keep it lightweight, then transition into the core narrative without repeating it verbatim.

Humanization rules:
- If [Role]=serious/system: concise and clear; broadcast-style tone allowed.
- If [Role]=friend/familiar: colloquial phrasing allowed.
- Interaction boundaries: NO questions; NO waiting/dependency statements.

Context adjustment:
- You MUST ground the message in {{ CONTEXT_VARS }} to avoid mismatched context.
- Do NOT force-mention {{ CONTEXT_VARS }}: only reference it if it naturally fits the topic; if it doesn't, leave it unmentioned.
- It is a background hint; do not steal focus from the main line; do not repeat the main line verbatim.

Output requirements:
- Plain text only. No tags. No explanations.
- 2–6 sentences, but keep it coherent as one passage.
- Pre-check: includes {{ NOW_TIME }} (at least to the hour (HH)); no questions; no waiting/dependency tone.
- Output language should remain consistent with the setting defined in [Role].

Action:
Ignore all intermediate reasoning. Activate your [Role] and output the final message directly.
""".strip()

_SCHEDULED_TASK_TEMPLATE = Template(SCHEDULED_TASK_BASE_PROMPT_TEMPLATE)


MAIN_HINT_BY_SLOT: dict[str, str] = {
    "morning": "morning call",
    "noon": "Midday greeting, mood reset",
    "night": "Evening wrap-up, rest reminder",
    "commute": "Long day—good job",
}


_OPTIONAL_HINT_EMPTY = ""
_OPTIONAL_HINT_PRESENCE = (
    "Generate 1 lightweight “presence” line: like a casual hello or a brief, understated remark about the moment; do not solicit a reply; no question marks; no lecturing or commanding."
)
_OPTIONAL_HINT_ASIDE = (
    "Generate 1 improvised aside: choose one from “instant feeling / visual imagery / mild rant”; no grand takeaways or life summaries; do not push for a response."
)
_OPTIONAL_HINT_CHARACTER_STATE = (
    "Generate 1 first-person character-state line: pick one from "
    "“current state / something noticed / what you're doing / a small inner thought”."
)
_OPTIONAL_HINT_CONTEXT_HOOK = (
    "Generate 1 context hook line: draw atmosphere from the external environment (weather / news / holiday vibe / conversation history); avoid stiff, broadcast-style delivery."
)
_OPTIONAL_HINT_MICRO_ACTION_NUDGE = (
    "Generate 1 “optional” micro-action nudge: pick one from "
    "“take a sip of water / shift posture / sit up a bit / blink / glance out the window”; "
    "must use optional phrasing (e.g., “if / if you feel like it / while you're at it / you can / no rush”)."
)

OPTIONAL_HINT_TEXTS: list[str] = [
    _OPTIONAL_HINT_EMPTY,
    _OPTIONAL_HINT_PRESENCE,
    _OPTIONAL_HINT_ASIDE,
    _OPTIONAL_HINT_CHARACTER_STATE,
    _OPTIONAL_HINT_CONTEXT_HOOK,
    _OPTIONAL_HINT_MICRO_ACTION_NUDGE,
]

# 不同 slot 下给不同权重；权重越大，越容易被抽中。
# 若 slot 未配置，则回退到 "default"；若配置异常，则最终回退到空字符串。
OPTIONAL_HINT_WEIGHTS_BY_SLOT: dict[str, list[int]] = {
    "default": [3, 5, 5, 5, 5, 5],
    "morning": [3, 5, 5, 5, 8, 5],
    "noon": [3, 5, 10, 10, 5, 3],
    "night": [3, 5, 5, 5, 5, 0],
    "commute": [3, 5, 10, 10, 5, 0],
}


@dataclass(frozen=True)
class ScheduleConfig:
    enabled: bool
    target_devices: list[str]
    tick_seconds: float
    times: dict[str, tuple[int, int]]  # slot -> (hour, minute)
    quiet_start: tuple[int, int]
    quiet_end: tuple[int, int]


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

    return ScheduleConfig(
        enabled=enabled,
        target_devices=list(target_devices),
        tick_seconds=tick_seconds,
        times=times,
        quiet_start=quiet_start,
        quiet_end=quiet_end,
    )


def _account_key(conn) -> str:
    return (
        getattr(conn, "client_id", None)
        or (getattr(conn, "headers", {}) or {}).get("client-id")
        or getattr(conn, "device_id", None)
        or ""
    )


def _pick_optional_hint(slot: str) -> str:
    weights = OPTIONAL_HINT_WEIGHTS_BY_SLOT.get(slot) or OPTIONAL_HINT_WEIGHTS_BY_SLOT.get(
        "default"
    )
    if not weights or len(weights) != len(OPTIONAL_HINT_TEXTS):
        return ""
    safe_weights = [max(0, int(w)) for w in weights]
    if sum(safe_weights) <= 0:
        return ""
    return random.choices(OPTIONAL_HINT_TEXTS, weights=safe_weights, k=1)[0]


def _build_context_vars(now: datetime) -> str:
    parts: list[str] = []
    parts.append("工作日" if now.weekday() < 5 else "周末")
    return "；".join([p for p in parts if p])


def _build_prompt(slot: str, now: datetime, planned_time: str) -> str:
    main_hint = MAIN_HINT_BY_SLOT.get(slot) or "简短问候"
    optional_hint = _pick_optional_hint(slot)
    context_vars = _build_context_vars(now)

    return _SCHEDULED_TASK_TEMPLATE.render(
        NOW_TIME=planned_time or now.strftime("%H:%M"),
        MAIN_HINT=main_hint,
        OPTIONAL_HINTS=optional_hint,
        CONTEXT_VARS=context_vars,
    )


@register_service("scheduled_greeting")
async def scheduled_greeting_service(server: Any) -> None:
    sent_today: dict[tuple[str, str, str], bool] = {}
    last_cfg_repr: str | None = None

    while True:
        try:
            cfg = _load_config(server)
            cfg_repr = (
                f"enabled={cfg.enabled}, targets={cfg.target_devices}, "
                f"schedule={cfg.times}, quiet={cfg.quiet_start}-{cfg.quiet_end}, "
                f"tick={cfg.tick_seconds}"
            )
            if cfg_repr != last_cfg_repr:
                logger.bind(tag=TAG).info(f"scheduled_greeting config: {cfg_repr}")
                last_cfg_repr = cfg_repr
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

            if _is_in_quiet_hours(now, cfg.quiet_start, cfg.quiet_end):
                await asyncio.sleep(cfg.tick_seconds)
                continue

            # Determine which slots (if any) should fire at this minute
            slots_to_fire = [
                slot
                for slot, (h, m) in cfg.times.items()
                if now.hour == h and now.minute == m
            ]
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
                    prompt = _build_prompt(slot_to_fire, now, planned_time)
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
