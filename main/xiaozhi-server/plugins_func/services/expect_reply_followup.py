from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from config.logger import setup_logging
from jinja2 import Template

from .registry import register_service

TAG = __name__
logger = setup_logging()


@dataclass(frozen=True)
class FollowupConfig:
    enabled: bool
    silence_seconds: float
    tick_seconds: float
    max_sentences: int
    max_per_session: int
    close_after_followup: bool


def _load_config(server) -> FollowupConfig:
    plugins = (getattr(server, "config", None) or {}).get("plugins", {})
    cfg = plugins.get("expect_reply_followup", {}) if isinstance(plugins, dict) else {}

    enabled = bool(cfg.get("enabled", False))
    silence_seconds = float(cfg.get("silence_seconds", 12.0))
    tick_seconds = float(cfg.get("tick_seconds", 1.0))
    max_sentences = int(cfg.get("max_sentences", 2))
    max_per_session = int(cfg.get("max_per_session", 1))
    close_after_followup = bool(cfg.get("close_after_followup", True))

    if max_sentences < 1:
        max_sentences = 1
    if max_sentences > 2:
        max_sentences = 2
    if max_per_session < 0:
        max_per_session = 0
    if silence_seconds < 1.0:
        silence_seconds = 1.0
    if tick_seconds < 0.2:
        tick_seconds = 0.2

    return FollowupConfig(
        enabled=enabled,
        silence_seconds=silence_seconds,
        tick_seconds=tick_seconds,
        max_sentences=max_sentences,
        max_per_session=max_per_session,
        close_after_followup=close_after_followup,
    )


FOLLOWUP_PROMPT_TEMPLATE = Template(
    """\
[System Prompt]
Your previous response ended with <Q>, indicating a need for a user answer. However, the user has remained silent.

Based on your current persona, please generate a "soft follow-up" to naturally conclude the interaction without applying pressure or asking further questions.

Hard Rules:
- Output ONLY plain text within 1 - {{ MAX_SENTENCES }} sentence(s).
- Output language should remain consistent with the setting defined in [Role].
- Do NOT use question marks (? / ？) or write any interrogative sentences.
- Do NOT express a tone of "waiting for" or "depending on" a user reply (e.g., avoid "I will wait for you," "Waiting for your response").
- Do NOT output any tags or control characters (including <Q>).
- Do NOT call tools; do NOT output JSON.


Context for Reference (Use to ensure natural flow; do not repeat content verbatim):
- Current Time: {{ NOW_TIME }}
- User's Last Input: {{ LAST_USER }}
- Your Last Input: {{ LAST_ASSISTANT }}
""".strip()
)


def _extract_last_contents(conn) -> tuple[str, str]:
    last_user = ""
    last_assistant = ""
    try:
        dialogue = getattr(getattr(conn, "dialogue", None), "dialogue", []) or []
        for m in reversed(dialogue):
            role = getattr(m, "role", None)
            content = getattr(m, "content", None)
            if not content:
                continue
            if not last_assistant and role == "assistant":
                last_assistant = str(content)
                continue
            if not last_user and role == "user":
                last_user = str(content)
                break
    except Exception:
        pass
    return last_user, last_assistant


@dataclass
class _Pending:
    started_at_ms: float
    user_activity_snapshot_ms: float
    sentence_id: str | None
    tts_stop_time_ms: float


@register_service("expect_reply_followup")
async def expect_reply_followup_service(server: Any) -> None:
    pending: dict[int, _Pending] = {}
    sent_count: dict[int, int] = {}
    last_cfg_repr: str | None = None

    while True:
        try:
            cfg = _load_config(server)
            cfg_repr = (
                f"enabled={cfg.enabled}, silence={cfg.silence_seconds}s, tick={cfg.tick_seconds}s, "
                f"max_sentences={cfg.max_sentences}, max_per_session={cfg.max_per_session}, "
                f"close_after_followup={cfg.close_after_followup}"
            )
            if cfg_repr != last_cfg_repr:
                logger.bind(tag=TAG).info(f"expect_reply_followup config: {cfg_repr}")
                last_cfg_repr = cfg_repr
            if not cfg.enabled or cfg.max_per_session <= 0:
                pending.clear()
                sent_count.clear()
                await asyncio.sleep(1.0)
                continue

            conns = getattr(server, "active_connections_by_device", {}) or {}
            if not conns:
                pending.clear()
                sent_count.clear()
                await asyncio.sleep(cfg.tick_seconds)
                continue

            now_ms = time.time() * 1000
            active_ids: set[int] = set()

            for device_id, conn in list(conns.items()):
                if not conn:
                    continue
                conn_id = id(conn)
                active_ids.add(conn_id)

                if getattr(conn, "stop_event", None) and conn.stop_event.is_set():
                    pending.pop(conn_id, None)
                    continue
                if getattr(conn, "close_after_chat", False):
                    pending.pop(conn_id, None)
                    continue
                if not getattr(conn, "tts", None) or not getattr(conn, "llm", None):
                    pending.pop(conn_id, None)
                    continue
                if getattr(conn, "client_is_speaking", False):
                    # 只在 TTS 结束后开始计时
                    continue
                if not getattr(conn, "llm_finish_task", True):
                    continue

                if not getattr(conn, "expect_user_reply", False):
                    pending.pop(conn_id, None)
                    continue
                # 仅对“用户参与过的会话”启用跟进：避免定时播报等服务端触发误入跟进流程。
                if float(getattr(conn, "last_user_activity_time", 0.0) or 0.0) <= 0.0:
                    pending.pop(conn_id, None)
                    continue

                if sent_count.get(conn_id, 0) >= cfg.max_per_session:
                    pending.pop(conn_id, None)
                    continue

                # 跟进计时从“本次播报结束(stop)”开始，而不是从“开始说话/开始生成”开始：
                # 否则 silence_seconds 会在 TTS 播放期间被消耗，导致播报一结束就立刻触发。
                current_sentence_id = getattr(conn, "sentence_id", None)
                last_stop_sentence_id = getattr(conn, "last_tts_sentence_id", None)
                last_stop_ms = float(getattr(conn, "last_tts_stop_time", 0.0) or 0.0)
                if (
                    not current_sentence_id
                    or last_stop_sentence_id != current_sentence_id
                    or last_stop_ms <= 0.0
                ):
                    pending.pop(conn_id, None)
                    continue

                p = pending.get(conn_id)
                if not p:
                    pending[conn_id] = _Pending(
                        started_at_ms=last_stop_ms,
                        user_activity_snapshot_ms=float(
                            getattr(conn, "last_user_activity_time", 0.0) or 0.0
                        ),
                        sentence_id=str(current_sentence_id),
                        tts_stop_time_ms=last_stop_ms,
                    )
                    logger.bind(tag=TAG).info(
                        f"expect_reply followup armed: device={device_id}, silence={cfg.silence_seconds}s"
                    )
                    continue

                # 如果句子 id 变了（新一轮输出），或 stop 时间变了（新一轮播报结束），重置计时。
                if (
                    p.sentence_id != str(current_sentence_id)
                    or abs(p.tts_stop_time_ms - last_stop_ms) > 0.5
                ):
                    pending[conn_id] = _Pending(
                        started_at_ms=last_stop_ms,
                        user_activity_snapshot_ms=float(
                            getattr(conn, "last_user_activity_time", 0.0) or 0.0
                        ),
                        sentence_id=str(current_sentence_id),
                        tts_stop_time_ms=last_stop_ms,
                    )
                    logger.bind(tag=TAG).info(
                        f"expect_reply followup re-armed(tts stop updated): device={device_id}, silence={cfg.silence_seconds}s"
                    )
                    continue

                # 如果用户有新输入，就取消本次跟进
                current_user_activity_ms = float(
                    getattr(conn, "last_user_activity_time", 0.0) or 0.0
                )
                if current_user_activity_ms != p.user_activity_snapshot_ms:
                    logger.bind(tag=TAG).info(
                        f"expect_reply followup canceled(user activity): device={device_id}"
                    )
                    pending.pop(conn_id, None)
                    continue

                if now_ms - p.started_at_ms < cfg.silence_seconds * 1000:
                    continue

                # 触发跟进：先解除期待，避免重复进入
                pending.pop(conn_id, None)
                conn.expect_user_reply = False
                sent_count[conn_id] = sent_count.get(conn_id, 0) + 1

                last_user, last_assistant = _extract_last_contents(conn)
                followup_prompt = FOLLOWUP_PROMPT_TEMPLATE.render(
                    NOW_TIME=datetime.now().strftime("%H:%M"),
                    LAST_USER=last_user,
                    LAST_ASSISTANT=last_assistant,
                    MAX_SENTENCES=cfg.max_sentences,
                )

                logger.bind(tag=TAG).info(
                    f"expect_reply followup trigger: device={device_id}, close_after={cfg.close_after_followup}"
                )

                if cfg.close_after_followup:
                    conn.close_after_chat = True

                # 使用 to_thread 避免阻塞服务循环
                await asyncio.to_thread(conn.chat, followup_prompt)

            # 清理已失活连接
            for conn_id in list(pending.keys()):
                if conn_id not in active_ids:
                    pending.pop(conn_id, None)
            for conn_id in list(sent_count.keys()):
                if conn_id not in active_ids:
                    sent_count.pop(conn_id, None)

        except Exception as loop_err:
            logger.bind(tag=TAG).opt(exception=True).error(
                f"expect_reply_followup loop error: {loop_err}"
            )

        await asyncio.sleep(cfg.tick_seconds if "cfg" in locals() else 1.0)
