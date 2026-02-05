from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from config.logger import setup_logging
from plugins_func.register import Action, ActionResponse, ToolType, register_function

from core.providers.tools.mcp_endpoint.mcp_endpoint_handler import call_mcp_endpoint_tool

TAG = __name__
logger = setup_logging()


DEFAULT_VISION_TOOL_NAME = "vision_assistant"


@dataclass(frozen=True)
class _EventSpec:
    event: str
    command: str
    required_keys: tuple[str, ...]


_WAKE_UP_COMMAND = """\
根据当前画面，判断用户是否已经起床（已坐起/离床/明显清醒）。
可能会出现没人的情况
只输出一个JSON对象，禁止输出任何其他文字/markdown/代码块/解释。
JSON格式必须严格为：
{"awake":true/false/unknown/nobody,"confidence":0-1,"evidence":"一句话依据"}
confidence为0到1的小数。
evidence用中文且不超过25字。
""".strip()


EVENT_SPECS: dict[str, _EventSpec] = {
    "wake_up": _EventSpec(
        event="wake_up",
        command=_WAKE_UP_COMMAND,
        required_keys=("awake", "confidence", "evidence"),
    ),
}


CONFIRM_EVENT_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "confirm_event",
        "description": "基于视觉MCP工具进行事件确认（固定枚举事件），返回结构化JSON结果，供后续对话使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "event": {
                    "type": "string",
                    "description": "要确认的事件类型（固定枚举）。",
                    "enum": sorted(list(EVENT_SPECS.keys())),
                },
            },
            "required": ["event"],
        },
    },
}

WAKE_CHECK_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "wake_check",
        "description": "确认用户是否已起床/清醒（confirm_event 的便捷别名）。输出固定格式：wake_up_check_result(flag): awake=..., confidence=..., evidence=...",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _try_parse_json_dict(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    extracted = _extract_first_json_object(text)
    if not extracted:
        return None
    try:
        obj = json.loads(extracted)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _normalize_wake_up(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    if not {"awake", "confidence", "evidence"}.issubset(payload.keys()):
        return None

    awake_raw = payload.get("awake")
    if isinstance(awake_raw, bool):
        awake = awake_raw
    elif isinstance(awake_raw, str):
        s = awake_raw.strip().lower()
        if s in ("unknown", "nobody"):
            awake = s
        elif s in ("true", "1", "yes", "y"):
            awake = True
        elif s in ("false", "0", "no", "n"):
            awake = False
        else:
            return None
    else:
        return None

    conf_raw = payload.get("confidence")
    try:
        confidence = float(conf_raw)
    except Exception:
        confidence = 0.0
    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0

    evidence = str(payload.get("evidence", "") or "").strip()
    if len(evidence) > 80:
        evidence = evidence[:80]

    return {"awake": awake, "confidence": confidence, "evidence": evidence}


def _format_wake_up_check_result(
    payload: dict[str, Any] | None,
    *,
    ok: bool,
    min_confidence: float,
    error: str | None = None,
) -> str:
    awake = "unknown"
    confidence = 0.0
    evidence = ""
    if ok and isinstance(payload, dict):
        awake_val = payload.get("awake")
        if isinstance(awake_val, bool):
            awake = "true" if awake_val else "false"
        elif isinstance(awake_val, str):
            awake = awake_val.strip().lower() or "unknown"
        elif awake_val is not None:
            awake = str(awake_val).strip().lower() or "unknown"
        try:
            confidence = float(payload.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        evidence = str(payload.get("evidence", "") or "").strip()
    else:
        evidence = (str(error or "") or "unavailable").strip()

    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0
    if min_confidence < 0.0:
        min_confidence = 0.0
    if min_confidence > 1.0:
        min_confidence = 1.0

    if not ok:
        flag = "unavailable"
    elif awake == "false":
        flag = "unwake"
    elif awake in ("unknown", "nobody"):
        flag = awake
    elif confidence < float(min_confidence):
        flag = "low_confidence"
    else:
        flag = "ok"

    return (
        f"wake_up_check_result({flag}): awake={awake}, confidence={confidence:.2f}, "
        f"evidence={evidence}"
    )


def _pick_prompt_key(properties: dict[str, Any]) -> str | None:
    for k in ("prompt", "query", "text", "instruction", "input", "message", "content", "question"):
        v = properties.get(k)
        if isinstance(v, dict) and v.get("type") == "string":
            return k
    # Fallback: single string property
    string_keys = [
        k for k, v in properties.items() if isinstance(v, dict) and v.get("type") == "string"
    ]
    if len(string_keys) == 1:
        return string_keys[0]
    return None


def _pick_command_enum_value(values: list[Any]) -> str | None:
    enum_vals = [v for v in values if isinstance(v, str) and v.strip()]
    if not enum_vals:
        return None

    def score(s: str) -> int:
        t = s.lower()
        if any(w in t for w in ("analyze", "analysis", "ask", "query", "chat", "describe", "vision")):
            return 100
        if any(w in t for w in ("detect", "classify", "infer", "judge")):
            return 80
        if any(w in t for w in ("capture", "snapshot", "photo", "image")):
            return 60
        return 10

    return sorted(enum_vals, key=score, reverse=True)[0]


def _build_vision_tool_args(conn, tool_name: str, prompt_text: str) -> list[dict[str, Any]]:
    """
    Try to adapt to different MCP tool input schemas.
    - Some tools accept {"command": <free-text>} (no enum).
    - Some tools accept {"command": <enum>, "prompt"/"query"/"text": <free-text>}.
    """
    mcp_client = getattr(conn, "mcp_endpoint_client", None)
    tool = getattr(mcp_client, "tools", {}).get(tool_name) if mcp_client else None
    schema = (tool or {}).get("inputSchema", {}) if isinstance(tool, dict) else {}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = schema.get("required", []) if isinstance(schema, dict) else []
    if not isinstance(properties, dict):
        properties = {}
    if not isinstance(required, list):
        required = []

    prompt_key = _pick_prompt_key(properties)
    cmd_spec = properties.get("command") if isinstance(properties.get("command"), dict) else None
    cmd_enum = cmd_spec.get("enum") if isinstance(cmd_spec, dict) else None
    cmd_value = _pick_command_enum_value(cmd_enum) if isinstance(cmd_enum, list) else None

    candidates: list[dict[str, Any]] = []
    if cmd_value and prompt_key:
        candidates.append({"command": cmd_value, prompt_key: prompt_text})
    if prompt_key and "command" not in required:
        candidates.append({prompt_key: prompt_text})
    if "command" in properties and not cmd_value:
        candidates.append({"command": prompt_text})

    if not candidates:
        candidates = [{"command": prompt_text}]

    # De-dup while keeping order
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in candidates:
        key = json.dumps(c, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _classify_wake_up_from_text(text: str) -> dict[str, Any]:
    """
    Heuristic classification based on vision tool's free-form description.
    This is a fallback when the vision tool cannot emit the expected JSON schema.
    """
    t = (text or "").strip()
    snippet = " ".join(t.split())
    if len(snippet) > 60:
        snippet = snippet[:60]
    lower = t.lower()

    nobody_markers = [
        "nobody",
        "no person",
        "no one",
        "empty room",
        "无人",
        "没人",
        "空",
        "没有人",
        "未检测到人",
        "未发现人",
    ]
    asleep_markers = [
        "sleeping",
        "asleep",
        "lying in bed",
        "eyes closed",
        "闭眼",
        "睡着",
        "在睡",
        "躺在床上",
        "躺在床",
        "被子",
        "枕头",
    ]
    awake_markers = [
        "holding",
        "standing",
        "sitting",
        "talking",
        "smiling",
        "wearing",
        "glasses",
        "earbuds",
        "shaving",
        "刷牙",
        "洗漱",
        "刮胡",
        "在走动",
        "站着",
        "坐着",
        "起床",
    ]

    for m in nobody_markers:
        if m in lower or m in t:
            return {"awake": "nobody", "confidence": 0.7, "evidence": snippet}
    for m in asleep_markers:
        if m in lower or m in t:
            return {"awake": False, "confidence": 0.6, "evidence": snippet}
    for m in awake_markers:
        if m in lower or m in t:
            return {"awake": True, "confidence": 0.6, "evidence": snippet}

    if not t:
        return {"awake": "unknown", "confidence": 0.0, "evidence": ""}
    return {"awake": "unknown", "confidence": 0.2, "evidence": snippet}


async def _vision_tool_available(conn, tool_name: str) -> bool:
    mcp_client = getattr(conn, "mcp_endpoint_client", None)
    if not mcp_client:
        return False
    if not getattr(mcp_client, "ready", False):
        return False
    return mcp_client.has_tool(tool_name)


async def run_confirm_event(
    conn,
    event: str,
    *,
    tool_name: str = DEFAULT_VISION_TOOL_NAME,
    timeout_seconds: float = 10.0,
    retries: int = 1,
) -> dict[str, Any]:
    spec = EVENT_SPECS.get(str(event))
    if not spec:
        return {"event": str(event), "ok": False, "error": "unsupported_event"}

    if not await _vision_tool_available(conn, tool_name):
        return {
            "event": spec.event,
            "ok": False,
            "error": "vision_tool_unavailable",
            "tool_name": tool_name,
        }

    last_raw: str = ""
    last_err: str = ""
    attempts = max(1, int(retries) + 1)
    for attempt in range(attempts):
        prompt_text = spec.command
        if attempt > 0 and last_raw:
            prompt_text = (
                prompt_text
                + "\n\n上次输出不合规，请严格按要求只输出一个JSON对象：\n"
                + last_raw[:500]
            )
        try:
            for args in _build_vision_tool_args(conn, tool_name, prompt_text):
                raw_text = await call_mcp_endpoint_tool(
                    conn.mcp_endpoint_client,
                    tool_name,
                    args,
                    timeout=int(max(1, timeout_seconds)),
                )
                last_raw = str(raw_text or "")

                parsed = _try_parse_json_dict(last_raw)
                # Some tools wrap output as {"success":true,"result":"..."}.
                wrapped_result_text: str | None = None
                if parsed and "success" in parsed:
                    if parsed.get("success") is False:
                        last_err = str(parsed.get("error") or "vision_tool_error")
                        continue
                    if isinstance(parsed.get("result"), str):
                        wrapped_result_text = parsed["result"]
                        nested = _try_parse_json_dict(wrapped_result_text)
                        if nested:
                            parsed = nested

                # Some tools may return {"error":"..."} without "success".
                if (
                    spec.event == "wake_up"
                    and isinstance(parsed, dict)
                    and parsed.get("error")
                    and not any(k in parsed for k in ("awake", "confidence", "evidence", "result", "success"))
                ):
                    last_err = str(parsed.get("error") or "vision_tool_error")
                    continue

                if spec.event == "wake_up" and parsed:
                    normalized = _normalize_wake_up(parsed)
                    if normalized is not None:
                        return {
                            "event": spec.event,
                            "ok": True,
                            "data": normalized,
                            "raw": last_raw[:2000],
                        }

                # Fallback: if we only got a free-form description, use heuristic classification.
                if spec.event == "wake_up":
                    desc = None
                    if wrapped_result_text:
                        desc = wrapped_result_text
                    elif isinstance(parsed, dict) and isinstance(parsed.get("result"), str):
                        desc = parsed.get("result")
                    else:
                        desc = last_raw
                    normalized = _classify_wake_up_from_text(str(desc or ""))
                    return {
                        "event": spec.event,
                        "ok": True,
                        "data": normalized,
                        "raw": last_raw[:2000],
                    }

                last_err = "invalid_json_schema"
        except Exception as e:
            last_err = str(e) or "vision_call_failed"

    return {"event": spec.event, "ok": False, "error": last_err, "raw": last_raw[:2000]}


@register_function("confirm_event", CONFIRM_EVENT_FUNCTION_DESC, ToolType.SYSTEM_CTL)
async def confirm_event(conn, event: str):
    plugins_cfg = conn.config.get("plugins", {}) if isinstance(conn.config, dict) else {}
    cfg = plugins_cfg.get("confirm_event", {}) if isinstance(plugins_cfg, dict) else {}

    tool_name = str(cfg.get("tool_name", DEFAULT_VISION_TOOL_NAME) or DEFAULT_VISION_TOOL_NAME)
    try:
        timeout_seconds = float(cfg.get("timeout_seconds", 10.0))
    except Exception:
        timeout_seconds = 10.0
    try:
        retries = int(cfg.get("retries", 1))
    except Exception:
        retries = 1

    result = await run_confirm_event(
        conn,
        event,
        tool_name=tool_name,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    logger.bind(tag=TAG).info(f"confirm_event done: event={event}, ok={result.get('ok')}")

    # Always REQLLM: tool output should be consumed by the model, not spoken directly.
    return ActionResponse(
        action=Action.REQLLM,
        result=json.dumps(result, ensure_ascii=False),
        response=None,
    )


@register_function("wake_check", WAKE_CHECK_FUNCTION_DESC, ToolType.SYSTEM_CTL)
async def wake_check(conn):
    plugins_cfg = conn.config.get("plugins", {}) if isinstance(conn.config, dict) else {}
    cfg = plugins_cfg.get("confirm_event", {}) if isinstance(plugins_cfg, dict) else {}
    wrappers_cfg = cfg.get("wrappers", {}) if isinstance(cfg, dict) else {}
    wrapper_cfg = wrappers_cfg.get("wake_check", {}) if isinstance(wrappers_cfg, dict) else {}
    if not isinstance(wrapper_cfg, dict):
        wrapper_cfg = {}
    sched_cfg = (
        # Prefer the new `morning_wake_check` (scenario overrides) over legacy `wake_check`.
        (plugins_cfg.get("scheduled_greeting", {}) or {}).get("morning_wake_check")
        or (plugins_cfg.get("scheduled_greeting", {}) or {}).get("wake_check", {})
        if isinstance(plugins_cfg, dict)
        else {}
    )
    if not isinstance(sched_cfg, dict):
        sched_cfg = {}

    tool_name = str(
        wrapper_cfg.get("tool_name")
        or sched_cfg.get("tool_name")
        or cfg.get("tool_name", DEFAULT_VISION_TOOL_NAME)
        or DEFAULT_VISION_TOOL_NAME
    )
    try:
        timeout_seconds = float(
            wrapper_cfg.get(
                "timeout_seconds",
                sched_cfg.get("timeout_seconds", cfg.get("timeout_seconds", 10.0)),
            )
        )
    except Exception:
        timeout_seconds = 10.0
    try:
        retries = int(
            wrapper_cfg.get("retries", sched_cfg.get("retries", cfg.get("retries", 1)))
        )
    except Exception:
        retries = 1
    try:
        min_confidence = float(
            wrapper_cfg.get(
                "min_confidence",
                sched_cfg.get("min_confidence", cfg.get("min_confidence", 0.6)),
            )
        )
    except Exception:
        min_confidence = 0.6

    result = await run_confirm_event(
        conn,
        "wake_up",
        tool_name=tool_name,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )

    ok = bool(isinstance(result, dict) and result.get("ok"))
    if ok:
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        text = _format_wake_up_check_result(
            data, ok=True, min_confidence=min_confidence, error=None
        )
    else:
        err = ""
        if isinstance(result, dict):
            err = str(result.get("error") or "")
        text = _format_wake_up_check_result(
            None, ok=False, min_confidence=min_confidence, error=err or "unavailable"
        )
    logger.bind(tag=TAG).info(f"wake_check done: {text}")

    return ActionResponse(
        action=Action.REQLLM,
        result=text,
        response=None,
    )
