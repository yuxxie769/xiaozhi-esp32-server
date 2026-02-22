from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

_POLICY_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="proactive_policy")
_policy_inflight_lock = threading.Lock()
_policy_inflight: Future | None = None
_MAX_META_CHARS = 6000
_MAX_POLICY_PROMPT_CONTEXT_CHARS = 8000
_MAX_TOOL_ARGS_CHARS = 1200
_MAX_TOOL_RESULT_CHARS = 2000
_MAX_TOOL_RECORDS_IN_FINAL_PROMPT = 8
_MAX_DECISION_REASON_CHARS = 255


@dataclass(frozen=True)
class PolicyLLMConfig:
    enabled: bool
    timeout_seconds: float
    max_rounds: int
    tool_whitelist: tuple[str, ...]
    max_tool_calls_per_round: int


@dataclass(frozen=True)
class GateConfig:
    enabled: bool
    fail_open: bool
    policy_llm: PolicyLLMConfig


_POLICY_SYSTEM_PROMPT = """\
You are a strict speech gate for proactive assistant speech.

Task:
- Decide whether assistant should speak right now.
- You may call tools when needed.
- If a vision tool like `vision_assistant` is available, you may call it directly.
- When calling tools, build arguments exactly according to the tool schema.

Output rules:
- Final answer must be exactly one JSON object.
- Do not output markdown, comments, explanations, or extra text.
- JSON schema:
  {"speak":true/false,"cooldown_sec":0-600,"reason":"<=255 chars"}

Decision rules:
- Use scenario + context + tool results.
- Prefer using the provided context over assumptions. The user prompt may include a
  `[Task Engine Attempts] ... [/Task Engine Attempts]` block. If present, use it.
- Use attempt records as trend signals: repeated "nobody/unknown/unwake" outcomes, repeated low-confidence
  results, "camera blocked/遮挡" evidence, repeated silent/no-response outcomes, or repeated unknown/unavailable/nobody result_code should generally
  make you less likely to speak again.
- When the context suggests the user is likely not present or not engaging, prefer `speak=false`
  rather than repeating similar proactive messages.
- When context suggests the user is present/engaged or this is an early attempt, it can be
  reasonable to speak.
- If uncertain, choose conservative behavior.
""".strip()


def _load_config(server: Any) -> GateConfig:
    plugins = (getattr(server, "config", None) or {}).get("plugins", {})
    cfg = plugins.get("proactive_speech_gate", {}) if isinstance(plugins, dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}

    enabled = bool(cfg.get("enabled", True))
    fail_open = bool(cfg.get("fail_open", True))

    policy_cfg = cfg.get("policy_llm", {})
    if not isinstance(policy_cfg, dict):
        policy_cfg = {}

    policy_enabled = bool(policy_cfg.get("enabled", True))
    try:
        timeout_seconds = float(policy_cfg.get("timeout_seconds", 6.0))
    except Exception:
        timeout_seconds = 6.0
    if timeout_seconds < 1.0:
        timeout_seconds = 1.0

    try:
        max_rounds = int(policy_cfg.get("max_rounds", 3))
    except Exception:
        max_rounds = 3
    if max_rounds < 1:
        max_rounds = 1
    if max_rounds > 8:
        max_rounds = 8

    raw_whitelist = policy_cfg.get("tool_whitelist", ["vision_assistant"])
    if not isinstance(raw_whitelist, list):
        raw_whitelist = ["vision_assistant"]
    whitelist: list[str] = []
    for item in raw_whitelist:
        name = str(item or "").strip()
        if not name:
            continue
        whitelist.append(name)
    if not whitelist:
        whitelist = ["vision_assistant"]

    try:
        max_tool_calls_per_round = int(policy_cfg.get("max_tool_calls_per_round", 1))
    except Exception:
        max_tool_calls_per_round = 1
    if max_tool_calls_per_round < 1:
        max_tool_calls_per_round = 1
    if max_tool_calls_per_round > 8:
        max_tool_calls_per_round = 8

    return GateConfig(
        enabled=enabled,
        fail_open=fail_open,
        policy_llm=PolicyLLMConfig(
            enabled=policy_enabled,
            timeout_seconds=timeout_seconds,
            max_rounds=max_rounds,
            tool_whitelist=tuple(whitelist),
            max_tool_calls_per_round=max_tool_calls_per_round,
        ),
    )


def _extract_first_json_object(text: str) -> str | None:
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


def _try_parse_json_dict(text: str) -> dict[str, Any] | None:
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


def _normalize_policy_decision(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if "speak" not in payload:
        return None

    speak_raw = payload.get("speak")
    if isinstance(speak_raw, bool):
        speak = speak_raw
    elif isinstance(speak_raw, str):
        s = speak_raw.strip().lower()
        if s in ("true", "1", "yes", "y"):
            speak = True
        elif s in ("false", "0", "no", "n"):
            speak = False
        else:
            return None
    else:
        return None

    try:
        cooldown_sec = int(payload.get("cooldown_sec", 0))
    except Exception:
        cooldown_sec = 0
    if cooldown_sec < 0:
        cooldown_sec = 0
    if cooldown_sec > 600:
        cooldown_sec = 600

    reason = str(payload.get("reason", "") or "").strip()
    if len(reason) > _MAX_DECISION_REASON_CHARS:
        reason = reason[:_MAX_DECISION_REASON_CHARS]

    return {
        "speak": speak,
        "cooldown_sec": cooldown_sec,
        "reason": reason,
    }


def _can_chat_hard(conn: Any) -> tuple[bool, str]:
    if not conn:
        return False, "conn_none"
    if getattr(conn, "stop_event", None) and conn.stop_event.is_set():
        return False, "stop_event"
    if getattr(conn, "close_after_chat", False):
        return False, "close_after_chat"
    if not getattr(conn, "tts", None) or not getattr(conn, "llm", None):
        return False, "missing_tts_or_llm"
    if getattr(conn, "client_is_speaking", False):
        return False, "client_is_speaking"
    if not getattr(conn, "llm_finish_task", True):
        return False, "llm_busy"
    return True, "ok"


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return "{}"


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _truncate_text(value: Any, max_chars: int) -> str:
    text = _to_text(value).strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


def _build_policy_user_prompt(*, scenario: str, prompt: str, meta: dict[str, Any], gate_tool_call_mode: str) -> str:
    meta_text = _safe_json_dumps(meta)
    if len(meta_text) > _MAX_META_CHARS:
        meta_text = meta_text[:_MAX_META_CHARS] + "...(truncated)"

    original_prompt_context = _truncate_text(prompt, _MAX_POLICY_PROMPT_CONTEXT_CHARS)

    return (
        f"Now: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n"
        f"Scenario: {scenario}\\n"
        f"GateToolCallMode: {gate_tool_call_mode}\\n"
        f"Meta: {meta_text}\\n"
        "OriginalUserPrompt:\\n"
        "[BEGIN_USER_PROMPT]\\n"
        f"{original_prompt_context}\\n"
        "[END_USER_PROMPT]\\n"
        "Return only JSON with keys: speak, cooldown_sec, reason."
    )


def _build_proactive_chat_prompt(
    *,
    original_prompt: str,
    scenario: str,
    decision_reason: str,
    tool_call_records: list[dict[str, Any]],
) -> str:
    compact_records: list[dict[str, Any]] = []
    for record in (tool_call_records or [])[:_MAX_TOOL_RECORDS_IN_FINAL_PROMPT]:
        if not isinstance(record, dict):
            continue
        compact_records.append(
            {
                "tool": str(record.get("tool") or ""),
                "status": str(record.get("status") or ""),
                "arguments": _truncate_text(record.get("arguments"), _MAX_TOOL_ARGS_CHARS),
                "result": _truncate_text(record.get("result"), _MAX_TOOL_RESULT_CHARS),
            }
        )

    gate_context = {
        "scenario": str(scenario or ""),
        "gate_decision": {"speak": True, "reason": str(decision_reason or "")},
        "tool_call_results": compact_records,
    }
    gate_context_json = _safe_json_dumps(gate_context)

    return (
        "[Proactive Gate Context]\n"
        f"{gate_context_json}\n"
        "[/Proactive Gate Context]\n\n"
        "[Proactive Gate Instruction]\n"
        "门控已判定：现在应当进行主动说话。请将上面的工具结果作为上下文，组织自然回复。"
        "不要提及门控判断、函数调用或以上上下文块。\n"
        "[/Proactive Gate Instruction]\n\n"
        f"{str(original_prompt or '')}"
    )


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _merge_tool_calls(tool_calls_list: list[dict[str, Any]], delta_tool_calls: Any) -> None:
    if not delta_tool_calls:
        return

    for raw_tc in list(delta_tool_calls):
        index = _get_value(raw_tc, "index", None)
        tc_id = _get_value(raw_tc, "id", "") or ""
        function_obj = _get_value(raw_tc, "function", {})
        name = _get_value(function_obj, "name", "") or ""
        arguments = _get_value(function_obj, "arguments", "")
        if not isinstance(arguments, str):
            arguments = _to_text(arguments)

        # Keep behavior aligned with connection._merge_tool_calls:
        # - if index is absent and name exists -> start a new tool call
        # - if index is absent and name missing -> append arguments to last tool call
        if index is None:
            if name:
                tool_index = len(tool_calls_list)
            else:
                tool_index = len(tool_calls_list) - 1 if tool_calls_list else 0
        else:
            try:
                tool_index = int(index)
            except Exception:
                tool_index = len(tool_calls_list)
            if tool_index < 0:
                tool_index = len(tool_calls_list)

        while len(tool_calls_list) <= tool_index:
            tool_calls_list.append({"id": "", "name": "", "arguments": ""})

        slot = tool_calls_list[tool_index]
        if tc_id:
            slot["id"] = tc_id
        if name:
            slot["name"] = name
        if arguments:
            slot["arguments"] = str(slot.get("arguments") or "") + arguments


def _run_policy_round_sync(conn: Any, dialogue: list[dict[str, Any]], functions: list[dict[str, Any]] | None) -> dict[str, Any]:
    llm = getattr(conn, "llm", None)
    if llm is None:
        raise RuntimeError("policy_llm_missing")

    text_parts: list[str] = []
    tool_calls_list: list[dict[str, Any]] = []

    if functions:
        responses = llm.response_with_functions("", dialogue, functions=functions)
        for response in responses:
            if isinstance(response, tuple) and len(response) == 2:
                content, tools_call = response
            elif isinstance(response, dict):
                content = response.get("content")
                tools_call = response.get("tool_calls")
            else:
                content = response
                tools_call = None

            if isinstance(content, str) and content:
                text_parts.append(content)
            if tools_call:
                _merge_tool_calls(tool_calls_list, tools_call)
    else:
        responses = llm.response("", dialogue)
        for token in responses:
            text = _to_text(token)
            if text:
                text_parts.append(text)

    normalized_tool_calls: list[dict[str, Any]] = []
    for item in tool_calls_list:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        tc_id = str(item.get("id") or "").strip() or uuid.uuid4().hex
        arguments = item.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = _to_text(arguments)
        normalized_tool_calls.append(
            {
                "id": tc_id,
                "name": name,
                "arguments": arguments,
            }
        )

    return {
        "text": "".join(text_parts).strip(),
        "tool_calls": normalized_tool_calls,
    }


def _clear_policy_inflight(done_future: Future) -> None:
    global _policy_inflight
    with _policy_inflight_lock:
        if _policy_inflight is done_future:
            _policy_inflight = None


def _submit_policy_round(
    conn: Any,
    dialogue: list[dict[str, Any]],
    functions: list[dict[str, Any]] | None,
) -> Future:
    global _policy_inflight
    with _policy_inflight_lock:
        if _policy_inflight is not None and not _policy_inflight.done():
            raise RuntimeError("policy_llm_busy")
        future = _POLICY_EXECUTOR.submit(_run_policy_round_sync, conn, dialogue, functions)
        future.add_done_callback(_clear_policy_inflight)
        _policy_inflight = future
        return future


async def _execute_tool_call(conn: Any, tool_call: dict[str, Any]) -> str:
    func_handler = getattr(conn, "func_handler", None)
    if not func_handler:
        return _safe_json_dumps(
            {
                "ok": False,
                "error": "tool_handler_unavailable",
                "tool": tool_call.get("name", ""),
            }
        )

    try:
        result = await func_handler.handle_llm_function_call(
            conn,
            {
                "id": str(tool_call.get("id") or ""),
                "name": str(tool_call.get("name") or ""),
                "arguments": str(tool_call.get("arguments") or ""),
            },
        )
    except Exception as e:
        return _safe_json_dumps(
            {
                "ok": False,
                "error": "tool_call_exception",
                "tool": tool_call.get("name", ""),
                "detail": str(e),
            }
        )

    if result is None:
        return _safe_json_dumps(
            {
                "ok": False,
                "error": "tool_call_failed",
                "tool": tool_call.get("name", ""),
            }
        )

    action = getattr(result, "action", None)
    action_name = action.name if hasattr(action, "name") else str(action or "")

    result_text = _to_text(getattr(result, "result", None))
    response_text = _to_text(getattr(result, "response", None))

    if action_name == "REQLLM" and result_text:
        return result_text

    return _safe_json_dumps(
        {
            "ok": action_name not in ("ERROR", "NOTFOUND"),
            "action": action_name,
            "result": result_text,
            "response": response_text,
        }
    )


def _get_allowed_functions(
    conn: Any,
    whitelist: tuple[str, ...],
    gate_tool_call_mode: str,
) -> list[dict[str, Any]]:
    if gate_tool_call_mode != "allow":
        return []

    func_handler = getattr(conn, "func_handler", None)
    if not func_handler:
        return []

    try:
        functions = func_handler.get_functions()
    except Exception:
        return []

    if not isinstance(functions, list):
        return []

    whitelist_set = {str(name or "").strip() for name in whitelist if str(name or "").strip()}
    if not whitelist_set:
        return []

    filtered: list[dict[str, Any]] = []
    for tool in functions:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = str(fn.get("name") or "").strip()
        if not name or name not in whitelist_set:
            continue
        filtered.append(tool)
    return filtered


async def _decide_with_policy_llm(
    conn: Any,
    *,
    cfg: PolicyLLMConfig,
    scenario: str,
    prompt: str,
    meta: dict[str, Any],
    gate_tool_call_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = {
        "tool_calls_used": 0,
        "tool_not_allowed_blocked": 0,
        "max_rounds_reached": False,
        "allowed_tools": [],
        "tool_call_records": [],
    }

    if not cfg.enabled:
        return (
            {"speak": True, "cooldown_sec": 0, "reason": "policy_llm_disabled"},
            metrics,
        )

    if gate_tool_call_mode not in ("allow", "off"):
        gate_tool_call_mode = "allow"

    allowed_functions = _get_allowed_functions(conn, cfg.tool_whitelist, gate_tool_call_mode)
    allowed_tool_names = [
        str((tool.get("function") or {}).get("name") or "")
        for tool in allowed_functions
        if isinstance(tool, dict)
    ]
    allowed_tool_set = {name for name in allowed_tool_names if name}
    metrics["allowed_tools"] = sorted(list(allowed_tool_set))

    dialogue: list[dict[str, Any]] = [
        {"role": "system", "content": _POLICY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_policy_user_prompt(
                scenario=scenario,
                prompt=prompt,
                meta=meta,
                gate_tool_call_mode=gate_tool_call_mode,
            ),
        },
    ]

    for round_index in range(cfg.max_rounds):
        functions_for_round = allowed_functions if gate_tool_call_mode == "allow" and allowed_functions else None
        future = _submit_policy_round(conn, dialogue, functions_for_round)
        try:
            round_output = await asyncio.wait_for(
                asyncio.wrap_future(future),
                timeout=cfg.timeout_seconds,
            )
        except asyncio.TimeoutError as e:
            raise RuntimeError("policy_llm_timeout") from e

        text = str((round_output or {}).get("text") or "")
        tool_calls = (round_output or {}).get("tool_calls")
        tool_calls = tool_calls if isinstance(tool_calls, list) else []

        if tool_calls and gate_tool_call_mode == "allow":
            assistant_tool_calls: list[dict[str, Any]] = []
            tool_messages: list[dict[str, Any]] = []

            limited_calls = tool_calls[: cfg.max_tool_calls_per_round]
            if len(tool_calls) > len(limited_calls):
                metrics["tool_not_allowed_blocked"] += len(tool_calls) - len(limited_calls)
                for dropped_call in tool_calls[cfg.max_tool_calls_per_round :]:
                    metrics["tool_call_records"].append(
                        {
                            "tool": str(dropped_call.get("name") or ""),
                            "status": "dropped_by_limit",
                            "arguments": _truncate_text(dropped_call.get("arguments"), _MAX_TOOL_ARGS_CHARS),
                            "result": _safe_json_dumps(
                                {"ok": False, "error": "tool_call_limited"}
                            ),
                        }
                    )

            for idx, call in enumerate(limited_calls):
                tool_name = str(call.get("name") or "").strip()
                tool_id = str(call.get("id") or "").strip() or uuid.uuid4().hex
                tool_args = str(call.get("arguments") or "")

                assistant_tool_calls.append(
                    {
                        "id": tool_id,
                        "function": {
                            "name": tool_name,
                            "arguments": tool_args,
                        },
                        "type": "function",
                        "index": idx,
                    }
                )

                if tool_name not in allowed_tool_set:
                    metrics["tool_not_allowed_blocked"] += 1
                    tool_result_text = _safe_json_dumps(
                        {
                            "ok": False,
                            "error": "tool_not_allowed",
                            "tool": tool_name,
                            "allowed_tools": sorted(list(allowed_tool_set)),
                        }
                    )
                    metrics["tool_call_records"].append(
                        {
                            "tool": tool_name,
                            "status": "blocked_not_allowed",
                            "arguments": _truncate_text(tool_args, _MAX_TOOL_ARGS_CHARS),
                            "result": _truncate_text(tool_result_text, _MAX_TOOL_RESULT_CHARS),
                        }
                    )
                else:
                    metrics["tool_calls_used"] += 1
                    tool_result_text = await _execute_tool_call(
                        conn,
                        {
                            "id": tool_id,
                            "name": tool_name,
                            "arguments": tool_args,
                        },
                    )
                    metrics["tool_call_records"].append(
                        {
                            "tool": tool_name,
                            "status": "executed",
                            "arguments": _truncate_text(tool_args, _MAX_TOOL_ARGS_CHARS),
                            "result": _truncate_text(tool_result_text, _MAX_TOOL_RESULT_CHARS),
                        }
                    )

                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": tool_result_text,
                    }
                )

            if assistant_tool_calls:
                dialogue.append({"role": "assistant", "tool_calls": assistant_tool_calls})
                dialogue.extend(tool_messages)
                continue

        parsed = _try_parse_json_dict(text)
        normalized = _normalize_policy_decision(parsed or {})
        if normalized is not None:
            return normalized, metrics

        if text:
            dialogue.append({"role": "assistant", "content": text})
        dialogue.append(
            {
                "role": "user",
                "content": (
                    "Your last output is invalid. Return exactly one JSON object with keys "
                    f"speak, cooldown_sec, reason (reason <= {_MAX_DECISION_REASON_CHARS} chars)."
                ),
            }
        )

        if round_index + 1 >= cfg.max_rounds:
            metrics["max_rounds_reached"] = True

    raise RuntimeError("policy_llm_max_rounds_reached")


async def proactive_chat(
    server: Any,
    conn: Any,
    prompt: str,
    *,
    scenario: str,
    meta: dict | None = None,
    gate_tool_call_mode: str = "allow",
) -> bool:
    start_ms = time.time() * 1000
    cfg = _load_config(server)
    logger.bind(tag=TAG).info(
        f"proactive_chat start: scenario={scenario}, gate_tool_call_mode={gate_tool_call_mode}, "
        f"gate_enabled={cfg.enabled}"
    )

    if not cfg.enabled:
        await asyncio.to_thread(conn.chat, prompt)
        return True

    can_chat, hard_reason = _can_chat_hard(conn)
    if not can_chat:
        logger.bind(tag=TAG).info(
            f"proactive_chat skipped: scenario={scenario}, reason={hard_reason}"
        )
        return False

    meta_obj = meta if isinstance(meta, dict) else {}
    fail_open_used = False
    llm_speak = True
    llm_reason = ""
    tool_calls_used = 0
    blocked_calls = 0
    max_rounds_reached = False
    allowed_tools: list[str] = []
    tool_call_records: list[dict[str, Any]] = []
    enriched_prompt_used = False
    final_prompt_to_send = str(prompt or "")

    try:
        decision, metrics = await _decide_with_policy_llm(
            conn,
            cfg=cfg.policy_llm,
            scenario=str(scenario or ""),
            prompt=str(prompt or ""),
            meta=meta_obj,
            gate_tool_call_mode=str(gate_tool_call_mode or "allow"),
        )
        llm_speak = bool(decision.get("speak", True))
        llm_reason = str(decision.get("reason", "") or "")
        tool_calls_used = int(metrics.get("tool_calls_used") or 0)
        blocked_calls = int(metrics.get("tool_not_allowed_blocked") or 0)
        max_rounds_reached = bool(metrics.get("max_rounds_reached", False))
        allowed_tools = list(metrics.get("allowed_tools") or [])
        tool_call_records = list(metrics.get("tool_call_records") or [])
    except Exception as e:
        if cfg.fail_open:
            fail_open_used = True
            llm_speak = True
            llm_reason = f"fail_open:{e}"
        else:
            llm_speak = False
            llm_reason = f"gate_error:{e}"

    latency_ms = int(time.time() * 1000 - start_ms)
    logger.bind(tag=TAG).info(
        f"proactive_chat decision: scenario={scenario}, gate_tool_call_mode={gate_tool_call_mode}, "
        f"allowed_tools={allowed_tools}, tool_calls_used={tool_calls_used}, "
        f"tool_not_allowed_blocked={blocked_calls}, max_rounds_reached={max_rounds_reached}, "
        f"llm_speak={llm_speak}, reason={llm_reason}, fail_open_used={fail_open_used}, "
        f"tool_context_items={len(tool_call_records)}, "
        f"latency_ms={latency_ms}"
    )

    if not llm_speak:
        return False

    if not fail_open_used:
        final_prompt_to_send = _build_proactive_chat_prompt(
            original_prompt=str(prompt or ""),
            scenario=str(scenario or ""),
            decision_reason=llm_reason,
            tool_call_records=tool_call_records,
        )
        enriched_prompt_used = True

    logger.bind(tag=TAG).info(
        f"proactive_chat final_prompt: scenario={scenario}, enriched={enriched_prompt_used}, "
        f"tool_context_items={len(tool_call_records)}"
    )
    await asyncio.to_thread(conn.chat, final_prompt_to_send)
    return True
