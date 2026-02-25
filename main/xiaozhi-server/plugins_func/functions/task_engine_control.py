from __future__ import annotations

import json
from typing import Any

from plugins_func.register import Action, ActionResponse, ToolType, register_function


TASK_ENGINE_CONTROL_DESC = {
    "type": "function",
    "function": {
        "name": "task_engine_control",
        "description": (
            "Control-plane API for Task Engine. Submit a state correction request for the current "
            "task_engine wake_up attempt bound to this connection. "
            "IMPORTANT: This tool does not accept fabricated evidence. "
            "For verify_mode='vision_confirm', you must pass a real tool_call_id from an actual tool call "
            "(wake_check)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "description": "Operation name. Currently supports: wake_up_mark_ok",
                },
                "instance_id": {
                    "type": "integer",
                    "description": "Task engine instance_id from [Task Engine Binding].",
                },
                "attempt_id": {
                    "type": "integer",
                    "description": "Task engine attempt_id from [Task Engine Binding].",
                },
                "verify_mode": {
                    "type": "string",
                    "description": "Evidence mode: user_confirm or vision_confirm.",
                    "enum": ["user_confirm", "vision_confirm"],
                },
                "user_quote": {
                    "type": "string",
                    "description": (
                        "Required when verify_mode='user_confirm'. Must be a short exact quote from a recent user message."
                    ),
                },
                "vision_tool_call_id": {
                    "type": "string",
                    "description": (
                        "Required when verify_mode='vision_confirm'. Must be a real tool_call_id from dialogue tool_calls "
                        "(the id returned by wake_check)."
                    ),
                },
                "note": {
                    "type": "string",
                    "description": "Optional audit note.",
                },
            },
            "required": ["op", "instance_id", "attempt_id", "verify_mode"],
        },
    },
}


@register_function("task_engine_control", TASK_ENGINE_CONTROL_DESC, ToolType.SYSTEM_CTL)
def task_engine_control(
    conn,
    op: str,
    instance_id: int,
    attempt_id: int,
    verify_mode: str,
    user_quote: str | None = None,
    vision_tool_call_id: str | None = None,
    note: str | None = None,
):
    server = getattr(conn, "server", None)
    if server is None:
        payload = {
            "ok": False,
            "error_code": "server_unavailable",
            "message": "connection has no server reference",
            "retryable": False,
            "next_steps": [],
        }
        return ActionResponse(Action.REQLLM, json.dumps(payload, ensure_ascii=False), None)

    try:
        from core.task_engine.control import handle_request

        req: dict[str, Any] = {
            "op": op,
            "instance_id": instance_id,
            "attempt_id": attempt_id,
            "verify_mode": verify_mode,
            "user_quote": user_quote or "",
            "vision_tool_call_id": vision_tool_call_id or "",
            "note": note or "",
        }
        result = handle_request(server, conn, req)
    except Exception as e:
        result = {
            "ok": False,
            "error_code": "control_exception",
            "message": str(e),
            "retryable": True,
            "next_steps": [],
        }

    return ActionResponse(Action.REQLLM, json.dumps(result, ensure_ascii=False), None)
