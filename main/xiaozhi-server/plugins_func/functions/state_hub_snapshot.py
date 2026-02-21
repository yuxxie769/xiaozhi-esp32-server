from __future__ import annotations

import json
from plugins_func.register import register_function, ToolType, ActionResponse, Action

from core.state_hub.registry import get_state_hub


state_hub_snapshot_desc = {
    "type": "function", 
    "function": {
        "name": "state_hub_snapshot",
        "description": "Capture the key status snapshot (highlight) of the Home Assistant State Hub. Return structured data containing connected/outdated/highlight, along with explanations for the fields field.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_highlight": {
                    "type": "integer",
                    "description": "最多返回多少条 highlight（1-80，默认 30）",
                }
            },
            "required": [],
        },
    },
}

FIELDS_DESC = {
    "connected": "Whether the connection with Home Assistant has been established.",
    "outdated": "Whether the current data is offline/expired.",
    "highlight": "Key status list (compressed and sorted according to cropping rules; only includes entities with expose_to_llm=true).",
    "highlight[].n": "Entity readable name.",
    "highlight[].s": "Current state of the entity.",
    "highlight[].dc": "Entity type.",
    "highlight[].u": "unit_of_measurement of state.",
    "notes": "Alert shall be given if the connected or outdated is false."
}
 

@register_function("state_hub_snapshot", state_hub_snapshot_desc, ToolType.SYSTEM_CTL)
def state_hub_snapshot(conn, max_highlight: int = 30):
    hub = get_state_hub()
    if not hub:
        payload = {
            "connected": False,
            "outdated": True,
            "highlight": [],
        }
    else:
        # Override max_highlight for this call (cap at 80).
        v = hub.view_highlight()
        hl = v.get("highlight") or []
        try:
            n = int(max_highlight)
        except Exception:
            n = 30
        n = max(1, min(n, 80))
        # Trim fields for LLM: drop age_s and entity_id (id). Only keep key fields.
        trimmed = []
        for it in hl[:n]:
            if not isinstance(it, dict):
                continue
            name = it.get("n")
            if not name:
                continue
            s = it.get("s")
            if s is None:
                continue
            out = {"n": name, "s": s}
            if it.get("dc"):
                out["dc"] = it.get("dc")
            if it.get("u"):
                out["u"] = it.get("u")
            trimmed.append(out)
        payload = {
            "connected": bool(v.get("connected", False)),
            "outdated": bool(v.get("outdated", True)),
            "highlight": trimmed,
        }

    payload["fields"] = FIELDS_DESC

    # function_call: return JSON to main LLM (REQLLM)
    if getattr(conn, "intent_type", "") == "function_call":
        return ActionResponse(Action.REQLLM, json.dumps(payload, ensure_ascii=False), None)

    # intent_llm: return local summary text directly (RESPONSE) to avoid extra LLM layer.
    if hub:
        text = hub.local_summary_text()
    else:
        text = "我目前未连接到 Home Assistant，当前也没有可用的离线数据。"
    return ActionResponse(Action.RESPONSE, None, text)
