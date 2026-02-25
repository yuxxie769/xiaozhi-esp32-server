from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from config.logger import setup_logging

from .store import TaskStore

TAG = __name__
logger = setup_logging()


def _now_ms() -> int:
    return int(datetime.now().timestamp() * 1000)


def _safe_int(v: Any, default: int | None = None) -> int | None:
    try:
        return int(v)
    except Exception:
        return default


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _try_json_load(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    s = raw.strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _result(
    *,
    ok: bool,
    error_code: str = "",
    message: str = "",
    retryable: bool = False,
    next_steps: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": bool(ok),
        "error_code": str(error_code or ""),
        "message": str(message or ""),
        "retryable": bool(retryable),
        "next_steps": list(next_steps or []),
    }
    payload.update(extra)
    return payload


@dataclass(frozen=True)
class ControlConfig:
    enabled: bool
    require_task_engine_enabled: bool
    ctx_ttl_seconds: int
    allow_user_confirm_codes: tuple[str, ...]
    require_vision_confirm_codes: tuple[str, ...]
    vision_min_confidence: float
    allowed_vision_evidence_tools: tuple[str, ...]
    recent_user_messages: int
    max_quote_chars: int


def _load_control_cfg(server: Any) -> ControlConfig:
    plugins = (getattr(server, "config", None) or {}).get("plugins", {})
    cfg = plugins.get("task_engine_control", {}) if isinstance(plugins, dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}

    enabled = bool(cfg.get("enabled", False))
    require_task_engine_enabled = bool(cfg.get("require_task_engine_enabled", False))

    ctx_ttl_seconds = _safe_int(cfg.get("ctx_ttl_seconds", 1800), 1800) or 1800
    if ctx_ttl_seconds < 10:
        ctx_ttl_seconds = 10
    if ctx_ttl_seconds > 24 * 3600:
        ctx_ttl_seconds = 24 * 3600

    allow_user_confirm_codes = cfg.get(
        "allow_user_confirm_codes", ["nobody", "unknown", "unavailable"]
    )
    if not isinstance(allow_user_confirm_codes, list):
        allow_user_confirm_codes = ["nobody", "unknown", "unavailable"]
    allow_user_confirm_codes_t = tuple(
        str(x).strip() for x in allow_user_confirm_codes if str(x).strip()
    )

    require_vision_confirm_codes = cfg.get("require_vision_confirm_codes", ["unwake"])
    if not isinstance(require_vision_confirm_codes, list):
        require_vision_confirm_codes = ["unwake"]
    require_vision_confirm_codes_t = tuple(
        str(x).strip() for x in require_vision_confirm_codes if str(x).strip()
    )

    vision_min_confidence = _safe_float(cfg.get("vision_min_confidence", 0.6), 0.6)
    if vision_min_confidence < 0.0:
        vision_min_confidence = 0.0
    if vision_min_confidence > 1.0:
        vision_min_confidence = 1.0

    allowed_vision_evidence_tools = cfg.get("allowed_vision_evidence_tools", ["wake_check"])
    if not isinstance(allowed_vision_evidence_tools, list):
        allowed_vision_evidence_tools = ["wake_check"]
    allowed_vision_evidence_tools_t = tuple(
        str(x).strip() for x in allowed_vision_evidence_tools if str(x).strip()
    ) or ("wake_check",)

    recent_user_messages = _safe_int(cfg.get("recent_user_messages", 8), 8) or 8
    if recent_user_messages < 1:
        recent_user_messages = 1
    if recent_user_messages > 50:
        recent_user_messages = 50

    max_quote_chars = _safe_int(cfg.get("max_quote_chars", 80), 80) or 80
    if max_quote_chars < 10:
        max_quote_chars = 10
    if max_quote_chars > 300:
        max_quote_chars = 300

    return ControlConfig(
        enabled=enabled,
        require_task_engine_enabled=require_task_engine_enabled,
        ctx_ttl_seconds=int(ctx_ttl_seconds),
        allow_user_confirm_codes=allow_user_confirm_codes_t,
        require_vision_confirm_codes=require_vision_confirm_codes_t,
        vision_min_confidence=float(vision_min_confidence),
        allowed_vision_evidence_tools=allowed_vision_evidence_tools_t,
        recent_user_messages=int(recent_user_messages),
        max_quote_chars=int(max_quote_chars),
    )


def _load_engine_state(server: Any) -> tuple[bool, str]:
    plugins = (getattr(server, "config", None) or {}).get("plugins", {})
    te = plugins.get("task_engine", {}) if isinstance(plugins, dict) else {}
    enabled = bool(isinstance(te, dict) and te.get("enabled", False))
    db_path = "data/tasks.db"
    if isinstance(te, dict) and te.get("db_path"):
        db_path = str(te.get("db_path") or db_path)
    return enabled, db_path


def _get_binding(conn: Any) -> dict[str, Any] | None:
    b = getattr(conn, "_task_engine_binding", None)
    return b if isinstance(b, dict) else None


def _clear_binding(conn: Any) -> None:
    try:
        delattr(conn, "_task_engine_binding")
    except Exception:
        try:
            setattr(conn, "_task_engine_binding", None)
        except Exception:
            pass


def _user_quote_in_recent_user_messages(conn: Any, quote: str, *, limit: int) -> bool:
    q = str(quote or "")
    if not q:
        return False
    dialogue = getattr(getattr(conn, "dialogue", None), "dialogue", None) or []
    seen = 0
    for msg in reversed(dialogue):
        if getattr(msg, "role", None) != "user":
            continue
        seen += 1
        content = getattr(msg, "content", "")
        if isinstance(content, str) and q in content:
            return True
        if seen >= int(limit):
            break
    return False


def _find_tool_call_name(conn: Any, tool_call_id: str) -> str:
    tc_id = str(tool_call_id or "").strip()
    if not tc_id:
        return ""
    dialogue = getattr(getattr(conn, "dialogue", None), "dialogue", None) or []
    for msg in reversed(dialogue):
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            if str(tc.get("id") or "").strip() != tc_id:
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            return str(fn.get("name") or "").strip()
    return ""


def _find_tool_result_content(conn: Any, tool_call_id: str) -> str:
    tc_id = str(tool_call_id or "").strip()
    if not tc_id:
        return ""
    dialogue = getattr(getattr(conn, "dialogue", None), "dialogue", None) or []
    for msg in reversed(dialogue):
        if getattr(msg, "role", None) != "tool":
            continue
        if str(getattr(msg, "tool_call_id", "") or "").strip() != tc_id:
            continue
        content = getattr(msg, "content", "")
        return content if isinstance(content, str) else str(content)
    return ""


def _validate_wake_check_wake_up_text(
    raw_text: str,
    *,
    min_confidence: float,
) -> tuple[bool, str]:
    text = str(raw_text or "").strip()
    if not text:
        return False, "tool_result_empty"

    m = re.search(
        r"wake_up_check_result\((?P<flag>[^)]+)\)\s*:\s*awake=(?P<awake>[^,]+),\s*confidence=(?P<confidence>[-+]?\d*\.?\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return False, "tool_result_invalid_format"

    flag = str(m.group("flag") or "").strip().lower()
    awake_raw = str(m.group("awake") or "").strip().lower()
    conf_raw = m.group("confidence")

    if awake_raw not in ("true", "1", "yes", "y"):
        return False, "tool_result_awake_not_true"

    conf = _safe_float(conf_raw, 0.0)
    if conf < float(min_confidence):
        return False, "tool_result_low_confidence"

    if flag and flag not in ("ok", "wake", "awake", "true"):
        return False, "tool_result_not_ok"
    return True, "ok"


def handle_request(server: Any, conn: Any, request: dict[str, Any]) -> dict[str, Any]:
    now_ms = _now_ms()
    cfg = _load_control_cfg(server)

    if not cfg.enabled:
        return _result(
            ok=False,
            error_code="control_disabled",
            message="task_engine_control is disabled",
            retryable=False,
            next_steps=["Set plugins.task_engine_control.enabled=true to enable this control tool."],
        )

    task_engine_enabled, db_path = _load_engine_state(server)
    if cfg.require_task_engine_enabled and not task_engine_enabled:
        return _result(
            ok=False,
            error_code="task_engine_disabled",
            message="task_engine is disabled",
            retryable=False,
        )

    if not isinstance(request, dict):
        return _result(ok=False, error_code="invalid_request", message="request must be an object")

    op = str(request.get("op") or "").strip()
    if op != "wake_up_mark_ok":
        return _result(
            ok=False,
            error_code="unsupported_op",
            message="unsupported op",
            retryable=False,
            op=op,
            supported_ops=["wake_up_mark_ok"],
        )

    binding = _get_binding(conn)
    if not binding:
        return _result(
            ok=False,
            error_code="no_active_binding",
            message="no active task_engine binding on this connection",
            retryable=False,
        )

    expires_at_ms = _safe_int(binding.get("expires_at_ms"), 0) or 0
    if expires_at_ms and now_ms > expires_at_ms:
        return _result(
            ok=False,
            error_code="binding_expired",
            message="task_engine binding expired",
            retryable=False,
            expires_at_ms=int(expires_at_ms),
            now_ms=int(now_ms),
        )

    req_instance_id = _safe_int(request.get("instance_id"))
    req_attempt_id = _safe_int(request.get("attempt_id"))
    if req_instance_id is None or req_attempt_id is None:
        return _result(
            ok=False,
            error_code="missing_params",
            message="instance_id and attempt_id are required",
            retryable=False,
        )

    bind_instance_id = _safe_int(binding.get("instance_id"))
    bind_attempt_id = _safe_int(binding.get("attempt_id"))
    if bind_instance_id != req_instance_id or bind_attempt_id != req_attempt_id:
        return _result(
            ok=False,
            error_code="stale_or_mismatch",
            message="binding does not match request; the task may have moved to a new attempt",
            retryable=False,
            binding={"instance_id": bind_instance_id, "attempt_id": bind_attempt_id},
            request_ids={"instance_id": req_instance_id, "attempt_id": req_attempt_id},
        )

    task_type = str(binding.get("task_type") or "").strip()
    if task_type != "wake_up":
        return _result(
            ok=False,
            error_code="wrong_task_type",
            message="binding is not for wake_up",
            retryable=False,
            task_type=task_type,
        )

    verify_mode = str(request.get("verify_mode") or "").strip()
    if verify_mode not in ("user_confirm", "vision_confirm"):
        return _result(
            ok=False,
            error_code="invalid_params",
            message="verify_mode must be user_confirm or vision_confirm",
            retryable=False,
        )

    store = TaskStore(db_path)
    try:
        store.init_schema()
    except Exception as e:
        return _result(
            ok=False,
            error_code="store_unavailable",
            message=f"failed to init store: {e}",
            retryable=True,
        )

    inst = store.get_instance_by_id(int(req_instance_id))
    if not inst:
        return _result(ok=False, error_code="instance_not_found", message="instance not found", retryable=False)

    inst_task_type = str(inst.get("task_type") or "").strip()
    if inst_task_type != "wake_up":
        return _result(
            ok=False,
            error_code="wrong_task_type",
            message="instance task_type is not wake_up",
            retryable=False,
            task_type=inst_task_type,
        )

    inst_status = str(inst.get("status") or "").strip()
    if inst_status not in ("PENDING", "IN_PROGRESS"):
        return _result(
            ok=False,
            error_code="instance_not_active",
            message="instance is not active",
            retryable=False,
            instance_status=inst_status,
        )

    account_id = str(inst.get("account_id") or "")
    task_row = store.get_task(account_id=account_id, task_type="wake_up")
    if not task_row or int(task_row.get("enabled") or 0) != 1:
        return _result(
            ok=False,
            error_code="task_disabled_or_missing",
            message="wake_up task is disabled or missing",
            retryable=False,
        )

    latest_attempt_id = store.get_latest_attempt_id(instance_id=int(req_instance_id))
    if latest_attempt_id is None:
        return _result(
            ok=False,
            error_code="attempt_not_found",
            message="no attempts found for instance",
            retryable=False,
        )
    if int(latest_attempt_id) != int(req_attempt_id):
        return _result(
            ok=False,
            error_code="attempt_not_latest",
            message="attempt is not the latest; a new attempt may have started",
            retryable=False,
            latest_attempt_id=int(latest_attempt_id),
        )

    attempt = store.get_attempt_by_id(attempt_id=int(req_attempt_id))
    if not attempt:
        return _result(ok=False, error_code="attempt_not_found", message="attempt not found", retryable=False)
    if int(attempt.get("instance_id") or 0) != int(req_instance_id):
        return _result(
            ok=False,
            error_code="stale_or_mismatch",
            message="attempt does not belong to instance",
            retryable=False,
        )

    current_result_code = str(attempt.get("result_code") or "").strip() or "unknown"
    binding_result_code = str(binding.get("result_code") or "").strip()
    if binding_result_code and binding_result_code != current_result_code:
        return _result(
            ok=False,
            error_code="stale_or_mismatch",
            message="attempt result_code changed; binding is stale",
            retryable=False,
            attempt_result_code=current_result_code,
            binding_result_code=binding_result_code,
        )

    if current_result_code == "ok":
        return _result(
            ok=True,
            error_code="",
            message="already ok",
            retryable=False,
            updated=False,
            instance_id=int(req_instance_id),
            attempt_id=int(req_attempt_id),
            new_result_code="ok",
        )

    # Decide evidence requirements by the attempt result_code.
    allow_user_confirm = current_result_code in set(cfg.allow_user_confirm_codes)
    require_vision_confirm = current_result_code in set(cfg.require_vision_confirm_codes)

    if require_vision_confirm and verify_mode != "vision_confirm":
        return _result(
            ok=False,
            error_code="vision_required",
            message="visual confirmation is required for this attempt result",
            retryable=True,
            next_steps=[
                "Call wake_check() first to get wake-up visual evidence.",
                "Pass the SAME tool_call_id from that wake_check call to task_engine_control(verify_mode='vision_confirm', vision_tool_call_id=...).",
            ],
            attempt_result_code=current_result_code,
        )

    if not allow_user_confirm and not require_vision_confirm:
        return _result(
            ok=False,
            error_code="unsupported_result_code",
            message="this attempt result_code is not eligible for override",
            retryable=False,
            attempt_result_code=current_result_code,
        )

    user_quote = str(request.get("user_quote") or "").strip()
    vision_tool_call_id = str(request.get("vision_tool_call_id") or "").strip()
    note = str(request.get("note") or "").strip()

    if verify_mode == "user_confirm":
        if not allow_user_confirm:
            return _result(
                ok=False,
                error_code="user_confirm_not_allowed",
                message="user_confirm is not allowed for this attempt",
                retryable=False,
                attempt_result_code=current_result_code,
            )
        if not user_quote:
            return _result(
                ok=False,
                error_code="missing_evidence",
                message="user_quote is required for user_confirm",
                retryable=False,
            )
        if len(user_quote) > int(cfg.max_quote_chars):
            return _result(
                ok=False,
                error_code="invalid_evidence",
                message="user_quote too long",
                retryable=False,
                max_quote_chars=int(cfg.max_quote_chars),
            )
        if not _user_quote_in_recent_user_messages(
            conn, user_quote, limit=int(cfg.recent_user_messages)
        ):
            return _result(
                ok=False,
                error_code="evidence_invalid",
                message="user_quote not found in recent user messages",
                retryable=False,
            )

    if verify_mode == "vision_confirm":
        if not vision_tool_call_id:
            return _result(
                ok=False,
                error_code="missing_evidence",
                message=(
                    "vision_tool_call_id is required for vision_confirm; "
                    "use the real tool_call_id from wake_check."
                ),
                retryable=False,
            )
        tool_name = _find_tool_call_name(conn, vision_tool_call_id)
        if not tool_name:
            return _result(
                ok=False,
                error_code="evidence_invalid",
                message="tool_call_id not found in dialogue tool_calls",
                retryable=False,
            )
        if tool_name not in set(cfg.allowed_vision_evidence_tools):
            return _result(
                ok=False,
                error_code="tool_not_allowed",
                message="tool is not allowed as vision evidence",
                retryable=False,
                tool_name=tool_name,
                allowed_tools=list(cfg.allowed_vision_evidence_tools),
            )
        raw = _find_tool_result_content(conn, vision_tool_call_id)
        if tool_name == "wake_check":
            ok_ev, reason = _validate_wake_check_wake_up_text(
                raw, min_confidence=float(cfg.vision_min_confidence)
            )
            if not ok_ev:
                return _result(
                    ok=False,
                    error_code="vision_evidence_insufficient",
                    message="visual evidence is insufficient",
                    retryable=True,
                    detail=reason,
                    min_confidence=float(cfg.vision_min_confidence),
                )
        else:
            return _result(
                ok=False,
                error_code="tool_not_supported",
                message="tool evidence validation not implemented",
                retryable=False,
                tool_name=tool_name,
            )

    # Apply override: mark attempt ok, complete instance.
    original_result_json = _try_json_load(attempt.get("result_json"))
    original_decision_json = _try_json_load(attempt.get("decision_json"))
    original_decision_code = str(attempt.get("decision_code") or "").strip() or ""

    user_quote_hash = ""
    if user_quote:
        user_quote_hash = hashlib.sha256(user_quote.encode("utf-8")).hexdigest()[:16]

    override = {
        "by": "task_engine_control",
        "op": op,
        "verify_mode": verify_mode,
        "at_ms": int(now_ms),
        "note": note,
        "user_quote_hash": user_quote_hash,
        "vision_tool_call_id": vision_tool_call_id,
        "original_result_code": current_result_code,
        "original_decision_code": original_decision_code,
    }

    new_result_json = dict(original_result_json)
    new_result_json["override"] = override
    new_decision_json = dict(original_decision_json)
    new_decision_json["override"] = override
    if original_decision_code:
        new_decision_json["original_decision_code"] = original_decision_code

    try:
        guarded_result = store.override_wake_up_attempt_to_ok_guarded(
            instance_id=int(req_instance_id),
            attempt_id=int(req_attempt_id),
            result_json=new_result_json,
            decision_json=new_decision_json,
            now_ms=int(now_ms),
        )
    except Exception as e:
        logger.bind(tag=TAG).warning(
            f"task_engine_control override failed: instance_id={req_instance_id}, "
            f"attempt_id={req_attempt_id}, error={e}"
        )
        return _result(
            ok=False,
            error_code="override_failed",
            message=f"override failed: {e}",
            retryable=True,
        )

    if guarded_result == "already_ok":
        return _result(
            ok=True,
            error_code="",
            message="already ok",
            retryable=False,
            updated=False,
            instance_id=int(req_instance_id),
            attempt_id=int(req_attempt_id),
            new_result_code="ok",
        )
    if guarded_result == "attempt_not_found":
        return _result(
            ok=False,
            error_code="attempt_not_found",
            message="attempt not found",
            retryable=False,
        )
    if guarded_result == "attempt_not_latest":
        return _result(
            ok=False,
            error_code="attempt_not_latest",
            message="attempt is not the latest; a new attempt may have started",
            retryable=False,
        )
    if guarded_result == "instance_not_active":
        return _result(
            ok=False,
            error_code="instance_not_active",
            message="instance is not active",
            retryable=False,
        )
    if guarded_result == "task_disabled_or_missing":
        return _result(
            ok=False,
            error_code="task_disabled_or_missing",
            message="wake_up task is disabled or missing",
            retryable=False,
        )
    if guarded_result == "wrong_task_type":
        return _result(
            ok=False,
            error_code="wrong_task_type",
            message="instance task_type is not wake_up",
            retryable=False,
        )
    if guarded_result != "updated":
        return _result(
            ok=False,
            error_code="stale_or_mismatch",
            message="attempt state changed before override",
            retryable=False,
            detail=guarded_result,
        )

    _clear_binding(conn)
    logger.bind(tag=TAG).info(
        f"task_engine_control override ok: instance_id={req_instance_id}, attempt_id={req_attempt_id}, "
        f"from={current_result_code} via={verify_mode}"
    )
    return _result(
        ok=True,
        error_code="",
        message="override applied",
        retryable=False,
        updated=True,
        instance_id=int(req_instance_id),
        attempt_id=int(req_attempt_id),
        new_result_code="ok",
        previous_result_code=current_result_code,
    )
