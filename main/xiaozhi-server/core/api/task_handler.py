from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from aiohttp import web

from core.api.base_handler import BaseHandler
from core.task_engine.store import TaskStore


TAG = __name__


def _now_ms() -> int:
    return int(datetime.now().timestamp() * 1000)


def _today_key() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _parse_hhmm_to_ms(hhmm: str) -> Optional[int]:
    s = str(hhmm or "").strip()
    if not s:
        return None
    try:
        hh, mm = s.split(":")
        h = int(hh)
        m = int(mm)
        if h == 24 and m == 0:
            h, m = 0, 0
        if not (0 <= h <= 23 and 0 <= m <= 59):
            return None
        now = datetime.now()
        dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _parse_date_hhmm_to_ms(date_key: str | None, hhmm: str) -> Optional[int]:
    """
    Parse HH:MM into an epoch-ms timestamp using the local timezone.
    - If date_key is a YYYY-MM-DD string, use that date; otherwise use today.
    - Treat "24:00" as next-day 00:00.
    """
    s = str(hhmm or "").strip()
    if not s:
        return None
    try:
        hh, mm = s.split(":")
        h = int(hh)
        m = int(mm)
        if not (0 <= m <= 59):
            return None
        base = None
        if isinstance(date_key, str) and len(date_key) == 10:
            try:
                base = datetime.strptime(date_key, "%Y-%m-%d")
            except Exception:
                base = None
        if base is None:
            base = datetime.now()

        if h == 24 and m == 0:
            dt = base.replace(hour=0, minute=0, second=0, microsecond=0)
            return int((dt.timestamp() + 24 * 3600) * 1000)

        if not (0 <= h <= 23):
            return None
        dt = base.replace(hour=h, minute=m, second=0, microsecond=0)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _json_loads(s: str | None) -> dict[str, Any]:
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out.get(k) or {}, v)
        else:
            out[k] = v
    return out


def _load_task_type_defaults(config: dict[str, Any], task_type: str) -> dict[str, Any]:
    plugins = (config or {}).get("plugins", {}) if isinstance(config, dict) else {}
    te = plugins.get("task_engine", {}) if isinstance(plugins, dict) else {}
    task_types = te.get("task_types", {}) if isinstance(te, dict) else {}
    if not isinstance(task_types, dict):
        task_types = {}

    base = task_types.get("default") if isinstance(task_types.get("default"), dict) else {}
    specific = task_types.get(task_type) if isinstance(task_types.get(task_type), dict) else {}
    merged = _deep_merge_dict(base, specific)

    legacy = te.get(task_type, {}) if isinstance(te.get(task_type), dict) else {}
    if task_type == "wake_up" and not legacy:
        legacy = te.get("wake_up", {}) if isinstance(te.get("wake_up"), dict) else {}
    if legacy:
        merged = _deep_merge_dict(merged, legacy)
    return merged if isinstance(merged, dict) else {}


class TaskApiHandler(BaseHandler):
    def __init__(self, config: dict):
        super().__init__(config)

    def _store(self) -> TaskStore:
        plugins = (self.config or {}).get("plugins", {}) if isinstance(self.config, dict) else {}
        te = plugins.get("task_engine", {}) if isinstance(plugins, dict) else {}
        db_path = str(te.get("db_path", "data/tasks.db") or "data/tasks.db")
        store = TaskStore(db_path)
        store.init_schema()
        return store

    def _ok(self, data: Any):
        resp = web.json_response({"success": True, "data": data}, dumps=lambda o: json.dumps(o, ensure_ascii=False))
        self._add_cors_headers(resp)
        return resp

    def _err(self, message: str, *, status: int = 400):
        resp = web.json_response({"success": False, "message": message}, status=status)
        self._add_cors_headers(resp)
        return resp

    async def handle_list(self, request):
        try:
            store = self._store()
            tasks = store.list_tasks(limit=200)
            out = []
            for t in tasks:
                item = dict(t)
                item["policy"] = _json_loads(item.get("policy_json"))
                item.pop("policy_json", None)
                # Attach latest instance (if any)
                inst = store.get_latest_instance(
                    account_id=str(item.get("account_id") or ""),
                    task_type=str(item.get("task_type") or ""),
                )
                item["latest_instance"] = inst
                out.append(item)
            return self._ok(out)
        except Exception as e:
            return self._err(str(e) or "internal_error", status=500)

    async def handle_list_account(self, request):
        account_id = str(request.match_info.get("account_id") or "").strip()
        if not account_id:
            return self._err("account_id required", status=400)
        try:
            store = self._store()
            tasks = store.list_tasks_by_account(account_id=account_id, limit=200)
            out = []
            for t in tasks:
                item = dict(t)
                item["policy"] = _json_loads(item.get("policy_json"))
                item.pop("policy_json", None)
                inst = store.get_latest_instance(
                    account_id=str(item.get("account_id") or ""),
                    task_type=str(item.get("task_type") or ""),
                )
                item["latest_instance"] = inst
                out.append(item)
            return self._ok(out)
        except Exception as e:
            return self._err(str(e) or "internal_error", status=500)

    async def handle_list_instances(self, request):
        account_id = str(request.match_info.get("account_id") or "").strip()
        task_type = str(request.query.get("task_type") or "").strip()
        if not account_id:
            return self._err("account_id required", status=400)
        try:
            store = self._store()
            rows = store.list_instances_by_account(account_id=account_id, task_type=task_type or None, limit=200)
            return self._ok(rows)
        except Exception as e:
            return self._err(str(e) or "internal_error", status=500)

    async def handle_upsert(self, request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        account_id = str(payload.get("account_id") or "").strip()
        task_type = str(payload.get("task_type") or "").strip()
        enabled = bool(payload.get("enabled", True))
        if not account_id:
            return self._err("account_id required", status=400)
        if not task_type:
            return self._err("task_type required", status=400)

        policy = payload.get("policy")
        if isinstance(policy, str):
            policy = _json_loads(policy)
        if not isinstance(policy, dict):
            policy = {}

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

        def _canonicalize_policy(
            *,
            task_type: str,
            policy: dict[str, Any],
            existing_policy: dict[str, Any],
            defaults_cfg: dict[str, Any],
        ) -> dict[str, Any]:
            merged = _deep_merge_dict(existing_policy or {}, policy or {})
            # Always persist "effective" policy by filling missing keys from config defaults.
            # This keeps tasks.policy_json self-contained for inspection/debugging.
            for k, v in (defaults_cfg or {}).items():
                if k not in merged:
                    merged[k] = v

            if task_type != "wake_up":
                return merged

            defaults = {
                "window_minutes": merged.get("window_minutes", 30),
                "cooldown_sec": merged.get("cooldown_sec", 300),
                "offline_retry_sec": merged.get("offline_retry_sec", 300),
                "max_attempts": merged.get("max_attempts", 6),
                "nudge_enabled": merged.get("nudge_enabled", True),
            }
            for k, v in defaults.items():
                if k not in merged:
                    merged[k] = v

            # Normalize types/ranges.
            merged["window_minutes"] = max(1, _safe_int(merged.get("window_minutes"), 30))
            merged["cooldown_sec"] = max(1, _safe_int(merged.get("cooldown_sec"), 300))
            merged["offline_retry_sec"] = max(1, _safe_int(merged.get("offline_retry_sec"), merged["cooldown_sec"]))
            merged["max_attempts"] = max(0, _safe_int(merged.get("max_attempts"), 6))
            merged["nudge_enabled"] = bool(merged.get("nudge_enabled", True))

            device_id = merged.get("device_id")
            if isinstance(device_id, str):
                merged["device_id"] = device_id.strip()

            # `min_confidence` is intentionally not used by task_engine (wake_up included).
            merged.pop("min_confidence", None)
            return merged

        try:
            store = self._store()
            existing = store.get_task(account_id=account_id, task_type=task_type)
            existing_policy = _json_loads((existing or {}).get("policy_json"))
            defaults_cfg = _load_task_type_defaults(self.config, task_type)
            policy_to_store = _canonicalize_policy(
                task_type=task_type,
                policy=policy,
                existing_policy=existing_policy,
                defaults_cfg=defaults_cfg,
            )

            t = store.upsert_task(
                account_id=account_id,
                task_type=task_type,
                enabled=enabled,
                policy=policy_to_store,
                now_ms=_now_ms(),
            )
            out = dict(t)
            out["policy"] = _json_loads(out.get("policy_json"))
            out.pop("policy_json", None)
            return self._ok(out)
        except Exception as e:
            return self._err(str(e) or "internal_error", status=500)

    async def handle_get(self, request):
        account_id = str(request.match_info.get("account_id") or "").strip()
        task_type = str(request.match_info.get("task_type") or "").strip()
        if not task_type:
            task_type = str(request.query.get("task_type") or "").strip()
        if not task_type:
            return self._err("task_type required", status=400)
        try:
            store = self._store()
            t = store.get_task(account_id=account_id, task_type=task_type)
            if not t:
                return self._err("task_not_found", status=404)
            inst = store.get_latest_instance(account_id=account_id, task_type=task_type)
            out = dict(t)
            out["policy"] = _json_loads(out.get("policy_json"))
            out.pop("policy_json", None)
            out["latest_instance"] = inst
            return self._ok(out)
        except Exception as e:
            return self._err(str(e) or "internal_error", status=500)

    async def handle_delete_task(self, request):
        account_id = str(request.match_info.get("account_id") or "").strip()
        task_type = str(request.match_info.get("task_type") or "").strip()
        if not task_type:
            task_type = str(request.query.get("task_type") or "").strip()
        if not task_type:
            return self._err("task_type required", status=400)
        try:
            store = self._store()
            deleted = store.delete_task(account_id=account_id, task_type=task_type)
            if deleted <= 0:
                return self._err("task_not_found", status=404)
            return self._ok({"deleted": int(deleted), "account_id": account_id, "task_type": task_type})
        except Exception as e:
            return self._err(str(e) or "internal_error", status=500)

    async def handle_kickoff(self, request):
        account_id = str(request.match_info.get("account_id") or "").strip()
        task_type = str(request.match_info.get("task_type") or "").strip()
        if not task_type:
            task_type = str(request.query.get("task_type") or "").strip()
        if not task_type:
            return self._err("task_type required", status=400)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        now_ms = _now_ms()

        instance_key = str(payload.get("instance_key") or "").strip() or _today_key()
        planned_at_ms_raw = payload.get("planned_at_ms")
        planned_time = payload.get("planned_time")
        if planned_at_ms_raw is None and planned_time:
            planned_at_ms_raw = _parse_date_hhmm_to_ms(instance_key, str(planned_time))
        try:
            planned_at_ms_raw = int(planned_at_ms_raw) if planned_at_ms_raw is not None else None
        except Exception:
            planned_at_ms_raw = None

        touch_planned = planned_at_ms_raw is not None
        touch_window = "window_minutes" in payload
        touch_max_runs = "max_runs" in payload

        next_action_at_ms_raw = payload.get("next_action_at_ms")
        try:
            next_action_at_ms_raw = int(next_action_at_ms_raw) if next_action_at_ms_raw is not None else None
        except Exception:
            next_action_at_ms_raw = None

        store = self._store()
        t = store.get_task(account_id=account_id, task_type=task_type)
        if not t:
            return self._err("task_not_found", status=404)

        existing_inst = store.get_instance_by_key(
            account_id=account_id,
            task_type=task_type,
            instance_key=instance_key,
        )
        if existing_inst:
            status = str(existing_inst.get("status") or "").strip().upper()
            if status in ("COMPLETED", "ABANDONED", "CANCELED"):
                resp = web.json_response(
                    {
                        "success": False,
                        "message": "instance_terminal",
                        "data": {"instance_key": instance_key, "status": status},
                    },
                    status=409,
                )
                self._add_cors_headers(resp)
                return resp
            resume_from_paused = status == "PAUSED"
        else:
            resume_from_paused = False

        policy = _json_loads((t or {}).get("policy_json"))
        if task_type == "wake_up" and not str(policy.get("device_id") or "").strip():
            return self._err("device_id required for wake_up (set it in POST /tasks policy)", status=400)

        # Decide schedule/window values.
        if existing_inst and not touch_planned and not touch_window:
            planned_at_ms = int(existing_inst.get("planned_at_ms") or now_ms)
            window_start_at_ms = int(existing_inst.get("window_start_at_ms") or planned_at_ms)
            window_end_at_ms = int(existing_inst.get("window_end_at_ms") or planned_at_ms)
        else:
            planned_at_ms = (
                int(planned_at_ms_raw)
                if planned_at_ms_raw is not None
                else int(existing_inst.get("planned_at_ms") or now_ms)
                if existing_inst
                else now_ms
            )

            defaults_cfg = _load_task_type_defaults(self.config, task_type)
            window_minutes = payload.get(
                "window_minutes",
                policy.get("window_minutes", defaults_cfg.get("window_minutes", 30)),
            )
            try:
                window_minutes = int(window_minutes)
            except Exception:
                window_minutes = 30
            if window_minutes < 1:
                window_minutes = 1

            window_start_at_ms = planned_at_ms
            window_end_at_ms = planned_at_ms + window_minutes * 60 * 1000

        # Decide max_runs.
        max_runs = payload.get("max_runs") if touch_max_runs else None
        if max_runs is not None:
            try:
                max_runs = int(max_runs)
            except Exception:
                max_runs = None
        if max_runs is None and existing_inst:
            try:
                max_runs = int(existing_inst.get("max_runs") or 0) or None
            except Exception:
                max_runs = None
        if max_runs is None:
            defaults_cfg = _load_task_type_defaults(self.config, task_type)
            max_runs = defaults_cfg.get("max_runs")
        if max_runs is None:
            if task_type == "wake_up":
                defaults_cfg = _load_task_type_defaults(self.config, task_type)
                cooldown_sec = policy.get("cooldown_sec", defaults_cfg.get("cooldown_sec", 300))
                offline_retry_sec = policy.get("offline_retry_sec", defaults_cfg.get("offline_retry_sec", 300))
                try:
                    cooldown_sec = int(cooldown_sec)
                except Exception:
                    cooldown_sec = 300
                try:
                    offline_retry_sec = int(offline_retry_sec)
                except Exception:
                    offline_retry_sec = cooldown_sec
                if cooldown_sec < 1:
                    cooldown_sec = 1
                if offline_retry_sec < 1:
                    offline_retry_sec = 1
                interval_sec = min(cooldown_sec, offline_retry_sec)
                window_sec = max(1, int((window_end_at_ms - window_start_at_ms) / 1000))
                max_runs = max(1, (window_sec + interval_sec - 1) // interval_sec)
            else:
                max_runs = 20
        try:
            max_runs = int(max_runs)
        except Exception:
            max_runs = 20
        if max_runs < 1:
            max_runs = 1

        # Decide next_action.
        next_action_at_ms = int(next_action_at_ms_raw) if next_action_at_ms_raw is not None else now_ms
        if next_action_at_ms < int(window_start_at_ms):
            next_action_at_ms = int(window_start_at_ms)
        if next_action_at_ms > int(window_end_at_ms):
            next_action_at_ms = int(window_end_at_ms)

        instance = store.ensure_instance(
            account_id=account_id,
            task_type=task_type,
            instance_key=instance_key,
            status="PENDING",
            planned_at_ms=planned_at_ms,
            window_start_at_ms=window_start_at_ms,
            window_end_at_ms=window_end_at_ms,
            next_action_at_ms=next_action_at_ms,
            max_runs=int(max_runs),
            now_ms=now_ms,
        )
        if resume_from_paused and instance:
            store.set_instance_status(instance_id=int(instance["instance_id"]), status="PENDING", now_ms=now_ms)
            instance = store.get_instance_by_id(int(instance["instance_id"])) or instance
        return self._ok(instance)

    async def handle_run(self, request):
        account_id = str(request.match_info.get("account_id") or "").strip()
        task_type = str(request.match_info.get("task_type") or "").strip()
        if not task_type:
            task_type = str(request.query.get("task_type") or "").strip()
        if not task_type:
            return self._err("task_type required", status=400)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        instance_key = str(payload.get("instance_key") or request.query.get("instance_key") or "").strip() or _today_key()
        now_ms = _now_ms()
        store = self._store()
        t = store.get_task(account_id=account_id, task_type=task_type)
        if not t:
            return self._err("task_not_found", status=404)
        policy = _json_loads((t or {}).get("policy_json"))
        if task_type == "wake_up" and not str(policy.get("device_id") or "").strip():
            return self._err("device_id required for wake_up (set it in POST /tasks policy)", status=400)
        inst = store.get_instance_by_key(account_id=account_id, task_type=task_type, instance_key=instance_key)
        if not inst:
            return self._err("instance_not_found (call kickoff first)", status=404)

        status = str(inst.get("status") or "").strip().upper()
        if status in ("COMPLETED", "ABANDONED", "CANCELED"):
            return self._err("instance_terminal", status=409)
        if status == "PAUSED":
            store.set_instance_status(
                instance_id=int(inst["instance_id"]),
                status="PENDING",
                next_action_at_ms=now_ms,
                now_ms=now_ms,
            )
        else:
            store.set_instance_next_action(instance_id=int(inst["instance_id"]), next_action_at_ms=now_ms, now_ms=now_ms)
        return self._ok({"instance_id": int(inst.get("instance_id") or 0), "next_action_at_ms": now_ms})

    async def handle_pause(self, request):
        account_id = str(request.match_info.get("account_id") or "").strip()
        task_type = str(request.match_info.get("task_type") or "").strip()
        if not task_type:
            task_type = str(request.query.get("task_type") or "").strip()
        if not task_type:
            return self._err("task_type required", status=400)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        instance_key = str(payload.get("instance_key") or request.query.get("instance_key") or "").strip() or _today_key()
        now_ms = _now_ms()
        store = self._store()
        inst = store.get_instance_by_key(account_id=account_id, task_type=task_type, instance_key=instance_key)
        if not inst:
            return self._err("instance_not_found", status=404)
        status = str(inst.get("status") or "").strip().upper()
        if status in ("COMPLETED", "ABANDONED", "CANCELED"):
            return self._err("instance_terminal", status=409)
        store.set_instance_status(instance_id=int(inst["instance_id"]), status="PAUSED", now_ms=now_ms)
        return self._ok({"instance_id": int(inst["instance_id"]), "status": "PAUSED"})

    async def handle_cancel(self, request):
        account_id = str(request.match_info.get("account_id") or "").strip()
        task_type = str(request.match_info.get("task_type") or "").strip()
        if not task_type:
            task_type = str(request.query.get("task_type") or "").strip()
        if not task_type:
            return self._err("task_type required", status=400)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        instance_key = str(payload.get("instance_key") or request.query.get("instance_key") or "").strip() or _today_key()
        now_ms = _now_ms()
        store = self._store()
        inst = store.get_instance_by_key(account_id=account_id, task_type=task_type, instance_key=instance_key)
        if not inst:
            return self._err("instance_not_found", status=404)
        status = str(inst.get("status") or "").strip().upper()
        if status in ("COMPLETED", "ABANDONED"):
            return self._err("instance_terminal", status=409)
        if status == "CANCELED":
            return self._ok({"instance_id": int(inst["instance_id"]), "status": "CANCELED"})
        store.set_instance_status(instance_id=int(inst["instance_id"]), status="CANCELED", now_ms=now_ms)
        return self._ok({"instance_id": int(inst["instance_id"]), "status": "CANCELED"})

    async def handle_attempts(self, request):
        account_id = str(request.match_info.get("account_id") or "").strip()
        task_type = str(request.match_info.get("task_type") or "").strip()
        if not task_type:
            task_type = str(request.query.get("task_type") or "").strip()
        if not task_type:
            return self._err("task_type required", status=400)
        store = self._store()
        instance_key = str(request.query.get("instance_key") or "").strip()
        if instance_key:
            inst = store.get_instance_by_key(account_id=account_id, task_type=task_type, instance_key=instance_key)
        else:
            attempts = store.list_attempts_by_account_task(account_id=account_id, task_type=task_type, limit=200)
            return self._ok({"scope": "all", "attempts": attempts})
        if not inst:
            return self._err("instance_not_found", status=404)
        attempts = store.list_attempts(instance_id=int(inst["instance_id"]), limit=100)
        return self._ok({"scope": "instance", "instance": inst, "attempts": attempts})

    async def handle_delete_instance(self, request):
        account_id = str(request.match_info.get("account_id") or "").strip()
        task_type = str(request.match_info.get("task_type") or "").strip()
        instance_key = str(request.match_info.get("instance_key") or "").strip()
        if not task_type:
            task_type = str(request.query.get("task_type") or "").strip()
        if not task_type:
            return self._err("task_type required", status=400)
        if not instance_key:
            instance_key = str(request.query.get("instance_key") or "").strip()
        if not instance_key:
            return self._err("instance_key required", status=400)

        store = self._store()
        inst = store.get_instance_by_key(account_id=account_id, task_type=task_type, instance_key=instance_key)
        if not inst:
            return self._err("instance_not_found", status=404)

        deleted = store.delete_instance_by_key(account_id=account_id, task_type=task_type, instance_key=instance_key)
        return self._ok({"deleted": int(deleted), "instance_key": instance_key})
