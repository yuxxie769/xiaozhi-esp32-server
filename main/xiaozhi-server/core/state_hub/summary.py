from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _domain(entity_id: str) -> str:
    parts = str(entity_id or "").split(".", 1)
    return parts[0] if len(parts) == 2 else ""


def _pick(d: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        if k in d:
            out[k] = d.get(k)
    return out


class ClipRules:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg if isinstance(cfg, dict) else {}
        rep = cfg.get("report", {}) if isinstance(cfg.get("report", {}), dict) else {}

        self.exclude_domains = set(
            str(x)
            for x in (
                rep.get("exclude_domains")
                or ["diagnostic", "update", "button", "input_text"]
            )
            if x
        )
        self.exclude_entity_id_regex: List[re.Pattern] = []
        for pat in rep.get("exclude_entity_id_regex") or [r"^sensor\..*_linkquality$", r"^sensor\..*_rssi$"]:
            try:
                self.exclude_entity_id_regex.append(re.compile(str(pat)))
            except Exception:
                continue

        self.attributes_allowlist_by_domain: Dict[str, List[str]] = {}
        raw_map = rep.get("attributes_allowlist_by_domain", {})
        if isinstance(raw_map, dict):
            for dom, keys in raw_map.items():
                if isinstance(keys, list):
                    self.attributes_allowlist_by_domain[str(dom)] = [str(k) for k in keys if k]

        # Reasonable defaults for the compressed HA subscribe_entities payload you observed.
        self.attributes_allowlist_by_domain.setdefault(
            "sensor",
            ["friendly_name", "device_class", "unit_of_measurement", "state_class"],
        )
        self.attributes_allowlist_by_domain.setdefault(
            "switch",
            ["friendly_name", "device_class"],
        )
        self.attributes_allowlist_by_domain.setdefault(
            "binary_sensor",
            ["friendly_name", "device_class"],
        )
        self.attributes_allowlist_by_domain.setdefault("light", ["friendly_name", "device_class"])
        self.attributes_allowlist_by_domain.setdefault("climate", ["friendly_name", "device_class"])
        self.attributes_allowlist_by_domain.setdefault("person", ["friendly_name"])

        try:
            self.max_highlight = int(rep.get("max_highlight", 30))
        except Exception:
            self.max_highlight = 30
        self.max_highlight = max(1, min(self.max_highlight, 80))


def is_excluded_entity(entity_id: str, rules: ClipRules) -> bool:
    eid = str(entity_id or "").strip()
    if not eid:
        return True
    dom = _domain(eid)
    if dom in rules.exclude_domains:
        return True
    for rx in rules.exclude_entity_id_regex:
        try:
            if rx.search(eid):
                return True
        except Exception:
            continue
    return False


def normalize_entity(
    entity_id: str,
    raw: Dict[str, Any],
    rules: ClipRules,
) -> Dict[str, Any]:
    """Normalize HA's compressed entity payload into our minimal fields."""
    eid = str(entity_id or "").strip()
    dom = _domain(eid)
    state = raw.get("s")
    attrs = raw.get("a") if isinstance(raw.get("a"), dict) else {}
    allow = rules.attributes_allowlist_by_domain.get(dom) or ["friendly_name"]
    picked = _pick(attrs, allow)

    out: Dict[str, Any] = {"id": eid, "s": state}
    if "friendly_name" in picked and picked.get("friendly_name"):
        out["n"] = picked.get("friendly_name")
    dc = picked.get("device_class")
    if dc:
        out["dc"] = dc
    u = picked.get("unit_of_measurement")
    if u:
        out["u"] = u
    return out


def _priority(entity_id: str, normalized: Dict[str, Any]) -> Tuple[int, str]:
    """Lower is higher priority."""
    dom = _domain(entity_id)
    s = str(normalized.get("s") or "").lower()
    if dom == "binary_sensor" and s == "on":
        return (0, entity_id)
    if dom in ("light", "switch") and s == "on":
        return (1, entity_id)
    if dom == "climate" and s not in ("off", "idle", ""):
        return (2, entity_id)
    if dom == "person" and s in ("home", "not_home"):
        return (3, entity_id)
    return (9, entity_id)


def build_highlight(
    entities: Dict[str, Dict[str, Any]],
    *,
    exposure: Dict[str, bool],
    rules: ClipRules,
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    max_items = int(max_items or rules.max_highlight)
    max_items = max(1, min(max_items, 80))

    items: List[Tuple[Tuple[int, str], Dict[str, Any]]] = []
    for eid, raw in (entities or {}).items():
        if not isinstance(raw, dict):
            continue
        if is_excluded_entity(eid, rules):
            continue
        if exposure.get(eid) is False:
            continue
        normalized = normalize_entity(eid, raw, rules)
        items.append((_priority(eid, normalized), normalized))

    items.sort(key=lambda x: x[0])
    return [x[1] for x in items[:max_items]]


def build_entity_rows(
    entities: Dict[str, Dict[str, Any]],
    *,
    exposure: Dict[str, bool],
    rules: ClipRules,
) -> List[Dict[str, Any]]:
    """Build a clipped list of entities for UI (key fields only).

    Note: Unlike highlight, this list includes both exposed and hidden entities, with a
    `expose_to_llm` boolean to drive toggles.
    """
    rows: List[Dict[str, Any]] = []
    for eid, raw in (entities or {}).items():
        if not isinstance(raw, dict):
            continue
        if is_excluded_entity(eid, rules):
            continue
        normalized = normalize_entity(eid, raw, rules)
        normalized["expose_to_llm"] = bool(exposure.get(eid, True))
        rows.append(normalized)

    def _key(it: Dict[str, Any]) -> Tuple[str, str]:
        eid = str(it.get("id") or "")
        dom = _domain(eid)
        name = str(it.get("n") or "")
        return (dom, name or eid)

    rows.sort(key=_key)
    return rows


def build_local_summary_text(connected: bool, outdated: bool, age_s: Optional[int], highlight: List[Dict[str, Any]]) -> str:
    if not highlight:
        if not connected:
            return "我目前未连接到 Home Assistant，当前也没有可用的离线数据。"
        return "我已连接到 Home Assistant，但当前没有可汇报的关键状态。"

    top = highlight[:8]
    parts = []
    for it in top:
        n = it.get("n")
        s = it.get("s")
        if not n or s is None:
            continue
        parts.append(f"{n}:{s}")
    status = "在线" if (connected and not outdated) else "离线数据"
    return f"State Hub {status}：{'; '.join(parts)}"
