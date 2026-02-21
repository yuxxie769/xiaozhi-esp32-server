"""Home Assistant State Hub (WS-only).

Provides a lightweight, always-on bridge that:
- Extracts an allowlist from HA using `extract_from_target`
- Subscribes to those entities via `subscribe_entities`
- Maintains the latest state in a single in-memory store
- Optionally persists the latest store snapshot under data/ for offline viewing
"""

