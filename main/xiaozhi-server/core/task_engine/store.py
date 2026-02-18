from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Iterable


_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS tasks (
  account_id     TEXT    NOT NULL,
  task_type      TEXT    NOT NULL,
  enabled        INTEGER NOT NULL,
  policy_json    TEXT    NOT NULL DEFAULT '{}',
  created_at_ms  INTEGER NOT NULL,
  updated_at_ms  INTEGER NOT NULL,
  PRIMARY KEY (account_id, task_type)
);

CREATE TABLE IF NOT EXISTS task_instances (
  instance_id         INTEGER PRIMARY KEY AUTOINCREMENT,
  account_id          TEXT    NOT NULL,
  task_type           TEXT    NOT NULL,
  instance_key        TEXT    NOT NULL,
  status              TEXT    NOT NULL,
  planned_at_ms       INTEGER NOT NULL,
  window_start_at_ms  INTEGER NOT NULL,
  window_end_at_ms    INTEGER NOT NULL,
  next_action_at_ms   INTEGER NOT NULL,
  run_count           INTEGER NOT NULL DEFAULT 0,
  max_runs            INTEGER NOT NULL DEFAULT 20,
  attempt_count       INTEGER NOT NULL DEFAULT 0,
  created_at_ms       INTEGER NOT NULL,
  updated_at_ms       INTEGER NOT NULL,
  UNIQUE (account_id, task_type, instance_key),
  FOREIGN KEY (account_id, task_type)
    REFERENCES tasks(account_id, task_type)
    ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_task_instances_due
ON task_instances(status, next_action_at_ms);

CREATE INDEX IF NOT EXISTS idx_task_instances_account
ON task_instances(account_id, task_type);

CREATE TABLE IF NOT EXISTS task_attempts (
  attempt_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  instance_id    INTEGER NOT NULL,
  at_ms          INTEGER NOT NULL,
  result_code    TEXT    NOT NULL,
  result_json    TEXT    NOT NULL DEFAULT '{}',
  decision_code  TEXT    NOT NULL,
  decision_json  TEXT    NOT NULL DEFAULT '{}',
  FOREIGN KEY (instance_id)
    REFERENCES task_instances(instance_id)
    ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_task_attempts_instance_time
ON task_attempts(instance_id, at_ms DESC);
""".strip()


def _json_loads(s: str | None) -> dict[str, Any]:
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return "{}"


class TaskStore:
    def __init__(self, db_path: str):
        self.db_path = str(db_path or "").strip() or os.path.join("data", "tasks.db")

    def init_schema(self) -> None:
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        self._migrate_add_columns(
            conn,
            "task_instances",
            {
                "run_count": "run_count INTEGER NOT NULL DEFAULT 0",
                "max_runs": "max_runs INTEGER NOT NULL DEFAULT 20",
            },
        )

    def _migrate_add_columns(
        self, conn: sqlite3.Connection, table: str, columns: dict[str, str]
    ) -> None:
        table = str(table or "").strip()
        if not table:
            return
        existing = {
            str(r["name"])
            for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
            if r and "name" in r.keys()
        }
        for name, ddl in (columns or {}).items():
            if name in existing:
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def upsert_task(
        self,
        *,
        account_id: str,
        task_type: str,
        enabled: bool,
        policy: dict[str, Any] | None,
        now_ms: int,
    ) -> dict[str, Any]:
        account_id = str(account_id or "").strip()
        task_type = str(task_type or "").strip()
        if not account_id or not task_type:
            raise ValueError("account_id/task_type required")

        policy_obj = policy if isinstance(policy, dict) else {}

        with self._connect() as conn:
            row = conn.execute(
                "SELECT policy_json, created_at_ms FROM tasks WHERE account_id=? AND task_type=?",
                (account_id, task_type),
            ).fetchone()
            created_at_ms = int(now_ms)
            merged_policy: dict[str, Any] = {}
            if row:
                created_at_ms = int(row["created_at_ms"] or now_ms)
                merged_policy = _json_loads(row["policy_json"])
            merged_policy.update(policy_obj)

            conn.execute(
                """
                INSERT INTO tasks(account_id, task_type, enabled, policy_json, created_at_ms, updated_at_ms)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(account_id, task_type) DO UPDATE SET
                  enabled=excluded.enabled,
                  policy_json=excluded.policy_json,
                  updated_at_ms=excluded.updated_at_ms
                """.strip(),
                (
                    account_id,
                    task_type,
                    1 if enabled else 0,
                    _json_dumps(merged_policy),
                    created_at_ms,
                    int(now_ms),
                ),
            )

            out = conn.execute(
                "SELECT * FROM tasks WHERE account_id=? AND task_type=?",
                (account_id, task_type),
            ).fetchone()

        return dict(out) if out else {}

    def get_task(self, *, account_id: str, task_type: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE account_id=? AND task_type=?",
                (str(account_id or ""), str(task_type or "")),
            ).fetchone()
        return dict(row) if row else None

    def delete_task(self, *, account_id: str, task_type: str) -> int:
        account_id = str(account_id or "").strip()
        task_type = str(task_type or "").strip()
        if not account_id or not task_type:
            raise ValueError("account_id/task_type required")
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM tasks WHERE account_id=? AND task_type=?",
                (account_id, task_type),
            )
            return int(cur.rowcount or 0)

    def list_tasks(self, *, limit: int = 200) -> list[dict[str, Any]]:
        limit = max(1, int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY updated_at_ms DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_tasks_by_account(self, *, account_id: str, limit: int = 200) -> list[dict[str, Any]]:
        limit = max(1, int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tasks
                WHERE account_id=?
                ORDER BY updated_at_ms DESC
                LIMIT ?
                """.strip(),
                (str(account_id or ""), limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_instances_by_account(
        self, *, account_id: str, task_type: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        limit = max(1, int(limit))
        account_id = str(account_id or "").strip()
        task_type = str(task_type or "").strip()
        with self._connect() as conn:
            if task_type:
                rows = conn.execute(
                    """
                    SELECT * FROM task_instances
                    WHERE account_id=? AND task_type=?
                    ORDER BY planned_at_ms DESC
                    LIMIT ?
                    """.strip(),
                    (account_id, task_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM task_instances
                    WHERE account_id=?
                    ORDER BY planned_at_ms DESC
                    LIMIT ?
                    """.strip(),
                    (account_id, limit),
                ).fetchall()
        return [dict(r) for r in rows]

    def ensure_instance(
        self,
        *,
        account_id: str,
        task_type: str,
        instance_key: str,
        status: str,
        planned_at_ms: int,
        window_start_at_ms: int,
        window_end_at_ms: int,
        next_action_at_ms: int,
        max_runs: int | None = None,
        now_ms: int,
    ) -> dict[str, Any]:
        account_id = str(account_id or "").strip()
        task_type = str(task_type or "").strip()
        instance_key = str(instance_key or "").strip()
        status = str(status or "").strip() or "PENDING"
        if not account_id or not task_type or not instance_key:
            raise ValueError("account_id/task_type/instance_key required")

        if max_runs is None:
            max_runs = 20
        try:
            max_runs = int(max_runs)
        except Exception:
            max_runs = 20
        if max_runs < 1:
            max_runs = 1

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO task_instances(
                  account_id, task_type, instance_key, status,
                  planned_at_ms, window_start_at_ms, window_end_at_ms,
                  next_action_at_ms, run_count, max_runs, attempt_count,
                  created_at_ms, updated_at_ms
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """.strip(),
                (
                    account_id,
                    task_type,
                    instance_key,
                    status,
                    int(planned_at_ms),
                    int(window_start_at_ms),
                    int(window_end_at_ms),
                    int(next_action_at_ms),
                    0,
                    int(max_runs),
                    0,
                    int(now_ms),
                    int(now_ms),
                ),
            )

            row = conn.execute(
                """
                SELECT * FROM task_instances
                WHERE account_id=? AND task_type=? AND instance_key=?
                """.strip(),
                (account_id, task_type, instance_key),
            ).fetchone()
            if not row:
                raise RuntimeError("failed to create task instance")

            # If instance exists and is non-terminal, allow updating schedule/window + next_action_at.
            if str(row["status"]) in ("PENDING", "IN_PROGRESS", "PAUSED"):
                conn.execute(
                    """
                    UPDATE task_instances
                    SET planned_at_ms=?, window_start_at_ms=?, window_end_at_ms=?,
                        next_action_at_ms=?, max_runs=?, updated_at_ms=?
                    WHERE instance_id=?
                    """.strip(),
                    (
                        int(planned_at_ms),
                        int(window_start_at_ms),
                        int(window_end_at_ms),
                        int(next_action_at_ms),
                        int(max_runs),
                        int(now_ms),
                        int(row["instance_id"]),
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM task_instances WHERE instance_id=?",
                    (int(row["instance_id"]),),
                ).fetchone()

        return dict(row) if row else {}

    def get_instance_by_id(self, instance_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM task_instances WHERE instance_id=?",
                (int(instance_id),),
            ).fetchone()
        return dict(row) if row else None

    def get_instance_by_key(
        self, *, account_id: str, task_type: str, instance_key: str
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM task_instances
                WHERE account_id=? AND task_type=? AND instance_key=?
                """.strip(),
                (str(account_id or ""), str(task_type or ""), str(instance_key or "")),
            ).fetchone()
        return dict(row) if row else None

    def get_latest_instance(self, *, account_id: str, task_type: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM task_instances
                WHERE account_id=? AND task_type=?
                ORDER BY planned_at_ms DESC
                LIMIT 1
                """.strip(),
                (str(account_id or ""), str(task_type or "")),
            ).fetchone()
        return dict(row) if row else None

    def delete_instance_by_key(self, *, account_id: str, task_type: str, instance_key: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM task_instances
                WHERE account_id=? AND task_type=? AND instance_key=?
                """.strip(),
                (str(account_id or ""), str(task_type or ""), str(instance_key or "")),
            )
            return int(cur.rowcount or 0)

    def list_due_instances(self, *, now_ms: int, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  i.*,
                  t.enabled AS task_enabled,
                  t.policy_json AS task_policy_json
                FROM task_instances i
                JOIN tasks t
                  ON t.account_id=i.account_id AND t.task_type=i.task_type
                WHERE t.enabled=1
                  AND i.status IN ('PENDING','IN_PROGRESS')
                  AND i.next_action_at_ms <= ?
                ORDER BY i.next_action_at_ms ASC
                LIMIT ?
                """.strip(),
                (int(now_ms), limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_attempts_by_account_task(
        self, *, account_id: str, task_type: str, limit: int = 200
    ) -> list[dict[str, Any]]:
        account_id = str(account_id or "").strip()
        task_type = str(task_type or "").strip()
        limit = max(1, int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  a.*,
                  i.instance_key AS instance_key,
                  i.status AS instance_status,
                  i.planned_at_ms AS planned_at_ms
                FROM task_attempts a
                JOIN task_instances i
                  ON i.instance_id = a.instance_id
                WHERE i.account_id=? AND i.task_type=?
                ORDER BY a.at_ms DESC
                LIMIT ?
                """.strip(),
                (account_id, task_type, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_instance_status(
        self, *, instance_id: int, status: str, now_ms: int, next_action_at_ms: int | None = None
    ) -> None:
        status = str(status or "").strip()
        if not status:
            raise ValueError("status required")
        with self._connect() as conn:
            if next_action_at_ms is None:
                conn.execute(
                    "UPDATE task_instances SET status=?, updated_at_ms=? WHERE instance_id=?",
                    (status, int(now_ms), int(instance_id)),
                )
            else:
                conn.execute(
                    """
                    UPDATE task_instances
                    SET status=?, next_action_at_ms=?, updated_at_ms=?
                    WHERE instance_id=?
                    """.strip(),
                    (status, int(next_action_at_ms), int(now_ms), int(instance_id)),
                )

    def set_instance_next_action(self, *, instance_id: int, next_action_at_ms: int, now_ms: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE task_instances
                SET next_action_at_ms=?, updated_at_ms=?
                WHERE instance_id=?
                """.strip(),
                (int(next_action_at_ms), int(now_ms), int(instance_id)),
            )

    def increment_attempt_count(self, *, instance_id: int, now_ms: int) -> int:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE task_instances
                SET attempt_count = attempt_count + 1,
                    updated_at_ms=?
                WHERE instance_id=?
                """.strip(),
                (int(now_ms), int(instance_id)),
            )
            row = conn.execute(
                "SELECT attempt_count FROM task_instances WHERE instance_id=?",
                (int(instance_id),),
            ).fetchone()
        return int(row["attempt_count"] or 0) if row else 0

    def increment_run_count(self, *, instance_id: int, now_ms: int) -> int:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE task_instances
                SET run_count = run_count + 1,
                    updated_at_ms=?
                WHERE instance_id=?
                """.strip(),
                (int(now_ms), int(instance_id)),
            )
            row = conn.execute(
                "SELECT run_count FROM task_instances WHERE instance_id=?",
                (int(instance_id),),
            ).fetchone()
        return int(row["run_count"] or 0) if row else 0

    def append_attempt(
        self,
        *,
        instance_id: int,
        at_ms: int,
        result_code: str,
        result_json: dict[str, Any] | None,
        decision_code: str,
        decision_json: dict[str, Any] | None,
    ) -> int:
        result_code = str(result_code or "").strip() or "unknown"
        decision_code = str(decision_code or "").strip() or "skip"
        res_obj = result_json if isinstance(result_json, dict) else {}
        dec_obj = decision_json if isinstance(decision_json, dict) else {}
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO task_attempts(
                  instance_id, at_ms,
                  result_code, result_json,
                  decision_code, decision_json
                ) VALUES(?,?,?,?,?,?)
                """.strip(),
                (
                    int(instance_id),
                    int(at_ms),
                    result_code,
                    _json_dumps(res_obj),
                    decision_code,
                    _json_dumps(dec_obj),
                ),
            )
            return int(cur.lastrowid or 0)

    def list_attempts(self, *, instance_id: int, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM task_attempts
                WHERE instance_id=?
                ORDER BY at_ms DESC
                LIMIT ?
                """.strip(),
                (int(instance_id), limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def execute_many(self, sql: str, params_seq: Iterable[tuple[Any, ...]]) -> None:
        with self._connect() as conn:
            conn.executemany(sql, list(params_seq))
