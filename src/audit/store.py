"""SQLite 기반 의사결정 감사 추적 저장소.

API 키나 인증 토큰은 저장하지 않는다. 사용자·도구 입력은 재현에 필요한
구조화 값만 JSON으로 남기고, 민감한 이름의 필드는 저장 직전에 마스킹한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config


_SECRET_TOKENS = (
    "api_key", "apikey", "secret", "token", "authorization", "password",
    "client_secret", "service_key", "openai",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _redact(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if any(token in lowered for token in _SECRET_TOKENS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str) and value.startswith("sk-"):
        return "[REDACTED]"
    return value


class DecisionAuditStore:
    """각 추천의 입력·모델·계산·근거를 하나의 run으로 묶는다."""

    def __init__(self, db_path: Path | str = config.DB_PATH):
        self.db_path = Path(db_path)
        self.ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as connection:
            connection.executescript("""
            CREATE TABLE IF NOT EXISTS decision_runs (
                decision_run_id TEXT PRIMARY KEY,
                session_id TEXT,
                property_id TEXT,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                code_version TEXT,
                data_version TEXT,
                simulation_seed INTEGER,
                model_versions_json TEXT,
                input_json TEXT,
                result_json TEXT,
                elapsed_ms REAL,
                error_type TEXT,
                error_message TEXT
            );
            CREATE TABLE IF NOT EXISTS decision_steps (
                step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_run_id TEXT NOT NULL,
                sequence_no INTEGER NOT NULL,
                stage TEXT NOT NULL,
                tool TEXT,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                elapsed_ms REAL,
                input_digest TEXT,
                output_digest TEXT,
                input_json TEXT,
                output_json TEXT,
                sql_text TEXT,
                sql_parameters_json TEXT,
                source_refs_json TEXT,
                fallback_json TEXT,
                error_type TEXT,
                error_message TEXT,
                FOREIGN KEY(decision_run_id) REFERENCES decision_runs(decision_run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_decision_steps_run
                ON decision_steps(decision_run_id, sequence_no);
            CREATE INDEX IF NOT EXISTS idx_decision_runs_property
                ON decision_runs(property_id, started_at DESC);
            """)

    def start_run(
        self,
        *,
        session_id: str | None,
        property_id: str | None,
        input_snapshot: dict,
        simulation_seed: int,
        model_versions: dict | None = None,
        data_version: str | None = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        safe_input = _redact(input_snapshot)
        code_version = os.getenv("APP_GIT_SHA") or os.getenv("GIT_COMMIT") or "workspace"
        with self._conn() as connection:
            connection.execute(
                """INSERT INTO decision_runs (
                    decision_run_id, session_id, property_id, status, started_at,
                    code_version, data_version, simulation_seed,
                    model_versions_json, input_json
                ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?)""",
                (run_id, session_id, property_id, _now(), code_version,
                 data_version, int(simulation_seed),
                 _json(_redact(model_versions or {})), _json(safe_input)),
            )
        return run_id

    def record_step(
        self,
        run_id: str,
        *,
        stage: str,
        tool: str | None = None,
        status: str = "ok",
        started_at: str | None = None,
        elapsed_ms: float | None = None,
        input_data: Any = None,
        output_data: Any = None,
        sql_text: str | None = None,
        sql_parameters: Any = None,
        source_refs: Any = None,
        fallback: Any = None,
        error: Exception | None = None,
    ) -> None:
        safe_input, safe_output = _redact(input_data), _redact(output_data)
        with self._conn() as connection:
            sequence = int(connection.execute(
                "SELECT COALESCE(MAX(sequence_no), 0) + 1 FROM decision_steps "
                "WHERE decision_run_id=?", (run_id,),
            ).fetchone()[0])
            connection.execute(
                """INSERT INTO decision_steps (
                    decision_run_id, sequence_no, stage, tool, status, started_at,
                    elapsed_ms, input_digest, output_digest, input_json, output_json,
                    sql_text, sql_parameters_json, source_refs_json, fallback_json,
                    error_type, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, sequence, stage, tool, status, started_at or _now(), elapsed_ms,
                 _digest(safe_input) if input_data is not None else None,
                 _digest(safe_output) if output_data is not None else None,
                 _json(safe_input) if input_data is not None else None,
                 _json(safe_output) if output_data is not None else None,
                 sql_text, _json(_redact(sql_parameters)) if sql_parameters is not None else None,
                 _json(_redact(source_refs)) if source_refs is not None else None,
                 _json(_redact(fallback)) if fallback is not None else None,
                 type(error).__name__ if error else None,
                 str(error)[:1000] if error else None),
            )

    def complete_run(self, run_id: str, result: dict, *, elapsed_ms: float) -> None:
        with self._conn() as connection:
            connection.execute(
                """UPDATE decision_runs SET status='completed', completed_at=?,
                   result_json=?, elapsed_ms=? WHERE decision_run_id=?""",
                (_now(), _json(_redact(result)), round(float(elapsed_ms), 3), run_id),
            )

    def fail_run(self, run_id: str, error: Exception, *, elapsed_ms: float) -> None:
        with self._conn() as connection:
            connection.execute(
                """UPDATE decision_runs SET status='failed', completed_at=?, elapsed_ms=?,
                   error_type=?, error_message=? WHERE decision_run_id=?""",
                (_now(), round(float(elapsed_ms), 3), type(error).__name__,
                 str(error)[:1000], run_id),
            )

    def fail_latest_running(
        self, *, session_id: str | None, property_id: str | None,
        error: Exception,
    ) -> str | None:
        """리포트 외곽에서 예외가 잡혀도 가장 최근 실행을 실패로 종결한다."""
        with self._conn() as connection:
            row = connection.execute(
                """SELECT decision_run_id FROM decision_runs
                   WHERE status='running' AND session_id IS ? AND property_id IS ?
                   ORDER BY started_at DESC LIMIT 1""",
                (session_id, property_id),
            ).fetchone()
        if row is None:
            return None
        self.fail_run(str(row[0]), error, elapsed_ms=0)
        return str(row[0])

    def get(self, run_id: str) -> dict | None:
        with self._conn() as connection:
            run = connection.execute(
                "SELECT * FROM decision_runs WHERE decision_run_id=?", (run_id,),
            ).fetchone()
            if run is None:
                return None
            steps = connection.execute(
                "SELECT * FROM decision_steps WHERE decision_run_id=? ORDER BY sequence_no",
                (run_id,),
            ).fetchall()
        payload = dict(run)
        for key in ("model_versions_json", "input_json", "result_json"):
            raw = payload.pop(key, None)
            payload[key.removesuffix("_json")] = json.loads(raw) if raw else None
        payload["steps"] = []
        for row in steps:
            item = dict(row)
            for key in ("input_json", "output_json", "sql_parameters_json",
                        "source_refs_json", "fallback_json"):
                raw = item.pop(key, None)
                item[key.removesuffix("_json")] = json.loads(raw) if raw else None
            payload["steps"].append(item)
        return payload


class timed_step:
    """작은 내부 단계의 실행시간 측정용 컨텍스트."""

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self.started) * 1000
