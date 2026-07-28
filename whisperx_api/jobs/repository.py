import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from whisperx_api.jobs.schemas import JobStage, JobStatus


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class JobRecord:
    id: str
    status: JobStatus
    stage: JobStage
    progress: float
    input_path: Path
    request: dict[str, Any]
    result: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    cancel_requested: bool
    worker_id: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class JobRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS transcription_job (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL,
                    input_path TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS transcription_job_status_created_at_idx
                    ON transcription_job(status, created_at);
                CREATE TABLE IF NOT EXISTS service_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            connection.commit()

    def create(self, *, input_path: Path, request: dict[str, Any]) -> JobRecord:
        job_id = f"job_{uuid4().hex}"
        now = utc_now()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO transcription_job (
                    id, status, stage, progress, input_path, request_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    JobStatus.QUEUED.value,
                    JobStage.QUEUED.value,
                    0,
                    str(input_path),
                    json.dumps(request),
                    now,
                    now,
                ),
            )
            connection.commit()
        record = self.get(job_id)
        assert record is not None
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM transcription_job WHERE id = ?",
                (job_id,),
            ).fetchone()
        return self._to_record(row) if row else None

    def list(
        self,
        *,
        status: JobStatus | None,
        limit: int,
        offset: int,
    ) -> list[JobRecord]:
        query = "SELECT * FROM transcription_job"
        params: list[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend((limit, offset))
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._to_record(row) for row in rows]

    def claim_next(self, worker_id: str) -> JobRecord | None:
        now = utc_now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT id FROM transcription_job
                WHERE status = ? AND cancel_requested = 0
                ORDER BY created_at
                LIMIT 1
                """,
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            connection.execute(
                """
                UPDATE transcription_job
                SET status = ?, stage = ?, progress = ?, worker_id = ?,
                    started_at = ?, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    JobStatus.RUNNING.value,
                    JobStage.DECODE.value,
                    1,
                    worker_id,
                    now,
                    now,
                    row["id"],
                    JobStatus.QUEUED.value,
                ),
            )
            connection.commit()
        return self.get(row["id"])

    def update_progress(self, job_id: str, stage: str, progress: float) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE transcription_job
                SET stage = ?, progress = MAX(progress, ?), updated_at = ?
                WHERE id = ? AND status IN (?, ?)
                """,
                (
                    stage,
                    max(0, min(100, progress)),
                    utc_now(),
                    job_id,
                    JobStatus.RUNNING.value,
                    JobStatus.CANCELLING.value,
                ),
            )
            connection.commit()

    def succeed(self, job_id: str, result: dict[str, Any]) -> None:
        now = utc_now()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE transcription_job
                SET status = ?, stage = ?, progress = 100, result_json = ?,
                    updated_at = ?, finished_at = ?
                WHERE id = ?
                """,
                (
                    JobStatus.SUCCEEDED.value,
                    JobStage.COMPLETE.value,
                    json.dumps(result, ensure_ascii=False),
                    now,
                    now,
                    job_id,
                ),
            )
            connection.commit()

    def fail(self, job_id: str, code: str, message: str) -> None:
        now = utc_now()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE transcription_job
                SET status = ?, error_code = ?, error_message = ?,
                    updated_at = ?, finished_at = ?
                WHERE id = ?
                """,
                (JobStatus.FAILED.value, code, message[:2000], now, now, job_id),
            )
            connection.commit()

    def request_cancel(self, job_id: str) -> JobRecord | None:
        existing = self.get(job_id)
        if existing is None or existing.status.terminal:
            return existing
        now = utc_now()
        new_status = (
            JobStatus.CANCELLED if existing.status == JobStatus.QUEUED else JobStatus.CANCELLING
        )
        finished_at = now if new_status == JobStatus.CANCELLED else None
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE transcription_job
                SET cancel_requested = 1, status = ?, updated_at = ?,
                    finished_at = COALESCE(?, finished_at)
                WHERE id = ?
                """,
                (new_status.value, now, finished_at, job_id),
            )
            connection.commit()
        return self.get(job_id)

    def cancel(self, job_id: str) -> None:
        now = utc_now()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE transcription_job
                SET status = ?, updated_at = ?, finished_at = ?
                WHERE id = ?
                """,
                (JobStatus.CANCELLED.value, now, now, job_id),
            )
            connection.commit()

    def heartbeat(self, worker_id: str) -> None:
        now = utc_now()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO service_state (key, value, updated_at)
                VALUES ('worker_heartbeat', ?, ?)
                ON CONFLICT(key) DO UPDATE
                SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (worker_id, now),
            )
            connection.commit()

    def get_state(self, key: str) -> tuple[str, datetime] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT value, updated_at FROM service_state WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        return str(row["value"]), datetime.fromisoformat(row["updated_at"])

    @staticmethod
    def _to_record(row: sqlite3.Row) -> JobRecord:
        def parse_datetime(value: str | None) -> datetime | None:
            return datetime.fromisoformat(value) if value else None

        return JobRecord(
            id=row["id"],
            status=JobStatus(row["status"]),
            stage=JobStage(row["stage"]),
            progress=float(row["progress"]),
            input_path=Path(row["input_path"]),
            request=json.loads(row["request_json"]),
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error_code=row["error_code"],
            error_message=row["error_message"],
            cancel_requested=bool(row["cancel_requested"]),
            worker_id=row["worker_id"],
            created_at=parse_datetime(row["created_at"]),
            updated_at=parse_datetime(row["updated_at"]),
            started_at=parse_datetime(row["started_at"]),
            finished_at=parse_datetime(row["finished_at"]),
        )
