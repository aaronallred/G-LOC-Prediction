"""Small reusable wall-clock timing recorder for experiment runs."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from .runtime import runtime_summary, synchronize_accelerator

logger = logging.getLogger(__name__)


class RunTimer:
    """Track named stages and write a machine-readable JSON report."""

    def __init__(self) -> None:
        self.started_at = datetime.now(timezone.utc)
        self._start = perf_counter()
        self.stages: list[dict[str, Any]] = []

    @contextmanager
    def track(self, name: str, **metadata: Any) -> Iterator[None]:
        synchronize_accelerator()
        start = perf_counter()
        record: dict[str, Any] = {"name": name, **metadata}
        try:
            yield
        except Exception as exc:
            record.update({"status": "failed", "error": str(exc)})
            raise
        else:
            record["status"] = "completed"
        finally:
            synchronize_accelerator()
            record["duration_seconds"] = round(perf_counter() - start, 6)
            self.stages.append(record)
            logger.info("Timing | %s | %.3f seconds", name, record["duration_seconds"])

    def report(self) -> dict[str, Any]:
        ended_at = datetime.now(timezone.utc)
        return {
            "started_at": self.started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "total_duration_seconds": round(perf_counter() - self._start, 6),
            "runtime": runtime_summary(),
            "stages": self.stages,
        }

    def save(self, path: str | Path) -> Path:
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.report(), indent=2), encoding="utf-8")
        logger.info("Saved run timing report to %s", output_path)
        return output_path
