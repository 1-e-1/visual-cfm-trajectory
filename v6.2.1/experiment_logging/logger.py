"""Append-only task summaries and lossless per-rollout NPZ logging."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .schema import SUMMARY_COLUMNS, TIMESERIES_FIELDS, UNITS
except ImportError:
    from schema import SUMMARY_COLUMNS, TIMESERIES_FIELDS, UNITS


def _csv_value(value: Any) -> Any:
    if value is None:
        return "NaN"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, bool):
        return int(value)
    return value


class ExperimentLogger:
    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dir = self.output_dir / "trajectories"
        self.trajectory_dir.mkdir(exist_ok=True)
        self.summary_path = self.output_dir / "summary.csv"
        metadata_path = self.output_dir / "schema_metadata.json"
        if not metadata_path.exists():
            metadata_path.write_text(
                json.dumps(
                    {"summary_units": UNITS, "timeseries_fields": TIMESERIES_FIELDS},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def save_timeseries(
        self,
        task_id: str,
        method: str,
        seed: int,
        arrays: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        missing = sorted(set(TIMESERIES_FIELDS) - set(arrays))
        if missing:
            raise ValueError(f"missing timeseries fields: {missing}")
        lengths = {key: len(np.asarray(value)) for key, value in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"timeseries lengths differ: {lengths}")
        path = self.trajectory_dir / f"{task_id}__{method}__seed{seed}.npz"
        payload = {key: np.asarray(value) for key, value in arrays.items()}
        payload["metadata_json"] = np.asarray(
            json.dumps(dict(metadata or {}), ensure_ascii=False)
        )
        np.savez_compressed(path, **payload)
        return path

    def append_summary(self, row: Mapping[str, Any]) -> None:
        unknown = sorted(set(row) - set(SUMMARY_COLUMNS))
        if unknown:
            raise ValueError(f"unknown summary columns: {unknown}")
        write_header = not self.summary_path.exists() or self.summary_path.stat().st_size == 0
        with self.summary_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({key: _csv_value(row.get(key)) for key in SUMMARY_COLUMNS})
