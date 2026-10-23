"""Project configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    return data
