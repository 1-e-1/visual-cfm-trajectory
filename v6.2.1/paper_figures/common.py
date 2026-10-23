"""Shared statistics, validation, provenance and export helpers."""

from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from constants import RNG_SEED


class MissingDataError(RuntimeError):
    def __init__(self, message: str, missing_fields: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.missing_fields = list(missing_fields)


def load_csv_or_json(path: Path | str) -> Any:
    path = Path(path)
    if not path.is_file():
        raise MissingDataError(f"input file not found: {path}", [str(path)])
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))
    if path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"unsupported tabular input: {path}")


def bootstrap_ci(values: Sequence[float], confidence: float = 0.95, repeats: int = 10000) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return math.nan, math.nan
    rng = np.random.default_rng(RNG_SEED)
    means = rng.choice(data, size=(repeats, len(data)), replace=True).mean(axis=1)
    alpha = 0.5 * (1.0 - confidence)
    return tuple(np.quantile(means, [alpha, 1.0 - alpha]).tolist())


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - radius, center + radius


def paired_bootstrap_diff(
    first: Sequence[float], second: Sequence[float], confidence: float = 0.95, repeats: int = 10000
) -> tuple[float, float, float]:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    difference = a[valid] - b[valid]
    if not len(difference):
        return math.nan, math.nan, math.nan
    low, high = bootstrap_ci(difference, confidence=confidence, repeats=repeats)
    return float(difference.mean()), low, high


def configure_paper_rcparams() -> None:
    installed = {item.name for item in fm.fontManager.ttflist}
    serif = "Times New Roman" if "Times New Roman" in installed else "DejaVu Serif"
    cjk = next((name for name in ("SimSun", "Noto Serif CJK SC", "Microsoft YaHei") if name in installed), None)
    if cjk is None:
        warnings.warn("No preferred CJK font found; Chinese glyphs may fall back.", stacklevel=2)
    matplotlib.rcParams.update(
        {
            "font.family": [serif] + ([cjk] if cjk else []),
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def validate_required_columns(rows: Sequence[Mapping[str, Any]], required: Iterable[str]) -> None:
    required = set(required)
    available = set().union(*(row.keys() for row in rows)) if rows else set()
    missing = sorted(required - available)
    if missing:
        raise MissingDataError(f"missing required columns: {missing}", missing)


def annotate_panel_labels(axes: Sequence[plt.Axes]) -> None:
    for index, axis in enumerate(np.ravel(axes)):
        axis.text(-0.12, 1.05, f"({chr(97 + index)})", transform=axis.transAxes, weight="bold", va="bottom")


def save_figure_all_formats(fig: plt.Figure, output_dir: Path | str, stem: str) -> list[str]:
    output_dir = Path(output_dir)
    paths = []
    for extension, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        directory = output_dir.parent / extension if output_dir.name in {"png", "pdf", "svg"} else output_dir / extension
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def data_required(axis: plt.Axes, message: str) -> None:
    axis.set_axis_off()
    axis.text(0.5, 0.56, "DATA REQUIRED", ha="center", va="center", weight="bold", color="#9B2C2C")
    axis.text(0.5, 0.42, message, ha="center", va="center", wrap=True, fontsize=7, color="#555555")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def image_or_placeholder(axis: plt.Axes, path: Path, label: str = "") -> bool:
    if not path.is_file():
        data_required(axis, f"Missing image: {path}")
        return False
    image = plt.imread(path)
    axis.imshow(image)
    axis.set_axis_off()
    if label:
        axis.set_title(label)
    return True
