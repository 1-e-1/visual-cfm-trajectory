"""Build all paper figures without failing the batch on missing experiment data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parent
SCRIPTS = sorted((ROOT / "scripts").glob("fig[0-9][0-9]_*.py"))
KNOWN_DATA_REQUIRED = {
    "fig13": "Experiment B nominal native-contact task failed validity audit",
}


def build_contact_sheet() -> None:
    figure_paths = sorted((ROOT / "png").glob("fig[0-9][0-9]_*.png"))
    if not figure_paths:
        return
    columns = 2
    tile_width = 900
    label_height = 42
    margin = 24
    tiles = []
    for path in figure_paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        image.thumbnail((tile_width, 520), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, image.height + label_height), "white")
        tile.paste(image, ((tile_width - image.width) // 2, label_height))
        ImageDraw.Draw(tile).text((8, 10), path.stem, fill="black")
        tiles.append(tile)
    rows = (len(tiles) + columns - 1) // columns
    row_heights = [
        max(tile.height for tile in tiles[row * columns : (row + 1) * columns])
        for row in range(rows)
    ]
    sheet = Image.new(
        "RGB",
        (
            columns * tile_width + (columns + 1) * margin,
            sum(row_heights) + (rows + 1) * margin,
        ),
        "white",
    )
    y = margin
    for row, row_height in enumerate(row_heights):
        for column, tile in enumerate(tiles[row * columns : (row + 1) * columns]):
            x = margin + column * (tile_width + margin)
            sheet.paste(ImageOps.expand(tile, border=1, fill="#cccccc"), (x, y))
        y += row_height + margin
    sheet.save(ROOT / "all_figures_contact_sheet.png", optimize=True)


def main() -> None:
    for name in ("data_cache", "png", "pdf", "svg"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    rows = []
    generated = 0
    skipped = 0
    for script in SCRIPTS:
        command = [sys.executable, str(script), "--output-dir", str(ROOT)]
        result = subprocess.run(command, cwd=ROOT.parent, capture_output=True, text=True)
        figure_id = script.stem.split("_")[0]
        status = (
            "data_required"
            if result.returncode == 0 and figure_id in KNOWN_DATA_REQUIRED
            else "generated" if result.returncode == 0 else "skipped"
        )
        if status in {"generated", "data_required"}:
            generated += 1
            print(f"{status.upper()} {script.stem}")
        else:
            skipped += 1
            print(f"SKIP {script.stem}: {result.stderr.strip()}")
        rows.append(
            {
                "figure": script.stem.split("_")[0],
                "script": str(script.relative_to(ROOT)),
                "status": status,
                "input_files": "see data_manifest.json",
                "missing_fields": (
                    result.stderr.strip()
                    if result.returncode
                    else KNOWN_DATA_REQUIRED.get(figure_id, "")
                ),
            }
        )
    with (ROOT / "figure_status.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    build_contact_sheet()
    print(json.dumps({"generated": generated, "skipped": skipped}, indent=2))
    raise SystemExit(0 if generated else 1)


if __name__ == "__main__":
    main()
