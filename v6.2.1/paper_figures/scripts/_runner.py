"""CLI adapter shared by the fourteen standalone figure scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


FIGURE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIGURE_DIR))

import builders


def run(number: int) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    builders.FIGURE_ROOT = args.output_dir.resolve()
    outputs = builders.BUILDERS[number](args.input)
    print("\n".join(outputs))
