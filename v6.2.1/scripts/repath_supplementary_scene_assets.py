"""Create an experiment-only USD copy with project-local asset references."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pxr import Sdf, UsdUtils


ASSET_FILENAMES = {
    "unitree_h1.usd",
    "zbot_6s_new_configured.usd",
    "robotiq_2f85.usd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("1111_copy_native_contact.usd"))
    parser.add_argument("--output", type=Path, default=Path("supplementary_experiment_scene.usd"))
    parser.add_argument(
        "--assets_dir",
        type=Path,
        default=Path("assets/source/assets"),
        help="Directory containing the three referenced robot assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    assets_dir = args.assets_dir.resolve()

    if source == output:
        raise ValueError("The output must be an isolated copy, not the source USD.")
    for filename in sorted(ASSET_FILENAMES):
        asset = assets_dir / filename
        if not asset.is_file():
            raise FileNotFoundError(asset)

    source_layer = Sdf.Layer.FindOrOpen(str(source))
    if source_layer is None:
        raise RuntimeError(f"Unable to open source layer: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not source_layer.Export(str(output)):
        raise RuntimeError(f"Unable to export isolated scene: {output}")

    output_layer = Sdf.Layer.FindOrOpen(str(output))
    if output_layer is None:
        raise RuntimeError(f"Unable to reopen output layer: {output}")

    replacements: dict[str, str] = {}

    def repath(asset_path: str) -> str:
        filename = Path(asset_path.replace("\\", "/")).name
        if filename not in ASSET_FILENAMES:
            return asset_path
        relative = os.path.relpath(assets_dir / filename, output.parent).replace("\\", "/")
        replacements[asset_path] = relative
        return relative

    UsdUtils.ModifyAssetPaths(output_layer, repath)
    output_layer.Save()

    if len(replacements) != len(ASSET_FILENAMES):
        missing = ASSET_FILENAMES.difference(Path(key.replace("\\", "/")).name for key in replacements)
        raise RuntimeError(f"Expected references were not found in the source layer: {sorted(missing)}")

    print(f"Created: {output}")
    for old, new in sorted(replacements.items()):
        print(f"  {old} -> {new}")


if __name__ == "__main__":
    main()
