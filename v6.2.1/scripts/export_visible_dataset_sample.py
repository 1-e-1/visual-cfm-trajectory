"""Export one real five-camera dataset sample with target-object visibility marks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


CAMERA_ZH = {
    "wrist": "腕部相机",
    "cube": "安装座相机",
    "chest": "胸部相机",
    "abdomen": "腹部相机",
    "head": "头部相机",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", default="left_000243")
    parser.add_argument("--target-semantic-id", type=int)
    parser.add_argument("--target-camera")
    parser.add_argument("--target-bbox", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--diagnostics", action="store_true")
    return parser.parse_args()


def configure_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def find_sample(data: np.lib.npyio.NpzFile, sample_id: str) -> int:
    sample_ids = [str(value) for value in data["sample_id"]]
    if sample_id not in sample_ids:
        raise ValueError(f"Sample {sample_id!r} not found in {data.files}")
    return sample_ids.index(sample_id)


def save_raw_views(
    rgb: np.ndarray, camera_names: list[str], output_dir: Path, sample_id: str
) -> None:
    for frame_idx in range(rgb.shape[1]):
        frame_dir = output_dir / f"frame_{frame_idx}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for camera_idx, camera_name in enumerate(camera_names):
            Image.fromarray(rgb[camera_idx, frame_idx]).save(
                frame_dir / f"{sample_id}_{camera_name}.png"
            )


def save_montage(
    rgb: np.ndarray,
    semantic: np.ndarray,
    camera_names: list[str],
    target_semantic_id: int | None,
    output_dir: Path,
    sample_id: str,
    frame_idx: int,
    target_camera: str | None = None,
    target_bbox: list[int] | None = None,
) -> list[dict[str, object]]:
    fig, axes = plt.subplots(1, len(camera_names), figsize=(16, 3.6), dpi=180)
    records: list[dict[str, object]] = []
    for camera_idx, (camera_name, axis) in enumerate(zip(camera_names, axes)):
        image = rgb[camera_idx, frame_idx]
        if target_semantic_id is not None:
            mask = semantic[camera_idx, frame_idx, ..., 0] == target_semantic_id
            bbox = bbox_from_mask(mask)
            pixel_count = int(mask.sum())
        else:
            bbox = target_bbox if camera_name == target_camera else None
            pixel_count = None
        axis.imshow(image)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            axis.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0 + 1,
                    y1 - y0 + 1,
                    fill=False,
                    edgecolor="#ff3b30",
                    linewidth=1.8,
                )
            )
        axis.set_title(f"{CAMERA_ZH.get(camera_name, camera_name)}\n{camera_name}", fontsize=10)
        axis.axis("off")
        record: dict[str, object] = {"camera": camera_name, "target_bbox_xyxy": bbox}
        if pixel_count is not None:
            record["target_pixels"] = pixel_count
        records.append(record)

    fig.suptitle(
        f"真实数据集样本 {sample_id} | 时刻 {frame_idx} | 红框：目标物块",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_dir / f"{sample_id}_five_camera_frame_{frame_idx}.png", bbox_inches="tight")
    plt.close(fig)
    return records


def save_rgb_mask_overview(
    rgb: np.ndarray,
    semantic: np.ndarray,
    camera_names: list[str],
    target_semantic_id: int,
    output_dir: Path,
    sample_id: str,
) -> None:
    fig, axes = plt.subplots(2, len(camera_names), figsize=(16, 6.2), dpi=180)
    for camera_idx, camera_name in enumerate(camera_names):
        image = rgb[camera_idx, 0]
        mask = semantic[camera_idx, 0, ..., 0] == target_semantic_id

        axes[0, camera_idx].imshow(image)
        axes[0, camera_idx].set_title(CAMERA_ZH.get(camera_name, camera_name), fontsize=10)
        axes[0, camera_idx].axis("off")

        overlay = image.astype(np.float32) / 255.0
        overlay[mask] = 0.35 * overlay[mask] + 0.65 * np.array([1.0, 0.15, 0.05])
        axes[1, camera_idx].imshow(overlay)
        axes[1, camera_idx].axis("off")

    axes[0, 0].set_ylabel("原始 RGB", fontsize=11)
    axes[1, 0].set_ylabel("目标语义掩码", fontsize=11)
    fig.suptitle(
        f"{sample_id}：五相机 RGB 与目标物块语义标注",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / f"{sample_id}_rgb_target_mask.png", bbox_inches="tight")
    plt.close(fig)


def save_semantic_id_overview(
    rgb: np.ndarray,
    semantic: np.ndarray,
    camera_names: list[str],
    output_dir: Path,
    sample_id: str,
) -> None:
    semantic_ids = sorted(
        int(value) for value in np.unique(semantic[:, 0, ..., 0]) if int(value) != 0
    )
    fig, axes = plt.subplots(
        len(semantic_ids), len(camera_names),
        figsize=(16, max(2.2, 2.25 * len(semantic_ids))),
        dpi=150,
        squeeze=False,
    )
    for row, semantic_id in enumerate(semantic_ids):
        for camera_idx, camera_name in enumerate(camera_names):
            image = rgb[camera_idx, 0].astype(np.float32) / 255.0
            mask = semantic[camera_idx, 0, ..., 0] == semantic_id
            faded = image * 0.18
            faded[mask] = image[mask]
            axes[row, camera_idx].imshow(faded)
            axes[row, camera_idx].axis("off")
            if row == 0:
                axes[row, camera_idx].set_title(
                    CAMERA_ZH.get(camera_name, camera_name), fontsize=9
                )
            if camera_idx == 0:
                axes[row, camera_idx].set_ylabel(
                    f"语义 ID {semantic_id}", fontsize=9, rotation=90
                )
    fig.suptitle(f"{sample_id}：各语义 ID 的实际像素区域", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / f"{sample_id}_semantic_id_diagnostic.png", bbox_inches="tight")
    plt.close(fig)


def save_camera_contact_sheets(
    data: np.lib.npyio.NpzFile, camera_names: list[str], output_dir: Path
) -> None:
    sample_ids = [str(value) for value in data["sample_id"]]
    for camera_idx, camera_name in enumerate(camera_names):
        columns = 7
        rows = int(np.ceil(len(sample_ids) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(14, 14), dpi=150, squeeze=False)
        for index, axis in enumerate(axes.flat):
            if index < len(sample_ids):
                axis.imshow(data["rgb"][index, camera_idx, 0])
                axis.set_title(sample_ids[index], fontsize=7)
            axis.axis("off")
        fig.suptitle(
            f"测试集{CAMERA_ZH.get(camera_name, camera_name)}候选样本（用于筛选清晰可见物块）",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output_dir / f"{camera_name}_candidate_contact_sheet.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_fonts()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.shard, allow_pickle=False) as data:
        sample_idx = find_sample(data, args.sample_id)
        camera_key = "camera_names" if "camera_names" in data.files else "camera_view_names"
        camera_names = [str(value) for value in data[camera_key]]
        rgb = data["rgb"][sample_idx]
        semantic = data["semantic"][sample_idx]

        save_raw_views(rgb, camera_names, args.output_dir, args.sample_id)
        visibility = {}
        for frame_idx in range(rgb.shape[1]):
            visibility[f"frame_{frame_idx}"] = save_montage(
                rgb,
                semantic,
                camera_names,
                args.target_semantic_id,
                args.output_dir,
                args.sample_id,
                frame_idx,
                args.target_camera,
                args.target_bbox,
            )
        if args.target_semantic_id is not None:
            save_rgb_mask_overview(
                rgb,
                semantic,
                camera_names,
                args.target_semantic_id,
                args.output_dir,
                args.sample_id,
            )
        if args.diagnostics:
            save_semantic_id_overview(
                rgb, semantic, camera_names, args.output_dir, args.sample_id
            )
            save_camera_contact_sheets(data, camera_names, args.output_dir)

        metadata = {
            "source_shard": str(args.shard.resolve()),
            "sample_index": sample_idx,
            "sample_id": args.sample_id,
            "camera_names": camera_names,
            "target_semantic_id": args.target_semantic_id,
            "target_camera": args.target_camera,
            "manual_target_bbox_xyxy": args.target_bbox,
            "object_type": int(data["object_type"][sample_idx]),
            "object_center_foot0": data["object_center_foot0"][sample_idx].tolist(),
            "object_size": data["object_size"][sample_idx].tolist(),
            "visibility": visibility,
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
