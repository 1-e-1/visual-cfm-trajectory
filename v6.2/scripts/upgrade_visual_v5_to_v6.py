"""Offline v5 -> v6 upgrade: add q_ik (P2), multi-label experts (T1), 15-dim condition.

Reuses the existing 5-view visual_five_view_v1 captures (no Isaac rendering needed).
No depth re-reconstruction is required: the stored online planning cloud
``scene_points`` (sampled from the online point-cloud fusion) is used to re-rank
expert candidates with the same environment + self clearance objective as the
online generator, and the differentiable FK computes the goal TCP position.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import multiprocessing as mp
import sys
from time import perf_counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.datasets.npz_dataset import write_npz_shards
from src.planning.ik_solver import solve_ik_candidates
from src.robot.torch_kinematics import DifferentiableZBotFK
from h1_1111_grasp.trajectory_safety import (
    environment_clearance_batch,
    link_segments,
    self_clearance_batch,
)

_EXPERT_CANDIDATES = 128
_IK_CANDIDATES = 4
_LABEL_COUNT = 8


def _wrap(values):
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def _sample_seed(sample_id: str, base: int) -> int:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return (base + int(digest[:8], 16)) % (2**31 - 1)


def _minimum_jerk_progress(point_count: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, point_count, dtype=np.float64)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _candidate_paths(q_start, q_goal_unwrapped, count, rng):
    if count < 1:
        raise ValueError("expert_candidates must be positive")
    direct_progress = _minimum_jerk_progress(64)
    direct = q_start[None] + direct_progress[:, None] * (q_goal_unwrapped - q_start)[None]
    paths = [direct]
    first_progress = _minimum_jerk_progress(33)
    second_progress = _minimum_jerk_progress(32)
    midpoint = 0.5 * (q_start + q_goal_unwrapped)
    for _ in range(count - 1):
        waypoint = midpoint + rng.uniform(-1.2, 1.2, size=6)
        waypoint = q_start + _wrap(waypoint - q_start)
        goal_from_waypoint = waypoint + _wrap(q_goal_unwrapped - waypoint)
        first = q_start[None] + first_progress[:, None] * (waypoint - q_start)[None]
        second = waypoint[None] + second_progress[:, None] * (goal_from_waypoint - waypoint)[None]
        paths.append(np.concatenate((first, second[1:]), axis=0))
    return _wrap(np.stack(paths))


def _rank_experts(fk_model, paths, planning_cloud, min_clearance_m: float = 0.01):
    flat_segments = link_segments(fk_model, paths.reshape(-1, 6), device="cpu").reshape(len(paths), 64, -1, 2, 3)
    environment_values = environment_clearance_batch(flat_segments[:, :, 1:], planning_cloud, capsule_radius_m=0.05)
    self_values = self_clearance_batch(
        flat_segments.reshape(-1, flat_segments.shape[2], 2, 3), capsule_radius_m=0.045
    ).reshape(len(paths), 64).min(axis=1)
    ranked = []
    for index, (environment, self_distance) in enumerate(zip(environment_values, self_values)):
        if min(environment, self_distance) < min_clearance_m:
            continue
        continuous = np.unwrap(paths[index], axis=0, period=2.0 * np.pi)
        path_length = float(np.abs(np.diff(continuous, axis=0)).sum())
        ranked.append((min(environment, self_distance) - 2.0e-4 * path_length, index))
    ranked.sort(reverse=True)
    return [index for _, index in ranked]


class _Worker:
    fk = None

    @classmethod
    def init(cls, candidate_count):
        torch.set_num_threads(2)
        global _EXPERT_CANDIDATES
        _EXPERT_CANDIDATES = int(candidate_count)
        cls.fk = DifferentiableZBotFK(str(WORKSPACE_ROOT / "configs" / "kinematic_chain.json"))
        cls.fk.eval()

    @classmethod
    def process(cls, task):
        side, row = task
        (
            sample_id, q_start, q_goal, stored_q, scene_points, scene_mask,
            duration, stored_condition, seed,
        ) = row
        q_start = np.asarray(q_start, dtype=np.float64)
        q_goal = np.asarray(q_goal, dtype=np.float64)
        stored_q = np.asarray(stored_q, dtype=np.float64)
        cloud = np.asarray(scene_points, dtype=np.float64).reshape(-1, 3)
        cloud = cloud[np.asarray(scene_mask, dtype=bool)]

        with torch.no_grad():
            p_goal, _ = cls.fk(torch.as_tensor(q_goal[None], dtype=torch.float32))
        p_goal = p_goal[0].numpy().astype(np.float64)

        # ---- P2: multi-restart IK candidates ----
        q_ik = solve_ik_candidates(
            cls.fk, p_goal, n_candidates=_IK_CANDIDATES, restarts=8, iterations=100,
            position_tolerance_m=0.005,
            self_clearance_fn=lambda q_row: float(
                self_clearance_batch(
                    link_segments(cls.fk, np.asarray(q_row, dtype=np.float64).reshape(1, 6), device="cpu"),
                    capsule_radius_m=0.045,
                ).min()
            ),
            seed=seed,
            device="cpu",
        )
        if len(q_ik) == 0:
            q_ik = _wrap(q_goal)[None]
        if len(q_ik) < _IK_CANDIDATES:
            q_ik = np.concatenate((q_ik, np.repeat(q_ik[:1], _IK_CANDIDATES - len(q_ik), axis=0)), axis=0)
        q_ik = q_ik.astype(np.float32)[:_IK_CANDIDATES]

        # ---- T1: multi-label experts ranked against the stored online cloud ----
        shortest_delta = _wrap(q_goal - q_start)
        q_goal_unwrapped = q_start + shortest_delta
        rng = np.random.default_rng(seed)
        paths = _candidate_paths(q_start, q_goal_unwrapped, _EXPERT_CANDIDATES, rng)
        paths = np.concatenate((paths, _wrap(stored_q)[None]), axis=0)  # _EXPERT_CANDIDATES + 1
        ranked = _rank_experts(cls.fk, paths, cloud)
        if len(ranked) < _LABEL_COUNT:
            selected = list(ranked) + [len(paths) - 1] * (_LABEL_COUNT - len(ranked))
        else:
            selected = ranked[:_LABEL_COUNT]
        q_experts = np.stack([_wrap(paths[i]).astype(np.float32) for i in selected[:_LABEL_COUNT]])

        condition = np.concatenate((q_start, q_ik[0], p_goal)).astype(np.float32)
        return {
            "sample_id": sample_id,
            "condition": condition,
            "q_experts": q_experts,
            "q_ik_candidates": q_ik,
            "expert_candidates": int(len(paths)),
        }


def _checkpoint_path(output_side, split, shard_path):
    return Path(output_side) / "_checkpoints" / f"{split}__{shard_path.stem}.npz"


def _save_checkpoint(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_id=np.asarray([r["sample_id"] for r in results], dtype="<U64"),
        condition=np.asarray([r["condition"] for r in results], dtype=np.float32),
        q_experts=np.asarray([r["q_experts"] for r in results], dtype=np.float32),
        q_ik_candidates=np.asarray([r["q_ik_candidates"] for r in results], dtype=np.float32),
        expert_candidates=np.asarray([r["expert_candidates"] for r in results], dtype=np.int64),
    )


def _load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        count = data["sample_id"].shape[0]
        return [
            {
                "sample_id": str(data["sample_id"][index]),
                "condition": data["condition"][index],
                "q_experts": data["q_experts"][index],
                "q_ik_candidates": data["q_ik_candidates"][index],
                "expert_candidates": int(data["expert_candidates"][index]),
            }
            for index in range(count)
        ]


def _write_upgrade_shards(
    samples, indices, output_dir, prefix, shard_size, camera_view_names, start_shard=0
):
    """Like write_npz_shards but keeps camera_view_names as a shard-level field
    and numbers output shards continuously across source shards."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for shard_index, start in enumerate(range(0, len(indices), shard_size)):
        selected = [samples[int(index)] for index in indices[start : start + shard_size]]
        if not selected:
            continue
        keys = selected[0].keys()
        arrays = {key: np.asarray([sample[key] for sample in selected]) for key in keys}
        arrays["camera_view_names"] = np.asarray(camera_view_names, dtype="<U32")
        path = output_dir / f"{prefix}_shard_{start_shard + shard_index:05d}.npz"
        np.savez_compressed(path, **arrays)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data" / "visual_five_view_v1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "visual_v6_upgrade")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="limit samples per split (0 = all)")
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--candidates", type=int, default=128)
    args = parser.parse_args()
    global _EXPERT_CANDIDATES
    _EXPERT_CANDIDATES = max(8, args.candidates)

    source_side = args.source / args.side
    output_side = args.output / args.side
    output_side.mkdir(parents=True, exist_ok=True)
    splits = ["train", "validation", "test"]
    split_paths = {s: sorted(source_side.glob(f"{s}_shard_*.npz")) for s in splits}
    total = sum(len(v) for v in split_paths.values())
    print(f"[UPGRADE] side={args.side} splits={ {k: len(v) for k, v in split_paths.items()} } candidates={_EXPERT_CANDIDATES}", flush=True)

    started = perf_counter()
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=_Worker.init, initargs=(_EXPERT_CANDIDATES,))
    outputs = {s: [] for s in splits}
    stats = {"samples": 0, "rank_counts": [], "padded_labels": 0}
    for split in splits:
        for shard_path in split_paths[split]:
            with np.load(shard_path, allow_pickle=False) as data:
                count = data["q"].shape[0]
                tasks = []
                for row in range(count):
                    sample_id = str(data["sample_id"][row])
                    tasks.append(
                        (
                            args.side,
                            (
                                sample_id,
                                data["q_start"][row],
                                data["q_goal"][row],
                                data["q"][row],
                                data["scene_points"][row],
                                data["scene_point_mask"][row],
                                data["duration"][row],
                                data["condition"][row],
                                _sample_seed(sample_id, args.seed),
                            ),
                        )
                    )
                if args.limit:
                    tasks = tasks[: args.limit]
                checkpoint = _checkpoint_path(output_side, split, shard_path)
                if checkpoint.is_file():
                    results = _load_checkpoint(checkpoint)
                    print(
                        f"[UPGRADE] {split} shard {shard_path.name} loaded from checkpoint ({len(results)} rows)",
                        flush=True,
                    )
                else:
                    results = pool.map(_Worker.process, tasks)
                    _save_checkpoint(checkpoint, results)
            for result in results:
                outputs[split].append(result)
                stats["samples"] += 1
                stats["rank_counts"].append(result["expert_candidates"])
                if result["q_experts"].shape[0] < _LABEL_COUNT:
                    stats["padded_labels"] += 1
            print(f"[UPGRADE] {split} shard {shard_path.name} done ({len(results)} rows)", flush=True)
    pool.close()
    pool.join()

    for split in splits:
        source_shards = split_paths[split]
        sample_by_id = {r["sample_id"]: r for r in outputs[split]}
        written = 0
        shard_counter = 0
        for shard_path in source_shards:
            with np.load(shard_path, allow_pickle=False) as data:
                n = data["q"].shape[0]
                ids = np.asarray([str(s) for s in data["sample_id"]])
                batch = []
                for row in range(n):
                    if ids[row] not in sample_by_id:
                        continue
                    r = sample_by_id[ids[row]]
                    item = {
                        key: data[key][row] for key in data.files
                        if key
                        not in (
                            "condition",
                            "q_experts",
                            "q_ik_candidates",
                            "expert_candidates",
                            "camera_view_names",
                        )
                    }
                    item["condition"] = r["condition"]
                    item["q_experts"] = r["q_experts"]
                    item["q_ik_candidates"] = r["q_ik_candidates"]
                    item["expert_candidates"] = r["expert_candidates"]
                    batch.append(item)
                if batch:
                    paths = _write_upgrade_shards(
                        batch,
                        np.arange(len(batch)),
                        output_side,
                        split,
                        args.shard_size,
                        camera_view_names=np.asarray(data["camera_view_names"], dtype="<U32"),
                        start_shard=shard_counter,
                    )
                    shard_counter += len(paths)
                    written += len(batch)
        print(f"[UPGRADE] {split} written={written} shards={shard_counter}", flush=True)

    elapsed = perf_counter() - started
    report = {
        "side": args.side,
        "source": str(source_side),
        "condition": "q_start[6] + q_ik[6] + p_goal[3] (v6)",
        "samples": stats["samples"],
        "elapsed_s": elapsed,
        "avg_rank_candidates": float(np.mean(stats["rank_counts"])) if stats["rank_counts"] else None,
        "min_rank_candidates": int(np.min(stats["rank_counts"])) if stats["rank_counts"] else None,
        "padded_label_samples": stats["padded_labels"],
    }
    (output_side / "generation_report.json").write_text(json.dumps(report, indent=2))
    print(f"[UPGRADE] done side={args.side} samples={stats['samples']} elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
