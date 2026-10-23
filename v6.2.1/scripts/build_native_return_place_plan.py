"""Build a native-contact pick, lift, return, and release demonstration plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "outputs/native_contact/native_contact_plan.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/native_contact/native_contact_return_place_plan.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs/native_contact/native_contact_return_place_plan.json",
    )
    parser.add_argument("--lift_hold_duration", type=float, default=2.0)
    args = parser.parse_args()

    with np.load(args.source, allow_pickle=False) as source:
        payload = {key: np.array(source[key], copy=True) for key in source.files}

    lift_path = np.asarray(payload["lift_q"], dtype=np.float32)
    q_lift = np.asarray(payload["q_lift"], dtype=np.float32)
    q_grasp = np.asarray(payload["q_grasp"], dtype=np.float32)
    grasp_center = np.asarray(payload["grasp_center_position_foot0"], dtype=np.float32)
    payload.update(
        {
            "place_center_position_foot0": grasp_center.copy(),
            "transfer_to_preplace_q": np.stack((q_lift, q_lift)),
            "transfer_to_preplace_duration": np.float32(args.lift_hold_duration),
            "descend_q": lift_path[::-1].copy(),
            "descend_duration": np.float32(payload["lift_duration"]),
            "q_preplace": q_lift.copy(),
            "q_place": q_grasp.copy(),
        }
    )
    payload.pop("terminal_place_correction_q", None)
    payload.pop("terminal_place_correction_duration", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    report = {
        "method": "native-contact pick-return-place",
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "attachment_joint": False,
        "place_center_equals_pickup_center": True,
        "lift_hold_duration_s": float(args.lift_hold_duration),
        "descend_is_exact_reverse_of_lift": True,
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
