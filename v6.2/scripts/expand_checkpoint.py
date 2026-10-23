"""Expand the single-arm 83-D H1 checkpoint to the dual-arm 95-D observation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def expand_first_layer(state: dict[str, torch.Tensor]) -> None:
    key = "mlp.0.weight"
    old = state[key]
    if tuple(old.shape) != (128, 83):
        raise ValueError(f"expected {key} shape (128, 83), got {tuple(old.shape)}")
    new = torch.zeros((old.shape[0], 95), dtype=old.dtype)
    new[:, :71] = old[:, :71]
    for joint_index in range(6):
        old_pos = 71 + joint_index
        old_vel = 77 + joint_index
        new_pos = 71 + 2 * joint_index
        new_vel = 83 + 2 * joint_index
        new[:, new_pos : new_pos + 2] = 0.5 * old[:, old_pos : old_pos + 1]
        new[:, new_vel : new_vel + 2] = 0.5 * old[:, old_vel : old_vel + 1]
    state[key] = new


def main() -> None:
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    expand_first_layer(checkpoint["actor_state_dict"])
    expand_first_layer(checkpoint["critic_state_dict"])
    checkpoint["optimizer_state_dict"] = None
    checkpoint["iter"] = 0
    if checkpoint.get("infos") is None:
        checkpoint["infos"] = {}
    checkpoint["infos"]["observation_expansion"] = {
        "source_dim": 83,
        "target_dim": 95,
        "mapping": "69 H1 + 2 xy + averaged duplicate of 6 position and 6 velocity features",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"[OK] expanded checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
