from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _expand(old: torch.Tensor) -> torch.Tensor:
    new = torch.zeros((old.shape[0], 95), dtype=old.dtype)
    new[:, :71] = old[:, :71]
    for joint_index in range(6):
        old_pos = 71 + joint_index
        old_vel = 77 + joint_index
        new_pos = 71 + 2 * joint_index
        new_vel = 83 + 2 * joint_index
        new[:, new_pos : new_pos + 2] = 0.5 * old[:, old_pos : old_pos + 1]
        new[:, new_vel : new_vel + 2] = 0.5 * old[:, old_vel : old_vel + 1]
    return new


def test_symmetric_dual_arm_mapping_preserves_single_arm_activation() -> None:
    torch.manual_seed(7)
    old = torch.randn(128, 83)
    old_obs = torch.randn(4, 83)
    interleaved_pos = old_obs[:, 71:77].repeat_interleave(2, dim=1)
    interleaved_vel = old_obs[:, 77:83].repeat_interleave(2, dim=1)
    new_obs = torch.cat([old_obs[:, :71], interleaved_pos, interleaved_vel], dim=1)
    assert torch.allclose(old_obs @ old.T, new_obs @ _expand(old).T, atol=1.0e-6)
