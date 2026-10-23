from __future__ import annotations

import torch

from src.models.flow_matching import ConditionalFlowMatcher
from src.robot.torch_kinematics import DifferentiableZBotFK
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.losses import FlowMatchingLoss


def test_flow_matching_forward_backward_and_checkpoint(tmp_path) -> None:
    model = ConditionalFlowMatcher(
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        time_embedding_dim=16,
    )
    fk = DifferentiableZBotFK("configs/kinematic_chain.json")
    criterion = FlowMatchingLoss(
        fk_model=fk,
        weights={
            "flow": 1.0,
            "start_consistency": 1.0,
            "final_tcp_position": 1.0,
            "final_tcp_orientation": 1.0,
            "velocity_smoothness": 0.01,
            "acceleration_smoothness": 0.01,
            "jerk_regularization": 0.01,
            "duration": 1.0,
        },
        velocity_limits=torch.ones(6) * 2.0,
        acceleration_limits=torch.ones(6) * 8.0,
        jerk_limits=torch.ones(6) * 80.0,
    )
    condition = torch.randn(2, 13)
    condition[:, 9:13] = torch.nn.functional.normalize(condition[:, 9:13], dim=-1)
    trajectory = torch.randn(2, 64, 6)
    trajectory[:, 0] = condition[:, :6]
    duration = torch.full((2,), 2.0)
    loss, metrics = criterion(model, trajectory, condition, duration)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is None or torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters())
    assert metrics["total"].item() == loss.item()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=3, best_validation_loss=1.25, config={"test": True})
    restored = ConditionalFlowMatcher(
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        time_embedding_dim=16,
    )
    checkpoint = load_checkpoint(path, restored, map_location="cpu")
    assert checkpoint["epoch"] == 3


def test_sampling_enforces_start_state() -> None:
    model = ConditionalFlowMatcher(hidden_dim=32, num_layers=1, num_heads=4, dropout=0.0, time_embedding_dim=16)
    condition = torch.randn(2, 13)
    trajectory, duration = model.sample(condition, steps=2, solver="heun")
    torch.testing.assert_close(trajectory[:, 0], condition[:, :6])
    assert trajectory.shape == (2, 64, 6)
    assert duration.shape == (2,)
