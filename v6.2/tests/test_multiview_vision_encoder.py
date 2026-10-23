from __future__ import annotations

import torch

from scripts.train_obstacle_vision import load_initial_state
from src.models.obstacle_flow_matching import ObstacleConditionalFlowMatcher, WristVisionEncoder


def _model(camera_views: int) -> ObstacleConditionalFlowMatcher:
    return ObstacleConditionalFlowMatcher(
        condition_dim=9,
        trajectory_points=64,
        dof=6,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        time_embedding_dim=16,
        obstacle_hidden_dim=16,
        obstacle_layers=1,
        obstacle_heads=4,
        vision_camera_views=camera_views,
    )


def test_visual_encoder_shapes_for_single_and_four_views() -> None:
    single = WristVisionEncoder(output_dim=16, temporal_frames=2, camera_views=1)
    four = WristVisionEncoder(output_dim=16, temporal_frames=2, camera_views=4)
    single_rgb = torch.zeros(2, 2, 32, 32, 3, dtype=torch.uint8)
    single_depth = torch.zeros(2, 2, 32, 32, 1)
    single_semantic = torch.zeros(2, 2, 32, 32, 1)
    four_rgb = single_rgb[:, None].repeat(1, 4, 1, 1, 1, 1)
    four_depth = single_depth[:, None].repeat(1, 4, 1, 1, 1, 1)
    four_semantic = single_semantic[:, None].repeat(1, 4, 1, 1, 1, 1)

    assert single(single_rgb, single_depth, single_semantic).shape == (2, 16)
    assert four(four_rgb, four_depth, four_semantic).shape == (2, 16)


def test_expanded_input_convolution_preserves_repeated_view_response() -> None:
    torch.manual_seed(7)
    single = _model(camera_views=1)
    four = _model(camera_views=4)
    checkpoint = {"model_state_dict": single.state_dict()}
    report = load_initial_state(four, checkpoint)
    assert report["input_convolution_expanded"]
    single.eval()
    four.eval()

    rgb = torch.randint(0, 256, (2, 2, 32, 32, 3), dtype=torch.uint8)
    depth = torch.rand(2, 2, 32, 32, 1)
    semantic = torch.randint(0, 9, (2, 2, 32, 32, 1))
    four_rgb = rgb[:, None].repeat(1, 4, 1, 1, 1, 1)
    four_depth = depth[:, None].repeat(1, 4, 1, 1, 1, 1)
    four_semantic = semantic[:, None].repeat(1, 4, 1, 1, 1, 1)

    with torch.no_grad():
        expected = single.vision_encoder(rgb, depth, semantic)
        actual = four.vision_encoder(four_rgb, four_depth, four_semantic)
    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-5)


def test_high_resolution_encoder_shape_and_checkpoint_initialization() -> None:
    torch.manual_seed(11)
    source = _model(camera_views=1)
    target = ObstacleConditionalFlowMatcher(
        condition_dim=9,
        trajectory_points=64,
        dof=6,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        time_embedding_dim=16,
        obstacle_hidden_dim=16,
        obstacle_layers=1,
        obstacle_heads=4,
        vision_camera_views=1,
        vision_backbone="residual_cnn_large",
    )
    report = load_initial_state(target, {"model_state_dict": source.state_dict()})

    assert report["vision_encoder_reinitialized"]
    torch.testing.assert_close(target.input_projection.weight, source.input_projection.weight)
    rgb = torch.zeros(1, 2, 256, 256, 3, dtype=torch.uint8)
    depth = torch.zeros(1, 2, 256, 256, 1)
    semantic = torch.zeros(1, 2, 256, 256, 1)
    assert target.vision_encoder(rgb, depth, semantic).shape == (1, 16)


def test_visual_sampling_encodes_static_observation_once() -> None:
    model = _model(camera_views=1).eval()
    calls = []
    hook = model.vision_encoder.register_forward_hook(lambda *_: calls.append(1))
    condition = torch.zeros(2, 9)
    rgb = torch.zeros(2, 2, 32, 32, 3, dtype=torch.uint8)
    depth = torch.zeros(2, 2, 32, 32, 1)
    semantic = torch.zeros(2, 2, 32, 32, 1)
    with torch.no_grad():
        trajectory, duration = model.sample_visual(
            condition, rgb, depth, semantic, steps=3, generator=torch.Generator().manual_seed(3)
        )
    hook.remove()
    assert calls == [1]
    assert trajectory.shape == (2, 64, 6)
    assert duration.shape == (2,)
