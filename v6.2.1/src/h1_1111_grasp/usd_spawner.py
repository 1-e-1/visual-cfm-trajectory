"""Isaac Lab spawner that references the robot sub-prim from 1111_copy.usd."""

from __future__ import annotations

from pxr import Sdf, Usd

from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone, create_prim, get_current_stage


ROBOT_SOURCE_PRIM = Sdf.Path("/World/UnitreeH1")


@clone
def spawn_1111_robot(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Reference only the assembled robot sub-prim, then apply native Isaac Lab properties."""
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = create_prim(
            prim_path,
            prim_type="Xform",
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
            stage=stage,
        )
        prim.GetReferences().AddReference(cfg.usd_path, ROBOT_SOURCE_PRIM)
    return _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
