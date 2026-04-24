from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.align.reconstruct import capture_single_frame
    from src.align.ply_registration import register_point_clouds
else:
    from .reconstruct import capture_single_frame
    from .ply_registration import register_point_clouds

source = Path("/home/rm/ljc/RM_Skills/outputs/kettle_source_right/point_cloud.ply")
target_ply = capture_single_frame(
    bbox_min=[-0.2, -0.2, 0.3],
    bbox_max=[0.2, 0.2, 0.7],
    output_dir=Path("outputs"),
    vis=False,
)
result = register_point_clouds(source, target_ply, voxel_size=0.01, icp_distance_factor=2.0, no_vis=True)
print(result)
