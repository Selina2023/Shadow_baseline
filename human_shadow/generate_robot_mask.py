
import os
import json
import logging
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from human_shadow.utils.file_utils import get_parent_folder_of_package
from human_shadow.virtual_twin.twin_robot import TwinRobot, get_mujoco_camera_params


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run_panda(
    camera_params,
    target_robot_state: dict,
    camera_res: int = 1080,
    render: bool = False,
    n_steps_short: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the Panda virtual twin for a single target state."""
    twin_robot = TwinRobot(
        "Panda",
        "Robotiq85",
        camera_params,
        camera_res=camera_res,
        render=render,
        n_steps_short=n_steps_short,
    )
    twin_robot.reset()
    robot_mask, gripper_mask, rgb_img = twin_robot.move_to_target_state(
        target_robot_state, init=True
    )
    return robot_mask, gripper_mask, rgb_img


def generate_franka_mask(
    resolution: str,
    ee_pos: np.ndarray,
    ee_quat_xyzw: np.ndarray,
    gripper_pos: float = 0.0,
    camera_pos: Optional[np.ndarray] = None,
    camera_quat_wxyz: Optional[np.ndarray] = None,
    camera_res: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate robot and gripper masks (and the rendered RGB image) for a Franka Panda pose.

    This function lets you configure both the Panda end-effector pose and the
    camera viewpoint. It reuses the existing real-world camera intrinsics from
    the calibration JSONs, but you can override the camera extrinsics.

    Args:
        resolution: ZED resolution key, e.g. "HD1080".
        ee_pos: End-effector position (x, y, z) in robot base frame.
        ee_quat_xyzw: End-effector orientation as quaternion (x, y, z, w).
        gripper_pos: Gripper command, typically in [0, 1].
        camera_pos: Optional camera position override in world frame (x, y, z).
        camera_quat_wxyz: Optional camera orientation override as quaternion (w, x, y, z).
        camera_res: Optional square output resolution for the RGB image; defaults
            to the vertical resolution associated with `resolution`.

    Returns:
        (robot_mask, gripper_mask, rgb_img)
    """
    project_folder = get_parent_folder_of_package("human_shadow")

    # Load extrinsics (and optionally override pose)
    camera_extrinsics_path = os.path.join(
        project_folder,
        f"human_shadow/camera/camera_calibration_data/hand_calib_{resolution}/cam_cal.json",
    )
    with open(camera_extrinsics_path, "r") as f:
        camera_extrinsics = json.load(f)

    if camera_pos is not None:
        camera_extrinsics[0]["camera_base_pos"] = np.asarray(
            camera_pos, dtype=float
        ).tolist()
    if camera_quat_wxyz is not None:
        rot = Rotation.from_quat(np.asarray(camera_quat_wxyz, dtype=float))
        camera_extrinsics[0]["camera_base_ori"] = rot.as_matrix().tolist()

    # Load intrinsics
    camera_intrinsics_path = os.path.join(
        project_folder,
        f"human_shadow/camera/intrinsics/camera_intrinsics_{resolution}.json",
    )
    with open(camera_intrinsics_path, "r") as f:
        camera_intrinsics = json.load(f)

    camera_params = get_mujoco_camera_params(
        resolution, camera_extrinsics, camera_intrinsics
    )

    if camera_res is None:
        # Use vertical resolution from ZED preset (height, width)
        camera_res = camera_params.resolution.value[0]

    target_robot_state = {
        "pos": np.asarray(ee_pos, dtype=float),
        "ori_xyzw": np.asarray(ee_quat_xyzw, dtype=float),
        "gripper_pos": float(gripper_pos),
    }

    return run_panda(camera_params, target_robot_state, camera_res=camera_res)


def main() -> None:
    """
    Example entry point: reproduces the previous hard-coded Panda pose and
    writes a simple debug visualization to disk.
    """
    resolution = "HD1080"

    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]
    )
    rotation_quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    ee_pos = np.array([0.5, 0.2, 0.0], dtype=float)

    robot_mask, gripper_mask, rgb_img = generate_franka_mask(
        resolution=resolution,
        ee_pos=ee_pos,
        ee_quat_xyzw=rotation_quat_xyzw,
        gripper_pos=0.0,
    )

    # Example: save a simple debug visualization where robot+gripper are blacked out
    try:
        import matplotlib.pyplot as plt

        masked_img = np.copy(rgb_img)
        mask = (robot_mask == 1) | (gripper_mask == 1)
        if mask.ndim == 3:
            # Collapse instance channel if needed
            mask = mask.any(axis=-1)
        masked_img[mask] = 0

        plt.figure(figsize=(6, 6))
        plt.imshow(masked_img)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("debug_franka_mask.png")
        plt.close()
    except Exception as e:
        logger.warning("Failed to save debug visualization: %s", e)


if __name__ == "__main__":
    main()