import argparse
import logging
import os
from typing import Optional

import cv2
import numpy as np

from human_shadow.generate_robot_mask import generate_franka_mask


logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone script to render a Franka Panda in the virtual twin "
            "and export its robot/gripper masks for a configurable pose and camera."
        )
    )

    # Robot configuration
    parser.add_argument(
        "--resolution",
        type=str,
        default="HD1080",
        help='ZED resolution key used for calibration (e.g. "HD1080").',
    )
    parser.add_argument(
        "--ee-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        required=True,
        help="End-effector position in robot base frame (meters).",
    )
    parser.add_argument(
        "--ee-quat-xyzw",
        type=float,
        nargs=4,
        metavar=("QX", "QY", "QZ", "QW"),
        required=True,
        help="End-effector orientation as quaternion (x y z w).",
    )
    parser.add_argument(
        "--gripper-pos",
        type=float,
        default=0.0,
        help="Gripper command/value (forwarded directly to the virtual twin).",
    )

    # Camera configuration (optional overrides)
    parser.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        metavar=("CX", "CY", "CZ"),
        help="Optional camera position override (x y z). "
             "If omitted, uses the calibrated camera pose.",
    )
    parser.add_argument(
        "--camera-quat-wxyz",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
        help="Optional camera orientation override as quaternion (w x y z). "
             "If omitted, uses the calibrated camera orientation.",
    )
    parser.add_argument(
        "--camera-res",
        type=int,
        default=None,
        help="Optional square RGB output resolution. "
             "Defaults to the calibrated vertical resolution.",
    )

    # Output options
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="franka_mask",
        help="Prefix for output files (PNG + NPY).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where output files are saved.",
    )

    return parser


def save_outputs(
    output_dir: str,
    prefix: str,
    robot_mask: np.ndarray,
    gripper_mask: np.ndarray,
    rgb_img: np.ndarray,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Ensure masks are uint8 when saving as npy (values are 0/1 already)
    robot_mask_out = robot_mask.astype(np.uint8)
    gripper_mask_out = gripper_mask.astype(np.uint8)

    robot_mask_path = os.path.join(output_dir, f"{prefix}_robot_mask.npy")
    gripper_mask_path = os.path.join(output_dir, f"{prefix}_gripper_mask.npy")
    rgb_path = os.path.join(output_dir, f"{prefix}_rgb.png")

    np.save(robot_mask_path, robot_mask_out)
    np.save(gripper_mask_path, gripper_mask_out)

    # Convert RGB -> BGR for OpenCV
    rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(rgb_path, rgb_bgr)

    logger.info("Saved robot mask to %s", robot_mask_path)
    logger.info("Saved gripper mask to %s", gripper_mask_path)
    logger.info("Saved RGB image to %s", rgb_path)


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        parser = build_arg_parser()
        args = parser.parse_args()

    ee_pos = np.asarray(args.ee_pos, dtype=float)
    ee_quat_xyzw = np.asarray(args.ee_quat_xyzw, dtype=float)

    camera_pos = None
    camera_quat_wxyz = None
    if args.camera_pos is not None:
        camera_pos = np.asarray(args.camera_pos, dtype=float)
    if args.camera_quat_wxyz is not None:
        camera_quat_wxyz = np.asarray(args.camera_quat_wxyz, dtype=float)

    robot_mask, gripper_mask, rgb_img = generate_franka_mask(
        resolution=args.resolution,
        ee_pos=ee_pos,
        ee_quat_xyzw=ee_quat_xyzw,
        gripper_pos=args.gripper_pos,
        camera_pos=camera_pos,
        camera_quat_wxyz=camera_quat_wxyz,
        camera_res=args.camera_res,
    )

    save_outputs(
        output_dir=args.output_dir,
        prefix=args.output_prefix,
        robot_mask=robot_mask,
        gripper_mask=gripper_mask,
        rgb_img=rgb_img,
    )


if __name__ == "__main__":
    main()

