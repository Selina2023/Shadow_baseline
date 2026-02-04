
import pdb
import os
import json
import torch
from tqdm import tqdm
import numpy as np 
import logging
import argparse 
from typing import Tuple

import pandas as pd
import logging
import mediapy as media
from scipy.spatial.transform import Rotation

from robomimic.utils.torch_utils import matrix_to_rotation_6d
from human_shadow.utils.file_utils import get_parent_folder_of_package
from human_shadow.utils.transform_utils import transform_pt, invert_homogeneous_matrix
from human_shadow.virtual_twin.twin_robot import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def run_panda(camera_params, target_robot_state):
    twin_robot = TwinRobot("Panda", "Robotiq85", camera_params, camera_res=1080, render=False, n_steps_short=3)
    twin_robot.reset()
    robot_mask, gripper_mask, rgb_img = twin_robot.move_to_target_state(target_robot_state, init=True)
    return robot_mask, gripper_mask, rgb_img

def run_ur5e(camera_params, target_robot_state):
    twin_robot = TwinRobot("UR5e", "Robotiq85", camera_params, camera_res=1080, render=False, n_steps_short=3)
    twin_robot.reset()
    robot_mask, gripper_mask, rgb_img = twin_robot.move_to_target_state(target_robot_state, init=True)
    return robot_mask, gripper_mask, rgb_img

def run_iiwa(camera_params, target_robot_state):
    twin_robot = TwinRobot("IIWA", "Robotiq85", camera_params, camera_res=1080, render=False, n_steps_short=3)
    twin_robot.reset()
    robot_mask, gripper_mask, rgb_img = twin_robot.move_to_target_state(target_robot_state, init=True)
    return robot_mask, gripper_mask, rgb_img

def main():
    resolution = "HD1080"
    project_folder = get_parent_folder_of_package("human_shadow")

    # Extrinsics
    camera_extrinsics_path = os.path.join(project_folder, f"human_shadow/camera/camera_calibration_data/hand_calib_{resolution}/cam_cal.json")
    with open(camera_extrinsics_path, "r") as f:
        camera_extrinsics = json.load(f)
    T_cam2robot = get_transformation_matrix_from_extrinsics(camera_extrinsics)

    # Intrinsics
    camera_intrinsics_path = os.path.join(project_folder, f"human_shadow/camera/intrinsics/camera_intrinsics_{resolution}.json")
    with open(camera_intrinsics_path, "r") as f:
        camera_intrinsics = json.load(f)

    camera_params = get_mujoco_camera_params(resolution, camera_extrinsics, camera_intrinsics)
    
    rotation_matrix = np.array([[1, 0, 0], 
                                [0, -1, 0], 
                                [0, 0, -1]])
    rotation_quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    target_robot_state = {
            "pos": np.array([0.5, 0.2, 0.0]),
            "ori_xyzw": rotation_quat_xyzw,
            "gripper_pos": 0.0
        }
    robot_mask_panda, gripper_mask_panda, rgb_img_panda = run_panda(camera_params, target_robot_state)
    # robot_mask_ur5e, gripper_mask_ur5e, rgb_img_ur5e = run_ur5e(camera_params, target_robot_state)
    robot_mask_iiwa, gripper_mask_iiwa, rgb_img_iiwa = run_iiwa(camera_params, target_robot_state)

    rgb_img_with_mask = np.zeros_like(rgb_img_panda)
    rgb_img_with_mask[(robot_mask_panda == 0) & (gripper_mask_panda == 0) & (robot_mask_iiwa == 0) & (gripper_mask_iiwa == 0)] = rgb_img_panda[(robot_mask_panda == 0) & (gripper_mask_panda == 0) & (robot_mask_iiwa == 0) & (gripper_mask_iiwa == 0)]
    plt.imshow(rgb_img_with_mask)
    plt.savefig("debug.png")

if __name__ == "__main__":
    main()