"""
Main entry point for iCub robot exploration simulation.
Orchestrates initialization, setup, and execution of random grid exploration.
"""

import time
import pybullet as p  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt 
import numpy as np # pyright: ignore[reportMissingImports]
import cv2 # pyright: ignore[reportMissingImports]

from config.robot_config import GRID_POINTS, IK_PARAMS, ARM_CONTROL_PARAMS
from environment.simulation import (
    initialize_pybullet_gui,
    setup_simulation_physics,
    load_icub_robot,
    setup_camera,
)
from environment.grid import build_grid_state
from environment.visualization import create_visual_sphere
from robot.model import build_joint_and_link_maps, get_joint_indices, get_link_index
from robot.kinematics import solve_left_arm_ik_dls, get_link_position
from robot.arm_control import move_arm_to_joint_configuration
from agent.exploration import explore_grid_randomly
from scripts.l_image import get_camera_from_link

from config.robot_config import (
    LEFT_HAND_LINK_NAME,
    LEFT_ARM_JOINT_NAMES,
    SIMULATION_PARAMS,
)


def main():
    """Main execution function."""
    
    # ===== Initialize PyBullet =====
    print("[INITIALIZATION]")
    physics_client = initialize_pybullet_gui()
    print(f"Holaaa: {physics_client}")
    setup_simulation_physics()
    setup_camera()
    
    # ===== Load Robot =====
    icub_id = load_icub_robot()
    print(icub_id)
    
    # ===== Setup Grid Visualization =====
    # sphere_ids = []
    # for i, point in enumerate(GRID_POINTS):
    #     sphere_id = create_visual_sphere(point)
    #     sphere_ids.append(sphere_id)
    
    grid_state = build_grid_state(GRID_POINTS)
    
    # ===== Build Robot Joint/Link Maps =====
    joint_name_to_index, link_name_to_index, movable_joint_indices = build_joint_and_link_maps(icub_id)
    left_arm_joint_indices = get_joint_indices(
        joint_name_to_index,
        LEFT_ARM_JOINT_NAMES
    )
    left_hand_link_index = get_link_index(link_name_to_index, LEFT_HAND_LINK_NAME)
    
    # ===== Initial Positioning =====
    initial_state = 2
    target_pos = grid_state[initial_state]["pos"]
    
    q_solution, ik_error = solve_left_arm_ik_dls(
        robot_id=icub_id,
        target_pos=target_pos,
        end_effector_link_index=left_hand_link_index,
        left_arm_joint_indices=left_arm_joint_indices,
        movable_joint_indices=movable_joint_indices,
        max_iters=IK_PARAMS["max_iterations"],
        tolerance=IK_PARAMS["tolerance"],
        damping=IK_PARAMS["damping"],
        step_size=IK_PARAMS["step_size"],
        max_delta_q=IK_PARAMS["max_delta_q"],
    )
    
    move_arm_to_joint_configuration(
        robot_id=icub_id,
        joint_indices=left_arm_joint_indices,
        q_target=q_solution,
        steps=ARM_CONTROL_PARAMS["control_steps"],
        force=ARM_CONTROL_PARAMS["force"],
    )
    
    # ===== Exploration Phase =====
    trajectory = explore_grid_randomly(
        robot_id=icub_id,
        initial_state=initial_state,
        n_steps=20,
        grid_state=grid_state,
        end_effector_link_index=left_hand_link_index,
        arm_joint_indices=left_arm_joint_indices,
        movable_joint_indices=movable_joint_indices,
    )

    # ===== Keep Simulation Running =====
    print("\n[SIMULATION RUNNING - Press Ctrl+C to exit]")
    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    
    except KeyboardInterrupt:
        print("\nSimulación cerrada.")
        p.disconnect()


if __name__ == "__main__":
    main()
