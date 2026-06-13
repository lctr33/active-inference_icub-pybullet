"""
Agent exploration behavior - random grid exploration with inverse kinematics.
High-level interface for autonomous robot exploration.
"""

import random
import numpy as np # pyright: ignore[reportMissingImports]
from environment.grid import get_valid_actions, transition
from robot.kinematics import solve_left_arm_ik_dls, get_link_position
from robot.arm_control import move_arm_to_joint_configuration
from config.robot_config import IK_PARAMS, ARM_CONTROL_PARAMS
from robot.camera import camera_visualization_callback


def execute_random_discrete_step(
    robot_id,
    current_state,
    grid_state,
    end_effector_link_index,
    arm_joint_indices,
    movable_joint_indices,
    max_iters=500,
    tolerance=0.015,
):
    """
    Execute one step of exploration: choose a random valid action,
    solve IK to reach the new position, and move the arm.
    
    Args:
        robot_id: PyBullet body unique ID
        current_state: Current state index in the grid
        grid_state: Grid state mapping
        end_effector_link_index: Link index of the end-effector (l_hand)
        arm_joint_indices: Joint indices for the left arm
        movable_joint_indices: All movable joint indices in robot
        max_iters: IK max iterations
        tolerance: IK convergence tolerance
        
    Returns:
        tuple: (next_state, step_data) where step_data contains trajectory info
    """
    valid_actions = get_valid_actions(current_state, grid_state)
    
    if not valid_actions:
        raise RuntimeError(f"El estado {current_state} no tiene acciones válidas.")
    
    action = random.choice(valid_actions)
    next_state = transition(current_state, action, grid_state)
    
    target_pos = grid_state[next_state]["pos"]
    
    print("\n[DISCRETE STEP]")
    print(f"state_from = {current_state}")
    print(f"valid_actions = {valid_actions}")
    print(f"chosen_action = {action}")
    print(f"state_to = {next_state}")
    print(f"target_pos = {target_pos}")
    
    # Solve IK
    q_solution, ik_error = solve_left_arm_ik_dls(
        robot_id=robot_id,
        target_pos=target_pos,
        end_effector_link_index=end_effector_link_index,
        left_arm_joint_indices=arm_joint_indices,
        movable_joint_indices=movable_joint_indices,
        max_iters=max_iters,
        tolerance=tolerance,
        damping=IK_PARAMS["damping"],
        step_size=IK_PARAMS["step_size"],
        max_delta_q=IK_PARAMS["max_delta_q"],
    )
    
    # Execute motion
    move_arm_to_joint_configuration(
        robot_id=robot_id,
        joint_indices=arm_joint_indices,
        q_target=q_solution,
        steps=ARM_CONTROL_PARAMS["control_steps"],
        force=ARM_CONTROL_PARAMS["force"],
    )
    
    camera_visualization_callback()
    
    # Record final position and error
    reached_pos = get_link_position(robot_id, end_effector_link_index)
    final_error = np.linalg.norm(np.array(target_pos) - reached_pos)
    
    print("[STEP RESULT]")
    print(f"reached_pos = {reached_pos.tolist()}")
    print(f"error = {final_error:.6f} m")
    
    step_data = {
        "state_from": current_state,
        "action": action,
        "state_to": next_state,
        "target_pos": target_pos,
        "reached_pos": reached_pos.tolist(),
        "error": float(final_error),
        "q_solution": list(q_solution),
    }
    
    return next_state, step_data


def explore_grid_randomly(
    robot_id,
    initial_state,
    n_steps,
    grid_state,
    end_effector_link_index,
    arm_joint_indices,
    movable_joint_indices,
):
    """
    Execute multiple steps of random grid exploration.
    
    Args:
        robot_id: PyBullet body unique ID
        initial_state: Starting state index
        n_steps: Number of steps to execute
        grid_state: Grid state mapping
        end_effector_link_index: Link index of the end-effector
        arm_joint_indices: Joint indices for the left arm
        movable_joint_indices: All movable joint indices
        
    Returns:
        list: Trajectory of step data dictionaries
    """
    current_state = initial_state
    trajectory = []
    
    print("\n[EXPLORATION START]")
    print(f"initial_state = {initial_state}")
    print(f"n_steps = {n_steps}")
    
    for step in range(n_steps):
        print(f"\n========== STEP {step} ==========")
        
        next_state, step_data = execute_random_discrete_step(
            robot_id=robot_id,
            current_state=current_state,
            grid_state=grid_state,
            end_effector_link_index=end_effector_link_index,
            arm_joint_indices=arm_joint_indices,
            movable_joint_indices=movable_joint_indices,
        )
        
        
        
        step_data["step"] = step
        trajectory.append(step_data)
        
        current_state = next_state
    
    return trajectory
