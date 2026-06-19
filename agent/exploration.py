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
    target_state,
    max_iters=500,
    tolerance=0.015,
):
    """
    Execute one deterministic grid step: move the hand to target_state.

    Args:
        robot_id: PyBullet body unique ID
        current_state: Current state index in the grid
        grid_state: Grid state mapping
        end_effector_link_index: Link index of the end-effector (l_hand)
        arm_joint_indices: Joint indices for the left arm
        movable_joint_indices: All movable joint indices in robot
        target_state: Grid state to reach
        max_iters: IK max iterations
        tolerance: IK convergence tolerance

    Returns:
        tuple: (next_state, step_data)
    """
    if current_state not in grid_state:
        raise ValueError(f"current_state={current_state} no existe en grid_state.")

    if target_state not in grid_state:
        raise ValueError(f"target_state={target_state} no existe en grid_state.")

    valid_actions = get_valid_actions(current_state, grid_state)

    # Try to infer which discrete action connects current_state -> target_state.
    # If no valid action matches, the movement is treated as a direct target jump.
    action = None
    for candidate_action in valid_actions:
        candidate_next = transition(current_state, candidate_action, grid_state)
        if candidate_next == target_state:
            action = candidate_action
            break

    if action is None:
        action = f"direct_to_{target_state}"

    next_state = target_state
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
        "target_pos": list(target_pos),
        "reached_pos": reached_pos.tolist(),
        "ik_error": float(ik_error),
        "error": float(final_error),
        "q_solution": list(q_solution),
    }

    return next_state, step_data


def explore_grid_getting_close(
    robot_id,
    grid_state,
    end_effector_link_index,
    arm_joint_indices,
    movable_joint_indices,
    start_states=None,
    depth_stride=9,
    max_iters=500,
    tolerance=0.015,
):
    """
    Execute getting-close trajectories toward the robot face.

    Each trajectory follows:
        [back_state, middle_state, front_state]
        [i, i + depth_stride, i + 2 * depth_stride]

    By default, it explores start states 0..7 and omits 8 because IK fails
    for that approach trajectory.

    Args:
        robot_id: PyBullet body unique ID
        grid_state: Grid state mapping
        end_effector_link_index: Link index of the end-effector
        arm_joint_indices: Joint indices for the left arm
        movable_joint_indices: All movable joint indices
        start_states: Back-layer states used to start getting-close trajectories
        depth_stride: Index offset between depth layers
        max_iters: IK max iterations
        tolerance: IK convergence tolerance

    Returns:
        tuple:
            trajectory: list of step data dictionaries
            joint_config_by_state: dict {grid_state_index: joint_configuration}
    """
    if start_states is None:
        start_states = [2, 1, 0, 3, 4, 5, 6, 7, 8]

    trajectory = []
    joint_config_by_state = {}

    print("\n[GETTING CLOSE EXPLORATION START]")
    print(f"start_states = {start_states}")
    print(f"depth_stride = {depth_stride}")
    print("omitted_start_state = 8")

    global_step = 0

    for start_state in start_states:
        state_sequence = [
            start_state,
            start_state + depth_stride,
            start_state + 2 * depth_stride,
        ]

        for state in state_sequence:
            if state not in grid_state:
                raise ValueError(
                    f"state={state} no existe en grid_state. "
                    f"Secuencia inválida: {state_sequence}"
                )

        print("\n[GETTING CLOSE TRAJECTORY]")
        print(f"state_sequence = {state_sequence}")

        current_state = state_sequence[0]

        # Move explicitly to the first state of the trajectory.
        for local_step, target_state in enumerate(state_sequence):
            print(
                f"\n========== GLOBAL STEP {global_step} | "
                f"LOCAL STEP {local_step} | target_state={target_state} =========="
            )

            next_state, step_data = execute_random_discrete_step(
                robot_id=robot_id,
                current_state=current_state,
                grid_state=grid_state,
                end_effector_link_index=end_effector_link_index,
                arm_joint_indices=arm_joint_indices,
                movable_joint_indices=movable_joint_indices,
                target_state=target_state,
                max_iters=max_iters,
                tolerance=tolerance,
            )

            step_data["global_step"] = global_step
            step_data["local_step"] = local_step
            step_data["getting_close_sequence"] = list(state_sequence)

            trajectory.append(step_data)

            # Register joint configuration after completing the movement.
            joint_config_by_state[target_state] = list(step_data["q_solution"])

            print("[JOINT CONFIG REGISTERED]")
            print(f"joint_config_by_state[{target_state}] = {joint_config_by_state[target_state]}")

            current_state = next_state
            global_step += 1

    return trajectory, joint_config_by_state
