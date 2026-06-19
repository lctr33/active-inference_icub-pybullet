"""
Arm control - actuating and moving the left arm to joint configurations.
This module must not import from agent.* to avoid circular imports.
"""

import time
import numpy as np  # pyright: ignore[reportMissingImports]
import pybullet as p  # pyright: ignore[reportMissingImports]

from config.robot_config import ARM_CONTROL_PARAMS


TOUCH_STATES = {21, 22, 24, 25}

NO_TOUCH = 0
TOUCH = 1


def tactile_observation_from_grid_state(
    state_index,
    tactile_contact_states=None,
):
    """
    Compute tactile observation Ot from the discrete hand state.

    Ot = 1 if the hand is in one of the predefined tactile states.
    Ot = 0 otherwise.
    """
    if tactile_contact_states is None:
        tactile_contact_states = TOUCH_STATES

    ot = TOUCH if state_index in tactile_contact_states else NO_TOUCH

    info = {
        "state_index": state_index,
        "Ot": ot,
        "tactile_contact_states": sorted(list(tactile_contact_states)),
        "contact": bool(ot == TOUCH),
    }

    return ot, info


def move_arm_to_joint_configuration(
    robot_id,
    joint_indices,
    q_target,
    steps=None,
    force=None,
    position_gain=0.25,
    velocity_gain=1.0,
):
    """
    Move the arm to a target joint configuration using position control.
    Executes the motion through multiple simulation steps.
    """
    if steps is None:
        steps = ARM_CONTROL_PARAMS.get("control_steps", 480)

    if force is None:
        force = ARM_CONTROL_PARAMS.get("force", 80.0)

    for _ in range(steps):
        for j, qj in zip(joint_indices, q_target):
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(qj),
                force=float(force),
                positionGain=float(position_gain),
                velocityGain=float(velocity_gain),
            )

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def execute_discrete_step_from_joint_config(
    robot_id,
    current_state,
    target_state,
    grid_state,
    joint_config_by_state,
    end_effector_link_index,
    arm_joint_indices,
    segmentation_mask=None,
    hand_link_id=None,
    retino_grid_shape=(4, 4),
    tactile_contact_states=None,
):
    """
    Execute one grid movement using stored joint configurations.

    Compatibility wrapper:
    - moves the arm;
    - returns movement diagnostics;
    - does NOT compute Oq, Or or Ot.

    Oq, Or and Ot are now computed in the POMDP experiment loop.
    """
    del segmentation_mask
    del hand_link_id
    del retino_grid_shape

    if current_state not in grid_state:
        raise ValueError(f"current_state={current_state} no existe en grid_state.")

    if target_state not in grid_state:
        raise ValueError(f"target_state={target_state} no existe en grid_state.")

    if target_state not in joint_config_by_state:
        raise KeyError(
            f"No hay configuración articular registrada para target_state={target_state}."
        )

    target_pos = grid_state[target_state]["pos"]
    q_target = joint_config_by_state[target_state]

    print("\n[DISCRETE STEP FROM JOINT CONFIG]")
    print(f"state_from = {current_state}")
    print(f"state_to = {target_state}")
    print(f"target_pos = {target_pos}")

    move_arm_to_joint_configuration(
        robot_id=robot_id,
        joint_indices=arm_joint_indices,
        q_target=q_target,
    )

    link_state = p.getLinkState(
        robot_id,
        end_effector_link_index,
        computeForwardKinematics=True,
    )
    reached_pos = np.asarray(link_state[4], dtype=float)

    final_error = np.linalg.norm(
        np.asarray(target_pos, dtype=float) - reached_pos
    )

    ot, ot_info = tactile_observation_from_grid_state(
        state_index=target_state,
        tactile_contact_states=tactile_contact_states,
    )

    print("[STEP RESULT]")
    print(f"reached_pos = {reached_pos.tolist()}")
    print(f"error = {final_error:.6f} m")
    print(f"Ot = {ot}")

    step_data = {
        "state_from": current_state,
        "state_to": target_state,
        "target_pos": list(target_pos),
        "reached_pos": reached_pos.tolist(),
        "error": float(final_error),
        "q_target": list(q_target),

        # Kept for compatibility. The POMDP loop computes real observations.
        "Oq": None,
        "Oq_info": None,
        "Or": None,
        "Or_info": None,
        "Ot": ot,
        "Ot_info": ot_info,
        "observation": {
            "Oq": None,
            "Or": None,
            "Ot": ot,
        },
    }

    return target_state, step_data


def factorize_end_effector_state(
    state_index,
    states_per_depth=9,
    n_depth_levels=3,
    return_labels=False,
):
    """
    Factorize a global grid state index into Shand and Sdepth.

    depth layer 0: states 0..8
    depth layer 1: states 9..17
    depth layer 2: states 18..26
    """
    if not isinstance(state_index, int):
        raise TypeError(f"state_index debe ser int, recibido: {type(state_index)}")

    n_total_states = states_per_depth * n_depth_levels

    if state_index < 0 or state_index >= n_total_states:
        raise ValueError(
            f"state_index={state_index} fuera de rango. "
            f"Rango válido: [0, {n_total_states - 1}]"
        )

    shand = state_index % states_per_depth
    sdepth = state_index // states_per_depth

    result: dict = {
        "state_index": state_index,
        "Shand": shand,
        "Sdepth": sdepth,
    }

    if return_labels:
        depth_labels = {
            0: "back_far_from_face",
            1: "middle",
            2: "front_close_to_face",
        }

        result["Sdepth_label"] = depth_labels.get(sdepth, f"depth_{sdepth}")

    return result