import numpy as np # pyright: ignore[reportMissingImports]
import pybullet as p # pyright: ignore

from typing import List, Dict, Optional
from config.robot_config import ARM_CONTROL_PARAMS
from robot.arm_control import move_arm_to_joint_configuration

def move_hand_to_grid_state_from_joint_config(
    robot_id,
    target_state,
    joint_config_by_state,
    arm_joint_indices,
    control_steps=None,
    force=None,
):
    """
    Move the robot hand to a grid state using a previously stored joint configuration.

    Args:
        robot_id: PyBullet body unique ID.
        target_state: Grid state index.
        joint_config_by_state: dict {state_index: joint_configuration}.
        arm_joint_indices: Joint indices for the controlled arm.
        control_steps: Number of control steps.
        force: Motor force.

    Returns:
        list: Target joint configuration used.
    """
    if target_state not in joint_config_by_state:
        raise KeyError(
            f"No hay configuración articular registrada para target_state={target_state}."
        )

    q_target = joint_config_by_state[target_state]

    if control_steps is None:
        control_steps = ARM_CONTROL_PARAMS["control_steps"]

    if force is None:
        force = ARM_CONTROL_PARAMS["force"]

    move_arm_to_joint_configuration(
        robot_id=robot_id,
        joint_indices=arm_joint_indices,
        q_target=q_target,
        steps=control_steps,
        force=force,
    )

    return list(q_target)

def read_joint_configuration(robot_id, joint_indices):
    """
    Read current joint positions from PyBullet.

    Returns:
        np.ndarray: Current joint configuration.
    """
    q = []
    for joint_idx in joint_indices:
        joint_state = p.getJointState(robot_id, joint_idx)
        q.append(joint_state[0])

    return np.asarray(q, dtype=float)


def proprioceptive_observation_from_joint_config(
    robot_id,
    arm_joint_indices,
    joint_config_by_state,
):
    """
    Compute proprioceptive observation Oq as the closest stored joint prototype.

    Args:
        robot_id: PyBullet body unique ID.
        arm_joint_indices: Controlled arm joint indices.
        joint_config_by_state: dict {state_index: joint_configuration}.

    Returns:
        tuple:
            oq: proprioceptive observation index, using the grid state key.
            info: diagnostic dictionary.
    """
    q_current = read_joint_configuration(robot_id, arm_joint_indices)

    best_state = None
    best_distance = np.inf

    for state_idx, q_proto in joint_config_by_state.items():
        q_proto = np.asarray(q_proto, dtype=float)

        if q_proto.shape != q_current.shape:
            raise ValueError(
                f"Dimensión incompatible en state={state_idx}: "
                f"q_proto.shape={q_proto.shape}, q_current.shape={q_current.shape}"
            )

        dist = np.linalg.norm(q_current - q_proto)

        if dist < best_distance:
            best_distance = dist
            best_state = state_idx

    info = {
        "q_current": q_current.tolist(),
        "nearest_state": best_state,
        "joint_distance": float(best_distance),
    }

    return best_state, info

def retinotopic_observation_from_segmentation(
    segmentation_mask,
    robot_id,
    hand_link_id,
    grid_shape=(4, 4),
    image_width=None,
    image_height=None,
):
    """
    Compute retinotopic observation Or from a synthetic camera segmentation mask.

    The image is divided into a retinotopic grid, default 4x4.
    The function finds the centroid of the pixels belonging to the hand link
    and returns the corresponding retinotopic cell index.

    Cell indexing is row-major:

        0   1   2   3
        4   5   6   7
        8   9   10  11
        12  13  14  15

    Args:
        segmentation_mask: PyBullet segmentation mask from getCameraImage.
        robot_id: PyBullet body unique ID of the robot.
        hand_link_id: Link index of the hand.
        grid_shape: tuple (n_rows, n_cols), default (4, 4).
        image_width: Required if segmentation_mask is flat.
        image_height: Required if segmentation_mask is flat.

    Returns:
        tuple:
            or_obs: retinotopic observation index, or None if hand is not visible.
            info: diagnostic dictionary.
    """
    seg = np.asarray(segmentation_mask, dtype=np.int64)

    if seg.ndim == 1:
        if image_width is None or image_height is None:
            raise ValueError(
                "image_width e image_height son necesarios si segmentation_mask es 1D."
            )
        seg = seg.reshape((image_height, image_width))

    if seg.ndim != 2:
        raise ValueError(f"segmentation_mask debe ser 2D. Forma recibida: {seg.shape}")

    height, width = seg.shape
    n_rows, n_cols = grid_shape

    object_ids = seg & ((1 << 24) - 1)
    link_ids = (seg >> 24) - 1

    hand_mask = (object_ids == robot_id) & (link_ids == hand_link_id)

    ys, xs = np.where(hand_mask)

    if len(xs) == 0:
        info = {
            "visible": False,
            "centroid": None,
            "pixel_count": 0,
            "grid_shape": grid_shape,
            "image_shape": (height, width),
        }
        return None, info

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    cell_w = width / n_cols
    cell_h = height / n_rows

    col = int(cx // cell_w)
    row = int(cy // cell_h)

    col = min(max(col, 0), n_cols - 1)
    row = min(max(row, 0), n_rows - 1)

    or_obs = row * n_cols + col

    info = {
        "visible": True,
        "centroid": [cx, cy],
        "row": row,
        "col": col,
        "or_obs": or_obs,
        "pixel_count": int(len(xs)),
        "grid_shape": grid_shape,
        "image_shape": (height, width),
    }

    return or_obs, info