"""
Grid environment - discrete space and state management.
Handles grid construction and state transitions.
"""

from config.robot_config import ACTIONS, GRID_SHAPE


def build_grid_state(points):
    """
    Build a state representation for a discrete 3D grid from a list of positions.

    Each point is assigned:
        - state index
        - ix, iy, iz discrete coordinates
        - physical position [x, y, z]

    Args:
        points: List of [x, y, z] positions.
        grid_shape: Tuple (n_x, n_y, n_z). If None, uses GRID_SHAPE.

    Returns:
        dict: {index: {"index", "ix", "iy", "iz", "pos"}}
    """

    grid_shape = GRID_SHAPE

    n_x, n_y, n_z = grid_shape

    expected_points = n_x * n_y * n_z
    if len(points) != expected_points:
        raise ValueError(
            f"Expected {expected_points} points for grid_shape={grid_shape}, "
            f"but received {len(points)}."
        )

    grid_state = {}

    for idx, pos in enumerate(points):
        ix = idx // (n_y * n_z)
        remainder = idx % (n_y * n_z)

        iy = remainder // n_z
        iz = remainder % n_z

        grid_state[idx] = {
            "index": idx,
            "ix": ix,
            "iy": iy,
            "iz": iz,
            "pos": pos,
        }

    return grid_state


def get_valid_actions(current_state, grid_state, grid_shape=None):
    """
    Get the valid 3D actions that keep the agent within grid bounds.

    Args:
        current_state: Current state index.
        grid_state: Grid state mapping.
        grid_shape: Tuple (n_x, n_y, n_z). If None, uses GRID_SHAPE.

    Returns:
        list: Valid action names.
    """

    if grid_shape is None:
        grid_shape = GRID_SHAPE

    n_x, n_y, n_z = grid_shape

    ix = grid_state[current_state]["ix"]
    iy = grid_state[current_state]["iy"]
    iz = grid_state[current_state]["iz"]

    valid_actions = []

    for action, (d_ix, d_iy, d_iz) in ACTIONS.items():
        new_ix = ix + d_ix
        new_iy = iy + d_iy
        new_iz = iz + d_iz

        inside_x = 0 <= new_ix < n_x
        inside_y = 0 <= new_iy < n_y
        inside_z = 0 <= new_iz < n_z

        if inside_x and inside_y and inside_z:
            valid_actions.append(action)

    return valid_actions


def transition(current_state, action, grid_state, grid_shape=None):
    """
    Apply a discrete 3D action and return the new state index.

    Assumes the action has already been validated.

    Args:
        current_state: Current state index.
        action: Action name from ACTIONS.
        grid_state: Grid state mapping.
        grid_shape: Tuple (n_x, n_y, n_z). If None, uses GRID_SHAPE.

    Returns:
        int: New state index.
    """

    if grid_shape is None:
        grid_shape = GRID_SHAPE

    n_x, n_y, n_z = grid_shape

    ix = grid_state[current_state]["ix"]
    iy = grid_state[current_state]["iy"]
    iz = grid_state[current_state]["iz"]

    d_ix, d_iy, d_iz = ACTIONS[action]

    new_ix = ix + d_ix
    new_iy = iy + d_iy
    new_iz = iz + d_iz

    new_state = new_ix * (n_y * n_z) + new_iy * n_z + new_iz

    return new_state
