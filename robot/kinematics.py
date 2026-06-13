"""
Inverse Kinematics (IK) solver and low-level joint manipulation.
Implements Damped Least Squares (DLS) algorithm for solving IK.
"""

import pybullet as p  # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]

def get_joint_limits(robot_id, joint_indices):
    """
    Get the lower and upper joint limits for specified joints.
    
    Args:
        robot_id: PyBullet body unique ID
        joint_indices: List of joint indices
        
    Returns:
        tuple: (lower_limits, upper_limits) as numpy arrays
    """
    lower = []
    upper = []
    
    for j in joint_indices:
        info = p.getJointInfo(robot_id, j)
        lo = info[8]
        hi = info[9]
        
        lower.append(lo)
        upper.append(hi)
    
    return np.array(lower, dtype=float), np.array(upper, dtype=float)

def get_joint_positions(robot_id, joint_indices):
    """
    Get current joint positions (angles).
    
    Args:
        robot_id: PyBullet body unique ID
        joint_indices: List of joint indices
        
    Returns:
        list: Current joint positions
    """
    return [p.getJointState(robot_id, j)[0] for j in joint_indices]


def set_joint_positions_instant(robot_id, joint_indices, q):
    """
    Set joint positions instantly (teleport) without dynamic simulation.
    Used during IK iterations to update joint configuration.
    
    Args:
        robot_id: PyBullet body unique ID
        joint_indices: List of joint indices to set
        q: List of target positions (angles)
    """
    for j, qj in zip(joint_indices, q):
        p.resetJointState(robot_id, j, float(qj), targetVelocity=0.0)


def get_link_position(robot_id, link_index):
    """
    Get the current 3D position of a link (end-effector).
    
    Args:
        robot_id: PyBullet body unique ID
        link_index: Link/joint index
        
    Returns:
        np.ndarray: 3D position [x, y, z]
    """
    return np.array(
        p.getLinkState(robot_id, link_index, computeForwardKinematics=True)[0],
        dtype=float
    )


def print_link_pose(icub_id, link_index, label):
    """
    Print and return the pose of a link (position and orientation).
    
    Args:
        icub_id: PyBullet body unique ID
        link_index: Link/joint index
        label: Label for console output
        
    Returns:
        tuple: (position, orientation)
    """
    pos, orn = p.getLinkState(icub_id, link_index, computeForwardKinematics=True)[:2]
    print(f"{label}: pos={list(pos)} orn={list(orn)}")
    return pos, orn


def solve_left_arm_ik_dls(
    robot_id,
    target_pos,
    end_effector_link_index,
    left_arm_joint_indices,
    movable_joint_indices,
    max_iters=300,
    tolerance=0.01,
    damping=0.05,
    step_size=0.6,
    max_delta_q=0.08,
):
    """
    Solve inverse kinematics for the left arm using Damped Least Squares (DLS).
    Only modifies joints in left_arm_joint_indices; other joints remain fixed.
    
    Uses the Jacobian method:
        dq = J^T (J J^T + lambda^2 I)^-1 e
    
    Args:
        robot_id: PyBullet body unique ID
        target_pos: Target end-effector position [x, y, z]
        end_effector_link_index: Link index of end-effector (l_hand)
        left_arm_joint_indices: Joint indices for left arm
        movable_joint_indices: All movable joint indices in robot
        max_iters: Maximum number of iterations
        tolerance: Convergence tolerance (meters)
        damping: Damping factor (lambda) for regularization
        step_size: Step size for joint update
        max_delta_q: Maximum delta per iteration (joint limit)
        
    Returns:
        tuple: (q_solution, final_error) where q_solution is joint configuration
    """
    target_pos = np.array(target_pos, dtype=float)
    
    q = get_joint_positions(robot_id, left_arm_joint_indices)
    q = np.array(q, dtype=float)
    lower_limits, upper_limits = get_joint_limits(robot_id, left_arm_joint_indices)
    
    # Build mapping: joint_index -> column in full Jacobian
    dof_col_by_joint = {
        joint_index: dof_i
        for dof_i, joint_index in enumerate(movable_joint_indices)
    }
    
    left_arm_cols = [dof_col_by_joint[j] for j in left_arm_joint_indices]
    
    for it in range(max_iters):
        # Update left arm joints, keep rest of robot fixed
        set_joint_positions_instant(robot_id, left_arm_joint_indices, q)
        
        current_pos = get_link_position(robot_id, end_effector_link_index)
        error_vec = target_pos - current_pos
        error_norm = np.linalg.norm(error_vec)
        
        if error_norm < tolerance:
            print(f"[IK-DLS] Convergió en iteración {it}, error={error_norm:.6f} m")
            return q, error_norm
        
        # Get Jacobian from PyBullet (requires full robot state)
        all_q = [p.getJointState(robot_id, j)[0] for j in movable_joint_indices]
        all_dq = [0.0] * len(movable_joint_indices)
        all_ddq = [0.0] * len(movable_joint_indices)
        
        zero_local_pos = [0.0, 0.0, 0.0]
        
        jac_t, jac_r = p.calculateJacobian(
            bodyUniqueId=robot_id,
            linkIndex=end_effector_link_index,
            localPosition=zero_local_pos,
            objPositions=all_q,
            objVelocities=all_dq,
            objAccelerations=all_ddq
        )
        
        # Extract translational Jacobian for left arm columns
        J_full = np.array(jac_t, dtype=float)
        J = J_full[:, left_arm_cols]  # 3 x 7 for left arm
        
        # Damped Least Squares: dq = J^T (J J^T + lambda^2 I)^-1 e
        A = J @ J.T + (damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, error_vec)
        
        dq = step_size * dq
        dq = np.clip(dq, -max_delta_q, max_delta_q)
        
        q = q + dq
        q = np.clip(q, lower_limits, upper_limits)
    
    # Converged or max iterations reached
    set_joint_positions_instant(robot_id, left_arm_joint_indices, q)
    final_pos = get_link_position(robot_id, end_effector_link_index)
    final_error = np.linalg.norm(target_pos - final_pos)
    
    if final_error > tolerance:
        print(f"[IK-DLS] No convergió. Error final={final_error:.6f} m")
    
    return q, final_error
