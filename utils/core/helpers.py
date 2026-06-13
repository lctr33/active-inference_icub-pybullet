"""
Utility functions for iCub robot manipulation.
Includes helpers for working with joints, links, and basic robot queries.
"""

import pybullet as p  # pyright: ignore[reportMissingImports]
import numpy as np


def decode_name(x):
    """Convert bytes to UTF-8 string."""
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)


def get_link_indices_by_name(body_id):
    """
    Build a mapping from link names to their indices.
    
    Args:
        body_id: PyBullet body unique ID
        
    Returns:
        dict: Mapping of link_name -> joint_index
    """
    link_name_to_index = {}
    num_joints = p.getNumJoints(body_id)
    
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(body_id, joint_index)
        link_name = decode_name(joint_info[12])
        link_name_to_index[link_name] = joint_index
    
    return link_name_to_index


def get_link_global_pose(body_id, link_index):
    """
    Get the global position and orientation of a link.
    
    Args:
        body_id: PyBullet body unique ID
        link_index: Joint index corresponding to the link
        
    Returns:
        tuple: (position, orientation) where position is [x, y, z] and orientation is quaternion [x, y, z, w]
    """
    link_state = p.getLinkState(
        body_id,
        link_index,
        computeForwardKinematics=True
    )
    
    position = link_state[0]
    orientation = link_state[1]
    
    return position, orientation
