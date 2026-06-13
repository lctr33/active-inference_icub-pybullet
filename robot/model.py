"""
Robot model functions for mapping joints and links.
Handles construction of joint/link mappings and querying joint properties.
"""

import pybullet as p  # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
from utils.core.helpers import decode_name


def build_joint_and_link_maps(robot_id):
    """
    Build mappings between joint/link names and their PyBullet indices.
    Also identifies all movable joints.
    
    Args:
        robot_id: PyBullet body unique ID
        
    Returns:
        tuple: (joint_name_to_index, link_name_to_index, movable_joint_indices)
    """
    joint_name_to_index = {}
    link_name_to_index = {}
    movable_joint_indices = []
    
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        
        joint_name = decode_name(info[1])
        joint_type = info[2]
        link_name = decode_name(info[12])
        q_index = info[3]
        
        joint_name_to_index[joint_name] = j
        link_name_to_index[link_name] = j
        
        if q_index >= 0 and joint_type != p.JOINT_FIXED:
            movable_joint_indices.append(j)
    
    return joint_name_to_index, link_name_to_index, movable_joint_indices


def get_joint_indices(joint_name_to_index, joint_names):
    """
    Get PyBullet joint indices from joint names.
    
    Args:
        joint_name_to_index: Mapping from joint names to indices
        joint_names: List of joint names to look up
        
    Returns:
        list: Joint indices
        
    Raises:
        ValueError: If any joint name is not found
    """
    missing = [name for name in joint_names if name not in joint_name_to_index]
    
    if missing:
        raise ValueError(f"No se encontraron estos joints: {missing}")
    
    return [joint_name_to_index[name] for name in joint_names]


def get_link_index(link_name_to_index, link_name):
    """
    Get the PyBullet link index from a link name.
    
    Args:
        link_name_to_index: Mapping from link names to indices
        link_name: Name of the link to find
        
    Returns:
        int: Link index
        
    Raises:
        ValueError: If link name is not found
    """
    if link_name not in link_name_to_index:
        candidates = [name for name in link_name_to_index.keys() if "hand" in name or "l_" in name]
        raise ValueError(
            f"No se encontró el link '{link_name}'. "
            f"Candidatos posibles: {candidates}"
        )
    
    return link_name_to_index[link_name]



