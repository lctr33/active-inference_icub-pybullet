"""
Visualization utilities - creating visual markers in PyBullet.
Used for debugging and visualization of grid points and trajectories.
"""

import pybullet as p  # pyright: ignore[reportMissingImports]
from config.robot_config import SPHERE_VISUAL_PARAMS


def create_visual_sphere(position, radius=None, color=None):
    """
    Create and place a visual sphere in the simulation (no collision).
    Useful for marking waypoints, targets, or grid points.
    
    Args:
        position: 3D position [x, y, z] for the sphere center
        radius: Sphere radius (if None, uses default from config)
        color: RGBA color tuple (if None, uses default from config)
        
    Returns:
        int: PyBullet body unique ID of the sphere
    """
    if radius is None:
        radius = SPHERE_VISUAL_PARAMS["radius"]
    if color is None:
        color = SPHERE_VISUAL_PARAMS["color"]
    
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color
    )
    
    sphere_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=position
    )
    
    return sphere_id
