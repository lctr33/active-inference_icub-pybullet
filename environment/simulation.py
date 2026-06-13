"""
PyBullet simulation initialization and configuration.
Handles physics engine setup, robot loading, and camera configuration.
"""

import pybullet as p  # pyright: ignore[reportMissingImports]
from config.robot_config import ICUB_URDF, SIMULATION_PARAMS


def initialize_pybullet_gui():
    """
    Initialize PyBullet with GUI mode and default configuration.
    
    Returns:
        int: Physics client ID
    """
    width = SIMULATION_PARAMS["gui_width"]
    height = SIMULATION_PARAMS["gui_height"]
    
    physics_client = p.connect(p.GUI, options=f"--opengl2 --width={width} --height={height}")

    # Asegurar que la GUI/rendering están activos
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    
    # Disable debug visualizer overlays
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    
    return physics_client


def setup_simulation_physics():
    """Configure physics parameters (gravity, etc.)."""
    gravity = SIMULATION_PARAMS["gravity"]
    p.setGravity(*gravity)


def load_icub_robot():
    """
    Load the iCub robot from URDF file.
    
    Returns:
        int: PyBullet body unique ID for the robot
    """
    icub_id = p.loadURDF(
        ICUB_URDF,
        basePosition=[0, 0, 0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE
    )
    
    print(f"iCub cargado correctamente. body_id = {icub_id}")
    return icub_id


def setup_camera():
    """Configure the debug visualizer camera view."""
    params = SIMULATION_PARAMS
    p.resetDebugVisualizerCamera(
        cameraDistance=params["camera_distance"],
        cameraYaw=params["camera_yaw"],
        cameraPitch=params["camera_pitch"],
        cameraTargetPosition=params["camera_target"]
    )
