"""
Configuration for iCub robot and simulation environment.
Contains all constants and parameters for the robot model and grid.
"""

ICUB_URDF = ""

# Left arm joint configuration
LEFT_HAND_LINK_NAME = "l_hand"

LEFT_ARM_JOINT_NAMES = [
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow",
    "l_wrist_prosup",
    "l_wrist_pitch",
    "l_wrist_yaw",
]

# Joints that remain fixed during manipulation
FIXED_JOINT_NAMES = [
    "torso_pitch",
    "torso_roll",
    "torso_yaw",
    "neck_pitch",
    "neck_roll",
    "neck_yaw",
]

# Grid points for discrete end-effector positions (3x3 grid)
# GRID_POINTS = [
#     [-0.1832700999981297, -0.11942584200005119, 0.41546400000001654],
#     [-0.1832700999981297, -0.05942584200005118, 0.41546400000001654],
#     [-0.1832700999981297, 0.0005741579999488169, 0.41546400000001654],
#     [-0.1832700999981297, -0.11942584200005119, 0.35546400000001654],
#     [-0.1832700999981297, -0.05942584200005118, 0.35546400000001654],
#     [-0.1832700999981297, 0.0005741579999488169, 0.35546400000001654],
#     [-0.1832700999981297, -0.11942584200005119, 0.29546400000001655],
#     [-0.1832700999981297, -0.05942584200005118, 0.29546400000001655],
#     [-0.1832700999981297, 0.0005741579999488169, 0.29546400000001655],
# ]

GRID_SHAPE = (3, 3)  # rows, columns

# Grid points for discrete end-effector positions (3x3 grid)
GRID_POINTS = [
[-0.19327, -0.09945, 0.31546],
[-0.19327, -0.09945, 0.35546],
[-0.19327, -0.09945, 0.39546],
[-0.19327, -0.059449999999999996, 0.31546],
[-0.19327, -0.059449999999999996, 0.35546],
[-0.19327, -0.059449999999999996, 0.39546],
[-0.19327, -0.019449999999999995, 0.31546],
[-0.19327, -0.019449999999999995, 0.35546],
[-0.19327, -0.019449999999999995, 0.39546],
[-0.15327, -0.09945, 0.31546],
[-0.15327, -0.09945, 0.35546],
[-0.15327, -0.09945, 0.39546],
[-0.15327, -0.059449999999999996, 0.31546],
[-0.15327, -0.059449999999999996, 0.35546],
[-0.15327, -0.059449999999999996, 0.39546],
[-0.15327, -0.019449999999999995, 0.31546],
[-0.15327, -0.019449999999999995, 0.35546],
[-0.15327, -0.019449999999999995, 0.39546],
[-0.11326999999999998, -0.09945, 0.31546],
[-0.11326999999999998, -0.09945, 0.35546],
[-0.11326999999999998, -0.09945, 0.39546],
[-0.11326999999999998, -0.059449999999999996, 0.31546],
[-0.11326999999999998, -0.059449999999999996, 0.35546],
[-0.11326999999999998, -0.059449999999999996, 0.39546],
[-0.11326999999999998, -0.019449999999999995, 0.31546],
[-0.11326999999999998, -0.019449999999999995, 0.35546],
[-0.11326999999999998, -0.019449999999999995, 0.39546],
]

GRID_SHAPE = (3, 3, 3)  # rows, columns

# Discrete actions for grid navigation (row_delta, col_delta)

ACTIONS = {
    "x_neg": (-1, 0, 0),
    "x_pos": (1, 0, 0),

    "y_neg": (0, -1, 0),
    "y_pos": (0, 1, 0),

    "z_neg": (0, 0, -1),
    "z_pos": (0, 0, 1),
}

# PyBullet simulation parameters
SIMULATION_PARAMS = {
    "gui_width": 1400,
    "gui_height": 900,
    "gravity": (0, 0, -9.81),
    "camera_distance": 0.7,
    "camera_yaw": -135,
    "camera_pitch": -25,
    "camera_target": [0, 0, 0.05],
    "timestep": 1.0 / 240.0,  # 240 Hz simulation
}

# IK solver parameters
IK_PARAMS = {
    "max_iterations": 500,
    "tolerance": 0.01,
    "damping": 0.05,
    "step_size": 0.6,
    "max_delta_q": 0.06,
}

# Arm control parameters
ARM_CONTROL_PARAMS = {
    "control_steps": 480,
    "force": 100.0,
    "position_gain": 0.25,
    "velocity_gain": 1.0,
}

# Visualization parameters
SPHERE_VISUAL_PARAMS = {
    "radius": 0.01,
    "color": (0.0, 1.0, 0.0, 0.8),  # RGBA
}
