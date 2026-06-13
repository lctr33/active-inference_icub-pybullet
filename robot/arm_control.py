"""
Arm control - actuating and moving the left arm to joint configurations.
Provides high-level interface for arm trajectory execution.
"""

import pybullet as p  # pyright: ignore[reportMissingImports]
import time


def move_arm_to_joint_configuration(
    robot_id,
    joint_indices,
    q_target,
    steps=480,
    force=80.0,
    position_gain=0.25,
    velocity_gain=1.0,
):
    """
    Move the arm to a target joint configuration using position control.
    Executes the motion through multiple simulation steps.
    
    Args:
        robot_id: PyBullet body unique ID
        joint_indices: List of joint indices to control
        q_target: Target joint positions (angles)
        steps: Number of simulation steps to execute
        force: Maximum force applied by motors
        position_gain: Proportional gain for position control
        velocity_gain: Derivative gain for velocity control
    """
    for _ in range(steps):
        for j, qj in zip(joint_indices, q_target):
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(qj),
                force=force,
                positionGain=position_gain,
                velocityGain=velocity_gain,
            )
        
        p.stepSimulation()
        time.sleep(1.0 / 240.0)  # 240 Hz simulation
