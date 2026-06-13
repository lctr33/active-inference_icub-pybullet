import pybullet as p # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt

def get_camera_from_link(
    robot_id,
    link_idx,
    width=320,
    height=240,
    fov=60,
    near=0.01,
    far=5.0
):
    # 1. Pose global del link de la cámara
    link_state = p.getLinkState(robot_id, link_idx, computeForwardKinematics=True)
    cam_pos = np.array(link_state[4])   # worldLinkFramePosition
    cam_orn = link_state[5]             # worldLinkFrameOrientation quaternion

    # 2. Matriz de rotación del link
    rot = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)

    # Convención típica:
    # eje X local = forward, eje Z local = up
    # Esto puede variar según el URDF del iCub.
    forward = rot @ np.array([1, 0, 0])
    up = rot @ np.array([0, 0, 1])

    target = cam_pos + 1.0 * forward

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_pos.tolist(),
        cameraTargetPosition=target.tolist(),
        cameraUpVector=up.tolist()
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=width / height,
        nearVal=near,
        farVal=far
    )

    img = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
        # En DIRECT puedes usar: renderer=p.ER_TINY_RENDERER
    )

    rgb = np.reshape(img[2], (height, width, 4))[:, :, :3]
    depth = np.reshape(img[3], (height, width))
    seg = np.reshape(img[4], (height, width))

    return rgb, depth, seg