import numpy as np # pyright: ignore[reportMissingImports]
import pybullet as p # pyright: ignore[reportMissingImports]
import cv2 # pyright: ignore[reportMissingImports]

width, height = 320, 240
near, far = 0.01, 5.0

EYE_LINK_ID = 99
HAND_LINK_ID = 67

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("Vector with near-zero norm")
    return v / n

def render_eye_camera(robot_id, eye_link_id, hand_link_id, width, height, cid):
    eye_state = p.getLinkState(
    robot_id,
    eye_link_id,
    computeForwardKinematics=True,
    physicsClientId=cid
)

    eye_pos = np.array(eye_state[4])
    eye_orn = eye_state[5]

    R = np.array(p.getMatrixFromQuaternion(eye_orn)).reshape(3, 3)

    local_forward = np.array([0.0, 0.0, 1.0])
    local_up = np.array([0.0, -1.0, 0.0])

    direction = R @ local_forward
    direction = direction / np.linalg.norm(direction)

    up = R @ local_up
    up = up / np.linalg.norm(up)

    camera_eye = eye_pos + 0.02 * direction
    camera_target = camera_eye + 1.0 * direction

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_eye.tolist(),
        cameraTargetPosition=camera_target.tolist(),
        cameraUpVector=up.tolist()
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.01,
        farVal=5.0
    )

    p.addUserDebugLine(
        camera_eye.tolist(),
        camera_target.tolist(),
        [1, 0, 0],
        lineWidth=3,
        lifeTime=0.05,
        physicsClientId=cid
    )

    img = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        physicsClientId=cid
    )

    return img, camera_eye, camera_target, view_matrix
      
def decode_segmentation(seg_value):
    """
    Decodifica valores de PyBullet con:
    ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    """
    if seg_value == -1:
        return -1, -1

    object_id = seg_value & ((1 << 24) - 1)
    link_id = (seg_value >> 24) - 1
    return object_id, link_id


def colorize_segmentation(seg):
    """
    Convierte la máscara de segmentación en una imagen RGB visualizable.
    """
    h, w = seg.shape
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    unique_ids = np.unique(seg)

    for value in unique_ids:
        if value == -1:
            # Fondo negro
            color = np.array([0, 0, 0], dtype=np.uint8)
        else:
            # Color pseudoaleatorio determinista por id
            rng = np.random.default_rng(abs(int(value)) % (2**32))
            color = rng.integers(50, 255, size=3, dtype=np.uint8)

        seg_rgb[seg == value] = color

    return seg_rgb


def visualize_camera_buffers(img, width, height, near=0.01, far=5.0):
    rgb_rgba = np.asarray(img[2], dtype=np.uint8).reshape(height, width, 4)
    rgb = rgb_rgba[:, :, :3]
    depth_buffer = np.reshape(img[3], (height, width))
    seg = np.reshape(img[4], (height, width))

    # RGB: PyBullet entrega RGB, OpenCV usa BGR
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Depth buffer -> profundidad métrica aproximada
    depth_m = far * near / (far - (far - near) * depth_buffer)

    # Para visualizar, ignorar fondo donde depth_buffer == 1
    valid = depth_buffer < 0.999

    depth_vis = np.zeros_like(depth_buffer, dtype=np.uint8)
    if np.any(valid):
        d_valid = depth_m[valid]
        d_min, d_max = d_valid.min(), d_valid.max()

        depth_norm = (depth_m - d_min) / (d_max - d_min + 1e-9)
        depth_norm = np.clip(depth_norm, 0, 1)

        depth_vis = (255 * (1.0 - depth_norm)).astype(np.uint8)
        depth_vis[~valid] = 0

    # Segmentación coloreada
    seg_rgb = colorize_segmentation(seg)
    seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)

    cv2.imshow("RGB camera", rgb_bgr)
    cv2.imshow("Depth camera", depth_vis)
    cv2.imshow("Segmentation camera", seg_bgr)

    cv2.waitKey(1)

    return rgb, depth_m, seg

def camera_visualization_callback():
    img, camera_eye, camera_target, view_matrix = render_eye_camera(
        robot_id=0,
        eye_link_id=EYE_LINK_ID,
        hand_link_id=HAND_LINK_ID,
        width=width,
        height=height,
        cid=0
    )

    rgb, depth_m, seg = visualize_camera_buffers(
        img,
        width=width,
        height=height,
        near=near,
        far=far
    )