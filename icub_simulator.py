"""
Main entry point for iCub robot exploration simulation.
Orchestrates initialization, setup, and execution of random grid exploration.
"""

import time
import pybullet as p  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt 
import numpy as np # pyright: ignore[reportMissingImports]
import cv2 # pyright: ignore[reportMissingImports]

from config.robot_config import IK_PARAMS, ARM_CONTROL_PARAMS
from environment.simulation import (
    initialize_pybullet_gui,
    setup_simulation_physics,
    load_icub_robot,
    setup_camera,
)
from environment.grid import build_grid_state
from environment.visualization import create_visual_sphere
from robot.camera import HAND_LINK_ID, camera_visualization_callback, render_eye_camera, width, height
from robot.model import build_joint_and_link_maps, get_joint_indices, get_link_index
from robot.kinematics import solve_left_arm_ik_dls, get_link_position
from robot.arm_control import move_arm_to_joint_configuration
from agent.exploration import explore_grid_getting_close
from scripts.l_image import get_camera_from_link
from scripts.generate_grid_points import generar_malla_3x3x3
from agent.pomdp import (
    POMDPConfig,
    initialize_dirichlet_parameters,
    likelihoods_from_dirichlet,
    initialize_preferences,
    one_hot_state,
    infer_state,
    select_policy,
    update_dirichlet_parameters,
    idx_config_from_state,
    unflat_state,
    transition_state,
    predict_next_state_distribution,
)

from agent.utils import (
    move_hand_to_grid_state_from_joint_config,
    proprioceptive_observation_from_joint_config,
    retinotopic_observation_from_segmentation,
)

from config.robot_config import (
    LEFT_HAND_LINK_NAME,
    LEFT_ARM_JOINT_NAMES,
    SIMULATION_PARAMS,
)

# ===== POMDP Experiment Parameters =====

POMDP_STEPS = 50
POMDP_HORIZON = 3
POMDP_GAMMA = 1.0
POMDP_EPSILON_RANDOM_ACTION = 0.35

LEARN_WITH_EXECUTED_STATE = False

# Convención:
# Ot = 0 -> no tacto
# Ot = 1 -> tacto
#
# Estos índices son índices de configuración articular / estado plano 0..26.
TOUCH_CONFIG_INDICES = {21, 22, 24, 25}

# Estado inicial conocido. Usamos 8 porque ya confirmaste que existe configuración.
START_CONFIG_IDX = 2

import json
from pathlib import Path


def save_joint_config_by_state(joint_config_by_state, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": metadata or {},
        "joint_config_by_state": {
            str(int(state)): [float(q) for q in q_solution]
            for state, q_solution in joint_config_by_state.items()
        }
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_joint_config_by_state(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    joint_config_by_state = {
        int(state): [float(q) for q in q_solution]
        for state, q_solution in data["joint_config_by_state"].items()
    }

    metadata = data.get("metadata", {})

    return joint_config_by_state, metadata

def step_simulation_steps(n_steps: int = 120, sleep: bool = True):
    for _ in range(n_steps):
        p.stepSimulation()
        if sleep:
            time.sleep(1.0 / 240.0)


def tactile_observation_from_config_idx(config_idx: int) -> int:
    return int(config_idx in TOUCH_CONFIG_INDICES)


def validate_joint_config_table(joint_config_by_state, cfg: POMDPConfig):
    expected = set(range(cfg.n_states))
    found = set(joint_config_by_state.keys())
    missing = sorted(expected - found)

    if missing:
        raise RuntimeError(
            "joint_config_by_state está incompleto. "
            f"Faltan configuraciones para estados: {missing}"
        )

    print(f"[POMDP] joint_config_by_state completo: {cfg.n_states} estados.")


def capture_segmentation_mask_from_eye(
    robot_id: int,
    eye_link_id: int,
    width: int = 320,
    height: int = 240,
    near: float = 0.01,
    far: float = 5.0,
    fov: float = 60.0,
):
    """
    Fallback directo con PyBullet.

    Si tu robot.camera.render_eye_camera ya funciona mejor, puedes reemplazar
    esta función por esa llamada y regresar únicamente segmentation_mask.
    """
    eye_state = p.getLinkState(
        robot_id,
        eye_link_id,
        computeForwardKinematics=True,
    )

    eye_pos = np.asarray(eye_state[4], dtype=float)
    eye_orn = eye_state[5]

    rot = np.asarray(p.getMatrixFromQuaternion(eye_orn), dtype=float).reshape(3, 3)

    # Eje óptico. Si Or sale raro o la mano no aparece, cambia este eje.
    # Alternativas frecuentes:
    #   rot @ [1, 0, 0]
    #   rot @ [0, 0, 1]
    #   rot @ [0, 0, -1]
    forward = rot @ np.array([0.0, 0.0, 1.0])
    up = rot @ np.array([0.0, -1.0, 0.0])

    target = eye_pos + 0.02 * forward

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=eye_pos.tolist(),
        cameraTargetPosition=target.tolist(),
        cameraUpVector=up.tolist(),
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=width / height,
        nearVal=near,
        farVal=far,
    )

    _, _, _, _, segmentation_mask = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
    )

    return np.asarray(segmentation_mask, dtype=np.int64).reshape((height, width))


def observe_pomdp_modalities(
    robot_id: int,
    current_config_idx: int,
    arm_joint_indices,
    joint_config_by_state,
    eye_link_id: int,
    hand_link_id: int,
    cid: int,
):
    """
    Obtiene Oq, Or, Ot para el estado físico actual del robot.
    """
    oq, oq_info = proprioceptive_observation_from_joint_config(
        robot_id=robot_id,
        arm_joint_indices=arm_joint_indices,
        joint_config_by_state=joint_config_by_state,
    )

    if oq is None:
        raise RuntimeError(
            "Oq es None. Esto indica que joint_config_by_state está vacío "
            "o no contiene prototipos articulares válidos. "
            f"Diagnóstico Oq: {oq_info}"
        )

    oq_int = int(oq)

    img, camera_eye, camera_target, view_matrix = render_eye_camera(
        robot_id=robot_id,
        eye_link_id=eye_link_id,
        hand_link_id=hand_link_id,
        width=width,
        height=height,
        cid=cid,
    )

    segmentation_mask = np.asarray(img[4], dtype=np.int64).reshape((height, width))

    or_obs, or_info = retinotopic_observation_from_segmentation(
        segmentation_mask=segmentation_mask,
        robot_id=robot_id,
        hand_link_id=hand_link_id,
        grid_shape=(4, 4),
        image_width=width,
        image_height=height,
    )

    if or_obs is None:
        raise RuntimeError(
            "Or es None. La mano no fue visible en la segmentación. "
            f"camera_eye={camera_eye}, camera_target={camera_target}, "
            f"Diagnóstico Or: {or_info}"
        )

    or_int = int(or_obs)

    ot = tactile_observation_from_config_idx(current_config_idx)

    observations = {
        "q": oq_int,
        "r": or_int,
        "t": int(ot),
    }
    
    if not (0 <= oq_int < 27):
        raise RuntimeError(f"Oq fuera de rango: oq={oq_int}. Esperado: 0..26")

    if not (0 <= or_int < 16):
        raise RuntimeError(f"Or fuera de rango: or_obs={or_int}. Esperado: 0..15")

    info = {
        "oq_info": oq_info,
        "or_info": or_info,
        "camera_eye": camera_eye,
        "camera_target": camera_target,
        "current_config_idx": int(current_config_idx),
    }

    return observations, info


def run_active_inference_pomdp_experiment(
    robot_id: int,
    joint_config_by_state,
    arm_joint_indices,
    eye_link_id: int,
    hand_link_id: int,
):
    print("\n[POMDP EXPERIMENT]")

    cfg = POMDPConfig(
        n_rows=3,
        n_cols=3,
        n_depth=3,
        n_oq=27,
        n_or=16,
        n_ot=2,
    )

    validate_joint_config_table(joint_config_by_state, cfg)

    rng = np.random.default_rng(7)

    a_params = initialize_dirichlet_parameters(
        cfg=cfg,
        concentration=1.0,
        random_scale=1e-3,
        seed=7,
        structured_aq_strength=0.0,
    )

    A = likelihoods_from_dirichlet(a_params)

    C = initialize_preferences(
        cfg=cfg,
        tactile_preference_logit=4.0,
    )

    # Colocar el robot en un estado inicial.
    current_config_idx = START_CONFIG_IDX
    current_hand, current_depth = unflat_state(current_config_idx, cfg)

    print(
        f"[POMDP] Estado inicial: "
        f"idx={current_config_idx}, hand={current_hand}, depth={current_depth}"
    )

    move_hand_to_grid_state_from_joint_config(
        robot_id=robot_id,
        target_state=current_config_idx,
        joint_config_by_state=joint_config_by_state,
        arm_joint_indices=arm_joint_indices,
    )

    step_simulation_steps(120)

    camera_visualization_callback()

    # prior_qs = one_hot_state(current_hand, current_depth, cfg)
    # Una distribución uniforme como prior inicial, sin sesgo hacia el estado real.
    prior_qs = np.ones((cfg.n_hand, cfg.n_depth), dtype=float) / (
        cfg.n_hand * cfg.n_depth
    )

    history = []

    for t in range(POMDP_STEPS):
        print(f"\n[POMDP STEP {t:03d}]")

        # Estado físico observado al inicio del paso.
        observed_config_idx = current_config_idx
        observed_hand = current_hand
        observed_depth = current_depth

        # 1. Observar estado físico actual.
        observations, obs_info = observe_pomdp_modalities(
            robot_id=robot_id,
            current_config_idx=current_config_idx,
            arm_joint_indices=arm_joint_indices,
            joint_config_by_state=joint_config_by_state,
            eye_link_id=eye_link_id,
            hand_link_id=hand_link_id,
            cid=0,
        )

        print(
            f"obs: Oq={observations['q']}, "
            f"Or={observations['r']}, "
            f"Ot={observations['t']}"
        )

        # 2. Inferir estado actual con la A disponible al inicio del paso.
        qs = infer_state(
            observations=observations,
            A=A,
            prior_qs=prior_qs,
        )

        # 3. Aprender A_q, A_r, A_t.
        if LEARN_WITH_EXECUTED_STATE:
            qs_learning = one_hot_state(observed_hand, observed_depth, cfg)
        else:
            qs_learning = qs

        update_dirichlet_parameters(
            a_params=a_params,
            observations=observations,
            qs_for_learning=qs_learning,
            lr=1.0,
        )

        A = likelihoods_from_dirichlet(a_params)

        # 4. Reinferir estado actual usando la A recién actualizada.
        qs = infer_state(
            observations=observations,
            A=A,
            prior_qs=prior_qs,
        )

        map_flat = int(np.argmax(qs))
        map_hand, map_depth = np.unravel_index(map_flat, qs.shape)
        map_config_idx = idx_config_from_state(int(map_hand), int(map_depth), cfg)

        print(
            f"belief MAP: idx={map_config_idx}, "
            f"hand={int(map_hand)}, depth={int(map_depth)}, "
        )

        print(
            f"real/executed: idx={observed_config_idx}, "
            f"hand={observed_hand}, depth={observed_depth}"
        )

        # 5. Seleccionar acción local.
        action, policy_info = select_policy(
            qs=qs,
            A=A,
            C=C,
            cfg=cfg,
            horizon=POMDP_HORIZON,
            gamma=POMDP_GAMMA,
            epsilon_random_action=POMDP_EPSILON_RANDOM_ACTION,
            rng=rng,
        )

        print(
            f"action={action}, mode={policy_info['mode']}, "
            f"policy={policy_info.get('selected_policy')}"
        )

        # 6. Aplicar la acción al proceso generativo real.
        #
        # La acción se aplica desde el estado físico observado al inicio del paso,
        # no desde el MAP.
        target_hand, target_depth = transition_state(
            hand=observed_hand,
            depth=observed_depth,
            action=action,
            cfg=cfg,
            strict=False,
        )

        target_config_idx = idx_config_from_state(target_hand, target_depth, cfg)

        print(
            f"target: idx={target_config_idx}, "
            f"hand={target_hand}, depth={target_depth}"
        )

        move_hand_to_grid_state_from_joint_config(
            robot_id=robot_id,
            target_state=target_config_idx,
            joint_config_by_state=joint_config_by_state,
            arm_joint_indices=arm_joint_indices,
        )

        step_simulation_steps(120)

        # 7. Prior predictivo para el siguiente paso.
        prior_qs = predict_next_state_distribution(qs, action, cfg)

        selected_policy_idx = policy_info.get("selected_policy_idx")
        G_values = policy_info.get("G")
        q_pi = policy_info.get("q_pi")

        if G_values is not None:
            G_values = np.asarray(G_values, dtype=float)
            G_min = float(np.min(G_values))
            G_mean = float(np.mean(G_values))

            if selected_policy_idx is not None:
                G_selected = float(G_values[int(selected_policy_idx)])
            else:
                G_selected = np.nan
        else:
            G_min = np.nan
            G_mean = np.nan
            G_selected = np.nan

        if q_pi is not None:
            q_pi = np.asarray(q_pi, dtype=float)
            policy_entropy = float(-np.sum(q_pi * np.log(q_pi + 1e-16)))
        else:
            policy_entropy = np.nan

        state_entropy = float(-np.sum(qs * np.log(qs + 1e-16)))

        history.append(
            {
                "t": t,
                "observations": observations,
                "observed_config_idx": int(observed_config_idx),
                "observed_hand": int(observed_hand),
                "observed_depth": int(observed_depth),
                "map_config_idx": int(map_config_idx),
                "map_hand": int(map_hand),
                "map_depth": int(map_depth),
                "map_prob": float(np.max(qs)),
                "action": action,
                "target_config_idx": int(target_config_idx),
                "target_hand": int(target_hand),
                "target_depth": int(target_depth),
                "policy_mode": policy_info["mode"],
                "G_min": G_min,
                "G_mean": G_mean,
                "G_selected": G_selected,
                "policy_entropy": policy_entropy,
                "state_entropy": state_entropy,
            }
        )

        # 8. Actualizar estado físico registrado.
        current_hand = target_hand
        current_depth = target_depth
        current_config_idx = target_config_idx

    np.savez(
        "pomdp_learning_run.npz",
        a_q=a_params["q"],
        a_r=a_params["r"],
        a_t=a_params["t"],
        history=np.array(history, dtype=object),
    )

    print("\n[POMDP] Experimento terminado.")
    print("[POMDP] Parámetros guardados en pomdp_learning_run.npz")

    return history, a_params

def run_pomdp_with_loaded_a_params(
    robot_id: int,
    joint_config_by_state,
    arm_joint_indices,
    eye_link_id: int,
    hand_link_id: int,
    a_params,
    n_steps=POMDP_STEPS,
    epsilon_random_action=0.0,
):
    print("\n[POMDP LOADED POLICY RUN]")

    cfg = POMDPConfig(
        n_rows=3,
        n_cols=3,
        n_depth=3,
        n_oq=27,
        n_or=16,
        n_ot=2,
    )

    validate_joint_config_table(joint_config_by_state, cfg)

    # Usar los parámetros aprendidos cargados desde archivo.
    A = likelihoods_from_dirichlet(a_params)

    C = initialize_preferences(
        cfg=cfg,
        tactile_preference_logit=4.0,
    )

    rng = np.random.default_rng(7)

    # Estado inicial.
    current_config_idx = START_CONFIG_IDX
    current_hand, current_depth = unflat_state(current_config_idx, cfg)

    print(
        f"[POMDP LOADED] Estado inicial: "
        f"idx={current_config_idx}, hand={current_hand}, depth={current_depth}"
    )

    move_hand_to_grid_state_from_joint_config(
        robot_id=robot_id,
        target_state=current_config_idx,
        joint_config_by_state=joint_config_by_state,
        arm_joint_indices=arm_joint_indices,
    )

    step_simulation_steps(120)
    camera_visualization_callback()

    prior_qs = one_hot_state(current_hand, current_depth, cfg)

    acting_history = []

    for t in range(n_steps):
        print(f"\n[POMDP LOADED STEP {t:03d}]")

        # 1. Observar estado físico actual.
        observations, obs_info = observe_pomdp_modalities(
            robot_id=robot_id,
            current_config_idx=current_config_idx,
            arm_joint_indices=arm_joint_indices,
            joint_config_by_state=joint_config_by_state,
            eye_link_id=eye_link_id,
            hand_link_id=hand_link_id,
            cid=0,
        )

        print(
            f"obs: Oq={observations['q']}, "
            f"Or={observations['r']}, "
            f"Ot={observations['t']}"
        )

        # 2. Inferir estado con los A aprendidos.
        qs = infer_state(
            observations=observations,
            A=A,
            prior_qs=prior_qs,
        )

        map_flat = int(np.argmax(qs))
        map_hand, map_depth = np.unravel_index(map_flat, qs.shape)
        map_config_idx = idx_config_from_state(int(map_hand), int(map_depth), cfg)

        print(
            f"belief MAP: idx={map_config_idx}, "
            f"hand={int(map_hand)}, depth={int(map_depth)}, "
        )

        print(
            f"real/executed: idx={current_config_idx}, "
            f"hand={current_hand}, depth={current_depth}"
        )

        # 3. Seleccionar acción usando los parámetros cargados.
        # Aquí NO se actualiza a_params.
        action, policy_info = select_policy(
            qs=qs,
            A=A,
            C=C,
            cfg=cfg,
            horizon=POMDP_HORIZON,
            gamma=POMDP_GAMMA,
            epsilon_random_action=epsilon_random_action,
            rng=rng,
        )

        print(
            f"action={action}, mode={policy_info['mode']}, "
            f"policy={policy_info.get('selected_policy')}"
        )

        # 4. Aplicar acción al proceso generativo real.
        target_hand, target_depth = transition_state(
            hand=current_hand,
            depth=current_depth,
            action=action,
            cfg=cfg,
            strict=False,
        )

        target_config_idx = idx_config_from_state(target_hand, target_depth, cfg)

        print(
            f"target: idx={target_config_idx}, "
            f"hand={target_hand}, depth={target_depth}"
        )

        move_hand_to_grid_state_from_joint_config(
            robot_id=robot_id,
            target_state=target_config_idx,
            joint_config_by_state=joint_config_by_state,
            arm_joint_indices=arm_joint_indices,
        )

        step_simulation_steps(120)

        # 5. Prior predictivo para el siguiente paso.
        prior_qs = predict_next_state_distribution(qs, action, cfg)

        # 6. Actualizar estado físico registrado.
        current_hand = target_hand
        current_depth = target_depth
        current_config_idx = target_config_idx

        selected_policy_idx = policy_info.get("selected_policy_idx")
        G_values = policy_info.get("G")
        q_pi = policy_info.get("q_pi")

        if G_values is not None:
            G_values = np.asarray(G_values, dtype=float)
            G_min = float(np.min(G_values))
            G_mean = float(np.mean(G_values))

            if selected_policy_idx is not None:
                G_selected = float(G_values[int(selected_policy_idx)])
            else:
                G_selected = np.nan
        else:
            G_min = np.nan
            G_mean = np.nan
            G_selected = np.nan

        if q_pi is not None:
            q_pi = np.asarray(q_pi, dtype=float)
            policy_entropy = float(-np.sum(q_pi * np.log(q_pi + 1e-16)))
        else:
            policy_entropy = np.nan

        state_entropy = float(-np.sum(qs * np.log(qs + 1e-16)))

        acting_history.append(
            {
                "t": t,
                "observations": observations,
                "current_config_idx": int(obs_info["current_config_idx"]),
                "current_hand": int(current_hand),
                "current_depth": int(current_depth),
                "map_config_idx": int(map_config_idx),
                "map_hand": int(map_hand),
                "map_depth": int(map_depth),
                "map_prob": float(np.max(qs)),
                "action": action,
                "target_config_idx": int(target_config_idx),
                "target_hand": int(target_hand),
                "target_depth": int(target_depth),
                "policy_mode": policy_info["mode"],
                "G_min": G_min,
                "G_mean": G_mean,
                "G_selected": G_selected,
                "policy_entropy": policy_entropy,
                "state_entropy": state_entropy,
            }
        )

    print("\n[POMDP LOADED] Ejecución con parámetros cargados terminada.")

    return acting_history

POMDP_LEARNING_PATH = Path("pomdp_learning_run.npz")

def load_pomdp_learning_run(path=POMDP_LEARNING_PATH):
    path = Path(path)

    data = np.load(path, allow_pickle=True)

    a_params = {
        "q": data["a_q"],
        "r": data["a_r"],
        "t": data["a_t"],
    }

    history = data["history"].tolist()

    return history, a_params

def main():
    """Main execution function."""

    # ===== Initialize PyBullet =====
    print("[INITIALIZATION]")
    physics_client = initialize_pybullet_gui()
    setup_simulation_physics()
    setup_camera()

    # ===== Load Robot =====
    icub_id = load_icub_robot()
    print(icub_id)

    # ===== Setup Grid Visualization =====
    sphere_ids = []
    GRID_POINTS = generar_malla_3x3x3(spacing=0.04)

    for i, point in enumerate(GRID_POINTS):
        sphere_id = create_visual_sphere(point)
        sphere_ids.append(sphere_id)

    grid_state = build_grid_state(GRID_POINTS)

    # ===== Build Robot Joint/Link Maps =====
    joint_name_to_index, link_name_to_index, movable_joint_indices = build_joint_and_link_maps(icub_id)

    left_arm_joint_indices = get_joint_indices(
        joint_name_to_index,
        LEFT_ARM_JOINT_NAMES
    )

    left_hand_link_index = get_link_index(
        link_name_to_index,
        LEFT_HAND_LINK_NAME
    )

    print(f"[ROBOT] left_hand_link_index from map: {left_hand_link_index}")
    print(f"[ROBOT] HAND_LINK_ID used by camera/POMDP: {HAND_LINK_ID}")
    
    JOINT_CONFIG_PATH = "cache/joint_config_by_state.json"

    try:
        joint_config_by_state, metadata = load_joint_config_by_state(JOINT_CONFIG_PATH)
        trajectory = None

        print(f"[CACHE] joint_config_by_state cargado desde {JOINT_CONFIG_PATH}")
        print(f"[CACHE] metadata: {metadata}")

    except FileNotFoundError:
        print("[CACHE] No existe joint_config_by_state. Ejecutando exploración...")

        trajectory, joint_config_by_state = explore_grid_getting_close(
            robot_id=icub_id,
            grid_state=grid_state,
            end_effector_link_index=HAND_LINK_ID,
            arm_joint_indices=left_arm_joint_indices,
            movable_joint_indices=movable_joint_indices,
        )

        save_joint_config_by_state(
            joint_config_by_state,
            JOINT_CONFIG_PATH,
            metadata={
                "hand_link_id": HAND_LINK_ID,
                "n_states": len(grid_state),
                "arm_joint_indices": left_arm_joint_indices,
                "movable_joint_indices": movable_joint_indices,
            }
        )

    print(f"[CACHE] joint_config_by_state guardado en {JOINT_CONFIG_PATH}")
    print(f"[EXPLORATION] joint configs: {len(joint_config_by_state)}")

    # ===== Active Inference POMDP Phase =====
    #
    # Si tu ojo sintético no es el link 99, cambia este valor.
    # En tu camera.py anterior usabas EYE_LINK_ID = 99.
    EYE_LINK_ID = 99
    
    POMDP_LEARNING_PATH = Path("pomdp_learning_run.npz")

    if POMDP_LEARNING_PATH.exists():
        print(f"[CACHE] Cargando aprendizaje POMDP desde {POMDP_LEARNING_PATH}")

        learning_history, a_params = load_pomdp_learning_run(POMDP_LEARNING_PATH)

        print("[CACHE] Parámetros cargados:")
        print("A_q params shape:", a_params["q"].shape)
        print("A_r params shape:", a_params["r"].shape)
        print("A_t params shape:", a_params["t"].shape)
        print("learning history length:", len(learning_history))

        # El iCub actúa usando los parámetros aprendidos.
        history = run_pomdp_with_loaded_a_params(
            robot_id=icub_id,
            joint_config_by_state=joint_config_by_state,
            arm_joint_indices=left_arm_joint_indices,
            eye_link_id=EYE_LINK_ID,
            hand_link_id=HAND_LINK_ID,
            a_params=a_params,
            n_steps=POMDP_STEPS,
            epsilon_random_action=0.0,
        )

    else:
        print("[CACHE] No existe aprendizaje previo. Ejecutando experimento POMDP...")

        history, a_params = run_active_inference_pomdp_experiment(
            robot_id=icub_id,
            joint_config_by_state=joint_config_by_state,
            arm_joint_indices=left_arm_joint_indices,
            eye_link_id=EYE_LINK_ID,
            hand_link_id=HAND_LINK_ID,
        )

    # ===== Keep Simulation Running =====
    print("\n[SIMULATION RUNNING - Press Ctrl+C to exit]")
    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    except KeyboardInterrupt:
        print("\nSimulación cerrada.")
        p.disconnect()


if __name__ == "__main__":
    main()
