import numpy as np # pyright: ignore[reportMissingImports]
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence, Union

EPS = 1e-16

ACTION_STAY = "stay"
ACTION_UP = "up"
ACTION_DOWN = "down"
ACTION_LEFT = "left"
ACTION_RIGHT = "right"
ACTION_CLOSER = "closer"
ACTION_FARTHER = "farther"

LOCAL_ACTIONS = (
    ACTION_STAY,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_CLOSER,
    ACTION_FARTHER,
)

Axis = Optional[Union[int, Tuple[int, ...]]]


@dataclass(frozen=True)
class POMDPConfig:
    n_rows: int = 3
    n_cols: int = 3
    n_depth: int = 3
    n_oq: int = 27
    n_or: int = 16
    n_ot: int = 2

    @property
    def n_hand(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def n_states(self) -> int:
        return self.n_hand * self.n_depth


def softmax(x: np.ndarray, axis: Axis = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def normalize_prob(x: np.ndarray, axis: Axis = 0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    total = np.sum(x, axis=axis, keepdims=True)
    return x / np.maximum(total, EPS)


def flat_state(hand: int, depth: int, cfg: POMDPConfig) -> int:
    return depth * cfg.n_hand + hand


def unflat_state(idx: int, cfg: POMDPConfig) -> Tuple[int, int]:
    depth = idx // cfg.n_hand
    hand = idx % cfg.n_hand
    return hand, depth


def hand_to_row_col(hand: int, cfg: POMDPConfig) -> Tuple[int, int]:
    return hand // cfg.n_cols, hand % cfg.n_cols


def row_col_to_hand(row: int, col: int, cfg: POMDPConfig) -> int:
    return row * cfg.n_cols + col


def one_hot_state(hand: int, depth: int, cfg: POMDPConfig) -> np.ndarray:
    qs = np.zeros((cfg.n_hand, cfg.n_depth), dtype=float)
    qs[hand, depth] = 1.0
    return qs


def uniform_state_prior(cfg: POMDPConfig) -> np.ndarray:
    return np.ones((cfg.n_hand, cfg.n_depth), dtype=float) / cfg.n_states


def valid_actions(hand: int, depth: int, cfg: POMDPConfig) -> List[str]:
    row, col = hand_to_row_col(hand, cfg)

    actions = [ACTION_STAY]

    if row > 0:
        actions.append(ACTION_UP)
    if row < cfg.n_rows - 1:
        actions.append(ACTION_DOWN)
    if col > 0:
        actions.append(ACTION_LEFT)
    if col < cfg.n_cols - 1:
        actions.append(ACTION_RIGHT)

    if depth < cfg.n_depth - 1:
        actions.append(ACTION_CLOSER)
    if depth > 0:
        actions.append(ACTION_FARTHER)

    return actions


def transition_state(
    hand: int,
    depth: int,
    action: str,
    cfg: POMDPConfig,
    strict: bool = False,
) -> Tuple[int, int]:
    row, col = hand_to_row_col(hand, cfg)
    new_row, new_col, new_depth = row, col, depth

    if action == ACTION_STAY:
        pass
    elif action == ACTION_UP and row > 0:
        new_row -= 1
    elif action == ACTION_DOWN and row < cfg.n_rows - 1:
        new_row += 1
    elif action == ACTION_LEFT and col > 0:
        new_col -= 1
    elif action == ACTION_RIGHT and col < cfg.n_cols - 1:
        new_col += 1
    elif action == ACTION_CLOSER and depth < cfg.n_depth - 1:
        new_depth += 1
    elif action == ACTION_FARTHER and depth > 0:
        new_depth -= 1
    else:
        if strict:
            raise ValueError(
                f"Acción inválida: action={action}, hand={hand}, depth={depth}"
            )
        return hand, depth

    return row_col_to_hand(new_row, new_col, cfg), new_depth


def predict_next_state_distribution(
    qs: np.ndarray,
    action: str,
    cfg: POMDPConfig,
) -> np.ndarray:
    qs_next = np.zeros_like(qs, dtype=float)

    for hand in range(cfg.n_hand):
        for depth in range(cfg.n_depth):
            prob = qs[hand, depth]
            if prob <= 0:
                continue

            hand_next, depth_next = transition_state(
                hand=hand,
                depth=depth,
                action=action,
                cfg=cfg,
                strict=False,
            )
            qs_next[hand_next, depth_next] += prob

    return normalize_prob(qs_next, axis=None)


def generate_policies_from_state(
    hand: int,
    depth: int,
    cfg: POMDPConfig,
    horizon: int = 2,
) -> List[Tuple[str, ...]]:
    if horizon <= 0:
        return [tuple()]

    policies = []

    for action in valid_actions(hand, depth, cfg):
        h2, d2 = transition_state(hand, depth, action, cfg, strict=True)
        tails = generate_policies_from_state(h2, d2, cfg, horizon - 1)

        for tail in tails:
            policies.append((action,) + tail)

    return policies


def initialize_dirichlet_parameters(
    cfg: POMDPConfig,
    concentration: float = 1.0,
    random_scale: float = 1e-3,
    seed: Optional[int] = None,
    structured_aq_strength: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Initializes Dirichlet concentration parameters.

    structured_aq_strength = 0.0 means A_q starts uninformative.
    A small positive value, e.g. 1.0 or 2.0, gives A_q a weak prior
    that Oq roughly corresponds to the flat state index.
    """
    rng = np.random.default_rng(seed)

    a_q = concentration * np.ones((cfg.n_oq, cfg.n_hand, cfg.n_depth), dtype=float)
    a_r = concentration * np.ones((cfg.n_or, cfg.n_hand, cfg.n_depth), dtype=float)
    a_t = concentration * np.ones((cfg.n_ot, cfg.n_hand, cfg.n_depth), dtype=float)

    if random_scale > 0:
        a_q += random_scale * rng.random(a_q.shape)
        a_r += random_scale * rng.random(a_r.shape)
        a_t += random_scale * rng.random(a_t.shape)

    if structured_aq_strength > 0:
        for hand in range(cfg.n_hand):
            for depth in range(cfg.n_depth):
                oq = flat_state(hand, depth, cfg)
                a_q[oq, hand, depth] += structured_aq_strength

    return {"q": a_q, "r": a_r, "t": a_t}


def likelihoods_from_dirichlet(
    a_params: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    return {
        "q": normalize_prob(a_params["q"], axis=0),
        "r": normalize_prob(a_params["r"], axis=0),
        "t": normalize_prob(a_params["t"], axis=0),
    }


def observation_entropy_per_state(A: np.ndarray) -> np.ndarray:
    """
    A shape: (n_observations, n_hand, n_depth)
    Returns H[P(o|s)] with shape (n_hand, n_depth).
    """
    return -np.sum(A * np.log(A + EPS), axis=0)


def predict_observation_distribution(A: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """
    A shape:  (n_observations, n_hand, n_depth)
    qs shape: (n_hand, n_depth)
    """
    qo = np.sum(A * qs[None, :, :], axis=(1, 2))
    return normalize_prob(qo, axis=None)


def infer_state(
    observations: Dict[str, int],
    A: Dict[str, np.ndarray],
    prior_qs: np.ndarray,
) -> np.ndarray:
    """
    observations:
        {
            "q": oq,
            "r": or_obs,
            "t": ot
        }

    Returns:
        qs posterior with shape (n_hand, n_depth)
    """
    logp = np.log(prior_qs + EPS)

    oq = observations.get("q")
    if oq is not None:
        logp += np.log(A["q"][oq, :, :] + EPS)

    or_obs = observations.get("r")
    if or_obs is not None:
        logp += np.log(A["r"][or_obs, :, :] + EPS)

    ot = observations.get("t")
    if ot is not None:
        logp += np.log(A["t"][ot, :, :] + EPS)

    return softmax(logp, axis=None)


def initialize_preferences(
    cfg: POMDPConfig,
    tactile_preference_logit: float = 4.0,
) -> Dict[str, np.ndarray]:
    """
    C_q and C_r are uniform and should usually have zero planning weight.
    C_t prefers touch: Ot=1.
    """
    C_q = np.ones(cfg.n_oq, dtype=float) / cfg.n_oq
    C_r = np.ones(cfg.n_or, dtype=float) / cfg.n_or

    # Ot index convention:
    # 0 = no touch
    # 1 = touch
    C_t = softmax(np.array([0.0, tactile_preference_logit]), axis=None)

    return {"q": C_q, "r": C_r, "t": C_t}


def kl_divergence(q: np.ndarray, p: np.ndarray) -> float:
    q = normalize_prob(q, axis=None)
    p = normalize_prob(p, axis=None)
    return float(np.sum(q * (np.log(q + EPS) - np.log(p + EPS))))


def expected_free_energy(
    policy: Sequence[str],
    qs: np.ndarray,
    A: Dict[str, np.ndarray],
    C: Dict[str, np.ndarray],
    cfg: POMDPConfig,
    risk_weights: Optional[Dict[str, float]] = None,
    ambiguity_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Computes G(pi) = risk + ambiguity over the policy horizon.

    Default: only tactile modality affects planning.
    q and r are used for inference, but not for preferences.
    """
    if risk_weights is None:
        risk_weights = {"q": 0.0, "r": 0.0, "t": 1.0}

    if ambiguity_weights is None:
        ambiguity_weights = {"q": 0.0, "r": 0.0, "t": 1.0}

    qs_tau = np.array(qs, dtype=float)
    G = 0.0

    for action in policy:
        qs_tau = predict_next_state_distribution(qs_tau, action, cfg)

        for modality in ("q", "r", "t"):
            A_m = A[modality]
            qo = predict_observation_distribution(A_m, qs_tau)

            risk = kl_divergence(qo, C[modality])
            ambiguity = float(
                np.sum(qs_tau * observation_entropy_per_state(A_m))
            )

            G += risk_weights.get(modality, 0.0) * risk
            G += ambiguity_weights.get(modality, 0.0) * ambiguity

    return float(G)


def select_policy(
    qs: np.ndarray,
    A: Dict[str, np.ndarray],
    C: Dict[str, np.ndarray],
    cfg: POMDPConfig,
    horizon: int = 2,
    gamma: float = 8.0,
    epsilon_random_action: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[str, Dict]:
    """
    Selects first action of the most probable policy.

    Uses valid local policies from the MAP state.
    """
    rng_local = rng if rng is not None else np.random.default_rng()

    map_flat = int(np.argmax(qs))
    map_hand, map_depth = np.unravel_index(map_flat, qs.shape)

    first_valid_actions = valid_actions(
        hand=int(map_hand),
        depth=int(map_depth),
        cfg=cfg,
    )

    if rng_local.random() < epsilon_random_action:
        action = str(rng_local.choice(first_valid_actions))
        return action, {
            "mode": "epsilon_random",
            "map_hand": int(map_hand),
            "map_depth": int(map_depth),
            "valid_actions": first_valid_actions,
        }

    policies = generate_policies_from_state(
        hand=int(map_hand),
        depth=int(map_depth),
        cfg=cfg,
        horizon=horizon,
    )

    G = np.array(
        [expected_free_energy(policy, qs, A, C, cfg) for policy in policies],
        dtype=float,
    )

    q_pi = softmax(-gamma * G, axis=None)

    policy_idx = int(rng_local.choice(len(policies), p=q_pi))
    selected_policy = policies[policy_idx]

    return selected_policy[0], {
        "mode": "active_inference",
        "map_hand": int(map_hand),
        "map_depth": int(map_depth),
        "policies": policies,
        "G": G,
        "q_pi": q_pi,
        "selected_policy": selected_policy,
        "selected_policy_idx": policy_idx,
    }


def update_dirichlet_parameters(
    a_params: Dict[str, np.ndarray],
    observations: Dict[str, int],
    qs_for_learning: np.ndarray,
    lr: float = 1.0,
) -> None:
    """
    Dirichlet learning:

        a[o, s_hand, s_depth] += Q(s_hand, s_depth)

    Use qs_for_learning = posterior qs for unsupervised learning.
    During early debugging, qs_for_learning can be one_hot_state(target_hand, target_depth).
    """
    oq = observations.get("q")
    if oq is not None:
        a_params["q"][oq, :, :] += lr * qs_for_learning

    or_obs = observations.get("r")
    if or_obs is not None:
        a_params["r"][or_obs, :, :] += lr * qs_for_learning

    ot = observations.get("t")
    if ot is not None:
        a_params["t"][ot, :, :] += lr * qs_for_learning


def state_from_action(
    qs: np.ndarray,
    action: str,
    cfg: POMDPConfig,
) -> Tuple[int, int]:
    """
    Deterministic target state from MAP state and selected action.
    Used to send the command to PyBullet.
    """
    map_flat = int(np.argmax(qs))
    hand, depth = np.unravel_index(map_flat, qs.shape)
    return transition_state(hand, depth, action, cfg, strict=True)


def idx_config_from_state(hand: int, depth: int, cfg: POMDPConfig) -> int:
    """
    Default mapping from factorized state to stored joint-configuration index.
    Change this function if your joint_config_by_state uses a different order.
    """
    return flat_state(hand, depth, cfg)