"""
Results analysis for the iCub active inference POMDP experiment.

Reads:
    pomdp_learning_run.npz

Produces:
    results/observations_and_states.png
    results/free_energy_metrics.png
    results/At_touch_likelihood.png
    results/Ar_dominant_retinotopic.png
    results/Aq_likelihood.png
    results/summary.txt
"""

from pathlib import Path
from typing import Any

import numpy as np # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt

from agent.pomdp import (
    POMDPConfig,
    likelihoods_from_dirichlet,
    unflat_state,
    normalize_prob,
    expected_free_energy,
    generate_policies_from_state,
)


RESULTS_DIR = Path("/home/lctr/SEMESTRES/SEMESTRE_6/COGNITIVE_ROBOTICS/Act-inf-iCub/results")
DEFAULT_RUN_FILE = Path("/home/lctr/SEMESTRES/SEMESTRE_6/COGNITIVE_ROBOTICS/Act-inf-iCub/pomdp_learning_run.npz")

TOUCH_STATES = {21, 22, 24, 25}


def load_run(path: Path = DEFAULT_RUN_FILE):
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    data = np.load(path, allow_pickle=True)

    a_params = {
        "q": data["a_q"],
        "r": data["a_r"],
        "t": data["a_t"],
    }

    history = list(data["history"])

    return a_params, history


def get_history_value(history: list[dict[str, Any]], key: str, default=np.nan):
    values = []
    for item in history:
        if key in item:
            values.append(item[key])
        else:
            values.append(default)
    return np.asarray(values)


def extract_observations(history: list[dict[str, Any]]):
    oq, or_obs, ot = [], [], []

    for item in history:
        obs = item.get("observations", {})
        oq.append(obs.get("q", np.nan))
        or_obs.append(obs.get("r", np.nan))
        ot.append(obs.get("t", np.nan))

    return np.asarray(oq), np.asarray(or_obs), np.asarray(ot)


def entropy(p: np.ndarray, axis=None):
    p = normalize_prob(np.asarray(p, dtype=float), axis=axis)
    return -np.sum(p * np.log(p + 1e-16), axis=axis)


def reconstruct_G_if_missing(
    history: list[dict[str, Any]],
    a_params: dict[str, np.ndarray],
    cfg: POMDPConfig,
):
    """
    Approximate fallback.

    If G was not stored during the run, this recomputes G using the final learned A.
    This is not the true online evolution of G, but it helps inspect the learned model.
    """
    A = likelihoods_from_dirichlet(a_params)

    C_t = np.exp(np.array([0.0, 4.0]))
    C_t = C_t / C_t.sum()

    C = {
        "q": np.ones(cfg.n_oq) / cfg.n_oq,
        "r": np.ones(cfg.n_or) / cfg.n_or,
        "t": C_t,
    }

    G_min = []
    G_selected = []

    for item in history:
        idx = int(item.get("current_config_idx", item.get("target_config_idx", 0)))
        hand, depth = unflat_state(idx, cfg)

        qs = np.zeros((cfg.n_hand, cfg.n_depth), dtype=float)
        qs[hand, depth] = 1.0

        policies = generate_policies_from_state(
            hand=hand,
            depth=depth,
            cfg=cfg,
            horizon=3,
        )

        G = np.asarray([
            expected_free_energy(policy, qs, A, C, cfg)
            for policy in policies
        ])

        G_min.append(float(np.min(G)))

        selected_action = item.get("action")
        selected_candidates = [
            i for i, policy in enumerate(policies)
            if len(policy) > 0 and policy[0] == selected_action
        ]

        if selected_candidates:
            G_selected.append(float(np.min(G[selected_candidates])))
        else:
            G_selected.append(np.nan)

    return np.asarray(G_min), np.asarray(G_selected)


def plot_observations_and_states(history: list[dict[str, Any]], out_path: Path):
    t = np.arange(len(history))

    oq, or_obs, ot = extract_observations(history)

    current_idx = get_history_value(history, "current_config_idx")
    target_idx = get_history_value(history, "target_config_idx")
    map_idx = get_history_value(history, "map_config_idx")
    map_prob = get_history_value(history, "map_prob")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, oq, marker="o", label="Oq")
    axes[0].plot(t, or_obs, marker="o", label="Or")
    axes[0].set_ylabel("Observación")
    axes[0].set_title("Observaciones propioceptiva y retinotópica")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].step(t, ot, where="post", label="Ot")
    axes[1].set_ylabel("Ot")
    axes[1].set_yticks([0, 1])
    axes[1].set_title("Observación táctil")
    axes[1].grid(True)

    axes[2].plot(t, current_idx, marker="o", label="estado actual")
    axes[2].plot(t, target_idx, marker="x", label="estado objetivo")
    axes[2].plot(t, map_idx, linestyle="--", label="MAP belief")
    axes[2].set_ylabel("Estado idx")
    axes[2].set_title("Estados visitados e inferidos")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(t, map_prob, marker="o")
    axes[3].set_ylim(0.0, 1.05)
    axes[3].ticklabel_format(axis="y", useOffset=False)
    axes[3].set_ylabel("P(MAP)")
    axes[3].set_xlabel("Tiempo")
    axes[3].set_title("Confianza en el estado MAP")
    axes[3].grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_free_energy(history: list[dict[str, Any]], a_params, cfg, out_path: Path):
    t = np.arange(len(history))

    G_min = get_history_value(history, "G_min")
    G_selected = get_history_value(history, "G_selected")
    G_mean = get_history_value(history, "G_mean")
    policy_entropy = get_history_value(history, "policy_entropy")
    state_entropy = get_history_value(history, "state_entropy")

    missing_G = np.all(np.isnan(G_min))

    if missing_G:
        G_min, G_selected = reconstruct_G_if_missing(history, a_params, cfg)
        G_mean = np.full_like(G_min, np.nan)
        title_suffix = "reconstruida con A final"
    else:
        title_suffix = "online"

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t, G_min, marker="o", label="G mínima")
    axes[0].plot(t, G_selected, marker="o", label="G política seleccionada")
    if not np.all(np.isnan(G_mean)):
        axes[0].plot(t, G_mean, linestyle="--", label="G promedio")
    axes[0].set_ylabel("G")
    axes[0].set_title(f"Energía libre esperada por tiempo ({title_suffix})")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, policy_entropy, marker="o")
    axes[1].set_ylabel("H[Q(pi)]")
    axes[1].set_title("Entropía posterior de políticas")
    axes[1].grid(True)

    axes[2].plot(t, state_entropy, marker="o")
    axes[2].set_ylabel("H[Q(s)]")
    axes[2].set_xlabel("Tiempo")
    axes[2].set_title("Entropía posterior de estados")
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_At_touch_likelihood(A_t: np.ndarray, cfg: POMDPConfig, out_path: Path):
    """
    A_t shape: (2, n_hand, n_depth)
    We plot P(Ot=1 | state).
    """
    p_touch = A_t[1, :, :]  # hand x depth

    fig, axes = plt.subplots(1, cfg.n_depth, figsize=(13, 4))

    if cfg.n_depth == 1:
        axes = [axes]

    for depth in range(cfg.n_depth):
        mat = p_touch[:, depth].reshape(cfg.n_rows, cfg.n_cols)

        im = axes[depth].imshow(mat, vmin=0.0, vmax=1.0)
        axes[depth].set_title(f"P(Ot=1 | depth={depth})")
        axes[depth].set_xlabel("col")
        axes[depth].set_ylabel("row")

        for r in range(cfg.n_rows):
            for c in range(cfg.n_cols):
                hand = r * cfg.n_cols + c
                idx = depth * cfg.n_hand + hand
                axes[depth].text(
                    c, r,
                    f"{idx}\n{mat[r, c]:.2f}",
                    ha="center",
                    va="center",
                )

    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.suptitle("Matriz táctil aprendida A_t")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_Ar_dominant(A_r: np.ndarray, cfg: POMDPConfig, out_path: Path):
    """
    A_r shape: (16, n_hand, n_depth)
    Plot dominant retinotopic observation argmax Or for each state.
    """
    dominant_or = np.argmax(A_r, axis=0)  # hand x depth

    fig, axes = plt.subplots(1, cfg.n_depth, figsize=(13, 4))

    if cfg.n_depth == 1:
        axes = [axes]

    for depth in range(cfg.n_depth):
        mat = dominant_or[:, depth].reshape(cfg.n_rows, cfg.n_cols)

        im = axes[depth].imshow(mat, vmin=0, vmax=15)
        axes[depth].set_title(f"Or dominante | depth={depth}")
        axes[depth].set_xlabel("col")
        axes[depth].set_ylabel("row")

        for r in range(cfg.n_rows):
            for c in range(cfg.n_cols):
                hand = r * cfg.n_cols + c
                idx = depth * cfg.n_hand + hand
                axes[depth].text(
                    c, r,
                    f"{idx}\nOr={int(mat[r, c])}",
                    ha="center",
                    va="center",
                )

    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.suptitle("Mapa retinotópico aprendido A_r")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_Aq_likelihood(A_q: np.ndarray, cfg: POMDPConfig, out_path: Path):
    """
    A_q shape: (27, n_hand, n_depth)
    Flatten states into 27 columns.
    """
    Aq_flat = A_q.reshape(cfg.n_oq, cfg.n_states)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(Aq_flat, aspect="auto", vmin=0.0, vmax=np.max(Aq_flat))

    ax.set_title("Matriz propioceptiva aprendida A_q = P(Oq | estado)")
    ax.set_xlabel("Estado oculto plano idx")
    ax.set_ylabel("Oq")

    ax.set_xticks(np.arange(cfg.n_states))
    ax.set_yticks(np.arange(cfg.n_oq))
    ax.tick_params(axis="both", labelsize=6)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_action_sequence(history: list[dict[str, Any]], out_path: Path):
    actions = [item.get("action", "?") for item in history]
    unique_actions = sorted(set(actions))
    action_to_id = {a: i for i, a in enumerate(unique_actions)}
    y = np.asarray([action_to_id[a] for a in actions])
    t = np.arange(len(actions))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(t, y, where="post")
    ax.set_yticks(list(action_to_id.values()))
    ax.set_yticklabels(list(action_to_id.keys()))
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Acción")
    ax.set_title("Acción realizada")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_summary(history: list[dict[str, Any]], A, out_path: Path):
    oq, or_obs, ot = extract_observations(history)

    visited = get_history_value(history, "current_config_idx")
    visited = visited[~np.isnan(visited)].astype(int)

    touch_rate = float(np.mean(ot == 1)) if len(ot) else 0.0
    unique_states = sorted(set(visited.tolist()))

    p_touch = A["t"][1, :, :]
    learned_touch_flat = p_touch.reshape(-1)

    top_touch_states = np.argsort(learned_touch_flat)[::-1][:8]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Resumen del experimento POMDP iCub\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Pasos: {len(history)}\n")
        f.write(f"Estados visitados únicos: {len(unique_states)} / 27\n")
        f.write(f"Estados visitados: {unique_states}\n")
        f.write(f"Proporción de pasos con Ot=1: {touch_rate:.3f}\n\n")

        f.write("Estados con mayor P(Ot=1 | estado):\n")
        for idx in top_touch_states:
            hand = idx % 9
            depth = idx // 9
            f.write(
                f"  idx={idx:02d}, hand={hand}, depth={depth}, "
                f"P_touch={learned_touch_flat[idx]:.3f}\n"
            )

        f.write("\nObservaciones retinotópicas únicas:\n")
        f.write(f"  {sorted(set(or_obs.astype(int).tolist()))}\n")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    cfg = POMDPConfig(
        n_rows=3,
        n_cols=3,
        n_depth=3,
        n_oq=27,
        n_or=16,
        n_ot=2,
    )

    a_params, history = load_run(DEFAULT_RUN_FILE)
    A = likelihoods_from_dirichlet(a_params)

    plot_observations_and_states(
        history,
        RESULTS_DIR / "observations_and_states.png",
    )

    plot_free_energy(
        history,
        a_params,
        cfg,
        RESULTS_DIR / "free_energy_metrics.png",
    )

    plot_At_touch_likelihood(
        A["t"],
        cfg,
        RESULTS_DIR / "At_touch_likelihood.png",
    )

    plot_Ar_dominant(
        A["r"],
        cfg,
        RESULTS_DIR / "Ar_dominant_retinotopic.png",
    )

    plot_Aq_likelihood(
        A["q"],
        cfg,
        RESULTS_DIR / "Aq_likelihood.png",
    )

    plot_action_sequence(
        history,
        RESULTS_DIR / "actions.png",
    )

    write_summary(
        history,
        A,
        RESULTS_DIR / "summary.txt",
    )

    print("[RESULTS] Figuras guardadas en ./results")
    print("[RESULTS] Resumen guardado en ./results/summary.txt")


if __name__ == "__main__":
    main()