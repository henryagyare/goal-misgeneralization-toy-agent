# analysis/trajectory_overlay.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

@torch.no_grad()
def plot_pred_vs_true_trajectory(
    model,
    device,
    history,
    state_index=0,
    state_name="x",
    title=None,
    save_path=None,
    show=False,
):
    """
    Overlay predicted vs true next-state trajectory for a single dimension.

    state_index:
        0 = x
        1 = v
        2 = m
        3 = integrity
    """

    T = len(history)

    # Build true states and actions
    s = np.zeros((T, 4), dtype=np.float32)
    a = np.zeros((T, 1), dtype=np.float32)

    for i, h in enumerate(history):
        s[i, 0] = h["x"]
        s[i, 1] = h["v"]
        s[i, 2] = h["params"]["mass"]
        s[i, 3] = h["integrity"]
        a[i, 0] = h["action"]

    s_t = torch.tensor(s[:-1], dtype=torch.float32, device=device)
    a_t = torch.tensor(a[:-1], dtype=torch.float32, device=device)
    s_tp1_true = s[1:, state_index]

    # Model prediction
    s_tp1_pred = model(s_t, a_t)[:, state_index].cpu().numpy()

    timesteps = np.arange(len(s_tp1_true))

    # Plot
    fig = plt.figure()
    plt.plot(timesteps, s_tp1_true, label="True", linewidth=2)
    plt.plot(timesteps, s_tp1_pred, "--", label="Predicted", linewidth=2)

    plt.xlabel("Timestep")
    plt.ylabel(state_name)
    plt.title(title or f"Predicted vs True {state_name} Trajectory")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        _ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
