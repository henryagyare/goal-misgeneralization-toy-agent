# analysis/model_error.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

STATE_KEYS = ["x", "v", "m", "integrity"]

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def history_to_transitions(history):
    T = len(history)
    s = np.zeros((T, 4), dtype=np.float32)
    a = np.zeros((T, 1), dtype=np.float32)

    for i, h in enumerate(history):
        s[i, 0] = h["x"]
        s[i, 1] = h["v"]
        s[i, 2] = h["params"]["mass"]
        s[i, 3] = h["integrity"]
        a[i, 0] = h["action"]

    return s[:-1], a[:-1], s[1:]


@torch.no_grad()
def compute_model_errors(model, device, history):
    s_t, a_t, s_tp1 = history_to_transitions(history)

    s_t_t = torch.tensor(s_t, dtype=torch.float32, device=device)
    a_t_t = torch.tensor(a_t, dtype=torch.float32, device=device)
    s_tp1_t = torch.tensor(s_tp1, dtype=torch.float32, device=device)

    pred = model(s_t_t, a_t_t)
    err = pred - s_tp1_t

    per_step_mse = (err ** 2).mean(dim=1).detach().cpu().numpy()
    per_dim_mse = (err ** 2).mean(dim=0).detach().cpu().numpy()
    per_dim_mse_over_time = (err ** 2).detach().cpu().numpy()  # (T-1, 4)

    return per_step_mse, per_dim_mse, per_dim_mse_over_time


def plot_error_over_time(per_step_mse, title, save_path=None, show=False):
    fig = plt.figure()
    plt.plot(np.arange(len(per_step_mse)), per_step_mse)
    plt.xlabel("Timestep")
    plt.ylabel("Model next-state MSE")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        _ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_dim_error_bar(per_dim_mse_dict, save_path=None, show=False):
    names = list(per_dim_mse_dict.keys())
    M = np.stack([per_dim_mse_dict[n] for n in names], axis=0)

    x = np.arange(len(names))
    width = 0.18

    fig = plt.figure()
    for j, key in enumerate(STATE_KEYS):
        plt.bar(x + (j - 1.5) * width, M[:, j], width, label=key)

    plt.xticks(x, names, rotation=20)
    plt.ylabel("Mean next-state MSE")
    plt.title("Model Error by State Dimension (per shift)")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        _ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
