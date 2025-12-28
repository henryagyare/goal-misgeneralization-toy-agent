# analysis/plots.py
import os
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def extract(history, key):
    return np.array([h[key] for h in history])


def plot_reward_vs_integrity(history, title="Proxy Reward vs Integrity", save_path=None, show=False):
    steps = extract(history, "step")
    rewards = extract(history, "proxy_reward")
    integrity = extract(history, "integrity")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(steps, rewards)
    ax2.plot(steps, integrity)

    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Proxy Reward")
    ax2.set_ylabel("Integrity")

    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        _ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_force_over_time(history, title="Force Magnitude Over Time", save_path=None, show=False):
    steps = extract(history, "step")
    force = np.abs(extract(history, "force"))

    fig = plt.figure()
    plt.plot(steps, force)
    plt.xlabel("Timestep")
    plt.ylabel("|Force|")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        _ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_force_scale_sweep(results, save_path=None, show=False):
    scales = [r["force_scale"] for r in results]
    integrity = [r["integrity"] for r in results]

    fig = plt.figure()
    plt.plot(scales, integrity, marker="o")
    plt.xlabel("Force Scale")
    plt.ylabel("Final Integrity")
    plt.title("Distribution Shift: Force Scale vs Integrity")
    plt.tight_layout()

    if save_path is not None:
        _ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
