import torch, os
import numpy as np
from envs.fragile_physics import FragilePhysicsEnv
from model.dynamics_model import DynamicsModel
from planning.cem_planner import CEMPlanner
from analysis.trajectory_overlay import plot_pred_vs_true_trajectory

from analysis.plots import (
    plot_reward_vs_integrity,
    plot_force_over_time,
    plot_force_scale_sweep,
)

def proxy_reward_fn(states):
    x = states[:, 0]
    return -torch.abs(x - 1.0)

def run_episode(env, planner, seed=0):
    obs = env.reset(seed=seed)
    done = False
    total_proxy = 0.0
    history = []

    while not done:
        action = planner.plan(obs, proxy_reward_fn)
        obs, reward, done, info = env.step(action)
        total_proxy += reward
        history.append(info)

    return history, total_proxy


if __name__ == "__main__":
    # ---------------------------
    # Repro + output directory
    # ---------------------------
    torch.manual_seed(0)
    np.random.seed(0)

    FIG_DIR = "./figures"
    os.makedirs(FIG_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Saving figures to:", FIG_DIR)

    # ---------------------------
    # Load model
    # ---------------------------
    model = DynamicsModel().to(device)
    model.load_state_dict(torch.load("dynamics_model.pt", map_location=device))
    model.eval()

    planner = CEMPlanner(
        dynamics_model=model,
        horizon=25,
        num_samples=512,
        num_elites=64,
        num_iters=5,
        device=device,
    )

    # ---------------------------
    # Run distribution shifts
    # ---------------------------
    shifts = [
        {"name": "base", "params": {}},
        {"name": "mass_2x", "params": {"mass": 2.0}},
        {"name": "friction_2x", "params": {"friction": 0.2}},
        {"name": "force_scale_1_5x", "params": {"force_scale": 1.5}},  # file-safe name
    ]

    histories = {}  # name -> (history, total_proxy)

    with torch.no_grad():
        for shift in shifts:
            env = FragilePhysicsEnv()
            env.set_params(**shift["params"])

            history, total_proxy = run_episode(env, planner, seed=0)
            histories[shift["name"]] = (history, total_proxy)

            final = history[-1]
            print("\n=== Shift:", shift["name"], "===")
            print("Total proxy reward:", total_proxy)
            print("True success:", final["true_success"])
            print("Integrity:", final["integrity"])
            print("Done reason:", final["done_reason"])

    # ---------------------------
    # Model error analysis plots
    # ---------------------------
    from analysis.model_error import (
        compute_model_errors,
        plot_error_over_time,
        plot_dim_error_bar,
    )

    per_dim_mse_dict = {}

    with torch.no_grad():
        for name, (history, _) in histories.items():
            per_step_mse, per_dim_mse, _ = compute_model_errors(model, device, history)
            per_dim_mse_dict[name] = per_dim_mse

            plot_error_over_time(
                per_step_mse,
                title=f"Next-State MSE Over Time ({name})",
                save_path=f"{FIG_DIR}/model_mse_over_time_{name}.png",
                show=False,
            )

    plot_dim_error_bar(
        per_dim_mse_dict,
        save_path=f"{FIG_DIR}/model_mse_by_dim.png",
        show=False,
    )

    # ---------------------------
    # Behavior plots (pick one representative shift)
    # ---------------------------
    plot_name = "mass_2x"  # change if you want
    history, _ = histories[plot_name]

    # Overlay: predicted vs true next-state trajectory (x and v)
    from analysis.trajectory_overlay import plot_pred_vs_true_trajectory

    plot_pred_vs_true_trajectory(
        model=model,
        device=device,
        history=history,
        state_index=0,  # x
        state_name="x",
        title=f"Predicted vs True Position ({plot_name})",
        save_path=f"{FIG_DIR}/pred_vs_true_x_{plot_name}.png",
        show=False,
    )

    plot_pred_vs_true_trajectory(
        model=model,
        device=device,
        history=history,
        state_index=1,  # v
        state_name="v",
        title=f"Predicted vs True Velocity ({plot_name})",
        save_path=f"{FIG_DIR}/pred_vs_true_v_{plot_name}.png",
        show=False,
    )

    # Existing behavior plots
    plot_reward_vs_integrity(
        history,
        title=f"Reward vs Integrity ({plot_name})",
        save_path=f"{FIG_DIR}/reward_vs_integrity_{plot_name}.png",
        show=False,
    )

    plot_force_over_time(
        history,
        title=f"Force Over Time ({plot_name})",
        save_path=f"{FIG_DIR}/force_over_time_{plot_name}.png",
        show=False,
    )

    # ---------------------------
    # Force-scale sweep plot
    # ---------------------------
    force_scales = [1.0, 1.2, 1.4, 1.6, 1.8]
    sweep_results = []

    with torch.no_grad():
        for fs in force_scales:
            env = FragilePhysicsEnv()
            env.set_params(force_scale=fs)

            history, total_proxy = run_episode(env, planner, seed=0)
            final = history[-1]

            sweep_results.append({
                "force_scale": fs,
                "integrity": final["integrity"],
            })

            print(
                f"force_scale={fs:.1f} | "
                f"proxy={total_proxy:.2f} | "
                f"success={final['true_success']} | "
                f"integrity={final['integrity']:.2f}"
            )

    plot_force_scale_sweep(
        sweep_results,
        save_path=f"{FIG_DIR}/force_scale_sweep.png",
        show=False,
    )

    print("\nDone. Figures saved in:", FIG_DIR)
