import torch
from envs.fragile_physics import FragilePhysicsEnv
from model.dynamics_model import DynamicsModel
from planning.cem_planner import CEMPlanner


def run_episode(env, planner, device):
    obs = env.reset()
    done = False
    total_proxy = 0.0

    while not done:
        action = planner.plan(obs, proxy_reward_fn)
        obs, reward, done, info = env.step(action)
        total_proxy += reward

    return info, total_proxy


def proxy_reward_fn(states):
    x = states[:, 0]
    return -torch.abs(x - 1.0)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = FragilePhysicsEnv()

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

    info, total_proxy = run_episode(env, planner, device)

    print("Final info:", info)
    print("Total proxy reward:", total_proxy)
