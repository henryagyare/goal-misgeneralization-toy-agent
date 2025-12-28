import numpy as np
import torch
from envs.fragile_physics import FragilePhysicsEnv


def collect_rollouts(
    num_episodes=1000,
    max_steps=200,
    seed=0,
):
    env = FragilePhysicsEnv()
    rng = np.random.default_rng(seed)

    states = []
    actions = []
    next_states = []

    for _ in range(num_episodes):
        obs = env.reset()
        for _ in range(max_steps):
            action = rng.uniform(-1.0, 1.0)
            next_obs, _, done, _ = env.step(action)

            states.append(obs)
            actions.append([action])
            next_states.append(next_obs)

            obs = next_obs
            if done:
                break

    return (
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
        torch.tensor(next_states, dtype=torch.float32),
    )


if __name__ == "__main__":
    s, a, s_next = collect_rollouts()
    print("Collected:", s.shape[0], "transitions")
