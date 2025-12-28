import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import model
from model.dynamics_model import DynamicsModel
from experiments.collect_data import collect_rollouts


def train_dynamics(
    epochs=30,
    batch_size=256,
    lr=1e-3,
    device="cpu",
):
    states, actions, next_states = collect_rollouts()
    dataset = TensorDataset(states, actions, next_states)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DynamicsModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for s, a, s_next in loader:
            s = s.to(device)
            a = a.to(device)
            s_next = s_next.to(device)

            pred = model(s, a)
            loss = loss_fn(pred, s_next)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * s.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1:03d} | MSE: {avg_loss:.6f}")

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = train_dynamics(device=device)   # <-- CAPTURE RETURN VALUE
    torch.save(model.state_dict(), "dynamics_model.pt")
    print("Saved dynamics_model.pt")
