import torch
import torch.nn as nn


def _activation(name):
    activations = {
        "Tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leakyrelu": nn.LeakyReLU,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


class VelocityNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation="Tanh"):
        super().__init__()
        dimensions = [in_out_dim + 1] + [hidden_dim] * n_hiddens + [in_out_dim]
        activation_type = _activation(activation)
        self.net = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dimensions[index], dimensions[index + 1]),
                    activation_type(),
                )
                for index in range(len(dimensions) - 2)
            ]
        )
        self.out = nn.Linear(dimensions[-2], dimensions[-1])

    def forward(self, time, state):
        time_state = torch.cat((time.expand(state.shape[0], 1), state), dim=1)
        hidden = time_state
        for layer in self.net:
            hidden = layer(hidden)
        return self.out(hidden)


class GrowthNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation="Tanh"):
        super().__init__()
        activation_type = _activation(activation)
        self.net = nn.Sequential(
            nn.Linear(in_out_dim + 1, hidden_dim),
            activation_type(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_type(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_type(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, time, state):
        time_state = torch.cat((time.expand(state.shape[0], 1), state), dim=1)
        return self.net(time_state)


class FNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation):
        super().__init__()
        self.v_net = VelocityNet(in_out_dim, hidden_dim, n_hiddens, activation)
        self.g_net = GrowthNet(in_out_dim, hidden_dim, activation)

    def forward(self, time, state):
        with torch.set_grad_enabled(True):
            state.requires_grad_(True)
            time.requires_grad_(True)
            velocity = self.v_net(time, state).float()
            growth = self.g_net(time, state).float()
        return velocity, growth


class ODEFunc(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, time, augmented_state):
        state, _ = augmented_state
        velocity, growth = self.model(time, state)
        return velocity.float(), growth.float()
