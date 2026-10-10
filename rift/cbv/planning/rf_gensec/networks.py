"""Standalone critics for the source-only release.

The supplied gym_carla/utils/net.py was empty. These new MLPs implement the
interfaces needed by RiskHJR, but historical checkpoint compatibility is not
claimed. Train a new critic, or restore the original networks for old weights.
"""
import torch
from torch import nn


def mlp(input_dim, dims):
    layers = []
    for width in dims:
        layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
        input_dim = width
    layers.append(nn.Linear(input_dim, 1))
    return nn.Sequential(*layers)


class CriticPPO(nn.Module):
    def __init__(self, dims, state_dim, action_dim=None):
        super().__init__()
        self.net = mlp(state_dim, dims)

    def forward(self, state):
        return self.net(state).squeeze(-1)


class CriticTwin(nn.Module):
    def __init__(self, dims, state_dim, action_dim):
        super().__init__()
        self.net1 = mlp(state_dim + action_dim, dims)
        self.net2 = mlp(state_dim + action_dim, dims)

    def get_q1_q2(self, state, action):
        inputs = torch.cat((state, action), dim=-1)
        return self.net1(inputs).squeeze(-1), self.net2(inputs).squeeze(-1)

    def get_q_max(self, state, action):
        q1, q2 = self.get_q1_q2(state, action)
        return torch.maximum(q1, q2)
