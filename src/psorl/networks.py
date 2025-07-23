import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = self.l1(state)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        action_probs = F.softmax(self.l3(x), dim=-1)
        return action_probs


class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 1)

    def forward(self, state):
        x = self.l1(state)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        value = self.l3(x)
        return value
