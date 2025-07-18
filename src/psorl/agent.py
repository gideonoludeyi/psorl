from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, action_log_prob, reward, next_state):
        self.buffer.append((state, action, action_log_prob, reward, next_state))

    def sample(self, batch_size: int, random_state=None):
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(self.buffer), batch_size, replace=False)
        states, actions, _, rewards, next_states = zip(
            *[self.buffer[i] for i in indices]
        )
        return (
            np.asarray(states),
            np.asarray(actions),
            np.asarray(rewards),
            np.asarray(next_states),
        )

    def __len__(self):
        return len(self.buffer)


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


class ActorCritic:
    def __init__(self, state_dim: int, action_dim: int, device: torch.device):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.device = device

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(
        self,
        replay_buffer: ReplayBuffer,
        gamma: float = 0.99,
        batch_size: int = 128,
        random_state=None,
    ):
        states, actions, rewards, next_states = replay_buffer.sample(
            batch_size=batch_size, random_state=random_state
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # --- Critic Update ---
        current_state_values = self.critic(states)
        # Get the next state value prediction from the critic
        # For the target, we don't want gradients to flow through next_state_value
        # if the episode is done, next_state_value is 0 (no future rewards)
        next_state_values = self.critic(next_states)
        # Calculate the TD target (R + gamma * V(s'))
        target_values = rewards + gamma * next_state_values

        # Critic loss: Mean Squared Error between predicted value and TD target
        critic_loss = F.mse_loss(current_state_values, target_values)
        # Perform backpropagation for the critic
        self.critic_optimizer.zero_grad()  # Clear previous gradients
        critic_loss.backward()  # Compute gradients
        self.critic_optimizer.step()  # Update critic network parameters

        # --- Actor Update ---
        # Recompute action probabilities with current actor network
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(actions).unsqueeze(1)

        # Calculate the Advantage (TD Error): TD_target - V(s)
        # It's crucial to detach target_values and current_state_values here to prevent
        # gradients from flowing back into the Critic network during the Actor's update.
        # The Actor's update should only depend on the value estimate, not train the critic.
        advantages = (target_values - current_state_values).detach()

        # Actor loss: Negative log-probability weighted by the advantage
        # We want to maximize expected reward, so we minimize negative expected reward.
        actor_loss = -action_log_probs * advantages
        # Perform backpropagation for the actor
        self.actor_optimizer.zero_grad()  # Clear previous gradients
        # Take the mean of the loss for proper gradient computation
        mean_loss = actor_loss.mean()
        mean_loss.backward()  # Compute gradients
        self.actor_optimizer.step()  # Update actor network parameters
