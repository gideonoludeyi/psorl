import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .agent import ReplayBuffer
from .networks import Actor, Critic
from .rl_algorithm import RLAlgorithm


class StochasticActorCriticLearning(RLAlgorithm):
    def __init__(self, state_dim: int, action_dim: int, device: torch.device):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.device = device

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
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
        **kwargs,
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
        next_state_values = self.critic(next_states)
        target_values = rewards + gamma * next_state_values

        critic_loss = F.mse_loss(current_state_values, target_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(actions).unsqueeze(1)

        advantages = (target_values - current_state_values).detach()

        actor_loss = -action_log_probs * advantages
        self.actor_optimizer.zero_grad()
        mean_loss = actor_loss.mean()
        mean_loss.backward()
        self.actor_optimizer.step()

    def get_actor_parameters(self) -> np.ndarray:
        return parameters_to_vector(self.actor.parameters()).detach().cpu().numpy()

    def set_actor_parameters(self, params: np.ndarray) -> None:
        vector_to_parameters(
            torch.FloatTensor(params).to(self.device), self.actor.parameters()
        )
