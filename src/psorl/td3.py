import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from psorl.agent import ReplayBuffer
from psorl.networks import Actor, Critic
from psorl.rl_algorithm import RLAlgorithm


class TD3(RLAlgorithm):
    """Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.

    TD3 is an off-policy, model-free reinforcement learning algorithm designed for
    continuous action spaces. It builds upon DDPG by introducing three key
    modifications to address overestimation bias and improve training stability:

    1.  **Twin Critics**: Uses two Q-networks and their target networks. The minimum
        of the two Q-value predictions from the target critics is used for the
        Bellman update, reducing overestimation bias.
    2.  **Delayed Policy Updates**: The policy (actor) network and its target
        networks are updated less frequently than the Q-networks, allowing critics
        to converge to more accurate value estimates.
    3.  **Target Policy Smoothing**: Adds clipped random noise to target actions
        during Q-value target calculation, making the Q-function more robust and
        preventing exploitation of erroneous peaks.

    Attributes:
        actor (Actor): The actor network that outputs actions.
        actor_target (Actor): The target actor network for stable updates.
        actor_optimizer (torch.optim.Optimizer): Optimizer for the actor network.
        critic_1 (Critic): The first critic network that estimates Q-values.
        critic_1_target (Critic): The target for the first critic network.
        critic_1_optimizer (torch.optim.Adam): Optimizer for the first critic.
        critic_2 (Critic): The second critic network that estimates Q-values.
        critic_2_target (Critic): The target for the second critic network.
        critic_2_optimizer (torch.optim.Adam): Optimizer for the second critic.
        max_action (float): The maximum possible action value in the environment.
        device (torch.device): The device (CPU or GPU) on which to run computations.
        total_it (int): Counter for total number of optimization iterations.
    """

    def __init__(
        self, state_dim: int, action_dim: int, max_action: float, device: torch.device
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)

        self.max_action = max_action
        self.device = device
        self.total_it = 0

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Selects an action based on the current state using the actor network.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            np.ndarray: The selected action.
        """
        state = torch.FloatTensor(np.reshape(state, (1, -1))).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(
        self,
        replay_buffer: ReplayBuffer,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        batch_size: int = 128,
        random_state=None,
    ):
        """Performs a single optimization step on the TD3 networks.

        Args:
            replay_buffer (ReplayBuffer): The replay buffer to sample experiences from.
            discount (float): Discount factor for future rewards (gamma).
            tau (float): Soft update coefficient for target networks.
            policy_noise (float): Standard deviation of Gaussian noise added to target actions for smoothing.
            noise_clip (float): Range to clip the target action noise.
            policy_freq (int): Frequency of policy and target network updates.
            batch_size (int): Number of samples to draw from the replay buffer.
            random_state (int, optional): Random seed for sampling from replay buffer. Defaults to None.
        """
        self.total_it += 1

        # Sample replay buffer
        states, actions, rewards, next_states = replay_buffer.sample(
            batch_size=batch_size, random_state=random_state
        )

        state = torch.FloatTensor(states).to(self.device)
        action = torch.FloatTensor(actions).to(self.device)
        reward = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_states).to(self.device)

        # Select action according to policy and add clipped noise
        noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)

        next_action = (self.actor_target(next_state) + noise).clamp(
            -self.max_action, self.max_action
        )

        # Compute the target Q value
        target_q1 = self.critic_1_target(next_state, next_action)
        target_q2 = self.critic_2_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + discount * target_q

        # Get current Q estimates
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )

        # Optimize the critics
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed policy updates
        if self.total_it % policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic_1.parameters(), self.critic_1_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

            for param, target_param in zip(
                self.critic_2.parameters(), self.critic_2_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

    def get_actor_parameters(self) -> np.ndarray:
        """Retrieves the current parameters of the actor network.

        Returns:
            np.ndarray: A flattened NumPy array of the actor's parameters.
        """
        return parameters_to_vector(self.actor.parameters()).detach().cpu().numpy()

    def set_actor_parameters(self, params: np.ndarray) -> None:
        """Sets the parameters of the actor network.

        Args:
            params (np.ndarray): A flattened NumPy array of parameters to set for the actor.
        """
        vector_to_parameters(
            torch.FloatTensor(params).to(self.device), self.actor.parameters()
        )
