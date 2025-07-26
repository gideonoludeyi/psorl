import random

import gymnasium as gym
import numpy as np
import torch

from .agent import ReplayBuffer
from .td3 import TD3


def td3_experiment(
    env_name: str,
    *,
    max_timesteps: int = 100_000,
    replay_buffer_capacity: int = 10_000,
    batch_size: int = 256,
    discount: float = 0.99,
    tau: float = 0.005,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_freq: int = 2,
    device=None,
    verbose: bool = False,
    seed: int | None = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(seed is not None, warn_only=seed is not None)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_name)
    action_dim = env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(
        state_dim=observation_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
    )

    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

    state, _ = env.reset(seed=seed)
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(max_timesteps):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < replay_buffer_capacity:
            action = env.action_space.sample()
        else:
            action = agent.select_action(np.asarray(state))

        # Perform action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store data in replay buffer
        replay_buffer.push(state, action, reward, next_state)

        state = next_state
        episode_reward += float(reward)

        # Train agent after collecting sufficient experience
        if t >= replay_buffer_capacity:
            agent.update(
                replay_buffer=replay_buffer,
                discount=discount,
                tau=tau,
                policy_noise=policy_noise,
                noise_clip=noise_clip,
                policy_freq=policy_freq,
                batch_size=batch_size,
            )

        if done:
            if verbose:
                print(
                    f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3E}"
                )
            # Reset environment
            state, _ = env.reset(seed=seed)
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
