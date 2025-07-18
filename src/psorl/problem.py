import gymnasium as gym
import torch
from pymoo.core.problem import Problem
from torch.nn.utils import vector_to_parameters

from .agent import ActorCritic, ReplayBuffer


def run_episode(agent: ActorCritic, env: gym.Env, replay_buffer: ReplayBuffer):
    """evaluate fitness of actor's policy on an environment"""
    total_reward = 0.0
    steps = 0
    observation, _ = env.reset(seed=42)  # initial state
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, log_prob = agent.select_action(observation)
        new_observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        # Only store state, action, reward, next_state - we'll recompute log_probs during training
        replay_buffer.push(observation, action, None, reward, new_observation)
        observation = new_observation
        steps += 1
    return total_reward, steps


class TheProblem(Problem):
    def __init__(
        self,
        env: gym.Env,
        agents: list[ActorCritic],
        replay_buffer: ReplayBuffer,
        device: torch.device,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env = env
        self.agents = agents
        self.replay_buffer = replay_buffer
        self.device = device

    def _evaluate(self, X, out, *args, **kwargs):
        """X is the set of solutions, not just one solution"""
        F = []
        steps_list = []
        for agent, x in zip(self.agents, X):
            vector_to_parameters(
                torch.FloatTensor(x).to(self.device), agent.actor.parameters()
            )
            total_reward, steps = run_episode(
                agent=agent, env=self.env, replay_buffer=self.replay_buffer
            )
            F.append(-total_reward)
            steps_list.append(steps)
        out["F"] = F
        out["steps"] = steps_list
