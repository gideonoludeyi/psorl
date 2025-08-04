import gymnasium as gym
import torch
from pymoo.core.problem import Problem

from .agent import ReplayBuffer
from .rl_algorithm import RLAlgorithm


def run_episode(
    agent: RLAlgorithm, env: gym.Env, replay_buffer: ReplayBuffer, *, seed=None
):
    """evaluate fitness of actor's policy on an environment"""
    total_reward = 0.0
    steps = 0
    observation, _ = env.reset(seed=seed)  # initial state
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = agent.select_action(observation)
        new_observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        replay_buffer.push(
            observation,
            action,
            reward,
            new_observation,
            done=terminated or truncated,
        )
        observation = new_observation
        steps += 1
    return total_reward, steps


class TheProblem(Problem):
    def __init__(
        self,
        env: gym.Env,
        agents: list[RLAlgorithm],
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

    def _evaluate(self, X, out, *args, seed=None, **kwargs):
        """X is the set of solutions, not just one solution"""
        F = []
        steps_list = []
        for agent, x in zip(self.agents, X):
            agent.set_actor_parameters(x)
            total_reward, steps = run_episode(
                agent=agent,
                env=self.env,
                replay_buffer=self.replay_buffer,
                seed=seed,
            )
            F.append(-total_reward)
            steps_list.append(steps)
        out["F"] = F
        out["steps"] = steps_list
