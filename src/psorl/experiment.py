import random

import gymnasium as gym
import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.population import Population

from .agent import ReplayBuffer
from .problem import TheProblem
from .td3 import TD3


def experiment(
    env_name: str,
    *,
    num_agents: int = 25,
    max_timesteps: int = 100_000,
    exploration_ratio: float = 0.25,
    replay_buffer_capacity: int = 10_000,
    batch_size: int = 128,
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

    agents = [
        TD3(
            state_dim=observation_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
        )
        for _ in range(num_agents)
    ]

    vector_encoded_actors = np.asarray(
        [agent.get_actor_parameters() for agent in agents]
    )

    pso = PSO(
        pop_size=len(vector_encoded_actors),
        sampling=Population.new(X=vector_encoded_actors),
        adaptive=False,
        pertube_best=False,
        seed=seed,
    )

    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

    problem = TheProblem(
        env=env,
        agents=agents,
        replay_buffer=replay_buffer,
        n_var=vector_encoded_actors[0].shape[0],
        xl=-2.0,
        xu=2.0,
        device=device,
    )

    pso.setup(problem, verbose=verbose)
    L = np.zeros(pso.pop_size)
    B = np.zeros(pso.pop_size)
    t, e, b = (0, 0, 0)
    pop = pso.ask()
    pop = pso.evaluator.eval(problem, pop, algorithm=pso)
    while t < max_timesteps:
        stage = 1 if (t < max_timesteps * exploration_ratio) else 2
        pso.tell(pop)
        index_list = ([e] * pso.pop_size) + list(range(pso.pop_size))
        for i in index_list:
            fitness, steps = pso.evaluator.eval(problem, pop[i], algorithm=pso).get(
                "F", "steps"
            )
            fitness = fitness[0]
            t += steps
            L[i] += steps
            if fitness > B[i]:
                B[i] = fitness
                L[i] = 0
                e = i
            if B[i] > np.max(B):
                if stage == 1:
                    b = i
                elif stage == 2:
                    pop[b] = pop[i].copy()
            if stage == 1:  # (optimize Pi by via RL)
                actor_params = pop[i].X
                agent = agents[i]
                agent.set_actor_parameters(actor_params)
                agent.update(
                    replay_buffer=replay_buffer,
                    discount=discount,
                    tau=tau,
                    policy_noise=policy_noise,
                    noise_clip=noise_clip,
                    policy_freq=policy_freq,
                    batch_size=batch_size,
                )
                pop[i].set("X", agent.get_actor_parameters())
            elif stage == 2:  # (optimize Pb via RL)
                actor_params = pop[b].X
                agent = agents[b]
                agent.set_actor_parameters(actor_params)
                agent.update(
                    replay_buffer=replay_buffer,
                    discount=discount,
                    tau=tau,
                    policy_noise=policy_noise,
                    noise_clip=noise_clip,
                    policy_freq=policy_freq,
                    batch_size=batch_size,
                )
                pop[b].set("X", agent.get_actor_parameters())
        if L[e] > L[b] or stage == 2:
            e = b
        # ask-and-tell is inverted because `ask()` does the PSO update
        # which happens at the end of the while-loop according to Two-Stage ERL (TERL)
        pso.pop = pop
        pop = pso.ask()

    return pso.result()
