import random

import gymnasium as gym
import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.population import Population
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .agent import ActorCritic, ReplayBuffer
from .problem import TheProblem

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True, warn_only=True)


def experiment(*, device=None, verbose=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("LunarLander-v3")
    action_dim = env.action_space.n
    observation_dim = env.observation_space.shape[0]

    agents = [
        ActorCritic(state_dim=observation_dim, action_dim=action_dim, device=device)
        for _ in range(25)
    ]

    vector_encoded_actors = np.asarray(
        [
            parameters_to_vector(agent.actor.parameters()).detach().cpu().numpy()
            for agent in agents
        ]
    )

    pso = PSO(
        pop_size=len(vector_encoded_actors),
        sampling=Population.new(X=vector_encoded_actors),
        adaptive=False,
        pertube_best=False,
        seed=0,
    )

    replay_buffer = ReplayBuffer(capacity=10_000)

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
    MAX_TIMESTEPS = 10_000 * 100
    EXPLORATION_RATIO = 0.25
    L = np.zeros(pso.pop_size)
    B = np.zeros(pso.pop_size)
    t, e, b = (0, 0, 0)
    pop = pso.ask()
    pop = pso.evaluator.eval(problem, pop, algorithm=pso)
    while t < MAX_TIMESTEPS:
        stage = 1 if (t < MAX_TIMESTEPS * EXPLORATION_RATIO) else 2
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
                vector_to_parameters(
                    torch.FloatTensor(actor_params).to(device), agent.actor.parameters()
                )
                agent.update(replay_buffer=replay_buffer, batch_size=128)
                pop[i].set(
                    "X",
                    parameters_to_vector(agent.actor.parameters())
                    .detach()
                    .cpu()
                    .numpy(),
                )
            elif stage == 2:  # (optimize Pb via RL)
                actor_params = pop[b].X
                agent = agents[b]
                vector_to_parameters(
                    torch.FloatTensor(actor_params).to(device), agent.actor.parameters()
                )
                agent.update(replay_buffer=replay_buffer, batch_size=128)
                pop[b].set(
                    "X",
                    parameters_to_vector(agent.actor.parameters())
                    .detach()
                    .cpu()
                    .numpy(),
                )
        if L[e] > L[b] or stage == 2:
            e = b
        # ask-and-tell is inverted because `ask()` does the PSO update
        # which happens at the end of the while-loop according to Two-Stage ERL (TERL)
        pso.pop = pop
        pop = pso.ask()

    result = pso.result()
    return result
