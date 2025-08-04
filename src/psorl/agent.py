from collections import deque
from typing import Literal, overload

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done=False):
        self.buffer.append((state, action, reward, next_state, 1 if done else 0))

    @overload
    def sample(
        self,
        batch_size: int,
        *,
        random_state=None,
        return_dones: Literal[True],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    @overload
    def sample(
        self,
        batch_size: int,
        *,
        random_state=None,
        return_dones: Literal[False] | None = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def sample(
        self,
        batch_size: int,
        *,
        random_state=None,
        return_dones: bool | None = False,
    ):
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices]
        )
        if return_dones:
            return (
                np.asarray(states),
                np.asarray(actions),
                np.asarray(rewards),
                np.asarray(next_states),
                np.asarray(dones),
            )
        else:
            return (
                np.asarray(states),
                np.asarray(actions),
                np.asarray(rewards),
                np.asarray(next_states),
            )

    def __len__(self):
        return len(self.buffer)
