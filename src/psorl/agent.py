from collections import deque

import numpy as np


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