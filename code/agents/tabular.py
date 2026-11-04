"""Tabular Q-learning on a hash of the local observation (window + coarse goal direction)."""
import pickle
from collections import defaultdict
import numpy as np


class TabularQAgent:
    recurrent = False

    def __init__(self, obs_dim, n_actions=4, lr=0.1, gamma=0.99, **_):
        self.nA, self.lr, self.gamma = n_actions, lr, gamma
        self.Q = defaultdict(lambda: np.zeros(n_actions, np.float32))

    @staticmethod
    def key(obs):
        # window bits + sign of (dx, dy): goal direction is discretised to 9 cases
        bits = (obs[:-2] > 0.5).astype(np.uint8)
        return np.packbits(bits).tobytes() + bytes(np.sign(obs[-2:]).astype(np.int8) + 1)

    def reset_state(self):
        pass

    def act(self, obs, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(self.nA))
        q = self.Q[self.key(obs)]
        return int(rng.choice(np.flatnonzero(q == q.max())))

    def observe(self, o, a, r, o2, done, truncated=False):
        q = self.Q[self.key(o)]
        target = r + (0.0 if done else self.gamma * self.Q[self.key(o2)].max())
        q[a] += self.lr * (target - q[a])

    def end_episode(self):
        pass

    def maybe_update(self, step):
        return None

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.Q.update(pickle.load(f))
