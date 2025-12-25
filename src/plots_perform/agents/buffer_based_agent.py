import numpy as np


class BufferBasedAgent(object):
    def __init__(self, env):
        self.RESEVOIR = 20000  # ms
        self.CUSHION = 8000  # ms
        self.env = env

    def forward(self, obs):
        if obs["buffer_size_ms"] < self.RESEVOIR:
            bit_rate = np.array(0)
        elif obs["buffer_size_ms"] >= self.RESEVOIR + self.CUSHION:
            bit_rate = self.env.action_space.n - 1
        else:
            bit_rate = (self.env.action_space.n - 1) * (obs["buffer_size_ms"] - self.RESEVOIR) / float(self.CUSHION)
        return int(bit_rate)
