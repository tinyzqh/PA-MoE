import numpy as np


class BolaBasedAgent(object):
    def __init__(self, env):
        self.BUFFER_TARGET_S = 30
        self.MINIMUM_BUFFER_S = 10
        self.VIDEO_BIT_RATE = env.unwrapped.VIDEO_BIT_RATE
        self.gp = 1 - 0 + (np.log(self.VIDEO_BIT_RATE[-1] / float(self.VIDEO_BIT_RATE[0])) - 0) / (self.BUFFER_TARGET_S / self.MINIMUM_BUFFER_S - 1)  # log
        self.vp = self.MINIMUM_BUFFER_S / (0 + self.gp - 1)

    def forward(self, obs):
        score = -65535
        for q in range(len(self.VIDEO_BIT_RATE)):
            # s = (self.vp * (np.log(self.VIDEO_BIT_RATE[q] / float(self.VIDEO_BIT_RATE[0])) + self.gp) - obs["buffer_size_ms"] / 1000) / obs["next_video_chunk_sizes"][q]
            s = (self.vp * (self.VIDEO_BIT_RATE[q] / 1000.0 + self.gp) - obs["buffer_size_ms"] / 1000) / obs["next_video_chunk_sizes"][q]  # lin
            if s >= score:
                score = s
                bit_rate = q
        return int(bit_rate)
