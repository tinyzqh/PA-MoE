import numpy as np


class RobustMPCAgent(object):
    def __init__(self, env):
        self.env = env
        self.total_video_chunk = self.env.unwrapped.TOTAL_VIDEO_CHUNCK
        self.env_video_chunk_sizes = self.env.unwrapped.video_chunk_sizes
        self.VIDEO_BIT_RATE = env.unwrapped.VIDEO_BIT_RATE
        self.MPC_FUTURE_CHUNK_COUNT = 3
        self.REBUF_PENALTY = self.env.unwrapped.REBUF_PENALTY  # 4.3  # 1 sec rebuffering -> 3 Mbps
        self.SMOOTH_PENALTY = self.env.unwrapped.SMOOTH_PENALTY

        self.DEFAULT_QUALITY = 1  # default video quality without agent

        self.past_throughputs = np.zeros(6)  # 最近6个chunk的吞吐量（单位：byte/ms）
        self.bandwidth_estimates = []
        self.estimation_errors = []

    def forward(self, obs):
        # 当前chunk的实际吞吐量（byte/ms）
        chunk_size = float(obs["selected_video_chunk_size_bytes"])
        chunk_delay = float(obs["delay_ms"])
        current_throughput = chunk_size / chunk_delay if chunk_delay > 0 else 1e-6

        # 更新带宽历史（左移，添加最新值）
        self.past_throughputs = np.roll(self.past_throughputs, -1)
        self.past_throughputs[-1] = current_throughput

        # 丢弃前面无效的 0（刚初始化时）
        valid_throughputs = self.past_throughputs[self.past_throughputs > 0]

        # 如果历史太少，保守处理
        if len(valid_throughputs) == 0:
            return 0

        # 如果之前有预测，计算预测误差
        if self.bandwidth_estimates:
            last_est = self.bandwidth_estimates[-1]
            error = abs(last_est - current_throughput) / current_throughput
            self.estimation_errors.append(error)
        else:
            self.estimation_errors.append(0.0)

        # 计算历史带宽的调和平均作为估计带宽
        harmonic_bw = len(valid_throughputs) / np.sum(1.0 / valid_throughputs)
        self.bandwidth_estimates.append(harmonic_bw)

        # 使用最大误差调整估计（稳健策略）
        recent_errors = self.estimation_errors[-5:]  # 最多看5个
        max_error = max(recent_errors)
        predicted_bandwidth = harmonic_bw / (1 + max_error)

        # future chunks length (try 4 if that many remaining)
        last_index = int(self.total_video_chunk - obs["remain_chunk"] - 1)
        future_chunk_length = self.MPC_FUTURE_CHUNK_COUNT
        if self.total_video_chunk - last_index < self.MPC_FUTURE_CHUNK_COUNT:
            future_chunk_length = self.total_video_chunk - last_index

        max_reward = -100000000
        start_buffer = obs["buffer_size_ms"] / 1000  # ms -> s
        download_time_every_step = []
        for position in range(future_chunk_length):
            download_time_current = []
            for action in range(0, self.env.action_space.n):
                index = last_index + position + 1  # e.g., if last chunk is 3, then first iter is 3+0+1=4
                if index < 0 or index > 47:
                    download_time = (0) / (predicted_bandwidth / 1000)  # this is MB/MB/s --> seconds
                else:
                    download_time = (self.env_video_chunk_sizes[action][index] / 1000000.0) / (predicted_bandwidth / 1000)  # this is MB/MB/s --> seconds
                download_time_current.append(download_time)
            download_time_every_step.append(download_time_current)

        reward_comparison = False
        send_data = 0
        self.last_bitrate = self.env.unwrapped.last_select_bitrate
        parents_pool = [[0.0, start_buffer, int(self.last_bitrate)]]
        for position in range(future_chunk_length):
            if position == future_chunk_length - 1:
                reward_comparison = True
            children_pool = []
            for parent in parents_pool:
                action = 0
                curr_buffer = parent[1]
                last_quality = parent[-1]
                curr_rebuffer_time = 0
                chunk_quality = action
                download_time = download_time_every_step[position][chunk_quality]
                if curr_buffer < download_time:
                    curr_rebuffer_time += download_time - curr_buffer
                    curr_buffer = 0.0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4

                # reward
                bitrate_sum = self.VIDEO_BIT_RATE[chunk_quality]
                smoothness_diffs = abs(self.VIDEO_BIT_RATE[chunk_quality] - self.VIDEO_BIT_RATE[last_quality])
                reward = (bitrate_sum / 1000.0) - (self.REBUF_PENALTY * curr_rebuffer_time) - (self.SMOOTH_PENALTY * smoothness_diffs / 1000.0)
                reward += parent[0]

                children = parent[:]
                children[0] = reward
                children[1] = curr_buffer
                children.append(action)
                children_pool.append(children)
                if (reward >= max_reward) and reward_comparison:
                    if send_data > children[3] and reward == max_reward:
                        send_data = send_data
                    else:
                        send_data = children[3]
                    max_reward = reward

                rebuffer_term = self.REBUF_PENALTY * (
                    max(download_time_every_step[position][action + 1] - parent[1], 0) - max(download_time_every_step[position][action] - parent[1], 0)
                )
                if action + 1 <= parent[-1]:
                    High_Maybe_Superior = (1.0 + 2 * self.SMOOTH_PENALTY) * (self.VIDEO_BIT_RATE[action] / 1000.0 - self.VIDEO_BIT_RATE[action + 1] / 1000.0) + rebuffer_term < 0.0
                else:
                    High_Maybe_Superior = (self.VIDEO_BIT_RATE[action] / 1000.0 - self.VIDEO_BIT_RATE[action + 1] / 1000.0) + rebuffer_term < 0.0

                while High_Maybe_Superior:
                    curr_buffer = parent[1]
                    last_quality = parent[-1]
                    curr_rebuffer_time = 0
                    chunk_quality = action + 1
                    download_time = download_time_every_step[position][chunk_quality]
                    if curr_buffer < download_time:
                        curr_rebuffer_time += download_time - curr_buffer
                        curr_buffer = 0
                    else:
                        curr_buffer -= download_time
                    curr_buffer += 4

                    # reward
                    bitrate_sum = self.VIDEO_BIT_RATE[chunk_quality]
                    smoothness_diffs = abs(self.VIDEO_BIT_RATE[chunk_quality] - self.VIDEO_BIT_RATE[last_quality])
                    reward = (bitrate_sum / 1000.0) - (self.REBUF_PENALTY * curr_rebuffer_time) - (self.SMOOTH_PENALTY * smoothness_diffs / 1000.0)
                    reward += parent[0]

                    children = parent[:]
                    children[0] = reward
                    children[1] = curr_buffer
                    children.append(chunk_quality)
                    children_pool.append(children)
                    if (reward >= max_reward) and reward_comparison:
                        if send_data > children[3] and reward == max_reward:
                            send_data = send_data
                        else:
                            send_data = children[3]
                        max_reward = reward

                    action += 1
                    if action + 1 == self.env.action_space.n:
                        break
                    # criterion terms
                    # theta = SMOOTH_PENALTY * (VIDEO_BIT_RATE[action+1]/1000. - VIDEO_BIT_RATE[action]/1000.)
                    rebuffer_term = self.REBUF_PENALTY * (
                        max(download_time_every_step[position][action + 1] - parent[1], 0) - max(download_time_every_step[position][action] - parent[1], 0)
                    )
                    if action + 1 <= parent[-1]:
                        High_Maybe_Superior = (1.0 + 2 * self.SMOOTH_PENALTY) * (
                            self.VIDEO_BIT_RATE[action] / 1000.0 - self.VIDEO_BIT_RATE[action + 1] / 1000.0
                        ) + rebuffer_term < 0
                    else:
                        High_Maybe_Superior = (self.VIDEO_BIT_RATE[action] / 1000.0 - self.VIDEO_BIT_RATE[action + 1] / 1000.0) + rebuffer_term < 0

            parents_pool = children_pool

        bit_rate = send_data
        return bit_rate
