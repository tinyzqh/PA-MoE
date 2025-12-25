import numpy as np


class RateBasedAgent(object):
    def __init__(self, env):
        self.env = env
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

        # 码率选择策略（从高到低尝试）
        buffer_ms = float(obs["buffer_size_ms"])
        next_chunk_sizes = obs["next_video_chunk_sizes"]

        selected_bitrate = 0
        for q in reversed(range(self.env.action_space.n)):
            estimated_download_time = next_chunk_sizes[q] / predicted_bandwidth
            if estimated_download_time <= buffer_ms:
                selected_bitrate = q
                break

        return selected_bitrate
