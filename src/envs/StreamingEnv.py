import math
import copy
import random
import numpy as np
import gymnasium as gym


from src.envs.utils import load_trace, load_video_size


LINK_RTT = 80  # millisec
NOISE_LOW = 0.9
M_IN_K = 1000.0
NOISE_HIGH = 1.1
BITRATE_LEVELS = 6
BITS_IN_BYTE = 8.0

REBUF_PENALTY = 1  # 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1

DEFAULT_QUALITY = 1  # default video quality without agent
B_IN_MB = 1000000.0
TOTAL_VIDEO_CHUNCK = 48
BUFFER_NORM_FACTOR = 10.0
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
PACKET_PAYLOAD_PORTION = 0.95
MILLISECONDS_IN_SECOND = 1000.0
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
VIDEO_BIT_RATE = np.array([300.0, 750.0, 1200.0, 1850.0, 2850.0, 4300.0])  # Kbps
CHUNK_TIL_VIDEO_END_CAP = 48.0

cof = 7.0


class AdaptiveStreamingEnv(object):
    def __init__(self, seed_num, content_model="documentary"):
        np.random.seed(seed_num)
        assert content_model in ["documentary", "live", "news"], "The Model of Content Out of Scope!"

        self.content_model = content_model

        self.content_type_weights = {
            "documentary": {"bitrate_weight": 1.0, "rebuf_penalty": 1.0, "smooth_penalty": cof},  # 标准权重
            "live": {"bitrate_weight": 1.0, "rebuf_penalty": cof, "smooth_penalty": 1.0},  # 直播可以接受较低码率  # 直播对延迟最敏感  # 可以容忍一定画质波动
            "news": {"bitrate_weight": cof, "rebuf_penalty": 1.0, "smooth_penalty": 1.0},  # 新闻类内容对码率要求低  # 新闻类内容对缓冲不太敏感  # 新闻类内容对画质波动不敏感
        }

        self.cooked_timestep_lists, self.cooked_bw_lists, trace_filename = load_trace("high")

        ## --------- select network bandwidth --------- ##
        self.trace_idx = np.random.randint(len(self.cooked_timestep_lists))
        self.cooked_timestep_seq = self.cooked_timestep_lists[self.trace_idx]
        self.cooked_bw = self.cooked_bw_lists[self.trace_idx]
        self.bw_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_bw_timestamp = self.cooked_timestep_seq[self.bw_ptr - 1]

        ## --------- Load Video Size --------- ##
        self.video_chunk_size = load_video_size()

        ## --------- Set Default Parameters --------- ##
        self.video_chunk_cnt = 0
        self.client_buffer_size = 0  # ms
        self.last_select_bitrate = random.randint(0, BITRATE_LEVELS - 1)
        self.state_info = 6
        self.state_length = 8
        self.action_space = gym.spaces.Discrete(BITRATE_LEVELS)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_info, self.state_length), dtype=np.float32)

    def seed(self, seed_num):
        self.seed_num = seed_num

    def reset(self):
        self.time_stamp = 0
        self.client_buffer_size = 0
        self.video_chunk_cnt = 0
        self.last_select_bitrate = random.randint(0, BITRATE_LEVELS - 1)
        self.state = np.zeros((self.state_info, self.state_length))
        delay_time, sleep_time, buffer_size, rebuffer_time, choose_video_chunk_size, next_video_chunk_size, end_of_video, video_chunk_remain = self._get_video_chunk(
            self.last_select_bitrate
        )
        self.state[0, -1] = VIDEO_BIT_RATE[self.last_select_bitrate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        self.state[1, -1] = (self.client_buffer_size / BUFFER_NORM_FACTOR) / 100  # 10 sec
        self.state[2, -1] = float(choose_video_chunk_size) / float(delay_time) / M_IN_K  # kilo byte / ms
        self.state[3, -1] = float(delay_time) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        self.state[4, : self.action_space.n] = np.array(next_video_chunk_size) / M_IN_K / M_IN_K  # mega byte
        self.state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        return copy.deepcopy(self.state[np.newaxis, ...])

    def step(self, action):
        bitrate = int(action)
        delay_time, sleep_time, buffer_size, rebuffer_time, choose_video_chunk_size, next_video_chunk_size, end_of_video, video_chunk_remain = self._get_video_chunk(bitrate)
        assert round(buffer_size * MILLISECONDS_IN_SECOND, 2) == round(self.client_buffer_size, 2), "Buffer Size Error!"
        cof = self.content_type_weights[self.content_model]
        bit_cof, reb_cof, smo_cof = cof["bitrate_weight"], cof["rebuf_penalty"], cof["smooth_penalty"]
        bitrate_reward = bit_cof * math.log(VIDEO_BIT_RATE[bitrate] / M_IN_K + 0.7) - (0.01 / (VIDEO_BIT_RATE[bitrate] / M_IN_K))  # range in [0, 1.6]
        rebuffer_time_reward = reb_cof * REBUF_PENALTY * rebuffer_time
        smooth_penalty_reward = smo_cof * SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bitrate] - VIDEO_BIT_RATE[self.last_select_bitrate]) / M_IN_K
        reward = bitrate_reward - rebuffer_time_reward - smooth_penalty_reward

        ## --------- Update Information --------- ##
        self.last_select_bitrate = bitrate
        self.state = np.roll(self.state, -1, axis=1)
        self.state[0, -1] = VIDEO_BIT_RATE[bitrate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        self.state[1, -1] = (self.client_buffer_size / BUFFER_NORM_FACTOR) / 100  # 10 sec
        self.state[2, -1] = float(choose_video_chunk_size) / float(delay_time) / M_IN_K  # kilo byte / ms
        self.state[3, -1] = float(delay_time) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        self.state[4, : self.action_space.n] = np.array(next_video_chunk_size) / M_IN_K / M_IN_K  # mega byte
        self.state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        return (
            copy.deepcopy(self.state[np.newaxis, ...]),
            np.array(reward),
            np.array(end_of_video),
            {
                "bitrate": bitrate,
                "delay": float(delay_time) / M_IN_K / BUFFER_NORM_FACTOR,
                "buffer_size": buffer_size / BUFFER_NORM_FACTOR,
                "buffer_sleep_time": sleep_time,
                "rebuffer": rebuffer_time,
                "choose_video_chunk_size": choose_video_chunk_size,
                "choose_video_chunk_size_per_time": float(choose_video_chunk_size) / float(delay_time) / M_IN_K,
                "bitrate_reward": bitrate_reward,
                "rebuffer_time_reward": -rebuffer_time_reward,
                "smooth_penalty_reward": -smooth_penalty_reward,
            },
        )

    def _get_video_chunk(self, quality):
        assert quality >= 0, "Video Quality Must be Greater 0!"
        assert quality < BITRATE_LEVELS, "Video Quality Must be Less than BITRATE_LEVELS!"
        selected_chunk_size = self.video_chunk_size[quality][self.video_chunk_cnt]

        ## --------- Process Video Chunk by Network --------- ##
        delay = 0.0  # ms
        video_chunk_have_processed = 0  # in bytes
        while True:
            throughput = self.cooked_bw[self.bw_ptr] * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_timestep_seq[self.bw_ptr] - self.last_bw_timestamp
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_have_processed + packet_payload > selected_chunk_size:
                fractional_time = (selected_chunk_size - video_chunk_have_processed) / throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_bw_timestamp += fractional_time
                assert self.last_bw_timestamp <= self.cooked_timestep_seq[self.bw_ptr], "bw timestamp must less than next ptr timestamp!"
                break

            video_chunk_have_processed += packet_payload
            delay += duration
            self.last_bw_timestamp = self.cooked_timestep_seq[self.bw_ptr]
            self.bw_ptr += 1
            if self.bw_ptr >= len(self.cooked_bw):
                self.bw_ptr = 1
                self.last_bw_timestamp = self.cooked_timestep_seq[self.bw_ptr - 1]

        ## --------- Add Noise For Link --------- ##
        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)  # add a multiplicative noise to the delay

        ## ---------- Process Buffer Time And Buffer Size --------- ##
        wait_rebuf_time = np.maximum(delay - self.client_buffer_size, 0.0)  # wait rebuffer time, ms
        self.client_buffer_size = np.maximum(self.client_buffer_size - delay, 0.0)  # update the buffer
        self.client_buffer_size += VIDEO_CHUNCK_LEN  # add in the new chunk

        ## --------- Sleep If Buffer Gets Too Large --------- ##
        sleep_time = 0  # ms
        if self.client_buffer_size > BUFFER_THRESH:  # buffer > 60s ---> sleep
            # We need to skip some network bandwidth here but not add up the delay

            ## --------- Compute Sleep Time --------- ##
            drain_buffer_time = self.client_buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * DRAIN_BUFFER_SLEEP_TIME
            self.client_buffer_size -= sleep_time
            while True:  # Network Move Forward
                sleep_duration = self.cooked_timestep_seq[self.bw_ptr] - self.last_bw_timestamp
                if sleep_duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_bw_timestamp += sleep_time / MILLISECONDS_IN_SECOND
                    break

                sleep_time -= sleep_duration * MILLISECONDS_IN_SECOND
                self.last_bw_timestamp = self.cooked_timestep_seq[self.bw_ptr]
                self.bw_ptr += 1

                if self.bw_ptr >= len(self.cooked_bw):
                    # loop back in the beginning, trace file starts with time 0.
                    self.bw_ptr = 1
                    self.last_bw_timestamp = self.cooked_timestep_seq[self.bw_ptr - 1]

        ## --------- Update Video Chunk Information --------- ##
        self.video_chunk_cnt += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_cnt
        end_of_video = False
        if self.video_chunk_cnt >= TOTAL_VIDEO_CHUNCK:
            end_of_video = True

            ## --------- Reset Buffer And Video Chunk Cnt --------- ##
            self.client_buffer_size = 0
            self.video_chunk_cnt = 0

            ## ---------Reset Select network bandwidth --------- ##
            self.trace_idx = np.random.randint(len(self.cooked_timestep_lists))
            self.cooked_timestep_seq = self.cooked_timestep_lists[self.trace_idx]
            self.cooked_bw = self.cooked_bw_lists[self.trace_idx]
            self.bw_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_bw_timestamp = self.cooked_timestep_seq[self.bw_ptr - 1]

        next_video_chunk_sizes = []
        for level in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_chunk_size[level][self.video_chunk_cnt])

        return (
            delay,
            sleep_time,
            copy.deepcopy(self.client_buffer_size / MILLISECONDS_IN_SECOND),
            copy.deepcopy(wait_rebuf_time / MILLISECONDS_IN_SECOND),
            selected_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
        )


if __name__ == "__main__":
    env = AdaptiveStreamingEnv(seed_num=1, content_model="live")
    obs = env.reset()
    done = False
    while not done:
        action = 0
        next_obs, reward, done, info = env.step(action)
        print(reward)
