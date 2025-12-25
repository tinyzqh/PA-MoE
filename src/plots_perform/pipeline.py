import time
import torch
import OptiVerse
import numpy as np
import gymnasium as gym
from pathlib import Path
from collections import deque

from src.plots_perform.plots.plot_qoe_cdf import plot_cdf
from src.plots_perform.plots.plot_smo_rebuf import smo_rebuf
from src.plots_perform.plots.plot_qoe_bar import plot_qoe_bar
from src.plots_perform.plots.plot_bitrate_smo import bitrate_smo
from src.plots_perform.plots.plot_bitrate_rebuf import bitrate_rebuf


from src.plots_perform.agents.bola_agent import BolaBasedAgent
from src.plots_perform.agents.pamoe_agent import Agent as PAMOE
from src.plots_perform.agents.ppo_agent import Agent as PPOAgent
from src.plots_perform.agents.rate_based_agent import RateBasedAgent
from src.plots_perform.agents.robust_mpc_agent import RobustMPCAgent
from src.plots_perform.agents.buffer_based_agent import BufferBasedAgent


from src.plots_perform.agents.meta_ppo.vae import Encoder
from src.plots_perform.agents.meta_ppo.vae import Decoder
from src.plots_perform.agents.meta_ppo.merina import Agent as MetaAgent
from src.plots_perform.agents.meta_ppo.vae import VariationalAutoencoder


S_LEN = 8
S_INFO = 6
action_dim = 6


def normalize_obs(env, state, global_state):
    global_state = np.roll(global_state, -1, axis=1)
    global_state[0, -1] = env.unwrapped.VIDEO_BIT_RATE[env.unwrapped.last_select_bitrate] / float(np.max(env.unwrapped.VIDEO_BIT_RATE))
    global_state[1, -1] = env.unwrapped.client_buffer_size / 10000.0
    global_state[2, -1] = state["selected_video_chunk_size_bytes"] / max(state["delay_ms"], 1) / 1000.0
    global_state[3, -1] = float(state["delay_ms"]) / 1000.0 / 10
    global_state[4, :action_dim] = state["next_video_chunk_sizes"] / 1e6
    global_state[5, -1] = np.minimum(state["remain_chunk"], env.unwrapped.TOTAL_VIDEO_CHUNCK) / float(env.unwrapped.TOTAL_VIDEO_CHUNCK)

    return global_state


def process(env, agent, vae_model, agent_name):
    reward_info = {agent_name: {"QoE": [], "Bitrate Reward": [], "Rebuffer Reward": [], "Rebuffer Time": [], "Smooth Penalty Reward": [], "Smooth": []}}

    tensor_obs_algorithms = ["PPO", "Pensieve", "Merina", "PAMoE", "Resin", "Resin-MoE"]

    # 记录所有决策延迟（毫秒）
    decision_latencies_ms = []

    measure_latency = True
    rule_based_agent = RateBasedAgent(env=env)

    for i in range(300):
        # print(i)
        global_state = np.zeros((S_INFO, S_LEN))
        obs, info = env.reset()
        if agent_name in tensor_obs_algorithms:
            global_state = normalize_obs(env, obs, global_state)
            obs = torch.Tensor(global_state)
        # if "Hybrid-Learning" in agent_name:
        #     rule_based_action = np.array([rule_based_agent.forward(obs)])
        #     next_rule_action = torch.Tensor(rule_based_action).long()

        #     global_state = normalize_obs(env, obs, global_state)
        #     obs = torch.Tensor(global_state)

        done = False

        episode_qoe = 0
        episode_bitrate_reward = 0
        episode_rebuffer_reward = 0
        episode_smooth_penalty_reward = 0

        episode_step = 0
        while not done:
            # ==== 计时：策略前向（可选含预处理） ====
            if measure_latency:
                t0 = time.perf_counter_ns()

            if agent_name in tensor_obs_algorithms:
                if agent_name == "Merina":
                    vae_input = obs.unsqueeze(0)[:, 2:4, :]
                    vae_encoder_mean, vae_encoder_var = model_vae.encoder(vae_input)
                    action, logprob, _, value = agent.get_max_action_and_value(obs, None, vae_encoder_mean)
                    # action, logprob, _, value = agent.get_action_and_value(obs, None, vae_encoder_mean)
                else:
                    action, logprob, _, value = agent.get_max_action_and_value(obs)
                    # action, logprob, _, value = agent.get_action_and_value(obs)
                action = int(action.cpu().numpy())
                # print(action)
            # elif "Hybrid-Learning" in agent_name:
            #     action, logprob, _, value = agent.get_max_action_and_value(obs, next_rule_action)
            else:
                action = agent.forward(obs)

            if measure_latency:
                t1 = time.perf_counter_ns()
                latency_ms = (t1 - t0) / 1e6  # 仅策略前向

                # # 如果需要把“上一轮的预处理时间”也算入决策延迟：
                # if include_preprocess and 'prep_t1' in locals() and 'prep_t0' in locals():
                #     latency_ms += (prep_t1 - prep_t0) / 1e6

                decision_latencies_ms.append(latency_ms)

            obs, reward, terminated, truncated, info = env.step(action)

            if agent_name in tensor_obs_algorithms:
                global_state = normalize_obs(env, obs, global_state)
                obs = torch.Tensor(global_state)
            # elif "Hybrid-Learning" in agent_name:
            #     rule_based_action = np.array([rule_based_agent.forward(obs)])
            #     next_rule_action = torch.Tensor(rule_based_action).long()
            #     global_state = normalize_obs(env, obs, global_state)
            #     obs = torch.Tensor(global_state)

            episode_qoe += reward
            episode_bitrate_reward += float(info["bitrate_reward"])
            episode_rebuffer_reward += float(info["rebuffer_time_reward"])
            episode_smooth_penalty_reward += float(info["smooth_penalty_reward"])

            done = terminated or truncated
            episode_step += 1

        reward_info[agent_name]["QoE"].append(episode_qoe / episode_step)
        reward_info[agent_name]["Bitrate Reward"].append(episode_bitrate_reward / episode_step)
        reward_info[agent_name]["Rebuffer Reward"].append(episode_rebuffer_reward / episode_step)
        reward_info[agent_name]["Rebuffer Time"].append(abs(episode_rebuffer_reward) / episode_step / env.env.env.REBUF_PENALTY)
        reward_info[agent_name]["Smooth Penalty Reward"].append(episode_smooth_penalty_reward / episode_step)
        reward_info[agent_name]["Smooth"].append(abs(episode_smooth_penalty_reward) / episode_step / env.env.env.SMOOTH_PENALTY)

    lat = np.array(decision_latencies_ms, dtype=np.float64)
    print("Agent: {} | mean_ms: {} | p50_ms: {} | p95_ms: {}".format(agent_name, float(lat.mean()), float(np.percentile(lat, 50)), float(np.percentile(lat, 95))))

    return reward_info


if __name__ == "__main__":
    setting_seed = 0

    trace_name = "test"

    reward_info_dict = {}

    # env = gym.make("VideoStreaming-v0", trace_name=trace_name, bandwidth_type="hybrid", qoe_type="normal", seed=setting_seed)
    # agent = BolaBasedAgent(env=env)
    # bola_reward_info = process(env, agent, None, "BOLA")
    # reward_info_dict.update(bola_reward_info)

    weight_path_root = Path(__file__).resolve().parent

    env = gym.make("VideoStreaming-v0", trace_name=trace_name, bandwidth_type="hybrid", qoe_type="normal", seed=setting_seed)
    agent = BufferBasedAgent(env=env)
    buffer_reward_info = process(env, agent, None, "Buffer-based")
    reward_info_dict.update(buffer_reward_info)

    env = gym.make("VideoStreaming-v0", trace_name=trace_name, bandwidth_type="hybrid", qoe_type="normal", seed=setting_seed)
    agent = RateBasedAgent(env=env)
    rate_reward_info = process(env, agent, None, "Rate-based")
    reward_info_dict.update(rate_reward_info)

    env = gym.make("VideoStreaming-v0", trace_name=trace_name, bandwidth_type="hybrid", qoe_type="normal", seed=setting_seed)
    agent = RobustMPCAgent(env=env)
    mpc_reward_info = process(env, agent, None, "RobustMPC")
    reward_info_dict.update(mpc_reward_info)

    env = gym.make("VideoStreaming-v0", trace_name=trace_name, bandwidth_type="hybrid", qoe_type="normal", seed=setting_seed)
    agent = PPOAgent(None, 6)
    agent.load_state_dict(torch.load(weight_path_root / "weights/Pensieve.pth"))
    agent.eval()
    ppo_reward_info = process(env, agent, None, "Pensieve")
    reward_info_dict.update(ppo_reward_info)

    env = gym.make("VideoStreaming-v0", trace_name=trace_name, bandwidth_type="hybrid", qoe_type="normal", seed=setting_seed)
    INPUT_DIM = 24
    OUTPUT_DIM = 6
    HIDDEN_DIM = 12
    LATENT_DIM = 12
    encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    decoder = Decoder(LATENT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model_vae = VariationalAutoencoder(encoder, decoder)
    model_vae.load_state_dict(torch.load(weight_path_root / "weights/MerinaVAE.pth"))
    model_vae.eval()
    agent = MetaAgent(None, 6)
    agent.load_state_dict(torch.load(weight_path_root / "weights/Merina.pth"))
    agent.eval()
    metarl_reward_info = process(env, agent, model_vae, "Merina")
    reward_info_dict.update(metarl_reward_info)

    env = gym.make("VideoStreaming-v0", trace_name=trace_name, bandwidth_type="hybrid", qoe_type="normal", seed=setting_seed)
    agent = PAMOE(None, 6)
    agent.load_state_dict(torch.load(weight_path_root / "weights/PAMoE.pth"))
    agent.eval()
    pamoe_reward_info = process(env, agent, None, "PAMoE")
    reward_info_dict.update(pamoe_reward_info)

    plot_cdf(data=reward_info_dict, x_label="Average Values of Chunk's QoE", y_label="CDF (Perc. of sessions)", index_name="QoE", save_file_name="cdf_{}.pdf".format(trace_name))
    plot_qoe_bar(data=reward_info_dict, y_label="QoE", x_label="QoE Components", save_file_name="qoe_bar_{}.pdf".format(trace_name))
    bitrate_rebuf(data=reward_info_dict, save_file_name="bitrate_rebuf_{}.pdf".format(trace_name))
    smo_rebuf(data=reward_info_dict, save_file_name="smo_rebuffer_{}.pdf".format(trace_name))
    bitrate_smo(data=reward_info_dict, save_file_name="bitrate_smo_{}.pdf".format(trace_name))
