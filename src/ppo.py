import os
import copy
import time
import tyro
import wandb
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
from datetime import datetime


from network.mlp.value import MLPVF
from network.mlp.policy import MLPPolicy
from network.moe.policy import MoEPolicy
from network.sparse_moe.policy import SparseMoEPolicy
from network.sparse_moe.value import SparseMoEVF
from network.distill_moe.policy import DistillMoEPolicy
from network.distill_moe.value import DistillMoEVF
from network.moe.value import MoEVF
from src.configs.ppo_config import Config

from src.envs.StreamingEnv import AdaptiveStreamingEnv
from src.utils.device_utils import set_cuda_configuration

from src.utils.statistic import WeightMagnitudeStatistic, RankStatistic, DormantStatistic, RouterStatistic
from src.utils.statistic import OverleapDormantStatistic, GradientInfoStatistic, OverleapGradientStatistic


def collect_feature_activity(net: nn.Module, net_name: str, store_dict: dict):
    if isinstance(net, MLPPolicy) or isinstance(net, MLPVF):
        net_activations = net.get_activations()

        for idx, expert_name in enumerate(["expert1", "expert2", "expert3"]):
            if expert_name in net_activations:
                for i in range(len(net_activations[expert_name])):
                    key = "expert_{}_{}_{}_layer".format(idx, net_name, i % 2)
                    if key not in store_dict:
                        store_dict.setdefault(key, net_activations[expert_name][i])
                    else:
                        store_dict[key] = torch.cat([store_dict[key], net_activations[expert_name][i]], dim=0)

    elif isinstance(net, MoEPolicy) or isinstance(net, MoEVF):
        net_activations = net.get_activations()

        for idx, expert_name in enumerate(["expert1", "expert2", "expert3"]):
            if expert_name in net_activations:
                for i in range(len(net_activations[expert_name])):
                    key = "expert_{}_{}_{}_layer".format(idx, net_name, i % 2)
                    if key not in store_dict:
                        store_dict.setdefault(key, net_activations[expert_name][i])
                    else:
                        store_dict[key] = torch.cat([store_dict[key], net_activations[expert_name][i]], dim=0)

        key = "{}_router".format(net_name)
        if key not in store_dict:
            store_dict.setdefault(key, net_activations["router"])
        else:
            store_dict[key] = torch.cat([store_dict[key], net_activations["router"]], dim=0)
    elif isinstance(net, SparseMoEPolicy) or isinstance(net, SparseMoEVF):
        net_activations = net.get_activations()

        for idx, expert_name in enumerate(["expert1", "expert2", "expert3"]):
            if expert_name in net_activations:
                for i in range(len(net_activations[expert_name])):
                    key = "expert_{}_{}_{}_layer".format(idx, net_name, i % 2)
                    if key not in store_dict:
                        store_dict.setdefault(key, net_activations[expert_name][i])
                    else:
                        store_dict[key] = torch.cat([store_dict[key], net_activations[expert_name][i]], dim=0)

        key = "{}_router".format(net_name)
        if key not in store_dict:
            store_dict.setdefault(key, net_activations["router"])
        else:
            store_dict[key] = torch.cat([store_dict[key], net_activations["router"]], dim=0)

    elif isinstance(net, DistillMoEPolicy) or isinstance(net, DistillMoEVF):
        net_activations = net.get_activations()

        for idx, expert_name in enumerate(["expert1", "expert2", "expert3"]):
            if expert_name in net_activations:
                for i in range(len(net_activations[expert_name])):
                    key = "expert_{}_{}_{}_layer".format(idx, net_name, i % 2)
                    if key not in store_dict:
                        store_dict.setdefault(key, net_activations[expert_name][i])
                    else:
                        store_dict[key] = torch.cat([store_dict[key], net_activations[expert_name][i]], dim=0)

        key = "{}_router".format(net_name)
        if key not in store_dict:
            store_dict.setdefault(key, net_activations["router"])
        else:
            store_dict[key] = torch.cat([store_dict[key], net_activations["router"]], dim=0)

    else:
        raise ValueError("The Type of {} is not support for net".format(type(net)))


def main(cfg: Config):
    cfg.batch_size = int(cfg.num_envs * cfg.num_steps)
    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.num_iterations = cfg.total_timesteps // cfg.batch_size

    ### --------- Set BandWidth Model --------- ###
    if cfg.envs_model == "normal":
        env_content_list = ["documentary", "documentary", "documentary", "documentary", "documentary"]
    elif cfg.envs_model == "change":
        env_content_list = ["documentary", "live", "news", "documentary", "live", "news", "documentary", "live", "news", "documentary"]
        # [smooth_penalty, rebuf_penalty, bitrate_weight, ]
    else:
        raise ValueError("Env Model {} Not Support!".format(cfg.envs_model))

    ### ------------------- Set Wandb ------------------- ###
    run_name = "{}_{}_{}_{}_{}_{}".format(cfg.exp_name, cfg.seed, cfg.envs_model, cfg.policy_type, cfg.value_type, datetime.now().strftime("%Y-%m-%d"))
    wandb.init(project=cfg.wandb_project_name, config=vars(cfg), name=run_name, monitor_gym=True, save_code=True, mode="online" if cfg.track else "disabled")
    ### ------------------- To get deterministic to work ------------------- ###
    if cfg.torch_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    ### ------------------- Set Random Seed ------------------- ###
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    # device = set_cuda_configuration(0 if torch.cuda.is_available() else -1)
    device = set_cuda_configuration(-1)
    ### ------------------- Env Setup ------------------- ###
    envs = AdaptiveStreamingEnv(seed_num=cfg.seed, content_model=env_content_list[0])

    ### ------------------- Set Agent ------------------- ###
    if cfg.policy_type == "mlp":
        policy_fun = MLPPolicy(envs, cfg.activation_name).to(device)
    elif cfg.policy_type == "moe":
        policy_fun = MoEPolicy(envs, cfg.activation_name).to(device)
    elif cfg.policy_type == "smoe":
        policy_fun = SparseMoEPolicy(envs, cfg.activation_name).to(device)
    elif cfg.policy_type == "dmoe":
        policy_fun = DistillMoEPolicy(envs, cfg.activation_name).to(device)
    else:
        raise ValueError("Not Support Policy Type {}".format(cfg.policy_type))

    if cfg.value_type == "mlp":
        value_fun = MLPVF(envs, cfg.activation_name).to(device)
    elif cfg.value_type == "moe":
        value_fun = MoEVF(envs, cfg.activation_name).to(device)
    elif cfg.value_type == "smoe":
        value_fun = SparseMoEVF(envs, cfg.activation_name).to(device)
    elif cfg.value_type == "dmoe":
        value_fun = DistillMoEVF(envs, cfg.activation_name).to(device)
    else:
        raise ValueError("Not Support Policy Type {}".format(cfg.value_type))

    ## ------------------- Set Optimizer ------------------- ###
    optimizer = optim.Adam(
        list(policy_fun.parameters()) + list(value_fun.parameters()), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2), eps=1e-5
    )

    # ALGO Logic: Storage setup
    assert cfg.num_envs == 1, "The number of env must equal to 1!"
    obs = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(cfg.num_envs).to(device)

    episode_reward_list = []
    episode_bitrate_list = []
    episode_delay_list = []
    episode_buffer_size_list = []
    episode_buffer_sleep_time_list = []
    episode_rebuffer_list = []
    episode_choose_video_chunk_size_list = []
    episode_choose_video_chunk_size_per_time_list = []
    episode_bitrate_reward_list = []
    episode_rebuffer_time_reward_list = []
    episode_smooth_penalty_reward_list = []

    short_term_feature_activity = {}
    for iteration in range(1, cfg.num_iterations + 1):

        # Annealing the rate if instructed to do so.
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            lrnow = frac * cfg.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        ## ---------- Sample Trajectory ---------- ##
        episode_step_cnt = 0
        episode_reward = 0
        episode_bitrate = 0
        episode_delay = 0
        episode_buffer_size = 0
        episode_buffer_sleep_time = 0
        episode_rebuffer = 0
        episode_choose_video_chunk_size = 0
        episode_choose_video_chunk_size_per_time = 0
        episode_bitrate_reward = 0
        episode_rebuffer_time_reward = 0
        episode_smooth_penalty_reward = 0

        for step in range(0, cfg.num_steps):
            global_step += cfg.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _ = policy_fun(next_obs)  # action shape: [1, act_dim], [1]
                collect_feature_activity(policy_fun, "policy", short_term_feature_activity)
                value = value_fun(next_obs)
                collect_feature_activity(value_fun, "value", short_term_feature_activity)

                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, infos = envs.step(action.cpu().numpy())
            episode_step_cnt += 1
            episode_reward += reward
            episode_bitrate += infos["bitrate"]
            episode_delay += infos["delay"]
            episode_buffer_size += infos["buffer_size"]
            episode_buffer_sleep_time += infos["buffer_sleep_time"]
            episode_rebuffer += infos["rebuffer"]
            episode_choose_video_chunk_size += infos["choose_video_chunk_size"]
            episode_choose_video_chunk_size_per_time += infos["choose_video_chunk_size_per_time"]
            episode_bitrate_reward += infos["bitrate_reward"]
            episode_rebuffer_time_reward += infos["rebuffer_time_reward"]
            episode_smooth_penalty_reward += infos["smooth_penalty_reward"]

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            if bool(done):
                next_obs = envs.reset()
                next_obs = torch.Tensor(next_obs).to(device)
                episode_reward_list.append(episode_reward / episode_step_cnt)
                episode_bitrate_list.append(episode_bitrate / episode_step_cnt)
                episode_delay_list.append(episode_delay / episode_step_cnt)
                episode_buffer_size_list.append(episode_buffer_size / episode_step_cnt)
                episode_buffer_sleep_time_list.append(episode_buffer_sleep_time / episode_step_cnt / 1000)
                episode_rebuffer_list.append(episode_rebuffer / episode_step_cnt)
                episode_choose_video_chunk_size_list.append(episode_choose_video_chunk_size / episode_step_cnt)
                episode_choose_video_chunk_size_per_time_list.append(episode_choose_video_chunk_size_per_time / episode_step_cnt)
                episode_bitrate_reward_list.append(episode_bitrate_reward / episode_step_cnt)
                episode_rebuffer_time_reward_list.append(episode_rebuffer_time_reward / episode_step_cnt)
                episode_smooth_penalty_reward_list.append(episode_smooth_penalty_reward / episode_step_cnt)

                if len(episode_reward_list) > 50:
                    logs = {
                        "SystemInfo/Episode BitRate": np.mean(episode_bitrate_list[-50:]),
                        "SystemInfo/Episode Delay": np.mean(episode_delay_list[-50:]),
                        "SystemInfo/Episode Buffer Size": np.mean(episode_buffer_size_list[-50:]),
                        "SystemInfo/Episode Sleep Time": np.mean(episode_buffer_sleep_time_list[-50:]),
                        "SystemInfo/Episode Rebuffer": np.mean(episode_rebuffer_list[-50:]),
                        "SystemInfo/Episode Choose Video Chunk Size": np.mean(episode_choose_video_chunk_size_list[-50:]),
                        "SystemInfo/Episode Choose Video Chunk Size Per Time": np.mean(episode_choose_video_chunk_size_per_time_list[-50:]),
                        "Reward/Episode Mean Return": np.mean(episode_reward_list[-50:]),
                        "Reward/Episode BitRate Reward": np.mean(episode_bitrate_reward_list[-50:]),
                        "Reward/Episode Rebuffer Reward": np.mean(episode_rebuffer_time_reward_list[-50:]),
                        "Reward/Episode Smooth Reward": np.mean(episode_smooth_penalty_reward_list[-50:]),
                    }
                    wandb.log(logs, step=global_step)

                episode_step_cnt = 0
                episode_reward = 0
                episode_bitrate = 0
                episode_delay = 0
                episode_buffer_size = 0
                episode_buffer_sleep_time = 0
                episode_rebuffer = 0
                episode_choose_video_chunk_size = 0
                episode_choose_video_chunk_size_per_time = 0
                episode_bitrate_reward = 0
                episode_rebuffer_time_reward = 0
                episode_smooth_penalty_reward = 0

        ## ---------- Weight Magnitude---------- ##
        weight_logs = {}
        if isinstance(policy_fun, MLPPolicy):
            weight_logs.update(WeightMagnitudeStatistic(copy.deepcopy(policy_fun.moe[0].moe_net), "policy"))
        elif isinstance(policy_fun, MoEPolicy) or isinstance(policy_fun, SparseMoEPolicy):
            weight_logs.update(WeightMagnitudeStatistic(copy.deepcopy(policy_fun.moe), "policy"))
        elif isinstance(policy_fun, DistillMoEPolicy):
            weight_logs.update(WeightMagnitudeStatistic(copy.deepcopy(policy_fun.moe), "policy"))
        else:
            raise ValueError("Not Support Policy Type {}!".format(type(policy_fun)))

        if isinstance(value_fun, MLPVF):
            weight_logs.update(WeightMagnitudeStatistic(copy.deepcopy(value_fun.moe[0].moe_net), "value"))
        elif isinstance(value_fun, MoEVF) or isinstance(value_fun, SparseMoEVF):
            weight_logs.update(WeightMagnitudeStatistic(copy.deepcopy(value_fun.moe), "value"))
        elif isinstance(value_fun, DistillMoEVF):
            weight_logs.update(WeightMagnitudeStatistic(copy.deepcopy(value_fun.moe), "value"))
        else:
            raise ValueError("Not Support Value Type {}!".format(type(value_fun)))

        wandb.log(weight_logs, step=global_step)

        ## ---------- Rank ---------- ##
        rank_logs = RankStatistic(short_term_feature_activity)
        wandb.log(rank_logs, step=global_step)

        ## ---------- Dormant Units ---------- ##
        dormant_logs = DormantStatistic(short_term_feature_activity, cfg.redo_tau)
        wandb.log(dormant_logs, step=global_step)

        ## Expert Router Ratio
        route_logs = RouterStatistic(short_term_feature_activity)
        wandb.log(route_logs, step=global_step)

        ## ---------- Reset short_term_feature_activity Dict And Previous Model ---------- ##
        short_term_feature_activity = {}

        ## ---------- Update Algorithm ---------- ##
        with torch.no_grad():
            next_value = value_fun(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy = policy_fun(b_obs[mb_inds], b_actions[mb_inds])
                newvalue = value_fun(b_obs[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                # distill loss
                if cfg.value_type == "dmoe" and cfg.policy_type == "dmoe":
                    # with torch.no_grad():
                    #     policy_embedding = policy_fun.get_moe_input(b_obs[mb_inds])
                    #     value_embedding = value_fun.get_moe_input(b_obs[mb_inds])
                    # policy_distill_loss = policy_fun.moe.distill_student(policy_embedding)
                    # value_distill_loss = value_fun.moe.distill_student(value_embedding)
                    # distill_loss = policy_distill_loss + value_distill_loss

                    # optimizer.zero_grad()
                    # distill_loss.backward()
                    # optimizer.step()

                    total_loss = loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    # if epoch % 2 == 0:
                    policy_fun.moe.add_expert_noise()
                    value_fun.moe.add_expert_noise()

                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        ## ---------- Loss Info ---------- ##
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        logs = {
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "charts/SPS": int(global_step / (time.time() - start_time)),
        }
        wandb.log(logs, step=global_step)

        ## --------- Change Env --------- ##
        change_times = len(env_content_list)
        if iteration % int(cfg.num_iterations / change_times) == 0 and iteration < cfg.num_iterations:
            print(
                "Change Iteration : {}; global step {}; bandwidth model {}".format(
                    iteration, global_step, env_content_list[int(iteration // (int(cfg.num_iterations / change_times)))]
                )
            )
            envs = AdaptiveStreamingEnv(seed_num=cfg.seed, content_model=env_content_list[int(iteration // (int(cfg.num_iterations / change_times)))])
            next_obs = envs.reset()
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(cfg.num_envs).to(device)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
