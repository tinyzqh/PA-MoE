import numpy as np
import matplotlib.pyplot as plt
from src.plots.plot_fun.exp_id import ExperimentId, ExperimentId_2, ExperimentId_3, ExperimentId_4, ExperimentId_5
from src.plots.wandb_data_fetcher import fetch_multiple_runs_data

colors = {"MoE": "red", "Ours": "blue"}
formats = {"MoE": "^--", "Ours": "s-."}


def calculate_iqm_list(reward_list):
    iqm_values = []
    for rewards in reward_list:
        sorted_rewards = np.sort(rewards)
        q1_index, q3_index = int(0.25 * len(sorted_rewards)), int(0.75 * len(sorted_rewards))
        iqm = np.mean(sorted_rewards[q1_index:q3_index])
        iqm_values.append(iqm)
    return np.mean(iqm_values), np.std(iqm_values) / np.sqrt(len(iqm_values))


def calculate_iqm_dict(reward_dict):
    # 用于存储 IQM 及其误差范围
    x_values = []
    iqm_values = []
    iqm_errors = []

    # 计算 IQM 和误差范围
    for x_value, rewards in reward_dict.items():
        iqm, error = calculate_iqm_list(rewards)
        x_values.append(x_value)
        iqm_values.append(iqm)
        iqm_errors.append(error)

    return x_values, iqm_values, iqm_errors


def plot_ablation_gradient_reward(ablation_lr_data, y_axis_name, x_axis_name, save_path):
    fig = plt.figure(figsize=(8, 6))

    for k, v in ablation_lr_data.items():
        x_values, iqm_values, iqm_errors = calculate_iqm_dict(v)
        print(k, x_values, iqm_values, iqm_errors)

        plt.errorbar(x_values, iqm_values, yerr=iqm_errors, fmt=formats[k], color=colors[k], elinewidth=1.5, capsize=5, capthick=1.5, markersize=12, linewidth=2, label=k)

    fig.axes[0].spines["right"].set_visible(False)
    fig.axes[0].spines["top"].set_visible(False)
    fig.axes[0].spines["left"].set_linewidth(2)
    fig.axes[0].spines["bottom"].set_linewidth(2)

    # 设置x轴为对数刻度
    plt.xscale("log", base=10)
    # fig.axes[0].set_xlim(0.0003, 0.01)
    fig.axes[0].spines["left"].set_position(("outward", 10))
    fig.axes[0].spines["bottom"].set_position(("outward", 10))

    # plt.rcParams['axes.formatter.useoffset'] = False

    fig.axes[0].tick_params(axis="both", labelsize=22)
    fig.axes[0].set_ylabel(y_axis_name, fontsize=18, fontweight="bold")
    fig.axes[0].set_xlabel(x_axis_name, fontsize=18, fontweight="bold")
    plt.legend(fontsize=18)
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")


moe_noise_0_001 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Reward/Episode Mean Return")

pamoe_noise_0_001 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Reward/Episode Mean Return")
pamoe_noise_0_0005 = fetch_multiple_runs_data(run_wandb_path=ExperimentId_2.wandb_path, run_ids=ExperimentId_2.DMOE_ID, metric_name="Reward/Episode Mean Return")
pamoe_noise_0_005 = fetch_multiple_runs_data(run_wandb_path=ExperimentId_3.wandb_path, run_ids=ExperimentId_3.DMOE_ID, metric_name="Reward/Episode Mean Return")

pamoe_lr_0_005 = fetch_multiple_runs_data(run_wandb_path=ExperimentId_5.wandb_path, run_ids=ExperimentId_5.DMOE_ID, metric_name="Reward/Episode Mean Return")
pamoe_lr_0_0001 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Reward/Episode Mean Return")
pamoe_lr_0_00005 = fetch_multiple_runs_data(run_wandb_path=ExperimentId_4.wandb_path, run_ids=ExperimentId_4.DMOE_ID, metric_name="Reward/Episode Mean Return")

ablation_gradient_plot = {
    "MoE": {0.0005: moe_noise_0_001, 0.001: moe_noise_0_001, 0.005: moe_noise_0_001},
    "Ours": {0.0005: pamoe_noise_0_0005, 0.001: pamoe_noise_0_001, 0.005: pamoe_noise_0_005},
}

# plot_ablation_gradient_reward(ablation_gradient_plot, y_axis_name="QoE", x_axis_name="Noise Injection Threshold", save_path="noise_threshold.pdf")

ablation_lr_plot = {
    "MoE": {0.005: moe_noise_0_001, 0.0001: moe_noise_0_001, 0.00005: moe_noise_0_001},
    "Ours": {0.005: pamoe_lr_0_005, 0.0001: pamoe_lr_0_0001, 0.00005: pamoe_lr_0_00005},
}
plot_ablation_gradient_reward(ablation_lr_plot, y_axis_name="QoE", x_axis_name="Learning Rate", save_path="lr_threshold.pdf")
