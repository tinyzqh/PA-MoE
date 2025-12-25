from src.plots.plot_fun.plot_1_1 import plot_1_1
from src.plots.plot_fun.exp_id import ExperimentId
from src.plots.wandb_data_fetcher import fetch_multiple_runs_data
from src.plots.plot_fun.plot_bar import plot_iqm_comparison


def get_single_data(index_name: str, xlabel: str, ylabel: str, title: str):
    """
    index_name: The index name of the data to be plotted.
    """

    MLP = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MLP_ID, metric_name=index_name)
    MOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name=index_name)
    SMOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name=index_name)
    PAMOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name=index_name)

    # Colors are chosen to be distinguishable and print-friendly
    RETURNCOLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue  # Orange  # Green  # Red

    # Line styles - using subtle variations
    LineStyle = ["-", "--", "-.", ":"]  # Solid  # Dashed  # Dash-dot  # Dotted

    qoe_data = {
        "MLP": (MLP, RETURNCOLORS[0], LineStyle[0]),
        "MoE": (MOE, RETURNCOLORS[1], LineStyle[1], xlabel, ylabel, title),
        "SMoE": (SMOE, RETURNCOLORS[2], LineStyle[2], xlabel, ylabel, title),
        "PA-MoE": (PAMOE, RETURNCOLORS[3], LineStyle[3], xlabel, ylabel, title),
    }
    return qoe_data


qoe_data = get_single_data(index_name="Reward/Episode Mean Return", xlabel="Step", ylabel="Episode Mean Return", title="QoE")
qoe_smooth_data = get_single_data(index_name="Reward/Episode Smooth Reward", xlabel="Step", ylabel="Smooth Return", title="Smooth")
qoe_rebuffer_data = get_single_data(index_name="Reward/Episode Rebuffer Reward", xlabel="Step", ylabel="Rebuffer Return", title="Rebuffer")
qoe_bitrate_data = get_single_data(index_name="Reward/Episode BitRate Reward", xlabel="Step", ylabel="BitRate Return", title="BitRate")


# plot_1_1(qoe_data, "QoE.pdf")
# plot_1_1(qoe_smooth_data, "QoE-Smooth.pdf")
# plot_1_1(qoe_bitrate_data, "QoE-BitRate.pdf")
# plot_1_1(qoe_rebuffer_data, "QoE-Rebuffer.pdf")


reward_sequences = {
    "QoE": {"MLP": list(qoe_data["MLP"][0]), "MoE": list(qoe_data["MoE"][0]), "SMoE": list(qoe_data["SMoE"][0]), "PA-MoE": list(qoe_data["PA-MoE"][0])},
    "BitRate": {
        "MLP": list(qoe_bitrate_data["MLP"][0]),
        "MoE": list(qoe_bitrate_data["MoE"][0]),
        "SMoE": list(qoe_bitrate_data["SMoE"][0]),
        "PA-MoE": list(qoe_bitrate_data["PA-MoE"][0]),
    },
    "Rebuffer": {
        "MLP": list(qoe_rebuffer_data["MLP"][0]),
        "MoE": list(qoe_rebuffer_data["MoE"][0]),
        "SMoE": list(qoe_rebuffer_data["SMoE"][0]),
        "PA-MoE": list(qoe_rebuffer_data["PA-MoE"][0]),
    },
    "Smooth": {
        "MLP": list(qoe_smooth_data["MLP"][0]),
        "MoE": list(qoe_smooth_data["MoE"][0]),
        "SMoE": list(qoe_smooth_data["SMoE"][0]),
        "PA-MoE": list(qoe_smooth_data["PA-MoE"][0]),
    },
}

plot_iqm_comparison(reward_sequences, "QoE-Bar.pdf")
