from src.plots.plot_fun.plot_1_1 import plot_1_1
from src.plots.plot_fun.exp_id import ExperimentId
from src.plots.wandb_data_fetcher import fetch_multiple_runs_data

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]  # Blue  # Red  # Green  # Purple

LineStyle = ["--", "-", "-.", ":"]

MLP = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MLP_ID, metric_name="SystemInfo/Episode BitRate", length_mode="truncate")
MLP_QOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MLP_ID, metric_name="Reward/Episode Mean Return", length_mode="truncate")

MOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="SystemInfo/Episode BitRate", length_mode="truncate")
MOE_QOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Reward/Episode Mean Return", length_mode="truncate")

SMOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="SystemInfo/Episode BitRate", length_mode="truncate")
SMOE_QOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Reward/Episode Mean Return", length_mode="truncate")

PAMOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="SystemInfo/Episode BitRate", length_mode="truncate")
PAMOE_QOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Reward/Episode Mean Return", length_mode="truncate")

MLP_5 = fetch_multiple_runs_data(
    run_wandb_path="tinyzqh-the-university-of-electro-communication/MOE-VS-Finall-18-Diverse-Cof/runs",
    run_ids=["i1pent7h", "2tibbrqy"],
    metric_name="SystemInfo/Episode BitRate",
    length_mode="truncate",
)
MOE_5 = fetch_multiple_runs_data(
    run_wandb_path="tinyzqh-the-university-of-electro-communication/MOE-VS-Finall-18-Diverse-Cof/runs",
    run_ids=["dm40ro8w", "tyd3l1c8"],
    metric_name="SystemInfo/Episode BitRate",
    length_mode="truncate",
)

MLP_7 = fetch_multiple_runs_data(
    run_wandb_path="tinyzqh-the-university-of-electro-communication/MOE-VS-Finall-18-Diverse-Cof/runs",
    run_ids=["4r3k8izr", "xkgphwkm"],
    metric_name="SystemInfo/Episode BitRate",
    length_mode="truncate",
)
MOE_7 = fetch_multiple_runs_data(
    run_wandb_path="tinyzqh-the-university-of-electro-communication/MOE-VS-Finall-18-Diverse-Cof/runs",
    run_ids=["ytdd5be6", "g313vii6"],
    metric_name="SystemInfo/Episode BitRate",
    length_mode="truncate",
)

question_data = {
    "MLP": (MLP, COLORS[0], LineStyle[0], "Step", "Action (Bitrate)", "Action Response to System Change"),
    "MoE": (MOE, COLORS[1], LineStyle[1], "Step", "Action (Bitrate)", "Action Response to System Change"),
}

question_qoe_data = {
    "MLP": (MLP_QOE, COLORS[0], LineStyle[0], "Step", "Episode Mean Return", "QoE Response to System Change"),
    "MoE": (MOE_QOE, COLORS[1], LineStyle[1], "Step", "Episode Mean Return", "QoE Response to System Change"),
    # "PA-MoE": (PAMOE_QOE, COLORS[2], LineStyle[2], "Step", "Episode Mean Return", "QoE Response to System Change"),
}

question_plus_data = {
    "MLP-5": (MLP_5, COLORS[0], LineStyle[0], "Step", "Action (Bitrate)", "Action Response to System Change"),
    "MLP-7": (MLP_7, COLORS[0], LineStyle[1], "Step", "Action (Bitrate)", "Action Response to System Change"),
    # "MLP-6": (MLP, COLORS[1], LineStyle[0], "Step", "Action (Bitrate)", "Action Response to System Change"),
    # "MoE-6": (MOE, COLORS[1], LineStyle[1], "Step", "Action (Bitrate)", "Action Response to System Change"),
    "MoE-5": (MOE_5, COLORS[3], LineStyle[0], "Step", "Action (Bitrate)", "Action Response to System Change"),
    "MoE-7": (MOE_7, COLORS[3], LineStyle[1], "Step", "Action (Bitrate)", "Action Response to System Change"),
}

answer_data = {
    # "MLP": (MLP, COLORS[0], LineStyle[0], "Step", "Action (Bitrate)", "Action Response to System Change"),
    "MoE": (MOE, COLORS[1], LineStyle[1], "Step", "Action (Bitrate)", "Action Response to System Change"),
    "SMoE": (SMOE, COLORS[2], LineStyle[2], "Step", "Action (Bitrate)", "Action Response to System Change"),
    "PA-MoE": (PAMOE, COLORS[3], LineStyle[3], "Step", "Action (Bitrate)", "Action Response to System Change"),
}

plot_1_1(question_data, "question.pdf", x_line_values=True, text_high=4.0)
plot_1_1(question_plus_data, "question_plus.pdf", x_line_values=True, text_high=4.0)
plot_1_1(question_qoe_data, "question_qoe.pdf", x_line_values=True, text_high=4.0)
plot_1_1(answer_data, "answer.pdf", x_line_values=True, text_high=4.5)
