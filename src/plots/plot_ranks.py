from src.plots.plot_fun.plot_1_4 import plot_1_4
from src.plots.plot_fun.exp_id import ExperimentId
from src.plots.wandb_data_fetcher import fetch_multiple_runs_data


COLORS = ["#779043", "#3b6291", "#943c39", "#624c7c", "#388498", "#bf7334", "#3f6899", "#9c403d", "#7d9847", "#675083", "#3b8ba1", "#c97937"]
LineStyle = ["-", "--", "-.", ":"]

moe_expert_0_policy_0_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Ranks/expert_0_policy_0_layer_rank")
moe_expert_0_policy_1_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Ranks/expert_0_policy_1_layer_rank")
moe_expert_0_value_0_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Ranks/expert_0_value_0_layer_rank")
moe_expert_0_value_1_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Ranks/expert_0_value_1_layer_rank")

smoe_expert_0_policy_0_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Ranks/expert_0_policy_0_layer_rank")
smoe_expert_0_policy_1_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Ranks/expert_0_policy_1_layer_rank")
smoe_expert_0_value_0_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Ranks/expert_0_value_0_layer_rank")
smoe_expert_0_value_1_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Ranks/expert_0_value_1_layer_rank")

pamoe_expert_0_policy_0_layer_rank = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Ranks/expert_0_policy_0_layer_rank"
)
pamoe_expert_0_policy_1_layer_rank = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Ranks/expert_0_policy_1_layer_rank"
)
pamoe_expert_0_value_0_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Ranks/expert_0_value_0_layer_rank")
pamoe_expert_0_value_1_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Ranks/expert_0_value_1_layer_rank")

Expert_0_Rank = {
    "Expert-0-Policy-0-Layer-Rank": {
        "MoE": (moe_expert_0_policy_0_layer_rank, COLORS[0], LineStyle[0], "Step", "Rank", "Expert-0-Policy-0-Rank"),
        "SMoE": (smoe_expert_0_policy_0_layer_rank, COLORS[1], LineStyle[1], "Step", "Rank", "Expert-0-Policy-0-Rank"),
        "PA-MoE": (pamoe_expert_0_policy_0_layer_rank, COLORS[2], LineStyle[2], "Step", "Rank", "Expert-0-Policy-0-Rank"),
    },
    "Expert-0-Policy-1-Layer-Rank": {
        "MoE": (moe_expert_0_policy_1_layer_rank, COLORS[0], LineStyle[0], "Step", "Rank", "Expert-0-Policy-1-Rank"),
        "SMoE": (smoe_expert_0_policy_1_layer_rank, COLORS[1], LineStyle[1], "Step", "Rank", "Expert-0-Policy-1-Rank"),
        "PA-MoE": (pamoe_expert_0_policy_1_layer_rank, COLORS[2], LineStyle[2], "Step", "Rank", "Expert-0-Policy-1-Rank"),
    },
    "Expert-0-Value-0-Layer-Rank": {
        "MoE": (moe_expert_0_value_0_layer_rank, COLORS[0], LineStyle[0], "Step", "Rank", "Expert-0-Value-0-Rank"),
        "SMoE": (smoe_expert_0_value_0_layer_rank, COLORS[1], LineStyle[1], "Step", "Rank", "Expert-0-Value-0-Rank"),
        "PA-MoE": (pamoe_expert_0_value_0_layer_rank, COLORS[2], LineStyle[2], "Step", "Rank", "Expert-0-Value-0-Rank"),
    },
    "Expert-0-Value-1-Layer-Rank": {
        "MoE": (moe_expert_0_value_1_layer_rank, COLORS[0], LineStyle[0], "Step", "Rank", "Expert-0-Value-1-Rank"),
        "SMoE": (smoe_expert_0_value_1_layer_rank, COLORS[1], LineStyle[1], "Step", "Rank", "Expert-0-Value-1-Rank"),
        "PA-MoE": (pamoe_expert_0_value_1_layer_rank, COLORS[2], LineStyle[2], "Step", "Rank", "Expert-0-Value-1-Rank"),
    },
}

plot_1_4(Expert_0_Rank, "expert_0_rank.pdf")


moe_expert_1_policy_0_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Ranks/expert_1_policy_0_layer_rank")
moe_expert_1_policy_1_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Ranks/expert_1_policy_1_layer_rank")
moe_expert_1_value_0_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Ranks/expert_1_value_0_layer_rank")
moe_expert_1_value_1_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Ranks/expert_1_value_1_layer_rank")

smoe_expert_1_policy_0_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Ranks/expert_1_policy_0_layer_rank")
smoe_expert_1_policy_1_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Ranks/expert_1_policy_1_layer_rank")
smoe_expert_1_value_0_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Ranks/expert_1_value_0_layer_rank")
smoe_expert_1_value_1_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Ranks/expert_1_value_1_layer_rank")

pamoe_expert_1_policy_0_layer_rank = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Ranks/expert_1_policy_0_layer_rank"
)
pamoe_expert_1_policy_1_layer_rank = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Ranks/expert_1_policy_1_layer_rank"
)
pamoe_expert_1_value_0_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Ranks/expert_1_value_0_layer_rank")
pamoe_expert_1_value_1_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Ranks/expert_1_value_1_layer_rank")

Expert_1_Rank = {
    "Expert-0-Policy-0-Layer-Rank": {
        "MoE": (moe_expert_1_policy_0_layer_rank, COLORS[0], LineStyle[0], "Step", "Rank", "Expert-1-Policy-0-Rank"),
        "SMoE": (smoe_expert_1_policy_0_layer_rank, COLORS[1], LineStyle[1], "Step", "Rank", "Expert-1-Policy-0-Rank"),
        "PA-MoE": (pamoe_expert_1_policy_0_layer_rank, COLORS[2], LineStyle[2], "Step", "Rank", "Expert-1-Policy-0-Rank"),
    },
    "Expert-0-Policy-1-Layer-Rank": {
        "MoE": (moe_expert_1_policy_1_layer_rank, COLORS[0], LineStyle[0], "Step", "Rank", "Expert-1-Policy-1-Rank"),
        "SMoE": (smoe_expert_1_policy_1_layer_rank, COLORS[1], LineStyle[1], "Step", "Rank", "Expert-1-Policy-1-Rank"),
        "PA-MoE": (pamoe_expert_1_policy_1_layer_rank, COLORS[2], LineStyle[2], "Step", "Rank", "Expert-1-Policy-1-Rank"),
    },
    "Expert-0-Value-0-Layer-Rank": {
        "MoE": (moe_expert_1_value_0_layer_rank, COLORS[0], LineStyle[0], "Step", "Rank", "Expert-1-Value-0-Rank"),
        "SMoE": (smoe_expert_1_value_0_layer_rank, COLORS[1], LineStyle[1], "Step", "Rank", "Expert-1-Value-0-Rank"),
        "PA-MoE": (pamoe_expert_1_value_0_layer_rank, COLORS[2], LineStyle[2], "Step", "Rank", "Expert-1-Value-0-Rank"),
    },
    "Expert-0-Value-1-Layer-Rank": {
        "MoE": (moe_expert_1_value_1_layer_rank, COLORS[0], LineStyle[0], "Step", "Rank", "Expert-1-Value-1-Rank"),
        "SMoE": (smoe_expert_1_value_1_layer_rank, COLORS[1], LineStyle[1], "Step", "Rank", "Expert-1-Value-1-Rank"),
        "PA-MoE": (pamoe_expert_1_value_1_layer_rank, COLORS[2], LineStyle[2], "Step", "Rank", "Expert-1-Value-1-Rank"),
    },
}

plot_1_4(Expert_1_Rank, "expert_1_rank.pdf")


moe_expert_2_policy_0_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Ranks/expert_2_policy_0_layer_rank")
moe_expert_2_policy_1_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Ranks/expert_2_policy_1_layer_rank")
moe_expert_2_value_0_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Ranks/expert_2_value_0_layer_rank")
moe_expert_2_value_1_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Ranks/expert_2_value_1_layer_rank")

smoe_expert_2_policy_0_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Ranks/expert_2_policy_0_layer_rank")
smoe_expert_2_policy_1_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Ranks/expert_2_policy_1_layer_rank")
smoe_expert_2_value_0_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Ranks/expert_2_value_0_layer_rank")
smoe_expert_2_value_1_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Ranks/expert_2_value_1_layer_rank")

pamoe_expert_2_policy_0_layer_rank = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Ranks/expert_2_policy_0_layer_rank"
)
pamoe_expert_2_policy_1_layer_rank = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Ranks/expert_2_policy_1_layer_rank"
)
pamoe_expert_2_value_0_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Ranks/expert_2_value_0_layer_rank")
pamoe_expert_2_value_1_layer_rank = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Ranks/expert_2_value_1_layer_rank")

Expert_2_Rank = {
    "Expert-0-Policy-0-Layer-Rank": {
        "MoE": (moe_expert_2_policy_0_layer_rank, COLORS[0], LineStyle[0], "Step", "Rank", "Expert-2-Policy-0-Rank"),
        "SMoE": (smoe_expert_2_policy_0_layer_rank, COLORS[1], LineStyle[1], "Step", "Rank", "Expert-2-Policy-0-Rank"),
        "PA-MoE": (pamoe_expert_2_policy_0_layer_rank, COLORS[2], LineStyle[2], "Step", "Rank", "Expert-2-Policy-0-Rank"),
    },
    "Expert-0-Policy-1-Layer-Rank": {
        "MoE": (moe_expert_2_policy_1_layer_rank, COLORS[0], LineStyle[0], "Step", "Rank", "Expert-2-Policy-1-Rank"),
        "SMoE": (smoe_expert_2_policy_1_layer_rank, COLORS[1], LineStyle[1], "Step", "Rank", "Expert-2-Policy-1-Rank"),
        "PA-MoE": (pamoe_expert_2_policy_1_layer_rank, COLORS[2], LineStyle[2], "Step", "Rank", "Expert-2-Policy-1-Rank"),
    },
    "Expert-0-Value-0-Layer-Rank": {
        "MoE": (moe_expert_2_value_0_layer_rank, COLORS[0], LineStyle[0], "Step", "Rank", "Expert-2-Value-0-Rank"),
        "SMoE": (smoe_expert_2_value_0_layer_rank, COLORS[1], LineStyle[1], "Step", "Rank", "Expert-2-Value-0-Rank"),
        "PA-MoE": (pamoe_expert_2_value_0_layer_rank, COLORS[2], LineStyle[2], "Step", "Rank", "Expert-2-Value-0-Rank"),
    },
    "Expert-0-Value-1-Layer-Rank": {
        "MoE": (moe_expert_2_value_1_layer_rank, COLORS[0], LineStyle[0], "Step", "Rank", "Expert-2-Value-1-Rank"),
        "SMoE": (smoe_expert_2_value_1_layer_rank, COLORS[1], LineStyle[1], "Step", "Rank", "Expert-2-Value-1-Rank"),
        "PA-MoE": (pamoe_expert_2_value_1_layer_rank, COLORS[2], LineStyle[2], "Step", "Rank", "Expert-2-Value-1-Rank"),
    },
}

plot_1_4(Expert_2_Rank, "expert_2_rank.pdf")
