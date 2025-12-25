import seaborn as sns

from src.plots.plot_fun.exp_id import ExperimentId
from src.plots.wandb_data_fetcher import fetch_multiple_runs_data
from src.plots.plot_fun.plot_1_4 import plot_1_4


COLORS = ["#4DAE48", "#0E986F", "#796CAD", "#377EB9", "#D65813", "#974F9F"]
LineStyle = ["-", "--", "-.", ":"]


moe_expert_0_policy_0_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Weights/expert_0_policy_0_layer_mean"
)
moe_expert_0_policy_1_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Weights/expert_0_policy_1_layer_mean"
)
moe_expert_0_value_0_layer_weight = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Weights/expert_0_value_0_layer_mean")
moe_expert_0_value_1_layer_weight = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Weights/expert_0_value_1_layer_mean")

smoe_expert_0_policy_0_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Weights/expert_0_policy_0_layer_mean"
)
smoe_expert_0_policy_1_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Weights/expert_0_policy_1_layer_mean"
)
smoe_expert_0_value_0_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Weights/expert_0_value_0_layer_mean"
)
smoe_expert_0_value_1_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Weights/expert_0_value_1_layer_mean"
)

pamoe_expert_0_policy_0_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Weights/expert_0_policy_0_layer_mean"
)
pamoe_expert_0_policy_1_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Weights/expert_0_policy_1_layer_mean"
)
pamoe_expert_0_value_0_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Weights/expert_0_value_0_layer_mean"
)
pamoe_expert_0_value_1_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Weights/expert_0_value_1_layer_mean"
)

Expert_0_Weights_Mean = {
    "Expert-0-Policy-0-Layer-Weights-Mean": {
        "MoE": (moe_expert_0_policy_0_layer_weight, COLORS[0], LineStyle[0], "Step", "Weight Mean", "Expert-0-Policy-0-W"),
        "SMoE": (smoe_expert_0_policy_0_layer_weight, COLORS[1], LineStyle[1], "Step", "Weight Mean", "Expert-0-Policy-0-W"),
        "PA-MoE": (pamoe_expert_0_policy_0_layer_weight, COLORS[2], LineStyle[2], "Step", "Weight Mean", "Expert-0-Policy-0-W"),
    },
    "Expert-0-Policy-1-Layer-Weights-Mean": {
        "MoE": (moe_expert_0_policy_1_layer_weight, COLORS[0], LineStyle[0], "Step", "Weight Mean", "Expert-0-Policy-1-W"),
        "SMoE": (smoe_expert_0_policy_1_layer_weight, COLORS[1], LineStyle[1], "Step", "Weight Mean", "Expert-0-Policy-1-W"),
        "PA-MoE": (pamoe_expert_0_policy_1_layer_weight, COLORS[2], LineStyle[2], "Step", "Weight Mean", "Expert-0-Policy-1-W"),
    },
    "Expert-0-Value-0-Layer-Weights-Mean": {
        "MoE": (moe_expert_0_value_0_layer_weight, COLORS[0], LineStyle[0], "Step", "Weight Mean", "Expert-0-Value-0-W"),
        "SMoE": (smoe_expert_0_value_0_layer_weight, COLORS[1], LineStyle[1], "Step", "Weight Mean", "Expert-0-Value-0-W"),
        "PA-MoE": (pamoe_expert_0_value_0_layer_weight, COLORS[2], LineStyle[2], "Step", "Weight Mean", "Expert-0-Value-0-W"),
    },
    "Expert-0-Value-1-Layer-Weights-Mean": {
        "MoE": (moe_expert_0_value_1_layer_weight, COLORS[0], LineStyle[0], "Step", "Weight Mean", "Expert-0-Value-1-W"),
        "SMoE": (smoe_expert_0_value_1_layer_weight, COLORS[1], LineStyle[1], "Step", "Weight Mean", "Expert-0-Value-1-W"),
        "PA-MoE": (pamoe_expert_0_value_1_layer_weight, COLORS[2], LineStyle[2], "Step", "Weight Mean", "Expert-0-Value-1-W"),
    },
}

plot_1_4(Expert_0_Weights_Mean, "expert_0_weights_mean.pdf")


moe_expert_1_policy_0_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Weights/expert_1_policy_0_layer_mean"
)
moe_expert_1_policy_1_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Weights/expert_1_policy_1_layer_mean"
)
moe_expert_1_value_0_layer_weight = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Weights/expert_1_value_0_layer_mean")
moe_expert_1_value_1_layer_weight = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Weights/expert_1_value_1_layer_mean")

smoe_expert_1_policy_0_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Weights/expert_1_policy_0_layer_mean"
)
smoe_expert_1_policy_1_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Weights/expert_1_policy_1_layer_mean"
)
smoe_expert_1_value_0_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Weights/expert_1_value_0_layer_mean"
)
smoe_expert_1_value_1_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Weights/expert_1_value_1_layer_mean"
)

pamoe_expert_1_policy_0_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Weights/expert_1_policy_0_layer_mean"
)
pamoe_expert_1_policy_1_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Weights/expert_1_policy_1_layer_mean"
)
pamoe_expert_1_value_0_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Weights/expert_1_value_0_layer_mean"
)
pamoe_expert_1_value_1_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Weights/expert_1_value_1_layer_mean"
)

Expert_1_Weights_Mean = {
    "Expert-1-Policy-0-Layer-Weights-Mean": {
        "MoE": (moe_expert_1_policy_0_layer_weight, COLORS[0], LineStyle[0], "Step", "Weight Mean", "Expert-1-Policy-0-W"),
        "SMoE": (smoe_expert_1_policy_0_layer_weight, COLORS[1], LineStyle[1], "Step", "Weight Mean", "Expert-1-Policy-0-W"),
        "PA-MoE": (pamoe_expert_1_policy_0_layer_weight, COLORS[2], LineStyle[2], "Step", "Weight Mean", "Expert-1-Policy-0-W"),
    },
    "Expert-1-Policy-1-Layer-Weights-Mean": {
        "MoE": (moe_expert_1_policy_1_layer_weight, COLORS[0], LineStyle[0], "Step", "Weight Mean", "Expert-1-Policy-1-W"),
        "SMoE": (smoe_expert_1_policy_1_layer_weight, COLORS[1], LineStyle[1], "Step", "Weight Mean", "Expert-1-Policy-1-W"),
        "PA-MoE": (pamoe_expert_1_policy_1_layer_weight, COLORS[2], LineStyle[2], "Step", "Weight Mean", "Expert-1-Policy-1-W"),
    },
    "Expert-1-Value-0-Layer-Weights-Mean": {
        "MoE": (moe_expert_1_value_0_layer_weight, COLORS[0], LineStyle[0], "Step", "Weight Mean", "Expert-1-Value-0-W"),
        "SMoE": (smoe_expert_1_value_0_layer_weight, COLORS[1], LineStyle[1], "Step", "Weight Mean", "Expert-1-Value-0-W"),
        "PA-MoE": (pamoe_expert_1_value_0_layer_weight, COLORS[2], LineStyle[2], "Step", "Weight Mean", "Expert-1-Value-0-W"),
    },
    "Expert-1-Value-1-Layer-Weights-Mean": {
        "MoE": (moe_expert_1_value_1_layer_weight, COLORS[0], LineStyle[0], "Step", "Weight Mean", "Expert-1-Value-1-W"),
        "SMoE": (smoe_expert_1_value_1_layer_weight, COLORS[1], LineStyle[1], "Step", "Weight Mean", "Expert-1-Value-1-W"),
        "PA-MoE": (pamoe_expert_1_value_1_layer_weight, COLORS[2], LineStyle[2], "Step", "Weight Mean", "Expert-1-Value-1-W"),
    },
}

plot_1_4(Expert_1_Weights_Mean, "expert_1_weights_mean.pdf")


moe_expert_2_policy_0_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Weights/expert_2_policy_0_layer_mean"
)
moe_expert_2_policy_1_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Weights/expert_2_policy_1_layer_mean"
)
moe_expert_2_value_0_layer_weight = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Weights/expert_2_value_0_layer_mean")
moe_expert_2_value_1_layer_weight = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Weights/expert_2_value_1_layer_mean")

smoe_expert_2_policy_0_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Weights/expert_2_policy_0_layer_mean"
)
smoe_expert_2_policy_1_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Weights/expert_2_policy_1_layer_mean"
)
smoe_expert_2_value_0_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Weights/expert_2_value_0_layer_mean"
)
smoe_expert_2_value_1_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Weights/expert_2_value_1_layer_mean"
)

pamoe_expert_2_policy_0_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Weights/expert_2_policy_0_layer_mean"
)
pamoe_expert_2_policy_1_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Weights/expert_2_policy_1_layer_mean"
)
pamoe_expert_2_value_0_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Weights/expert_2_value_0_layer_mean"
)
pamoe_expert_2_value_1_layer_weight = fetch_multiple_runs_data(
    run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Weights/expert_2_value_1_layer_mean"
)

Expert_2_Weights_Mean = {
    "Expert-1-Policy-0-Layer-Weights-Mean": {
        "MoE": (moe_expert_2_policy_0_layer_weight, COLORS[0], LineStyle[0], "Step", "Weight Mean", "Expert-2-Policy-0-W"),
        "SMoE": (smoe_expert_2_policy_0_layer_weight, COLORS[1], LineStyle[1], "Step", "Weight Mean", "Expert-2-Policy-0-W"),
        "PA-MoE": (pamoe_expert_2_policy_0_layer_weight, COLORS[2], LineStyle[2], "Step", "Weight Mean", "Expert-2-Policy-0-W"),
    },
    "Expert-1-Policy-1-Layer-Weights-Mean": {
        "MoE": (moe_expert_2_policy_1_layer_weight, COLORS[0], LineStyle[0], "Step", "Weight Mean", "Expert-2-Policy-1-W"),
        "SMoE": (smoe_expert_2_policy_1_layer_weight, COLORS[1], LineStyle[1], "Step", "Weight Mean", "Expert-2-Policy-1-W"),
        "PA-MoE": (pamoe_expert_2_policy_1_layer_weight, COLORS[2], LineStyle[2], "Step", "Weight Mean", "Expert-2-Policy-1-W"),
    },
    "Expert-1-Value-0-Layer-Weights-Mean": {
        "MoE": (moe_expert_2_value_0_layer_weight, COLORS[0], LineStyle[0], "Step", "Weight Mean", "Expert-2-Value-0-W"),
        "SMoE": (smoe_expert_2_value_0_layer_weight, COLORS[1], LineStyle[1], "Step", "Weight Mean", "Expert-2-Value-0-W"),
        "PA-MoE": (pamoe_expert_2_value_0_layer_weight, COLORS[2], LineStyle[2], "Step", "Weight Mean", "Expert-2-Value-0-W"),
    },
    "Expert-1-Value-1-Layer-Weights-Mean": {
        "MoE": (moe_expert_2_value_1_layer_weight, COLORS[0], LineStyle[0], "Step", "Weight Mean", "Expert-2-Value-1-W"),
        "SMoE": (smoe_expert_2_value_1_layer_weight, COLORS[1], LineStyle[1], "Step", "Weight Mean", "Expert-2-Value-1-W"),
        "PA-MoE": (pamoe_expert_2_value_1_layer_weight, COLORS[2], LineStyle[2], "Step", "Weight Mean", "Expert-2-Value-1-W"),
    },
}

plot_1_4(Expert_2_Weights_Mean, "expert_2_weights_mean.pdf")
