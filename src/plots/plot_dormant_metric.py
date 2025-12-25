from src.plots.plot_fun.plot_1_1 import plot_1_1
from src.plots.plot_fun.plot_1_4 import plot_1_4

from src.plots.plot_fun.exp_id import ExperimentId
from src.plots.wandb_data_fetcher import fetch_multiple_runs_data


MLP = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MLP_ID, metric_name="Dorman/total_dorman_fraction", length_mode="truncate")
MOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Dorman/total_dorman_fraction", length_mode="truncate")
SMOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Dorman/total_dorman_fraction", length_mode="truncate")
DMOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Dorman/total_dorman_fraction", length_mode="truncate")


RETURNCOLORS = ["#FFCC66", "#882255", "#EE7733", "#FFAE42", "#DD8844", "#CC3311"]
LineStyle = ["--", "-.", "-", "--"]


# --------- ToTal Dormant Fraction --------- #

dormant_total_data = {
    # "mlp": (MLP, RETURNCOLORS[0], LineStyle[0], "Step", "Rate", "Total Dormant Fraction"),
    "MoE": (MOE, RETURNCOLORS[1], LineStyle[1], "Step", "Rate (%)", "Total Dormant Fraction"),
    "SMoE": (SMOE, RETURNCOLORS[2], LineStyle[2], "Step", "Rate (%)", "Total Dormant Fraction"),
    "PA-MoE": (DMOE, RETURNCOLORS[3], LineStyle[3], "Step", "Rate (%)", "Total Dormant Fraction"),
}

plot_1_1(dormant_total_data, "dormant_total.pdf")


# --------- MOE --------- #

moe_expert_0_policy_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Dorman/expert_0_policy_0_layer_0.5_fraction")
moe_expert_0_policy_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Dorman/expert_0_policy_1_layer_0.5_fraction")
moe_expert_0_value_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Dorman/expert_0_value_0_layer_0.5_fraction")
moe_expert_0_value_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Dorman/expert_0_value_1_layer_0.5_fraction")

moe_expert_1_policy_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Dorman/expert_1_policy_0_layer_0.5_fraction")
moe_expert_1_policy_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Dorman/expert_1_policy_1_layer_0.5_fraction")
moe_expert_1_value_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Dorman/expert_1_value_0_layer_0.5_fraction")
moe_expert_1_value_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Dorman/expert_1_value_1_layer_0.5_fraction")

moe_expert_2_policy_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Dorman/expert_2_policy_0_layer_0.5_fraction")
moe_expert_2_policy_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Dorman/expert_2_policy_1_layer_0.5_fraction")
moe_expert_2_value_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Dorman/expert_2_value_0_layer_0.5_fraction")
moe_expert_2_value_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="Dorman/expert_2_value_1_layer_0.5_fraction")


# --------- SMOE --------- #

smoe_expert_0_policy_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Dorman/expert_0_policy_0_layer_0.5_fraction")
smoe_expert_0_policy_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Dorman/expert_0_policy_1_layer_0.5_fraction")
smoe_expert_0_value_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Dorman/expert_0_value_0_layer_0.5_fraction")
smoe_expert_0_value_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Dorman/expert_0_value_1_layer_0.5_fraction")

smoe_expert_1_policy_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Dorman/expert_1_policy_0_layer_0.5_fraction")
smoe_expert_1_policy_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Dorman/expert_1_policy_1_layer_0.5_fraction")
smoe_expert_1_value_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Dorman/expert_1_value_0_layer_0.5_fraction")
smoe_expert_1_value_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Dorman/expert_1_value_1_layer_0.5_fraction")

smoe_expert_2_policy_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Dorman/expert_2_policy_0_layer_0.5_fraction")
smoe_expert_2_policy_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Dorman/expert_2_policy_1_layer_0.5_fraction")
smoe_expert_2_value_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Dorman/expert_2_value_0_layer_0.5_fraction")
smoe_expert_2_value_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="Dorman/expert_2_value_1_layer_0.5_fraction")


# --------- DMOE --------- #

dmoe_expert_0_policy_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Dorman/expert_0_policy_0_layer_0.5_fraction")
dmoe_expert_0_policy_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Dorman/expert_0_policy_1_layer_0.5_fraction")
dmoe_expert_0_value_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Dorman/expert_0_value_0_layer_0.5_fraction")
dmoe_expert_0_value_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Dorman/expert_0_value_1_layer_0.5_fraction")

dmoe_expert_1_policy_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Dorman/expert_1_policy_0_layer_0.5_fraction")
dmoe_expert_1_policy_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Dorman/expert_1_policy_1_layer_0.5_fraction")
dmoe_expert_1_value_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Dorman/expert_1_value_0_layer_0.5_fraction")
dmoe_expert_1_value_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Dorman/expert_1_value_1_layer_0.5_fraction")

dmoe_expert_2_policy_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Dorman/expert_2_policy_0_layer_0.5_fraction")
dmoe_expert_2_policy_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Dorman/expert_2_policy_1_layer_0.5_fraction")
dmoe_expert_2_value_0 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Dorman/expert_2_value_0_layer_0.5_fraction")
dmoe_expert_2_value_1 = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="Dorman/expert_2_value_1_layer_0.5_fraction")

Expert_0_Dormant = {
    "Policy-0-Layer-Dormant": {
        "MoE": (moe_expert_0_policy_0, RETURNCOLORS[0], LineStyle[0], "Step", "Rate (%)", "Expert-0-Policy-0-LD"),
        "SMoE": (smoe_expert_0_policy_0, RETURNCOLORS[1], LineStyle[1], "Step", "Rate (%)", "Expert-0-Policy-0-LD"),
        "PA-MoE": (dmoe_expert_0_policy_0, RETURNCOLORS[2], LineStyle[2], "Step", "Rate (%)", "Expert-0-Policy-0-LD"),
    },
    "Policy-1-Layer-Dormant": {
        "MoE": (moe_expert_0_policy_1, RETURNCOLORS[0], LineStyle[0], "Step", "Rate (%)", "Expert-0-Policy-1-LD"),
        "SMoE": (smoe_expert_0_policy_1, RETURNCOLORS[1], LineStyle[1], "Step", "Rate (%)", "Expert-0-Policy-1-LD"),
        "PA-MoE": (dmoe_expert_0_policy_1, RETURNCOLORS[2], LineStyle[2], "Step", "Rate (%)", "Expert-0-Policy-1-LD"),
    },
    "Value-0-Layer-Dormant": {
        "MoE": (moe_expert_0_value_0, RETURNCOLORS[0], LineStyle[0], "Step", "Rate (%)", "Expert-0-Value-0-LD"),
        "SMoE": (smoe_expert_0_value_0, RETURNCOLORS[1], LineStyle[1], "Step", "Rate (%)", "Expert-0-Value-0-LD"),
        "PA-MoE": (dmoe_expert_0_value_0, RETURNCOLORS[2], LineStyle[2], "Step", "Rate (%)", "Expert-0-Value-0-LD"),
    },
    "Value-1-Layer-Dormant": {
        "MoE": (moe_expert_0_value_1, RETURNCOLORS[0], LineStyle[0], "Step", "Rate (%)", "Expert-0-Value-1-LD"),
        "SMoE": (smoe_expert_0_value_1, RETURNCOLORS[1], LineStyle[1], "Step", "Rate (%)", "Expert-0-Value-1-LD"),
        "PA-MoE": (dmoe_expert_0_value_1, RETURNCOLORS[2], LineStyle[2], "Step", "Rate (%)", "Expert-0-Value-1-LD"),
    },
}

plot_1_4(Expert_0_Dormant, "expert_0_dormant.pdf")

Expert_1_Dormant = {
    "Policy-0-Layer-Dormant": {
        "MoE": (moe_expert_1_policy_0, RETURNCOLORS[0], LineStyle[0], "Step", "Rate (%)", "Expert-1-Policy-0-LD"),
        "SMoE": (smoe_expert_1_policy_0, RETURNCOLORS[1], LineStyle[1], "Step", "Rate (%)", "Expert-1-Policy-0-LD"),
        "PA-MoE": (dmoe_expert_1_policy_0, RETURNCOLORS[2], LineStyle[2], "Step", "Rate (%)", "Expert-1-Policy-0-LD"),
    },
    "Policy-1-Layer-Dormant": {
        "MoE": (moe_expert_1_policy_1, RETURNCOLORS[0], LineStyle[0], "Step", "Rate (%)", "Expert-1-Policy-1-LD"),
        "SMoE": (smoe_expert_1_policy_1, RETURNCOLORS[1], LineStyle[1], "Step", "Rate (%)", "Expert-1-Policy-1-LD"),
        "PA-MoE": (dmoe_expert_1_policy_1, RETURNCOLORS[2], LineStyle[2], "Step", "Rate (%)", "Expert-1-Policy-1-LD"),
    },
    "Value-0-Layer-Dormant": {
        "MoE": (moe_expert_1_value_0, RETURNCOLORS[0], LineStyle[0], "Step", "Rate (%)", "Expert-1-Value-0-LD"),
        "SMoE": (smoe_expert_1_value_0, RETURNCOLORS[1], LineStyle[1], "Step", "Rate (%)", "Expert-1-Value-0-LD"),
        "PA-MoE": (dmoe_expert_1_value_0, RETURNCOLORS[2], LineStyle[2], "Step", "Rate (%)", "Expert-1-Value-0-LD"),
    },
    "Value-1-Layer-Dormant": {
        "MoE": (moe_expert_1_value_1, RETURNCOLORS[0], LineStyle[0], "Step", "Rate (%)", "Expert-1-Value-1-LD"),
        "SMoE": (smoe_expert_1_value_1, RETURNCOLORS[1], LineStyle[1], "Step", "Rate (%)", "Expert-1-Value-1-LD"),
        "PA-MoE": (dmoe_expert_1_value_1, RETURNCOLORS[2], LineStyle[2], "Step", "Rate (%)", "Expert-1-Value-1-LD"),
    },
}

plot_1_4(Expert_1_Dormant, "expert_1_dormant.pdf")

Expert_2_Dormant = {
    "Policy-0-Layer-Dormant": {
        "MoE": (moe_expert_2_policy_0, RETURNCOLORS[0], LineStyle[0], "Step", "Rate (%)", "Expert-2-Policy-0-LD"),
        "SMoE": (smoe_expert_2_policy_0, RETURNCOLORS[1], LineStyle[1], "Step", "Rate (%)", "Expert-2-Policy-0-LD"),
        "PA-MoE": (dmoe_expert_2_policy_0, RETURNCOLORS[2], LineStyle[2], "Step", "Rate (%)", "Expert-2-Policy-0-LD"),
    },
    "Policy-1-Layer-Dormant": {
        "MoE": (moe_expert_2_policy_1, RETURNCOLORS[0], LineStyle[0], "Step", "Rate (%)", "Expert-2-Policy-1-LD"),
        "SMoE": (smoe_expert_2_policy_1, RETURNCOLORS[1], LineStyle[1], "Step", "Rate (%)", "Expert-2-Policy-1-LD"),
        "PA-MoE": (dmoe_expert_2_policy_1, RETURNCOLORS[2], LineStyle[2], "Step", "Rate (%)", "Expert-2-Policy-1-LD"),
    },
    "Value-0-Layer-Dormant": {
        "MoE": (moe_expert_2_value_0, RETURNCOLORS[0], LineStyle[0], "Step", "Rate (%)", "Expert-2-Value-0-LD"),
        "SMoE": (smoe_expert_2_value_0, RETURNCOLORS[1], LineStyle[1], "Step", "Rate (%)", "Expert-2-Value-0-LD"),
        "PA-MoE": (dmoe_expert_2_value_0, RETURNCOLORS[2], LineStyle[2], "Step", "Rate (%)", "Expert-2-Value-0-LD"),
    },
    "Value-1-Layer-Dormant": {
        "MoE": (moe_expert_2_value_1, RETURNCOLORS[0], LineStyle[0], "Step", "Rate (%)", "Expert-2-Value-1-LD"),
        "SMoE": (smoe_expert_2_value_1, RETURNCOLORS[1], LineStyle[1], "Step", "Rate (%)", "Expert-2-Value-1-LD"),
        "PA-MoE": (dmoe_expert_2_value_1, RETURNCOLORS[2], LineStyle[2], "Step", "Rate (%)", "Expert-2-Value-1-LD"),
    },
}

plot_1_4(Expert_2_Dormant, "expert_2_dormant.pdf")
