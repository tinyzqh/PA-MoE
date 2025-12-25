import seaborn as sns

from src.plots.plot_fun.exp_id import ExperimentId
from src.plots.wandb_data_fetcher import fetch_multiple_runs_data
from src.plots.plot_fun.plot_1_3 import plot_1_3


RETURNCOLORS = sns.color_palette("deep")
LineStyle = ["-", "--", "-.", ":"]


moe_expert_0_policy_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="expert_0_policy_router")
moe_expert_1_policy_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="expert_1_policy_router")
moe_expert_2_policy_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="expert_2_policy_router")
moe_expert_0_value_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="expert_0_value_router")
moe_expert_1_value_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="expert_1_value_router")
moe_expert_2_value_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name="expert_2_value_router")


smoe_expert_0_policy_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="expert_0_policy_router")
smoe_expert_1_policy_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="expert_1_policy_router")
smoe_expert_2_policy_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="expert_2_policy_router")
smoe_expert_0_value_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="expert_0_value_router")
smoe_expert_1_value_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="expert_1_value_router")
smoe_expert_2_value_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name="expert_2_value_router")


dmoe_expert_0_policy_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="expert_0_policy_router")
dmoe_expert_1_policy_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="expert_1_policy_router")
dmoe_expert_2_policy_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="expert_2_policy_router")
dmoe_expert_0_value_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="expert_0_value_router")
dmoe_expert_1_value_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="expert_1_value_router")
dmoe_expert_2_value_router = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name="expert_2_value_router")


Router_Policy = {
    "Expert-Policy-0-Router": {
        "MoE": (moe_expert_0_policy_router, RETURNCOLORS[0], LineStyle[0], "Step", "Rate", "Policy Expert-0"),
        "SMoE": (smoe_expert_0_policy_router, RETURNCOLORS[1], LineStyle[1], "Step", "Rate", "Policy Expert-0"),
        "PA-MoE": (dmoe_expert_0_policy_router, RETURNCOLORS[2], LineStyle[2], "Step", "Rate", "Policy Expert-0"),
    },
    "Expert-Policy-1-Router": {
        "MoE": (moe_expert_1_policy_router, RETURNCOLORS[0], LineStyle[0], "Step", "Rate", "Policy Expert-1"),
        "SMoE": (smoe_expert_1_policy_router, RETURNCOLORS[1], LineStyle[1], "Step", "Rate", "Policy Expert-1"),
        "PA-MoE": (dmoe_expert_1_policy_router, RETURNCOLORS[2], LineStyle[2], "Step", "Rate", "Policy Expert-1"),
    },
    "Expert-Policy-2-Router": {
        "MoE": (moe_expert_2_policy_router, RETURNCOLORS[0], LineStyle[0], "Step", "Rate", "Policy Expert-2"),
        "SMoE": (smoe_expert_2_policy_router, RETURNCOLORS[1], LineStyle[1], "Step", "Rate", "Policy Expert-2"),
        "PA-MoE": (dmoe_expert_2_policy_router, RETURNCOLORS[2], LineStyle[2], "Step", "Rate", "Policy Expert-2"),
    },
}

plot_1_3(Router_Policy, "policy_router.pdf")

Router_Value = {
    "Expert-Value-0-Router": {
        "MoE": (moe_expert_0_value_router, RETURNCOLORS[0], LineStyle[0], "Step", "Rate", "Value Expert-0"),
        "SMoE": (smoe_expert_0_value_router, RETURNCOLORS[1], LineStyle[1], "Step", "Rate", "Value Expert-0"),
        "PA-MoE": (dmoe_expert_0_value_router, RETURNCOLORS[2], LineStyle[2], "Step", "Rate", "Value Expert-0"),
    },
    "Expert-Value-1-Router": {
        "MoE": (moe_expert_1_value_router, RETURNCOLORS[0], LineStyle[0], "Step", "Rate", "Value Expert-1"),
        "SMoE": (smoe_expert_1_value_router, RETURNCOLORS[1], LineStyle[1], "Step", "Rate", "Value Expert-1"),
        "PA-MoE": (dmoe_expert_1_value_router, RETURNCOLORS[2], LineStyle[2], "Step", "Rate", "Value Expert-1"),
    },
    "Expert-Value-2-Router": {
        "MoE": (moe_expert_2_value_router, RETURNCOLORS[0], LineStyle[0], "Step", "Rate", "Value Expert-2"),
        "SMoE": (smoe_expert_2_value_router, RETURNCOLORS[1], LineStyle[1], "Step", "Rate", "Value Expert-2"),
        "PA-MoE": (dmoe_expert_2_value_router, RETURNCOLORS[2], LineStyle[2], "Step", "Rate", "Value Expert-2"),
    },
}

plot_1_3(Router_Value, "value_router.pdf")
