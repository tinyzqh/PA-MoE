from src.plots.plot_fun.plot_3_2 import plot_3_2
from src.plots.plot_fun.exp_id import ExperimentId
from src.plots.wandb_data_fetcher import fetch_multiple_runs_data

RETURNCOLORS = ["#FFCC66", "#990000", "#B8860B", "#5C4033"]
LineStyle = ["--", "-.", ":", "-"]


def RequestAllData(index_name):
    MLP = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MLP_ID, metric_name=index_name, length_mode="truncate")
    MOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.MOE_ID, metric_name=index_name, length_mode="truncate")
    SMOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.SMOE_ID, metric_name=index_name, length_mode="truncate")
    DMOE = fetch_multiple_runs_data(run_wandb_path=ExperimentId.wandb_path, run_ids=ExperimentId.DMOE_ID, metric_name=index_name, length_mode="truncate")
    return_data = {"MLP": MLP, "MoE": MOE, "SMoE": SMOE, "PA-MoE": DMOE}
    return return_data


SystemDelay = RequestAllData(index_name="SystemInfo/Episode Delay")
SystemRebuffer = RequestAllData(index_name="SystemInfo/Episode Rebuffer")
SystemSleep = RequestAllData(index_name="SystemInfo/Episode Sleep Time")
SystemBitrate = RequestAllData(index_name="SystemInfo/Episode BitRate")
SystemBuffer = RequestAllData(index_name="SystemInfo/Episode Buffer Size")
SystemThroughput = RequestAllData(index_name="SystemInfo/Episode Choose Video Chunk Size Per Time")


system_info_data = {
    "Delay": {
        # "MLP": (SystemDelay["MLP"], RETURNCOLORS[0], LineStyle[0], "Step", "Second", "Delay"),
        "MoE": (SystemDelay["MoE"], RETURNCOLORS[1], LineStyle[1], "Step", "Second", "Delay"),
        "SMoE": (SystemDelay["SMoE"], RETURNCOLORS[2], LineStyle[2], "Step", "Second", "Delay"),
        "PA-MoE": (SystemDelay["PA-MoE"], RETURNCOLORS[3], LineStyle[3], "Step", "Second", "Delay"),
    },
    "Rebuffer": {
        # "MLP": (SystemRebuffer["MLP"], RETURNCOLORS[0], LineStyle[0], "Step", "Second", "Rebuffer"),
        "MoE": (SystemRebuffer["MoE"], RETURNCOLORS[1], LineStyle[1], "Step", "Second", "Rebuffer"),
        "SMoE": (SystemRebuffer["SMoE"], RETURNCOLORS[2], LineStyle[2], "Step", "Second", "Rebuffer"),
        "PA-MoE": (SystemRebuffer["PA-MoE"], RETURNCOLORS[3], LineStyle[3], "Step", "Second", "Rebuffer"),
    },
    "Sleep": {
        # "MLP": (SystemSleep["MLP"], RETURNCOLORS[0], LineStyle[0], "Step", "Second", "Sleep Time"),
        "MoE": (SystemSleep["MoE"], RETURNCOLORS[1], LineStyle[1], "Step", "Second", "Sleep Time"),
        "SMoE": (SystemSleep["SMoE"], RETURNCOLORS[2], LineStyle[2], "Step", "Second", "Sleep Time"),
        "PA-MoE": (SystemSleep["PA-MoE"], RETURNCOLORS[3], LineStyle[3], "Step", "Second", "Sleep Time"),
    },
    "BitRate": {
        # "MLP": (SystemBitrate["MLP"], RETURNCOLORS[0], LineStyle[0], "Step", "BitRate", "System BitRate"),
        "MoE": (SystemBitrate["MoE"], RETURNCOLORS[1], LineStyle[1], "Step", "Action (BitRate)", "System BitRate"),
        "SMoE": (SystemBitrate["SMoE"], RETURNCOLORS[2], LineStyle[2], "Step", "Action (BitRate)", "System BitRate"),
        "PA-MoE": (SystemBitrate["PA-MoE"], RETURNCOLORS[3], LineStyle[3], "Step", "Action (BitRate)", "System BitRate"),
    },
    "Buffer": {
        # "MLP": (SystemBuffer["MLP"], RETURNCOLORS[0], LineStyle[0], "Step", "Second", "System Buffer Size"),
        "MoE": (SystemBuffer["MoE"], RETURNCOLORS[1], LineStyle[1], "Step", "Second", "System Buffer Size"),
        "SMoE": (SystemBuffer["SMoE"], RETURNCOLORS[2], LineStyle[2], "Step", "Second", "System Buffer Size"),
        "PA-MoE": (SystemBuffer["PA-MoE"], RETURNCOLORS[3], LineStyle[3], "Step", "Second", "System Buffer Size"),
    },
    "Throughput": {
        # "MLP": (SystemThroughput["MLP"], RETURNCOLORS[0], LineStyle[0], "Step", "bps", "System Throughput"),
        "MoE": (SystemThroughput["MoE"], RETURNCOLORS[1], LineStyle[1], "Step", "bps", "System Throughput"),
        "SMoE": (SystemThroughput["SMoE"], RETURNCOLORS[2], LineStyle[2], "Step", "bps", "System Throughput"),
        "PA-MoE": (SystemThroughput["PA-MoE"], RETURNCOLORS[3], LineStyle[3], "Step", "bps", "System Throughput"),
    },
}


plot_3_2(system_info_data, "system_info.pdf")
