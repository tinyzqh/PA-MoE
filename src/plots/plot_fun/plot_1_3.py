import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.plots.utils import moving_average
from matplotlib.ticker import FormatStrFormatter


def plot_1_3(data, save_path):
    fig = plt.figure(figsize=(10, 3))
    axs = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
    metric_labels = list(data.keys())

    for idx, metric_label in enumerate(metric_labels):
        for alg_label, (y_datas, color, line_style, xlabel, ylabel, title) in data[metric_label].items():
            y_values = np.array(y_datas)
            y_values = moving_average(y_values, windowsize=20)
            y_mean = np.mean(y_values, axis=0)
            y_80 = np.quantile(y_values, q=4 / 5, axis=0)
            y_25 = np.quantile(y_values, q=1 / 4, axis=0)

            x_r_values = np.array(list(range(len(y_mean)))) * 2000
            sns.lineplot(x=x_r_values, y=y_mean, color=color, label=alg_label, linestyle=line_style, linewidth=2, estimator=np.mean, ax=axs[idx])
            axs[idx].fill_between(x_r_values, y_25, y_80, alpha=0.2, color=color)

            # NeurIPS style formatting
            axs[idx].set_title(title, pad=10, fontsize=14, fontweight="bold")
            axs[idx].spines["top"].set_visible(False)
            axs[idx].spines["right"].set_visible(False)
            axs[idx].spines["left"].set_linewidth(2)
            axs[idx].spines["bottom"].set_linewidth(2)
            axs[idx].spines["left"].set_position(("outward", 10))
            axs[idx].spines["bottom"].set_position(("outward", 10))

            axs[idx].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            axs[idx].tick_params(axis="both", labelsize=30)
            axs[idx].set_xlabel(xlabel, fontsize=16, fontweight="bold")
            axs[idx].set_ylabel(ylabel, fontsize=16, fontweight="bold")

            axs[idx].tick_params(axis="both", which="major", labelsize=14)
            axs[idx].tick_params(axis="both", which="minor", labelsize=14)
            axs[idx].set_xlabel(xlabel)
            axs[idx].set_ylabel(ylabel)
            axs[idx].legend(fontsize=13, loc="best")

    plt.tight_layout()
    if save_path.split(".")[-1] == "pdf":
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    elif save_path.split(".")[-1] == "png":
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        raise ValueError(f"Unsupported file extension: {save_path.split('.')[-1]}")
    plt.close()
