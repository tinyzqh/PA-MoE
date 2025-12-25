import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from src.plots.utils import moving_average


def plot_3_2(data, save_path):
    fig = plt.figure(figsize=(18, 24))
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]

    metric_labels = sorted(data.keys())
    for idx, metric_label in enumerate(metric_labels):
        for alg_label, (y_datas, color, line_style, xlabel, ylabel, title) in data[metric_label].items():
            y_values = np.array(y_datas)
            y_values = moving_average(y_values, windowsize=20)
            y_mean = np.mean(y_values, axis=0)
            y_80 = np.quantile(y_values, q=4 / 5, axis=0)
            y_20 = np.quantile(y_values, q=1 / 5, axis=0)

            x_r_values = np.array(list(range(len(y_mean)))) * 200
            sns.lineplot(x=x_r_values, y=y_mean, color=color, label=alg_label, linestyle=line_style, linewidth=3, estimator=np.median, ax=axs[idx])
            axs[idx].fill_between(x_r_values, y_20, y_80, alpha=0.15, color=color)

            axs[idx].set_title(title, x=0.5, y=1.0, fontsize=32, fontweight="bold")
            axs[idx].tick_params(axis="both", labelsize=22)
            axs[idx].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

            axs[idx].spines["right"].set_linewidth(2)
            axs[idx].spines["top"].set_linewidth(2)
            axs[idx].spines["left"].set_linewidth(2)
            axs[idx].spines["bottom"].set_linewidth(2)

            axs[idx].set_xlabel(xlabel, fontsize=32, fontweight="bold")
            axs[idx].set_ylabel(ylabel, fontsize=32, fontweight="bold")
            axs[idx].legend(fontsize=32)

    plt.tight_layout()
    if save_path.split(".")[-1] == "pdf":
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    elif save_path.split(".")[-1] == "png":
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        raise ValueError(f"Unsupported file extension: {save_path.split('.')[-1]}")
    plt.close()
