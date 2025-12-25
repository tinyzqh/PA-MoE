import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from src.plots.utils import moving_average


def plot_1_4(data: dict, save_path: str = None) -> None:
    """
    Create a figure with 4 subplots showing different metrics with moving averages and confidence intervals.

    Args:
        data: Dictionary containing plot data for each metric. Structure:
            {
                'metric_name': {
                    'algorithm_name': (
                        y_data,  # List of numpy arrays containing metric values
                        color,   # Color for the line
                        line_style,  # Line style (e.g., '-', '--', ':')
                        xlabel,  # Label for x-axis
                        ylabel,  # Label for y-axis
                        title    # Title for the subplot
                    )
                }
            }
        save_path: Optional path to save the figure. If None, figure is only displayed.

    Example:
        >>> # Prepare data
        >>> data = {
        ...     'Metric 1': {
        ...         'Algorithm1': (
        ...             [np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])],  # y_data
        ...             '#1f77b4',  # color
        ...             '-',        # line_style
        ...             'A',        # xlabel
        ...             'B',        # ylabel
        ...             'C'         # title
        ...         ),
        ...         'Algorithm2': (
        ...             [np.array([0.2, 0.3, 0.4]), np.array([0.3, 0.4, 0.5])],
        ...             '#ff7f0e',
        ...             '--',
        ...             'A',
        ...             'B',
        ...             'C'
        ...         )
        ...     },
        ...     'Metric 2': {
        ...         'Algorithm1': (
        ...             [np.array([1000, 2000, 3000]), np.array([2000, 3000, 4000])],
        ...             '#1f77b4',
        ...             '-',
        ...             'A',
        ...             'B',
        ...             'C'
        ...         ),
        ...         'Algorithm2': (
        ...             [np.array([2000, 3000, 4000]), np.array([3000, 4000, 5000])],
        ...             '#ff7f0e',
        ...             '--',
        ...             'A',
        ...             'B',
        ...             'C'
        ...         )
        ...     }
        ... }
        >>>
        >>> # Create and save the plot
        >>> plot_1_4(data, save_figure_name='metrics_comparison.png')

    Notes:
        - The function creates a figure with 4 subplots arranged horizontally
        - Each subplot shows a different metric
        - Data is smoothed using a moving average with window size 20
        - Confidence intervals are shown between 20th and 80th percentiles
        - The x-axis values are multiplied by 2000 to convert steps to actual values
        - The figure size is set to (31, 6) inches
    """

    fig = plt.figure(figsize=(31, 6))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)
    axs = [ax1, ax2, ax3, ax4]
    metric_labels = list(data.keys())

    for idx, metric_label in enumerate(metric_labels):
        for alg_label, (y_datas, color, line_style, xlabel, ylabel, title) in data[metric_label].items():
            y_values = np.array(y_datas)
            y_values = moving_average(y_values, windowsize=20)
            y_mean = np.mean(y_values, axis=0)
            y_80 = np.quantile(y_values, q=4 / 5, axis=0)
            y_20 = np.quantile(y_values, q=1 / 4, axis=0)

            x_r_values = np.array(list(range(len(y_mean)))) * 2000
            sns.lineplot(x=x_r_values, y=y_mean, color=color, label=alg_label, linestyle=line_style, linewidth=3, estimator=np.mean, ax=axs[idx])
            axs[idx].fill_between(x_r_values, y_20, y_80, alpha=0.3, color=color)

            axs[idx].set_title(title, x=0.5, y=1.0, fontsize=24, fontweight="bold")

            axs[idx].spines["right"].set_visible(False)
            axs[idx].spines["top"].set_visible(False)
            axs[idx].spines["left"].set_linewidth(2)
            axs[idx].spines["bottom"].set_linewidth(2)
            axs[idx].spines["left"].set_position(("outward", 10))
            axs[idx].spines["bottom"].set_position(("outward", 10))

            # Configure x-axis ticks
            max_x = x_r_values[-1]
            tick_positions = np.linspace(0, max_x, 6)  # 6 ticks including start and end
            axs[idx].set_xticks(tick_positions)
            axs[idx].set_xticklabels([f"{int(x/1000)}k" for x in tick_positions])

            axs[idx].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            axs[idx].tick_params(axis="both", labelsize=20)
            axs[idx].set_xlabel(xlabel, fontsize=22, fontweight="bold")
            axs[idx].set_ylabel(ylabel, fontsize=22, fontweight="bold")

            axs[idx].legend(fontsize=22, loc="upper left")

    plt.tight_layout()
    if save_path.split(".")[-1] == "pdf":
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    elif save_path.split(".")[-1] == "png":
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        raise ValueError(f"Unsupported file extension: {save_path.split('.')[-1]}")
    plt.close()
