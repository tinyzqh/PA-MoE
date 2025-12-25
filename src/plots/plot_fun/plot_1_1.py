import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]


def plot_1_1(data, save_path, x_line_values=None, text_high=None):
    """
    Plot a single figure with multiple lines and confidence intervals.

    Args:
        data (dict): Dictionary containing plot data with format:
            {
                'label_name': (y_data, color, line_style)
            }
            where:
            - y_data: numpy array of shape (n_runs, n_steps)
            - color: string or hex color code
            - line_style: string (e.g., '-', '--', '-.', ':')
            - xlabel: string
            - ylabel: string
            - title: string
        save_path (str): Path to save the figure

    Example:
        >>> plot_data = {
        ...     'Algorithm1': (np.random.rand(5, 100), '#1f77b4', '-', xlabel, ylabel, title),
        ...     'Algorithm2': (np.random.rand(5, 100), '#ff7f0e', '--', xlabel, ylabel, title)
        ... }
        >>> plot_single_fig(plot_data, 'Training Curves', 'output.png')
    """

    # ACM MM double-column figure size (width: 3.5 inches)
    plt.figure(figsize=(3.5, 2.5))

    for alg_label, (y_data, color, line_style, xlabel, ylabel, title) in data.items():
        y_mean = np.mean(y_data, axis=0)
        x_r_values = np.array(list(range(len(y_mean)))) * 200
        y_80 = np.quantile(y_data, q=4 / 5, axis=0)
        y_25 = np.quantile(y_data, q=1 / 4, axis=0)
        sns.lineplot(x=x_r_values, y=y_mean, color=color, label=alg_label, linestyle=line_style, linewidth=1.5, estimator=np.mean)
        plt.fill_between(x_r_values, y_25, y_80, alpha=0.15, color=color)

    plt.title(title, fontweight="bold", fontsize=8)
    plt.xlabel(xlabel, fontweight="bold", fontsize=8)
    plt.ylabel(ylabel, fontweight="bold", fontsize=8)

    # Remove top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    if x_line_values:
        # Add vertical lines and text labels
        plt.axvline(x=100 * 2000, color="gray", linestyle="--", alpha=0.5)
        plt.axvline(x=200 * 2000, color="gray", linestyle="--", alpha=0.5)
        plt.axvline(x=300 * 2000, color="gray", linestyle="--", alpha=0.5)
        plt.axvline(x=400 * 2000, color="gray", linestyle="--", alpha=0.5)
        plt.axvline(x=500 * 2000, color="gray", linestyle="--", alpha=0.5)
        plt.axvline(x=600 * 2000, color="gray", linestyle="--", alpha=0.5)
        plt.axvline(x=700 * 2000, color="gray", linestyle="--", alpha=0.5)
        plt.axvline(x=800 * 2000, color="gray", linestyle="--", alpha=0.5)
        plt.axvline(x=900 * 2000, color="gray", linestyle="--", alpha=0.5)

        plt.text(50 * 2000, text_high, "D", ha="center", fontsize=10, fontweight="bold", color="#1f77b4")
        plt.text(150 * 2000, text_high, "L", ha="center", fontsize=10, fontweight="bold", color="#ff7f0e")
        plt.text(250 * 2000, text_high, "N", ha="center", fontsize=10, fontweight="bold", color="#2ca02c")
        plt.text(350 * 2000, text_high, "D", ha="center", fontsize=10, fontweight="bold", color="#1f77b4")
        plt.text(450 * 2000, text_high, "L", ha="center", fontsize=10, fontweight="bold", color="#ff7f0e")
        plt.text(550 * 2000, text_high, "N", ha="center", fontsize=10, fontweight="bold", color="#2ca02c")
        plt.text(650 * 2000, text_high, "D", ha="center", fontsize=10, fontweight="bold", color="#1f77b4")
        plt.text(750 * 2000, text_high, "L", ha="center", fontsize=10, fontweight="bold", color="#ff7f0e")
        plt.text(850 * 2000, text_high, "N", ha="center", fontsize=10, fontweight="bold", color="#2ca02c")
        plt.text(950 * 2000, text_high, "D", ha="center", fontsize=10, fontweight="bold", color="#1f77b4")

    # Place legend inside the plot at best corner
    plt.legend(loc="best", frameon=True, fancybox=False, framealpha=0.5)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    if save_path.split(".")[-1] == "pdf":
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    elif save_path.split(".")[-1] == "png":
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        raise ValueError(f"Unsupported file extension: {save_path.split('.')[-1]}")
    plt.close()
