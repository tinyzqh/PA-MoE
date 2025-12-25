import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import matplotlib.colors as mcolors


def calculate_interquartile_mean(reward_sequences: List[np.ndarray]) -> Tuple[float, float]:
    """
    Calculate the Interquartile Mean (IQM) and its standard error for a list of reward sequences.

    Args:
        reward_sequences: List of numpy arrays containing reward values

    Returns:
        Tuple containing (IQM value, standard error)

    Example:
        >>> rewards = [np.array([1, 2, 3, 4, 5]), np.array([2, 3, 4, 5, 6])]
        >>> iqm, error = calculate_interquartile_mean(rewards)
    """
    if not isinstance(reward_sequences, list):
        raise TypeError("reward_sequences must be a list")
    if not all(isinstance(x, np.ndarray) for x in reward_sequences):
        raise TypeError("All elements in reward_sequences must be numpy arrays")

    iqm_values = []
    for rewards in reward_sequences:
        sorted_rewards = np.sort(rewards)
        q1_index = int(0.30 * len(sorted_rewards))
        q3_index = int(0.95 * len(sorted_rewards))
        iqm = np.mean(sorted_rewards[q1_index:q3_index])
        iqm_values.append(iqm)

    return np.mean(iqm_values), np.std(iqm_values) / np.sqrt(len(iqm_values))


def plot_iqm_comparison(reward_data: Dict[str, Dict[str, List[np.ndarray]]], save_path: str, figure_title: str = None) -> None:
    """
    Create a bar plot comparing Interquartile Mean (IQM) values across different methods and groups.

    Args:
        reward_data: Nested dictionary containing reward sequences:
            {
                'group_name': {
                    'method_name': [reward_sequence1, reward_sequence2, ...]
                }
            }
        save_path: Path where the figure should be saved
        figure_title: Optional title for the figure

    Example:
        >>> reward_data = {
        ...     'Group1': {
        ...         'Method1': [np.array([1,2,3]), np.array([2,3,4])],
        ...         'Method2': [np.array([2,3,4]), np.array([3,4,5])]
        ...     }
        ... }
        >>> plot_iqm_comparison(reward_data, 'output.png')
    """
    # Input validation
    if not isinstance(reward_data, dict):
        raise TypeError("reward_data must be a dictionary")
    if not all(isinstance(v, dict) for v in reward_data.values()):
        raise TypeError("All values in reward_data must be dictionaries")

    # Set up the figure with ACM MM double-column width (7 inches)
    fig = plt.figure(figsize=(3.5, 2.5))
    ax = fig.add_subplot(111)

    # Extract groups and methods
    groups = list(reward_data.keys())
    methods_per_group = [list(group_data.keys()) for group_data in reward_data.values()]
    n_groups = len(groups)
    n_methods = max(len(methods) for methods in methods_per_group)

    # Calculate IQM values and errors
    iqm_results = []
    error_results = []
    method_labels = []

    for group_methods in methods_per_group:
        group_iqms = []
        group_errors = []
        for method in group_methods:
            iqm, error = calculate_interquartile_mean(reward_data[groups[len(iqm_results)]][method])
            group_iqms.append(iqm)
            group_errors.append(error)
        iqm_results.append(group_iqms)
        error_results.append(group_errors)
        method_labels.append(group_methods)

    # Set up bar plot parameters
    bar_width = 0.25
    x_positions = np.arange(n_groups)

    # ACM MM style colors (print-friendly)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]  # Blue  # Orange  # Green  # Red  # Purple  # Brown  # Pink  # Gray

    # Bar patterns for better distinction in black and white
    patterns = ["/", "\\", "|", "-", "+", "x", "o", "O"]

    # Plot bars
    for i in range(n_methods):
        iqms = [group_iqms[i] if i < len(group_iqms) else 0 for group_iqms in iqm_results]
        errors = [group_errors[i] if i < len(group_errors) else 0 for group_errors in error_results]
        offset = bar_width * (i - n_methods / 2 + 0.5)

        ax.bar(
            x_positions + offset, iqms, bar_width, yerr=errors, capsize=3, color=colors[i % len(colors)], label=method_labels[0][i], hatch=patterns[i % len(patterns)], alpha=0.8
        )

    # ACM MM style formatting
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    # Set labels and ticks
    ax.set_ylabel("IQM", fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_xticks(x_positions, groups, fontweight="bold")
    ax.set_xticklabels(groups, fontweight="bold")

    # Add legend
    ax.legend(loc="best", frameon=True, fancybox=False, fontsize=11)

    # Add title if provided
    if figure_title:
        ax.set_title(figure_title, pad=10, fontsize=9)

    # Adjust layout and save
    plt.tight_layout()
    if save_path.split(".")[-1] == "pdf":
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    elif save_path.split(".")[-1] == "png":
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        raise ValueError(f"Unsupported file extension: {save_path.split('.')[-1]}")
    plt.close()
