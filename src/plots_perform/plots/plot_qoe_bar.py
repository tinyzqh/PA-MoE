import numpy as np
import matplotlib.pyplot as plt

COLOR_MAP = plt.cm.Set2


def plot_qoe_bar(data, y_label, x_label, save_file_name):
    """
    data: dict[str -> dict[str -> array-like]]
          外层 key 是方法名（如 "Buffer-based"）
          内层 key 是指标名（如 "QoE", "Bitrate Reward", "Rebuffer Time", "Smooth"）
          value 是一维数组，我们会取均值来画柱子。

    y_label: y 轴标签
    x_label: x 轴标签（比如 "QoE Components"）
    save_file_name: 图像保存路径
    """

    # 要画的指标顺序
    metric_keys = ["QoE", "Bitrate Reward", "Rebuffer Reward", "Smooth Penalty Reward"]
    metric_labels = ["QoE", "Bitrate Reward", "Rebuffer Reward", "Smooth Reward"]  # x 轴上的文字

    methods = list(data.keys())
    n_methods = len(methods)
    n_terms = len(metric_keys)

    # 每组（一个指标）在 x 轴上的中心位置
    group_centers = np.arange(n_terms)

    # 一组柱子的总宽度，剩下的是组与组之间的空隙
    total_group_width = 0.8
    bar_width = total_group_width / n_methods

    # 颜色和 hatch
    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, n_methods)]
    hatches = ["", "///", "\\\\", "xx", "...", "oo", "||", "//"]  # 够用就行

    fig, ax = plt.subplots(figsize=(7.5, 4))

    # 先遍历指标，再遍历方法
    for j, key in enumerate(metric_keys):
        center = group_centers[j]  # 当前这一组在 x 轴上的中心

        for i, method in enumerate(methods):
            # 这个方法在这一组中的偏移位置
            offset = (i - n_methods / 2) * bar_width + bar_width / 2
            x_pos = center + offset

            # 该方法在该指标上的均值
            y_val = np.mean(data[method][key])

            ax.bar(x_pos, y_val, width=bar_width, label=method if j == 0 else None, edgecolor="k", color=colors[i], alpha=0.6, hatch=hatches[i % len(hatches)])  # 只在第一组加 legend

    # 坐标轴与样式
    ax.set_xticks(group_centers)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel(y_label, fontsize=16, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=16, fontweight="bold")
    ax.spines["bottom"].set_linewidth(2.5)
    ax.spines["left"].set_linewidth(2.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    offset = 8
    ax.spines["bottom"].set_position(("outward", offset))
    ax.spines["left"].set_position(("outward", offset))

    ax.grid(axis="y", linestyle="--", alpha=0.5)

    ax.legend(fontsize=13, frameon=False, ncol=min(2, n_methods), handlelength=0.8, handleheight=0.8)

    fig.tight_layout()
    fig.savefig(save_file_name, dpi=300)
    plt.close(fig)
