import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

plt.switch_backend("agg")


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def smo_rebuf(data, save_file_name):
    plt.rcParams["axes.labelsize"] = 16
    font = {"size": 15}
    matplotlib.rc("font", **font)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(left=0.17, bottom=0.16, right=0.96, top=0.96)
    modern_academic_colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F"]
    markers = ["o", "x", "v", "^", ">", "<", "s", "p", "*", "h", "H", "D", "d", "1"]

    max_smo = 0
    SCHEMES_NAME = list(data.keys())
    for idx, scheme in enumerate(SCHEMES_NAME):
        mean_smo, low_smo, high_smo = mean_confidence_interval(data[scheme]["Smooth"])
        data[scheme]["QoE"]
        data[scheme]["Bitrate Reward"]
        data[scheme]["Rebuffer Reward"]
        data[scheme]["Smooth Penalty Reward"]
        print("Method {} | Mean Smoothness: {}, Std Smoothness {}".format(scheme, mean_smo, high_smo - mean_smo))
        mean_rebuf, low_rebuf, high_rebuf = mean_confidence_interval(data[scheme]["Rebuffer Time"])
        print("Method {} | Mean Rebuf: {}, Std Rebuf {}".format(scheme, mean_rebuf, high_rebuf - mean_rebuf))
        max_smo = max(high_smo, max_smo)
        ax.errorbar(
            mean_rebuf,
            mean_smo,
            xerr=high_rebuf - mean_rebuf,
            yerr=high_smo - mean_smo,
            color=modern_academic_colors[idx],
            marker=markers[idx],
            markersize=10,
            label=scheme,
            capsize=4,
        )
    ax.set_xlabel("Time Spent on Stall (s)")
    ax.set_ylabel("Bitrate Smoothness (mbps)")
    ax.set_ylim(0.05, max_smo + 0.05)

    ax.grid(linestyle="--", linewidth=1.0, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=12, ncol=3, edgecolor="white", loc="lower left")
    ax.invert_xaxis()
    ax.invert_yaxis()
    fig.savefig(save_file_name)
    # fig.savefig(outputs + '.pdf')
    plt.close()
