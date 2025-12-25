import numpy as np
import matplotlib.pyplot as plt


def plot_cdf(data, x_label, y_label, index_name, save_file_name):
    NUM_BINS = 500
    LINE_STY = [(0, (3, 5, 1, 5)), (0, (5, 10)), "-", ":", "--", "-.", ":", "-."]
    COLOR_MAP = plt.cm.rainbow

    fig = plt.figure()
    ax = fig.add_subplot(111)
    SCHEMES_NAME = list(data.keys())
    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(SCHEMES_NAME))]

    for i, scheme in enumerate(SCHEMES_NAME):
        print("{} | {} | {}".format(scheme, index_name, np.mean(data[scheme][index_name])))
        values, base = np.histogram(data[scheme][index_name], bins=NUM_BINS)
        cumulative = np.cumsum(values)
        cumulative = cumulative / np.max(cumulative)
        # / float(len(data[scheme][index_name]))
        # ax.plot(base[:-1], cumulative)
        ax.plot(base[:-1], cumulative, color=colors[i], linestyle=LINE_STY[i], linewidth=2.6, label=scheme)

    # for i, j in enumerate(ax.lines):
    #     plt.setp(j, color=colors[i], linestyle=LINE_STY[i], linewidth=2.6)

    ax.set_xlim(left=-1, right=2)

    # ax.legend(SCHEMES_NAME, loc="best", fontsize=12)
    ax.legend(loc="best", fontsize=16)
    plt.ylabel(y_label, fontsize=16, fontweight="bold")
    plt.xlabel(x_label, fontsize=16, fontweight="bold")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2.5)
    ax.spines["left"].set_linewidth(2.5)
    plt.grid()
    plt.savefig(save_file_name)
