import numpy as np

import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import copy


blue = [0.231, 0.549, 0.761]
blue2 = [0.141, 0.278, 0.588]
red = [0.882, 0.402, 0.188]
red2 = [0.89, 0.094, 0.082]

magma_3 = plt.cm.magma([0.1, 0.5, .7])


def plotting_init():
    """Set plotting hyperparameters"""
    sns.set_style("ticks", {"ytick.direction": "in", "xtick.direction": "in"})

    return blue, blue2, red, red2, magma_3


def matrix_plot(data, ax, items_n, vmin=None, vmax=None):
    """Plot a matrix

    :param data: Data matrix
    :param ax: axis
    :param items_n: Number of items
    :param vmin: Minimum value
    :param vmax: Maximum value
    :return: Return image axis for creation of colourbar
    """
    # Set nan color
    current_cmap = copy.copy(plt.cm.get_cmap("magma"))
    current_cmap.set_bad(color=[0.95] * 3)

    # Show data
    if vmin is None:
        im = ax.imshow(data, cmap=plt.cm.magma)
    else:
        im = ax.imshow(data, cmap=plt.cm.magma, vmin=vmin, vmax=vmax)

    # Add tick labels
    ax.set_xticks(range(items_n))
    ax.set_xticklabels(list(range(1, items_n // 2 + 1)) * 2, va='center', fontsize=10, fontweight="bold")
    ax.xaxis.set_tick_params(pad=12.)

    ax.set_yticks(range(items_n))
    ax.set_yticklabels(list(range(1, items_n // 2 + 1)) * 2, ha='center', fontsize=10, fontweight="bold")
    ax.tick_params(left=False, bottom=False)
    ax.yaxis.set_tick_params(pad=12.)

    # Color labels
    for n, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(blue if n < items_n // 2 else red)
    for n, ytick in enumerate(ax.get_yticklabels()):
        ytick.set_color(blue if n < items_n // 2 else red)

    # Add grouping lines next to labels
    for x in [True, False]:
        for color, rng in zip([blue, red], [[-0.35, 5.5], [5.5, 11.35]]):
            if x:
                line = lines.Line2D(rng, [11.55, 11.55], lw=3.25, color=color)
            else:
                line = lines.Line2D([-.55, -.55], rng, lw=3.25, color=color)
            line.set_clip_on(False)
            ax.add_line(line)

    for x in [True, False]:
        for color, rng in zip([blue, red], [[-0.35, 5.5], [5.5, 11.35]]):
            if x:
                line = lines.Line2D(rng, [-.55, -.55], lw=3.25, color=color)
            else:
                line = lines.Line2D([11.45, 11.45], rng, lw=3.25, color=color)
            line.set_clip_on(False)
            ax.add_line(line)

    # Set border color
    for pos in ["top", "right", "bottom", "left"]:
        ax.spines[pos].set_linewidth(1)
        ax.spines[pos].set_color([0.5] * 3)

    return im


def mds_plot(data, ax, items_n):
    """Plot MDS data

    :param data: MDS data
    :param ax: axis
    :param items_n: Number of items
    """
    d1, d2 = data

    # Plot connecting lines
    for group in range(2):
        start = group * items_n // 2
        end = items_n // 2 + group * items_n // 2
        color = blue if group == 0 else red
        ax.plot(d1[start:end], d2[start:end], linewidth=1.5, zorder=0, color=color)

    # Scatter and label data
    zorders = [20, 80, 20, 80, 20, 80, 80, 20, 80, 20, 80, 20]
    for item in range(items_n):
        color = blue if item < (items_n // 2) else red
        ax.scatter(d1[item], d2[item], s=250, edgecolor=color, linewidth=1.5, facecolor="w", alpha=1.,
                   zorder=zorders[item] + item)
        color = blue2 if item < (items_n // 2) else red2
        ax.text(d1[item], d2[item], item+1, ha='center', va='center', fontweight="bold", color=color,
                zorder=zorders[item] + item)

    # Set limits
    x_min, x_max = np.min(d1), np.max(d1)
    y_min, y_max = np.min(d2), np.max(d2)
    ax.set_xlim(1.2*x_min, 1.2*x_max)
    ax.set_ylim(1.2*y_min, 1.2*y_max)

    sns.despine(ax=ax)
