import argparse
import matplotlib.pyplot as plt
import numpy as np


def main(filename):
    # Setup matplotlib
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
    plt.rcParams["font.family"] = "Latin Modern Roman"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 6
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["ytick.left"] = False
    plt.rcParams["axes.titlesize"] = "medium"
    cm = 1 / 2.54

    # Load synergies
    synergies = np.load(filename)
    n_synergies = synergies.shape[0]
    n_dof = synergies.shape[2] // 2
    t = np.arange(synergies.shape[1]) * (1 / 20)

    # Create a figure and axes
    fig = plt.figure(figsize=(8.5*cm, 6*cm), constrained_layout=True)
    axes = []
    for k in range(n_synergies):
        axes.append([])
        for m in range(n_dof):
            ax = fig.add_subplot(n_dof + 1, n_synergies, m * n_synergies + k + 1)
            ax.tick_params(labelleft=False, labelbottom=False)
            axes[k].append(ax)

    # Plot time series of the synergies
    for m in range(n_dof):
        for k in range(n_synergies):
            axes[k][m].fill_between(t, np.zeros_like(t), +synergies[k, :, m], color="C6")
            axes[k][m].fill_between(t, np.zeros_like(t), -synergies[k, :, m+n_dof], color="C7")
            axes[k][m].plot([0, 10], [0, 0], lw=0.5, color="black")

    # Setup the plot range
    val_max = np.max(synergies) * 1.2
    for k in range(n_synergies):
        for m in range(n_dof):
            axes[k][m].set_xlim(0, t[-1])
            axes[k][m].set_ylim((-val_max, val_max))

    # Setup label and ticks
    for k in range(n_synergies):
        axes[k][0].set_title("Synergy{}".format(k + 1), pad=2)
        axes[k][-1].tick_params(labelbottom=True)
        axes[k][-1].set_xticks([0, t[-1]], [0, "{:.2f}".format(t[-1])])
        axes[k][-1].set_xlabel("Time [s]")
    axes[-1][0].tick_params(labelright=True)
    axes[-1][0].yaxis.set_label_position("left")
    axes[-1][0].yaxis.tick_right()
    labels = ["$\Delta p_x$ [m]", "$\Delta p_y$ [m]"]
    for m in range(n_dof):
        axes[0][m].set_ylabel(labels[m])

    # Compute positions
    pos = synergies[:, :, :2] - synergies[:, :, 2:]
    pos = np.cumsum(pos, axis=1)

    # Plot trajectories of the synergies
    for k in range(n_synergies):
        ax = fig.add_subplot(n_dof + 1, n_synergies, n_dof * n_synergies + k + 1)
        ax.axis("equal")
        ax.tick_params(labelleft=False)
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_xticks((-2, 0, 2))
        ax.set_yticks((-2, 0, 2))
        ax.set_xlabel("$p_x$ [m]")

        if k == 0:
            ax.set_ylabel("$p_y$ [m]")

        if k == n_synergies - 1:
            ax.tick_params(labelright=True)
            ax.yaxis.tick_right()

        ax.plot(-pos[k, :, 0], pos[k, :, 1], color="C{}".format(k), solid_capstyle="round")

    # Save the figure
    plt.savefig("result/synergies.pdf")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Filename of synergies (.npy)")
    args = parser.parse_args()

    main(args.filename)
