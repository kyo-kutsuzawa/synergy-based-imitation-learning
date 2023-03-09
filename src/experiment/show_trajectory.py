import argparse
import glob
import itertools
import os
import re
import numpy as np
import matplotlib.pyplot as plt


def main(dirname, outdir, plot):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
    plt.rcParams["font.family"] = "Latin Modern Roman"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 6
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["axes.titlesize"] = "medium"
    cm = 1 / 2.54

    labels = list(range(27))
    is_train= [
        True,  False, True,  True,  True,  False, False, True,  True,
        False, True,  True,  False, True,  True,  True,  False, False,
        True,  True,  False, False, False, True,  True,  True,  False
    ]

    # Define offset position
    offset_nn = np.array([0.02, 0.065])
    offset_robot = np.array([0.13, 0.25])

    patterns = {}
    for i, (l1, l2, l3) in enumerate(itertools.product(["j", "q", "k"], repeat=3)):
        patterns[i] = l1 + l2 + l3

    fig = plt.figure(figsize=(8.5*cm, 9*cm))
    n_col = int(np.ceil(np.sqrt(len(labels))))
    n_col = 4
    n_row = int(np.ceil(len(labels) / n_col))
    axes = []
    for i in range(len(labels)):
        ax = fig.add_subplot(n_row, n_col, i + 1)

        ax.set_xlim(-0.15, 0.15)
        ax.set_ylim( 0.18, 0.32)
        ax.invert_xaxis()
        ax.set_aspect("equal")

        if i != n_col * (n_row - 1):
            ax.tick_params(labelbottom=False)
            ax.tick_params(labelleft=False)
        else:
            ax.set_xlabel("$y$ [m]")
            ax.set_ylabel("$x$ [m]")

        ax.set_title("Target: {0[0]}-{0[1]}-{0[2]}".format(patterns[labels[i]]), pad=2)

        axes.append(ax)

    template = ".+?trajectory-\d+(.+?).csv"
    filelist = glob.glob(os.path.join(dirname, "trajectory-*.csv"))

    for filename in filelist:
        # Load a data
        data = np.loadtxt(filename, delimiter=",")
        positions = data[:, -2:] - offset_nn + offset_robot

        matched = re.match(template, filename)
        if matched:
            command_letters = matched.group(1)
            #print(command_letters)

            for i, letters in patterns.items():
                if letters == command_letters:
                    if is_train[i]:
                        color = "C0"
                    else:
                        color = "C1"

                    # Plot the data
                    axes[i].plot(positions[:, 0], positions[:, 1], color=color, lw=1)

    fig.subplots_adjust(bottom=0.08, right=0.98, top=0.98)

    # Save the figures
    figname = os.path.join(outdir, "trajectories-exp.pdf")
    fig.savefig(figname)

    # Visualize the figures
    if plot:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirname", type=str)
    parser.add_argument("--outdir", default="result", help="Output directory")
    parser.add_argument("--noplot", action="store_false", dest="plot")
    args = parser.parse_args()

    main(args.dirname, args.outdir, args.plot)
