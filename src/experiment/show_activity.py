import argparse
import itertools
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def main(dirname, show_patterns, outdir, plot):
    # Setup matplotlib
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
    plt.rcParams["font.family"] = "Latin Modern Roman"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 6
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["axes.titlesize"] = "medium"
    cm = 1 / 2.54

    dataset = {}
    max_value = 0.0
    filelist = glob.glob(os.path.join(dirname, "trajectory-*.csv"))
    filelist = sorted(filelist)
    for filename in filelist:
        letters = os.path.basename(filename)[-7:-4]
        data = np.loadtxt(filename, delimiter=",")
        data = data[:, 0:-2]
        dataset[letters] = data
        max_value = max(np.max(data), max_value)

    patterns = {}
    for i, (l1, l2, l3) in enumerate(itertools.product(["j", "q", "k"], repeat=3)):
        patterns[l1 + l2 + l3] = i
    n_patterns = len(show_patterns)

    times = np.linspace(0, 12.5, 250)

    fig = plt.figure(figsize=(8.5*cm, 6.0*cm))
    axes = [fig.add_subplot(n_patterns, 1, i + 1) for i in range(n_patterns)]

    for i, (ax, label) in enumerate(zip(axes, show_patterns)):
        data = dataset[label]
        n_synergies = data.shape[1]
        for j in range(n_synergies):
            ax.plot(times, data[:, j], lw=1, label="#{}".format(i+1), color="C{}".format(j))

        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(0.0, max_value * 1.2)
        ax.set_ylabel("{0[0]}-{0[1]}-{0[2]}".format(label))

        if i < n_patterns - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time [s]")

        axes.append(ax)

    ax_handles = []
    ax_labels = []
    for i in range(n_synergies):
        l, = axes[0].plot([0], [0], lw=1)
        label = "Synergy {}".format(i+1)
        ax_handles.append(l)
        ax_labels.append(label)
    axes[0].legend(ax_handles, ax_labels, bbox_to_anchor=(1.0, 1.0), loc="upper left")

    fig.subplots_adjust(hspace=0.3, right=0.75)

    figname = os.path.join(outdir, "activities-robot.pdf")
    plt.savefig(figname)
    if plot:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirname", type=str)
    parser.add_argument("--patterns", type=str, nargs="+", default=["jjj", "jqj", "qjk", "jqk"], help="Patterns to show")
    parser.add_argument("--outdir", default="result", help="Output directory")
    parser.add_argument("--noplot", action="store_false", dest="plot")
    args = parser.parse_args()

    main(args.dirname, args.patterns, args.outdir, args.plot)
