import argparse
import glob
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import dataio


def main(dirname, show_patterns, outdir, plot):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
    plt.rcParams["font.family"] = "Latin Modern Roman"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 6
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["axes.titlesize"] = "medium"
    cm = 1 / 2.54

    patterns = {}
    for i, (l1, l2, l3) in enumerate(itertools.product(["j", "q", "k"], repeat=3)):
        patterns[l1 + l2 + l3] = i
    n_patterns = len(show_patterns)

    fig = plt.figure(figsize=(8.5*cm, 6.0*cm))
    axes = [fig.add_subplot(n_patterns, 1, i + 1) for i in range(n_patterns)]

    max_value = 0.0
    filelist = glob.glob(os.path.join(dirname, "*.csv"))
    for filename in filelist:
        data = np.loadtxt(filename, delimiter=",", dtype=np.float32)

        times     = np.linspace(0, 100, data.shape[0])
        activity = dataio.synergy.read(data, "activity", dataio.n_markers)
        markers = dataio.synergy.read(data, "markers", dataio.n_markers)

        for n, letters in enumerate(show_patterns):
            idx = patterns[letters]

            if np.mean(markers[:, idx]) > 0.5:
                n_synergies = activity.shape[1]
                for i in range(n_synergies):
                    axes[n].plot(times, activity[:, i], lw=1, label="#{}".format(i+1), color="C{}".format(i))
                    max_value = max(np.max(activity[:, i]), max_value)

    for i, ax in enumerate(axes):
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(0.0, max_value * 1.2)
        ax.set_ylabel("{0[0]}-{0[1]}-{0[2]}".format(show_patterns[i]))
        if i < n_patterns - 1:
            ax.set_xticks([])
    axes[-1].set_xlabel("Time [\%]")

    handles = []
    labels = []
    for i in range(n_synergies):
        l, = axes[0].plot([0], [0], lw=1)
        label = "Synergy {}".format(i+1)
        handles.append(l)
        labels.append(label)
    axes[0].legend(handles, labels, bbox_to_anchor=(1.0, 1.0), loc="upper left")

    fig.subplots_adjust(hspace=0.1, right=0.75)

    figname = os.path.join(outdir, "activities-demo.pdf")
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
