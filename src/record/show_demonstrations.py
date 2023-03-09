import argparse
import glob
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import dataio

def main(dirname):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
    plt.rcParams["font.family"] = "Latin Modern Roman"
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

    # Create letter patterns
    patterns = {}
    for i, (l1, l2, l3) in enumerate(itertools.product(["j", "q", "k"], repeat=3)):
        patterns[i] = l1 + l2 + l3

    # Create a figure
    fig = plt.figure(figsize=(8.5*cm, 9*cm), constrained_layout=True)
    n_col = int(np.ceil(np.sqrt(len(labels))))
    n_col = 4
    n_row = int(np.ceil(len(labels) / n_col))
    axes = []
    for i in range(len(labels)):
        ax = fig.add_subplot(n_row, n_col, i + 1)

        ax.set_xlim(-0.05, 0.25)
        ax.set_ylim(0.01, 0.14)
        ax.set_aspect("equal")

        ax.set_title("Target: {0[0]}-{0[1]}-{0[2]}".format(patterns[labels[i]]), pad=2)

        ax.tick_params(labelbottom=False)
        ax.tick_params(labelleft=False)
        if i == n_col * (n_row - 1):
            ax.tick_params(labelbottom=True)
            ax.tick_params(labelleft=True)
            ax.set_xlabel("$y$ [m]")
            ax.set_ylabel("$x$ [m]")

        axes.append(ax)

    # Load and plot demonstrations
    filelist = glob.glob(os.path.join(dirname, "*.csv"))
    lengths = []
    for filename in filelist:
        data = np.loadtxt(filename, delimiter=",")
        pos = dataio.movement.read(data, "position", dataio.n_markers)
        markers = dataio.movement.read(data, "markers", dataio.n_markers)
        lengths.append(pos.shape[0])

        idx = np.argmax(np.sum(markers, axis=0))

        for i, letters in patterns.items():
            if idx == i:
                if is_train[i]:
                    color = "C0"
                else:
                    color = "C1"

                axes[i].plot(-pos[:, 0], pos[:, 1], color=color, lw=1)

    # Print the lengths of demonstrations
    print("Length: mean = {}, std = {}".format(np.mean(lengths), np.std(lengths)))

    # Save the figure
    os.makedirs("result", exist_ok=True)
    fig.savefig("result/demonstrations.pdf")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirname", type=str)
    args = parser.parse_args()

    main(args.dirname)
