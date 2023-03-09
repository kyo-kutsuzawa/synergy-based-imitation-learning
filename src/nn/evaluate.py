import argparse
import glob
import itertools
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import dataio
import model_base
import model_prop


def main(model_filename, outdir, plot):
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

    # Load model parameters from a json file
    model_dir = os.path.dirname(model_filename)
    with open(os.path.join(model_dir, "params.json"), "r") as f:
        model_params = json.load(f)

    with open(os.path.join(model_dir, "args.json"), "r") as f:
        args_param = json.load(f)

    if args_param["method"] == "proposed":
        mymodel = model_prop

        filename = os.path.join(args_param["dataset_train"], "synergy.npy")
        synergies = np.load(filename)

        model_params.pop("synergies_shape")
        model_params["synergies"] = synergies
        model = mymodel.Mymodel(**model_params)
        model.load_state_dict(mymodel.convert_model(torch.load(model_filename, map_location=torch.device("cpu"))))
        model.eval()

    if args_param["method"] == "baseline":
        mymodel = model_base

        model = mymodel.Mymodel(**model_params)
        model.load_state_dict(mymodel.convert_model(torch.load(model_filename, map_location=torch.device("cpu"))))
        model.eval()

    labels = list(range(27))
    is_train= [
        True,  False, True,  True,  True,  False, False, True,  True,
        False, True,  True,  False, True,  True,  True,  False, False,
        True,  True,  False, False, False, True,  True,  True,  False
    ]

    patterns = {}
    for i, (l1, l2, l3) in enumerate(itertools.product(["j", "q", "k"], repeat=3)):
        patterns[i] = l1 + l2 + l3

    # Load a dataset
    dirname = os.path.join(args_param["dataset_train"], "../all")
    filelist = glob.glob(os.path.join(dirname, "*.csv"))
    filelist = sorted(filelist)
    datalist_all = [np.loadtxt(filename, delimiter=",", dtype=np.float32) for filename in filelist]
    length = max([d.shape[0] for d in (datalist_all)])

    # Create a figure
    fig = plt.figure(figsize=(8.5*cm, 9*cm))
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

        if i != n_col * (n_row - 1):
            ax.tick_params(labelbottom=False)
            ax.tick_params(labelleft=False)
        else:
            ax.set_xlabel("$y$ [m]")
            ax.set_ylabel("$x$ [m]")

        axes.append(ax)

    # Evaluation loop
    for n, i in enumerate(labels):
        # Extract specified data
        datalist = []
        for data in datalist_all:
            markers = dataio.read(data, "markers", dataio.n_markers)
            if np.mean(markers[:, i]) > 0.5:
                datalist.append(data)

        for j, data in enumerate(datalist):
            # Reset the model
            model.reset_state()
            pt = dataio.read(data, "position", dataio.n_markers)[0]
            markers = dataio.read(data, "markers", dataio.n_markers)
            command = dataio.convert_writing(markers)[0]

            # Generate output sequences
            p = []
            for _ in range(length):
                xt = np.concatenate([pt, command])
                xt = torch.from_numpy(xt.reshape(1, -1).astype(np.float32))
                pt = model.forward(xt)
                p.append(pt.copy())
            p = np.stack(p, axis=0)

            if is_train[n]:
                color = "C0"
            else:
                color = "C1"

            axes[n].plot(-p[:, 0], p[:, 1], lw=1, color=color)

            dirname = os.path.join(outdir, "autoregression")
            os.makedirs(dirname, exist_ok=True)
            np.savetxt(os.path.join(dirname, "trajectory-{}{}.csv".format(j, patterns[i])), p, delimiter=",")

    fig.subplots_adjust(bottom=0.06, right=0.98, top=0.98)

    # Save the figures
    fig.savefig(os.path.join(outdir, "trajectories-nn.pdf"))

    # Visualize the figures
    if plot:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Trained model")
    parser.add_argument("--outdir", default="result", help="Output directory")
    parser.add_argument("--noplot", action="store_false", dest="plot")
    args = parser.parse_args()

    with torch.no_grad():
        main(args.model, args.outdir, args.plot)
