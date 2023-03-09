import argparse
import glob
import itertools
import os
import re
import numpy as np
import dataio


def main(result_dirname, demo_dirname):
    is_train= [
        True,  False, True,  True,  True,  False, False, True,  True,
        False, True,  True,  False, True,  True,  True,  False, False,
        True,  True,  False, False, False, True,  True,  True,  False
    ]

    patterns = {}
    for i, (l1, l2, l3) in enumerate(itertools.product(["j", "q", "k"], repeat=3)):
        patterns[i] = l1 + l2 + l3

    # Load demonstrations
    demonstrations = []
    for filename in glob.glob(os.path.join(demo_dirname, "*.csv")):
        data = np.loadtxt(filename, delimiter=",")
        pos = dataio.read(data, "position", dataio.n_markers)
        markers = dataio.read(data, "markers", dataio.n_markers)

        idx = np.argmax(np.sum(markers, axis=0))

        for i, letters in patterns.items():
            if idx == i:
                demonstrations.append((letters, pos))

    # Load NN trajectories
    results = []
    template = ".+?trajectory-\d+(.+?).csv"
    filelist = glob.glob(os.path.join(result_dirname, "trajectory-*.csv"))
    for filename in filelist:
        # Load a data
        data = np.loadtxt(filename, delimiter=",")
        positions = data[:, -2:]

        matched = re.match(template, filename)
        if matched:
            command_letters = matched.group(1)

            for i, letters in patterns.items():
                if letters == command_letters:
                    results.append((letters, positions, is_train[i]))

    # Classify the NN trajectories
    succeeded_trained = 0
    succeeded_unseen = 0
    n_trained = 0
    n_unseen = 0
    for (letters, result, trained) in results:
        # Compute similarity scores
        scores = []
        for (_, demo) in demonstrations:
            traj = result - result[0:1] + demo[0:1]
            length = min(result.shape[0], demo.shape[0])
            score = np.mean(np.square(traj[0:length] - demo[0:length]))
            scores.append(score)

        # Compute the score order
        idxs = np.argsort(scores)

        # Decide success/failure; succeeded if more than one of top-2 demonstrations were the correct letters
        n_hit = 0
        for i in range(2):
            if letters == demonstrations[idxs[i]][0]:
                n_hit += 1
        if n_hit >= 1:
            succeeded = True
        else:
            succeeded = False

        print("{}: ".format(letters), end="")
        for i in range(5):
            print(demonstrations[idxs[i]][0], end=", ")
        if trained:
            print("Trained ", end="")
            n_trained += 1
        else:
            print("Unseen  ", end="")
            n_unseen += 1
        if succeeded:
            print("Succeeded")
        else:
            print("Failed")

        if trained and succeeded:
            succeeded_trained += 1
        if (not trained) and succeeded:
            succeeded_unseen += 1

    print("Trained: {:2d}/{:2d} ({:.1f}%)".format(succeeded_trained, n_trained, succeeded_trained / n_trained * 100))
    print("New    : {:2d}/{:2d} ({:.1f}%)".format(succeeded_unseen, n_unseen, succeeded_unseen / n_unseen * 100))
    print("Total  : {:2d}/{:2d} ({:.1f}%)".format(succeeded_trained + succeeded_unseen, n_trained + n_unseen, (succeeded_trained + succeeded_unseen) / (n_trained + n_unseen) * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirname", type=str, help="Directory that contains trajectories")
    parser.add_argument("--demonstrations", type=str, default="dataset/all")
    args = parser.parse_args()

    main(args.dirname, args.demonstrations)
