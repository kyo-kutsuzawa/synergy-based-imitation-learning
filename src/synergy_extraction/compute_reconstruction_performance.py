import argparse
import glob
import itertools
import json
import os
import numpy as np
import dataio
import timevarying


def main(dirname):
    filename = os.path.join(dirname, "synergy.npy")
    synergies = np.load(filename)

    with open(os.path.join(dirname, "args.json"), "r") as f:
        params = json.load(f)
    refractory_period = int(params["synergy_length"] * 0.5)
    n_synergies_use = params["n_synergies_use"]

    patterns = {}
    for i, (l1, l2, l3) in enumerate(itertools.product(["j", "q", "k"], repeat=3)):
        patterns[l1 + l2 + l3] = i
    n_patterns = len(patterns)

    filelist = glob.glob(os.path.join(dirname, "*.csv"))
    dataset = [[] for _ in range(n_patterns)]
    for filename in filelist:
        data = np.loadtxt(filename, delimiter=",", dtype=np.float32)

        markers   = dataio.synergy.read(data, "markers", dataio.n_markers)
        velocity = dataio.synergy.read(data, "velocity", dataio.n_markers)

        idx = np.argmax(np.sum(markers, axis=0))
        dataset[idx].append(velocity)

    r2_dict = {}
    for label, data in zip(patterns, dataset):
        if len(data) == 0:
            continue

        data = timevarying.transform_nonnegative(data)
        delays, amplitude = timevarying.match_synergies(data, synergies, n_synergies_use, refractory_period)
        r2 = timevarying.compute_R2(data, synergies, amplitude, delays)

        r2_dict[label] = r2
        print(label, r2)

    is_train= [
        True,  False, True,  True,  True,  False, False, True,  True,
        False, True,  True,  False, True,  True,  True,  False, False,
        True,  True,  False, False, False, True,  True,  True,  False
    ]

    r2_learned = []
    r2_unlearned = []
    for label, idx in patterns.items():
        if is_train[idx]:
            r2_learned.append(r2_dict[label])
        else:
            r2_unlearned.append(r2_dict[label])

    print("train: mean={:.3f}, std={:.4f}".format(np.mean(r2_learned), np.std(r2_learned)))
    print("test : mean={:.3f}, std={:.4f}".format(np.mean(r2_unlearned), np.std(r2_unlearned)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirname", type=str)
    args = parser.parse_args()

    main(args.dirname)
