import argparse
import glob
import json
import os
import pickle
import numpy as np
import dataio
import timevarying


def main(dirname, outdir, n_synergies, synergy_length, n_synergies_use, lr, load):
    # Load a dataset
    dataset = []
    filelist = glob.glob(os.path.join(dirname, "*.csv"))
    filelist = sorted(filelist)
    for filename in filelist:
        data = np.loadtxt(filename, delimiter=",")
        dataset.append(data)

    # Extract movements
    n_markers = dataio.n_markers
    movements = [dataio.movement.read(data, "velocity", n_markers) for data in dataset]
    lengths = [mov.shape[0] for mov in movements]

    movements = timevarying.transform_nonnegative(movements)
    dof = movements[0].shape[1]

    if load is None:
        # Initialize synergies
        synergies = np.random.uniform(0.0, 1.0, (n_synergies, synergy_length, dof))

        # Extract motor synergies
        refractory_period = int(synergy_length * 0.5)
        r2_pre = -1.0
        r2 = 0.0
        i = 0
        while abs(r2 - r2_pre) > 1e-5:
            delays, amplitude = timevarying.match_synergies(movements, synergies, n_synergies_use, refractory_period)

            r2_pre = r2
            r2 = timevarying.compute_R2(movements, synergies, amplitude, delays)
            print("Iter {:4d}: R2 = {:10.8f}, delta R = {:10.8f}".format(i, r2, abs(r2 - r2_pre)))

            # Save synergies
            np.save(os.path.join(outdir, "synergy.npy"), synergies)

            synergies = timevarying.update_synergies(movements, synergies, amplitude, delays, lr)
            i += 1
    else:
        synergies = np.load(load)
        refractory_period = int(synergies.shape[1] * 0.5)
        i = -1

    # Compute synergy activities
    delays, amplitude = timevarying.match_synergies(movements, synergies, n_synergies_use, refractory_period)
    r2 = timevarying.compute_R2(movements, synergies, amplitude, delays)
    print("Iter {:4d}: R2 = {}".format(i+1, r2))

    # Save results
    for n in range(len(dataset)):
        data = dataset[n]

        # Convert activities into time-series
        activity = np.zeros((data.shape[0], n_synergies))
        for k in range(n_synergies):
            for ts, c in zip(delays[n][k], amplitude[n][k]):
                activity[ts, k] = c

        # Compute residual
        movements_reconstruct = timevarying.decode([delays[n]], [amplitude[n]], synergies, [lengths[n]])[0]
        movements_reconstruct = timevarying.inverse_transform_nonnegative([movements_reconstruct])[0]
        pos = dataio.movement.read(dataset[n], "position", n_markers)
        residual = np.zeros_like(pos)
        for t in range(pos.shape[0] - 1):
            residual[t] = (pos[t + 1] - pos[t]) - movements_reconstruct[t]

        # Create a data
        time = dataio.movement.read(data, "time", n_markers)
        position = dataio.movement.read(data, "position", n_markers)
        velocity = dataio.movement.read(data, "velocity", n_markers)
        markers = dataio.movement.read(data, "markers", n_markers)
        result = dataio.synergy.write(time, position, markers, velocity, activity, residual)

        # Save to a csv file
        filename = os.path.basename(filelist[n])
        filename = os.path.join(outdir, filename)
        np.savetxt(filename, result, delimiter=",")

    # Save synergies
    np.save(os.path.join(outdir, "synergy.npy"), synergies)

    # Save activities
    with open(os.path.join(outdir, "activity.pickle"), "wb") as f:
        activities = (delays, amplitude)
        pickle.dump(activities, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract time-varying synergies from a dataset")
    parser.add_argument("dirname", type=str, help="Input directory")
    parser.add_argument("outdir", type=str, help="Output directory")
    parser.add_argument("--n-synergies", type=int, default=4, help="Number of synergies in a repertory")
    parser.add_argument("--synergy-length", type=int, default=25, help="Time length of synergies")
    parser.add_argument("--n-synergies-use", type=int, default=70, help="Number of synergies used in a data")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate in the gradient descent")
    parser.add_argument("--load", type=str, default=None, help="Load existing synergies (.npy)")
    args = parser.parse_args()

    # Create an output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Save the arguments
    with open(os.path.join(args.outdir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    main(args.dirname, args.outdir, args.n_synergies, args.synergy_length, args.n_synergies_use, args.lr, args.load)
