import argparse
import glob
import os
import numpy as np
import dataio


def main(dirname, outdir, t0, t1, dt, tau):
    filelist = glob.glob(os.path.join(dirname, "*.csv"))
    filelist = sorted(filelist)

    os.makedirs(outdir, exist_ok=True)
    
    for i, filename in enumerate(filelist):
        filename_out = os.path.join(outdir, "data_converted{:0>2}.csv".format(i))

        # Preprocess data
        times, positions, velocity, markers = preprocess(filename, t0, t1, dt, tau)

        # Save data
        data = dataio.movement.write(time=times, position=positions, markers=markers, velocity=velocity)
        np.savetxt(filename_out, data, delimiter=",")
        print(filename_out, "saved.")


def preprocess(filename, t0, t1, dt, tau):
    data = np.loadtxt(filename, delimiter=",")

    # Extract data
    marker_ids = list(range(5, 32))
    times = dataio.realsense.read(data, "time")  # Time
    base = dataio.realsense.read(data, 0)  # Base position
    markers = dataio.realsense.read(data, marker_ids)  # marker positions
    positions = dataio.realsense.read(data, 1)  # Hand positions
    positions = positions[:, :3]

    # Replace outliers to NaN
    for i in range(data.shape[0]):
        if np.all(positions[i] == 0):
            positions[i] = np.nan

    # Remove outliers
    indices = ~np.isnan(positions).any(axis=1)
    times = times[indices]
    positions = positions[indices]
    markers = markers[indices]

    # Resample data
    values = np.concatenate([positions, markers], axis=1)
    times, values = resample(times, values, dt)
    positions, markers = np.split(values, [positions.shape[1]], axis=1)

    # Clip data
    i0 = np.where(times >= t0)[0][0]
    i1 = np.where(times <= t1)[0][-1]
    times = times[i0:i1]
    positions = positions[i0:i1]
    markers = markers[i0:i1]

    # Modify coordinate system
    positions = transform(positions, base)

    # Compute velocity
    velocity = np.diff(positions, axis=0)

    # Filter the velocity
    for i in range(velocity.shape[0]):
        velocity[i] = velocity[i - 1] * tau + velocity[i] * (1 - tau)

    # Padding the first sample
    vT = np.zeros_like(velocity[-1:])
    velocity = np.concatenate([velocity, vT], axis=0)

    # Convert marker data
    markers = encode_markers(markers)

    return times, positions, velocity, markers


def encode_markers(markers):
    encoded = []
    for i in range(0, markers.shape[1], 6):
        y = markers[:, i]
        y = np.where(np.isnan(y), 0.0, 1.0)
        encoded.append(y)

    encoded = np.stack(encoded, axis=1)

    return encoded


def resample(times, values, interval=1e-3):
    from scipy import interpolate

    f_list = []
    for i in range(values.shape[1]):
        f = interpolate.interp1d(times, values[:, i], kind="linear")
        f_list.append(f)

    times_new = np.arange(times[0], times[-1], interval)
    values_new = []
    for f in f_list:
        v = f(times_new)
        values_new.append(v)

    values_new = np.stack(values_new, axis=1)

    return times_new, values_new


def transform(positions, base):
    # Estimate the base position
    idx = np.all(~np.isnan(base), axis=1)
    base = np.mean(base[idx], axis=0)
    pos_base = base[0:3]

    # Transform the positions
    positions = positions - pos_base

    return positions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess measurement data")
    parser.add_argument("dirname", type=str, help="Input directory")
    parser.add_argument("outdir", type=str, help="Output directory")
    parser.add_argument("--t0", type=float, default=0.0, help="Start time [s]")
    parser.add_argument("--t1", type=float, default=1e9, help="End time [s]")
    parser.add_argument("--dt", type=float, default=0.05, help="Sampling interval of the new time series [s]")
    parser.add_argument("--tau", type=float, default=0.5, help="Cut-off angular frequency of the pseudo differentiation [rad/s]")
    args = parser.parse_args()
    print(args.__dict__)

    main(args.dirname, args.outdir, args.t0, args.t1, args.dt, args.tau)
