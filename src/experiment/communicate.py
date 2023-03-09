import argparse
import itertools
import json
import os
import signal
import sys
import numpy as np
import torch
import zmq

sys.path.append(os.path.join(os.path.dirname(__file__), "../nn"))
import model_base
import model_prop


def main(model_filename, letters):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5000")

    # Setup a command generator
    gen = command_generator(model_filename, letters)
    next(gen)

    while True:
        # Receive the current values
        state = socket.recv()
        state = np.frombuffer(state, np.float64)
        print("state:   ", state)

        # Get the next command
        cmd = gen.send(state)

        # Send the next command
        command = cmd.tobytes()
        socket.send(command)
        print("command: ", cmd)
        print()


def command_generator(model_filename, letters):
    patterns = []
    for l1, l2, l3 in itertools.product(["j", "q", "k"], repeat=3):
        patterns.append(l1 + l2 + l3)

    current_letters = letters
    if not current_letters in patterns:
        current_letters = patterns[0]

    task_signal = get_task_signal(current_letters)

    # Load model parameters from a json file
    model_dir = os.path.dirname(model_filename)
    with open(os.path.join(model_dir, "params.json"), "r") as f:
        model_params = json.load(f)

    # Load training parameters
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

    # Setup trajectory info
    cmd0 = np.array([0.02, 0.065])
    length = 250

    def myhandler(signal, frame):
        nonlocal y, p, c

        print("Interrupted!")

        y = torch.stack(y, dim=1).detach().numpy()[0]
        p = np.stack(p, axis=0)
        c = np.stack(c, axis=0)
        result = np.concatenate([y, p], axis=1)
        np.savetxt("trajectory-{}{}.csv".format(cnt_trials, current_letters), result, delimiter=",")
        np.savetxt("internal-{}{}.csv".format(cnt_trials, current_letters), c, delimiter=",")

        sys.exit()

    signal.signal(signal.SIGINT, myhandler)

    cnt_trials = 0
    while True:
        # Reset NN
        model.reset_state()
        t = 0
        cmd = cmd0
        y = []
        p = []
        c = []

        # Run NN
        finish_flag = np.array([0.0])
        while t < length:
            # Send/receive variables
            sending_values = np.concatenate([cmd - cmd0, finish_flag], axis=0)
            received_values = (yield sending_values)

            # Finish the current trial if the reset flag is on.
            if received_values[-1] > 0.5:
                break

            # Generate the next command
            obs = received_values[0:2] + cmd0
            xt = np.concatenate([obs, task_signal])
            xt = torch.from_numpy(xt.reshape(1, -1).astype(np.float32))
            pt = model.forward(xt)
            cmd = pt

            ct = model.get_internal_state().numpy().flatten().copy()

            p.append(pt.copy())
            y.append(model.activity)
            c.append(ct)

            t += 1

        y = torch.stack(y, dim=1).detach().numpy()[0]
        p = np.stack(p, axis=0)
        c = np.stack(c, axis=0)
        result = np.concatenate([y, p], axis=1)
        np.savetxt("trajectory-{}{}.csv".format(cnt_trials, current_letters), result, delimiter=",")
        np.savetxt("internal-{}{}.csv".format(cnt_trials, current_letters), c, delimiter=",")
        cnt_trials += 1

        # Keep the final command values
        finish_flag = np.array([1.0])
        while True:
            # Send/receive variables
            sending_values = np.concatenate([cmd - cmd0, finish_flag], axis=0)
            received_values = (yield sending_values)

            # Finish the current trial if the reset flag is on.
            if received_values[-1] > 0.5:
                break

        # Update the task signal
        current_letters = get_next_letters(current_letters, patterns)
        task_signal = get_task_signal(current_letters)


def get_next_letters(current_letters, patterns):
    idx = patterns.index(current_letters)
    next_idx = (idx + 1) % len(patterns)
    next_letters = patterns[next_idx]

    return next_letters


def get_task_signal(letters):
    task_signal = np.zeros(3, dtype=np.float32)
    for i in range(3):
        if letters[i] == "j":
            task_signal[i] = -1
        elif letters[i] == "q":
            task_signal[i] = 0
        elif letters[i] == "k":
            task_signal[i] = +1

    return task_signal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Trained model (.pth)")
    parser.add_argument("--letters", type=str, default="jqk")
    args = parser.parse_args()

    with torch.no_grad():
        main(args.model, args.letters)
