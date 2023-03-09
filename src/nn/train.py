import argparse
import json
import os
import numpy as np
import torch
import model_base
import model_prop


def train(method, dataset_train, dataset_validation, gpu, epochs, batchsize, lr, n_layers, n_units, alpha, offset, save_freq, outdir):
    # Enable GPU if specified
    if gpu >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        dataloader_generator = torch.Generator(device="cuda")
    else:
        dataloader_generator = None

    if method == "proposed":
        mymodel = model_prop

        # Load datasets
        train_dataset = mymodel.load_dataset(dataset_train,      use_torch=True, padding=True)
        val_dataset   = mymodel.load_dataset(dataset_validation, use_torch=True)

        # Load synergies
        filename = os.path.join(dataset_train, "synergy.npy")
        synergies = np.load(filename)

        # Setup a model
        in_dim  = train_dataset[0][0].shape[1]
        out_dim = synergies.shape[0]
        params = {
            "in_dim": in_dim,
            "out_dim": out_dim,
            "n_units": n_units,
            "n_layers": n_layers,
            "synergies_shape": synergies.shape,
            "offset": offset,
        }
        model = mymodel.Mymodel_train(**params)
        model.register_dec(synergies)

        # Setup a loss function
        loss_func = lambda y1, y2: mymodel.loss_func(y1, y2, alpha)

    elif method == "baseline":
        mymodel = model_base

        # Load datasets
        train_dataset = mymodel.load_dataset(dataset_train,      use_torch=True, padding=True)
        val_dataset   = mymodel.load_dataset(dataset_validation, use_torch=True)

        # Setup a model
        in_dim  = train_dataset[0][0].shape[1]
        out_dim = train_dataset[0][1].shape[1]
        params = {
            "in_dim": in_dim,
            "out_dim": out_dim,
            "n_units": n_units,
            "n_layers": n_layers,
        }
        model = mymodel.Mymodel_train(**params)

        # Setup a loss function
        loss_func = lambda y1, y2: mymodel.loss_func(y1, y2)

    # Save model parameters
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    # Setup data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, generator=dataloader_generator)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=1,                       generator=dataloader_generator)

    # Setup an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        # Training
        train_loss = 0.0
        for data_in, data_out in train_loader:
            # Reset the model
            optimizer.zero_grad()
            model.reset_state()

            # Forward computation
            y = model.forward(data_in)
            loss = loss_func(y, data_out)

            # Update parameters
            loss.backward()
            optimizer.step()

            # Accumulate the training loss
            train_loss += loss.item()

        # Show the training loss
        avg_train_loss = train_loss / len(train_loader)
        print("Epoch {:3d}:".format(epoch), end=" ")
        print("train loss = {:.5e}".format(avg_train_loss), end=" ")

        # Validation
        val_loss = 0.0
        for data_in, data_out in val_loader:
            # Forward computation
            with torch.no_grad():
                model.reset_state()
                y = model.forward(data_in)
                loss = loss_func(y, data_out)

            # Accumulate the validation loss
            val_loss += loss.item()

        # Show the validation loss
        avg_val_loss = val_loss / len(val_loader)
        print("validation loss = {:.5e}".format(avg_val_loss))

        # Save the latest model
        os.makedirs(outdir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(outdir, "nn_latest.pth"))

        if epoch % save_freq == 0:
            os.makedirs(outdir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(outdir, "nn_{}.pth".format(epoch)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["proposed", "baseline"])
    parser.add_argument("--dataset-train", type=str, default="dataset/train/", help="Training dataset")
    parser.add_argument("--dataset-validation", type=str, default="dataset/validation/", help="Validation dataset")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id to use; CPU is used if -1")
    parser.add_argument("--epochs", type=int, default=100000, help="Number of training epochs")
    parser.add_argument("--batchsize", type=int, default=4, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--n-layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--n-units", type=int, default=256, help="Number of hidden units")
    parser.add_argument("--alpha", type=float, default=0.1, help="Coefficient of synergy-activity regularization")
    parser.add_argument("--offset", type=float, default=-6.0, help="Offset for softplus activation at the output layer")
    parser.add_argument("--save-freq", type=int, default=10000, help="Saving frequency")
    parser.add_argument("--outdir", default="result", help="Output directory")
    args = parser.parse_args()

    # Make a result folder
    os.makedirs(args.outdir, exist_ok=True)

    # Save the arguments
    with open(os.path.join(args.outdir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    train(args.method, args.dataset_train, args.dataset_validation, args.gpu, args.epochs, args.batchsize, args.lr, args.n_layers, args.n_units, args.alpha, args.offset, args.save_freq, args.outdir)
