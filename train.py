import os
import time
import argparse
import random
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from trainer import *
from data import *
from plotter import Plotter

def parse_args():
    """
    Parse command-line arguments for training or inference.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including device, input files, hyperparameters, and output options.
    """
    parser = argparse.ArgumentParser(description="Transformer Masked Autoencoder Training")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "auto"], default="auto",
                        help="Choose device: cpu, gpu, or auto (default: auto)")
    parser.add_argument("inputs", type=str, nargs="*", default=["avgWires.csv"],
                        help="One or more input CSV files")
    parser.add_argument("--max_epochs", type=int, default=120,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for DataLoader")
    parser.add_argument("--outdir", type=str, default="outputs/local",
                        help="Directory to save models and plots")
    parser.add_argument("--end_name", type=str, default="",
                        help="Optional suffix to append to output files")
    parser.add_argument("--d_model", type=int, default=64,
                        help="Transformer embedding dimension (must be divisible by nhead)")
    parser.add_argument("--nhead", type=int, default=2,
                        help="Number of attention heads in the transformer")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of transformer encoder layers")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--no_train", action="store_true",
                        help="Skip training and only run inference using a saved model")
    parser.add_argument("--enable_progress_bar", action="store_true",
                        help="Enable progress bar during training (default: disabled)")
    return parser.parse_args()


# -----------------------------
def corrupt_input(x, seq_len=6):
    """
    Randomly mask one cluster in each input sequence to create a corrupted input.

    Parameters
    ----------
    x : torch.Tensor, shape [batch_size, seq_len, 2]
        Input sequences (avgWire, slope) for each cluster.
    seq_len : int
        Number of clusters in the sequence (default=6).

    Returns
    -------
    x_corrupted : torch.Tensor, shape [batch_size, seq_len-1, 2]
        Input sequences with one cluster removed per sample.
    mask_idx : torch.Tensor, shape [batch_size]
        Indices of the removed clusters.
    """
    batch_size = x.size(0)
    mask_idx = torch.randint(0, seq_len, (batch_size,), device=x.device)

    x_corrupted = []
    for i in range(batch_size):
        idx = mask_idx[i]
        xi = torch.cat([x[i, :idx], x[i, idx+1:]], dim=0)
        x_corrupted.append(xi.unsqueeze(0))
    x_corrupted = torch.cat(x_corrupted, dim=0)
    return x_corrupted, mask_idx

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

# -----------------------------
def main():
    """
    Main script to train or test the Transformer Masked Autoencoder.

    Workflow:
        1. Parse arguments and create output directories.
        2. Load CSV data files and convert to FeatureDataset.
        3. Split dataset into training and validation sets.
        4. Initialize the TransformerAutoencoder model.
        5. Train the model if not skipped.
        6. Run inference on the validation set and plot results.
    """
    set_seed(42)

    args = parse_args()

    inputs = args.inputs if args.inputs else ["avgWires.csv"]
    outDir = args.outdir
    maxEpochs = args.max_epochs
    batchSize = args.batch_size
    end_name = args.end_name
    doTraining = not args.no_train
    os.makedirs(outDir, exist_ok=True)

    # -----------------------------
    print('\n\nLoading data...')
    startT_data = time.time()

    events = []
    for fname in inputs:
        print(f"Loading data from {fname} ...")
        events.extend(read_file(fname))  # returns [N,12] array

    # Define plotter
    plotter = Plotter(print_dir=outDir, end_name=end_name)

    # Dataset: convert [N,12] → [N,6,2]
    dataset = FeatureDataset(events)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    print('\n\nTrain size:', train_size)
    print('Validation size:', val_size)

    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batchSize, shuffle=False)

    X_sample = next(iter(train_loader))
    print('X_sample shape:', X_sample.shape)  # e.g. torch.Size([32,6,2])

    endT_data = time.time()
    print(f'Loading data took {endT_data - startT_data:.2f}s \n\n')

    # -----------------------------
    if args.d_model % args.nhead != 0:
        raise ValueError(f"d_model ({args.d_model}) must be divisible by nhead ({args.nhead})")

    model = TransformerAutoencoder(
        seq_len=X_sample.shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        lr=args.lr
    )

    loss_tracker = LossTracker()

    # -----------------------------
    if doTraining:
        if args.device == "cpu":
            accelerator = "cpu"; devices = 1
        elif args.device == "gpu":
            if torch.cuda.is_available(): accelerator="gpu"; devices=1
            else: print("GPU not available. Falling back to CPU."); accelerator="cpu"; devices=1
        elif args.device == "auto":
            if torch.cuda.is_available(): accelerator="gpu"; devices="auto"
            else: accelerator="cpu"; devices=1
        else:
            raise ValueError(f"Unknown device option: {args.device}")

        print(f"Using accelerator={accelerator}, devices={devices}")

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy="auto",
            max_epochs=maxEpochs,
            enable_progress_bar=args.enable_progress_bar,
            log_every_n_steps=1000,
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            logger=False,
            callbacks=[loss_tracker]
        )

        print('\n\nTraining...')
        startT_train = time.time()
        trainer.fit(model, train_loader, val_loader)
        endT_train = time.time()
        print(f'Training took {(endT_train - startT_train)/60:.2f} minutes \n\n')

        plotter.plotTrainLoss(loss_tracker)

        # Save model
        model.to("cpu")
        torchscript_model = torch.jit.script(model)
        torchscript_model.save(f"{outDir}/tmae_{end_name}.pt")

    # -----------------------------
    # Load model for inference
    model_file = f"{outDir}/tmae_{end_name}.pt" if doTraining else "nets/tmae_default.pt"
    model = torch.jit.load(model_file)
    model.eval()

    all_preds = []
    all_targets = []

    startT_test = time.time()
    with torch.no_grad():
        for batch in val_loader:
            x_corrupted, mask_idx = corrupt_input(batch, seq_len=6)
            y_true = batch[torch.arange(batch.size(0)), mask_idx]  # [batch,2]
            y_pred = model(x_corrupted, mask_idx)                  # [batch,2]

            all_preds.append(y_pred.cpu())
            all_targets.append(y_true.cpu())

    endT_test = time.time()
    print(f'Test with {val_size} samples took {endT_test - startT_test:.2f}s \n\n')

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    plotter.plot_pred_target(all_preds, all_targets)
    plotter.plot_diff(all_preds, all_targets)

    print("Predictions shape:", all_preds.shape)  # [val_size,2]
    print("Targets shape:", all_targets.shape)
    print("First 10 predictions:\n", all_preds[:10].numpy())
    print("First 10 true values:\n", all_targets[:10].numpy())


if __name__ == "__main__":
    main()
