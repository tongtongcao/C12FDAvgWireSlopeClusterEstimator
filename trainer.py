import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import math
from pytorch_lightning.callbacks import Callback

# --------- Dataset ---------
# Input: each row has 12 numbers: first 6 are avgWire, last 6 are slope
# Output shape: [6, 2], each cluster is (avgWire, slope)
class FeatureDataset(Dataset):
    def __init__(self, events):
        """
        Parameters
        ----------
        events : array-like, shape [N, 12]
            First 6 columns are avgWire, last 6 columns are slope.
        """
        events = np.array(events, dtype=np.float32)
        avg = events[:, :6]       # [N, 6]
        slope = events[:, 6:]     # [N, 6]
        combined = np.stack([avg, slope], axis=2)  # [N, 6, 2]
        self.data = combined

    def __len__(self):
        """
        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        torch.Tensor, shape [6, 2]
            Features for one sample.
        """
        return torch.tensor(self.data[idx])


# --------- Positional Encoding ---------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Parameters
        ----------
        d_model : int
            Dimensionality of the model.
        max_len : int, optional
            Maximum sequence length.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape [batch, seq_len, d_model]
            Input sequence.

        Returns
        -------
        torch.Tensor, shape [batch, seq_len, d_model]
            Sequence with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


# --------- Transformer Autoencoder ---------
class TransformerAutoencoder(pl.LightningModule):
    def __init__(self, seq_len=6, d_model=64, nhead=2, num_layers=3, lr=2e-4):
        """
        Parameters
        ----------
        seq_len : int
            Length of the input sequence (number of clusters).
        d_model : int
            Dimension of transformer model.
        nhead : int
            Number of attention heads.
        num_layers : int
            Number of transformer encoder layers.
        lr : float
            Learning rate.
        """
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.lr = lr

        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)

        self.mask_pos_embedding = nn.Embedding(seq_len, 16)
        self.mask_proj = nn.Linear(d_model + 16, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)
        )

        self.criterion = nn.MSELoss()

    def corrupt_input(self, x, mask_idx=None):
        """
        Randomly remove one cluster from each sequence.

        Parameters
        ----------
        x : torch.Tensor, shape [batch, seq_len, 2]
            Full input sequence.
        mask_idx : torch.Tensor or None, optional
            Indices of clusters to mask. If None, randomly chosen.

        Returns
        -------
        x_corrupted : torch.Tensor, shape [batch, seq_len-1, 2]
            Input sequences with one cluster removed.
        mask_idx : torch.Tensor, shape [batch]
            Indices of removed clusters.
        """
        batch_size, seq_len, feat_dim = x.shape
        if mask_idx is None:
            mask_idx = torch.randint(0, seq_len, (batch_size,), device=x.device)

        x_corrupted = []
        for i in range(batch_size):
            idx = mask_idx[i]
            xi = torch.cat([x[i, :idx], x[i, idx+1:]], dim=0)
            x_corrupted.append(xi.unsqueeze(0))
        x_corrupted = torch.cat(x_corrupted, dim=0)
        return x_corrupted, mask_idx

    def forward(self, x, mask_idx):
        """
        Parameters
        ----------
        x : torch.Tensor, shape [batch, seq_len-1, 2]
            Input sequence with one cluster removed.
        mask_idx : torch.Tensor, shape [batch]
            Indices of removed clusters.

        Returns
        -------
        torch.Tensor, shape [batch, 2]
            Predicted features for the missing cluster.
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        mask_embed = self.mask_pos_embedding(mask_idx)
        mask_embed_expanded = mask_embed.unsqueeze(1).expand(-1, x.size(1), -1)

        x = torch.cat([x, mask_embed_expanded], dim=-1)
        x = self.mask_proj(x)

        x = x.permute(1, 0, 2)  # [seq_len-1, batch, d_model]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # [batch, seq_len-1, d_model]

        x_selected = x.mean(dim=1)
        y_pred = self.fc(x_selected)
        return y_pred

    def training_step(self, batch, batch_idx):
        x_full = batch
        x_corrupted, mask_idx = self.corrupt_input(x_full)
        y_true = x_full[torch.arange(x_full.size(0)), mask_idx]
        y_pred = self(x_corrupted, mask_idx)
        loss = self.criterion(y_pred, y_true)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_full = batch
        x_corrupted, mask_idx = self.corrupt_input(x_full)
        y_true = x_full[torch.arange(x_full.size(0)), mask_idx]
        y_pred = self(x_corrupted, mask_idx)
        loss = self.criterion(y_pred, y_true)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        """
        Returns
        -------
        torch.optim.Optimizer
            Adam optimizer for training.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# --------- Loss Tracking Callback ---------
class LossTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())
