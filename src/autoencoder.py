import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json


# ── Model Definition ─────────────────────────────────────────────

class HallucinationAutoencoder(nn.Module):
    """
    Autoencoder trained ONLY on factual samples.

    Why this works for anomaly detection:
    The autoencoder learns to compress and reconstruct 'normal'
    (factual) text embeddings. When it sees hallucinated text,
    the reconstruction error is high because the pattern doesn't
    match what it learned. High error = anomaly = hallucination.

    Architecture:
    Encoder: 384 → 256 → 128 → 32 → 16 (bottleneck)
    Decoder: 16  → 32  → 128 → 256 → 384
    """

    def __init__(self, input_dim=384, latent_dim=16, dropout=0.2):
        super(HallucinationAutoencoder, self).__init__()

        self.latent_dim = latent_dim

        # Encoder — compresses embedding to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        # Decoder — reconstructs embedding from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, input_dim)
            # No activation — output is unbounded (MSE loss)
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        return recon

    def encode(self, x):
        """Get latent representation only."""
        return self.encoder(x)

    def reconstruction_error(self, x):
        """
        Per-sample MSE reconstruction error.
        Higher = more anomalous = more likely hallucinated.
        """
        recon = self.forward(x)
        error = torch.mean((x - recon) ** 2, dim=1)
        return error


# ── Training Utilities ─────────────────────────────────────────────

def get_factual_embeddings(X_train, y_train):
    """
    Filter training data to keep ONLY factual samples (label=0).
    The autoencoder is trained unsupervised on 'normal' data only.
    """
    mask      = y_train == 0
    X_factual = X_train[mask]
    print(f"  Factual samples for AE training: {X_factual.shape[0]}")
    return X_factual


def prepare_ae_dataloader(X_factual, batch_size=32):
    """
    DataLoader for autoencoder — input = target (reconstruction).
    """
    X_t    = torch.FloatTensor(X_factual)
    ds     = TensorDataset(X_t, X_t)   # input and target are the same
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    print(f"  AE train batches: {len(loader)}")
    return loader


def train_ae_epoch(model, loader, optimizer, device):
    """One training epoch for the autoencoder."""
    model.train()
    total_loss = 0.0
    total      = 0
    criterion  = nn.MSELoss()

    for X_batch, _ in loader:
        X_batch = X_batch.to(device)

        optimizer.zero_grad()
        recon = model(X_batch)
        loss  = criterion(recon, X_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)
        total      += len(X_batch)

    return total_loss / total


def eval_ae(model, X, device, batch_size=64):
    """
    Compute reconstruction errors for all samples in X.
    Returns numpy array of per-sample MSE errors.
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            err   = model.reconstruction_error(batch)
            errors.append(err.cpu().numpy())

    return np.concatenate(errors)


def train_autoencoder(model, ae_loader, config, device, save_path):
    """
    Full autoencoder training loop with early stopping.
    """
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    best_loss        = float('inf')
    patience_counter = 0
    history          = {"train_loss": []}

    print(f"\n  Training AE for up to {config['epochs']} epochs...")
    print(f"  LR={config['lr']}, Latent={config['latent_dim']}, "
          f"Batch={config['batch_size']}")
    print("-" * 55)

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_ae_epoch(model, ae_loader, optimizer, device)
        scheduler.step(train_loss)
        history["train_loss"].append(train_loss)

        print(f"  Epoch {epoch:03d} | Train Loss: {train_loss:.6f}")

        if train_loss < best_loss:
            best_loss        = train_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best AE saved (loss={train_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    return history


def compute_threshold(model, X_factual, X_hallucinated, device):
    """
    Find the optimal reconstruction error threshold that best
    separates factual from hallucinated samples.

    Returns the threshold value and stats.
    """
    factual_errors      = eval_ae(model, X_factual, device)
    hallucinated_errors = eval_ae(model, X_hallucinated, device)

    # Threshold = midpoint between the two distribution means
    threshold = (factual_errors.mean() + hallucinated_errors.mean()) / 2

    stats = {
        "factual_mean":      float(factual_errors.mean()),
        "factual_std":       float(factual_errors.std()),
        "hallucinated_mean": float(hallucinated_errors.mean()),
        "hallucinated_std":  float(hallucinated_errors.std()),
        "threshold":         float(threshold)
    }

    print(f"\n  Factual error:      {stats['factual_mean']:.6f} "
          f"± {stats['factual_std']:.6f}")
    print(f"  Hallucinated error: {stats['hallucinated_mean']:.6f} "
          f"± {stats['hallucinated_std']:.6f}")
    print(f"  Threshold:          {threshold:.6f}")

    return threshold, stats


def run_ae_hyperparameter_search(X_factual, config_list, device, save_dir):
    """
    Search over AE hyperparameters.
    """
    import itertools

    search_space = {
        "lr":         [1e-3, 5e-4],
        "latent_dim": [16, 32],
        "batch_size": [32, 64],
    }

    keys   = list(search_space.keys())
    combos = list(itertools.product(*search_space.values()))

    print(f"\n{'='*55}")
    print(f"AE HYPERPARAMETER SEARCH — {len(combos)} configurations")
    print(f"{'='*55}")

    best_loss   = float('inf')
    best_config = None

    for i, values in enumerate(combos):
        config = dict(zip(keys, values))
        config["epochs"]   = 50
        config["patience"] = 7
        config["dropout"]  = 0.2

        print(f"\n[{i+1}/{len(combos)}] Config: {config}")

        ae_loader = prepare_ae_dataloader(
            X_factual, batch_size=config["batch_size"]
        )
        model = HallucinationAutoencoder(
            input_dim=384,
            latent_dim=config["latent_dim"],
            dropout=config["dropout"]
        ).to(device)

        save_path = os.path.join(save_dir, f"ae_search_{i+1}.pt")
        history   = train_autoencoder(
            model, ae_loader, config, device, save_path
        )

        final_loss = min(history["train_loss"])
        if final_loss < best_loss:
            best_loss   = final_loss
            best_config = config
            import shutil
            shutil.copy(save_path,
                        os.path.join(save_dir, "ae_best.pt"))

    print(f"\nBEST AE CONFIG: {best_config}")
    print(f"BEST AE LOSS:   {best_loss:.6f}")
    return best_config


if __name__ == "__main__":
    print("autoencoder.py loaded — import and use via Colab notebook.")