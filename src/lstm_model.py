import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json


# ── Model Definition ─────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier for hallucination detection.

    Input:  Sentence-BERT embeddings (batch, seq_len=1, embed_dim=384)
    Output: 3-class probabilities (factual / hallucinated / overconfident)

    Why BiLSTM?
    We treat each embedding as a single-step sequence. The BiLSTM
    learns to project the 384-dim embedding into a richer hidden
    representation before classification. For longer sequences
    (e.g. token-level embeddings) the bidirectionality becomes
    even more valuable.
    """

    def __init__(self, input_dim=384, hidden_dim=128, num_layers=2,
                 num_classes=3, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()

        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers

        # BiLSTM — processes embedding sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,    # doubles effective hidden size → 256
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),   # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
            # No softmax here — CrossEntropyLoss applies it internally
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)

        # Take the last time step output
        last_hidden = lstm_out[:, -1, :]   # (batch, hidden_dim * 2)

        logits = self.classifier(last_hidden)
        return logits


# ── Training Utilities ────────────────────────────────────────────

def get_class_weights(y_train, num_classes=3):
    """
    Compute inverse-frequency class weights to handle class imbalance.
    partially_true has only 13 samples — without this it gets ignored.
    """
    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    print(f"  Class weights: {dict(enumerate(weights.round(3)))}")
    return torch.FloatTensor(weights)


def prepare_dataloaders(X_train, X_val, y_train, y_val, batch_size=32):
    """
    Wrap numpy arrays into PyTorch DataLoaders.
    Adds a sequence dimension: (batch, 384) → (batch, 1, 384)
    """
    def to_loader(X, y, shuffle):
        X_t = torch.FloatTensor(X).unsqueeze(1)   # add seq_len dim
        y_t = torch.LongTensor(y)
        ds  = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader   = to_loader(X_val,   y_val,   shuffle=False)

    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, criterion, device):
    """Run one full training epoch. Returns avg loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    """Run one full validation epoch. Returns avg loss and accuracy."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits     = model(X_batch)
            loss       = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, config, device, save_path):
    """
    Full training loop with:
    - Weighted CrossEntropy for class imbalance
    - Adam optimizer with LR scheduler
    - Early stopping (patience=5)
    - Best model checkpointing
    """
    class_weights = get_class_weights(
        train_loader.dataset.tensors[1].numpy()
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=1e-5
    )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5,
    #     patience=3, verbose=True
    # )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=3
    )

    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    print(f"\n  Training for up to {config['epochs']} epochs...")
    print(f"  LR={config['lr']}, Dropout={config['dropout']}, "
          f"Batch={config['batch_size']}")
    print("-" * 55)

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {config['patience']} epochs)")
                break

    return history


# ── Hyperparameter Search ─────────────────────────────────────────

def run_hyperparameter_search(X_train, X_val, y_train, y_val,
                               device, save_dir):
    """
    Grid search over key hyperparameters.
    Trains one model per config and returns the best one.
    """
    search_space = {
        "lr":         [1e-3, 5e-4],
        "dropout":    [0.3, 0.4],
        "hidden_dim": [128, 256],
        "batch_size": [32, 64],
    }

    # Generate all combinations
    import itertools
    keys   = list(search_space.keys())
    combos = list(itertools.product(*search_space.values()))

    print(f"\n{'='*55}")
    print(f"HYPERPARAMETER SEARCH — {len(combos)} configurations")
    print(f"{'='*55}")

    best_val_loss = float('inf')
    best_config   = None
    results       = []

    for i, values in enumerate(combos):
        config = dict(zip(keys, values))
        config["epochs"]  = 30      # cap for search
        config["patience"] = 5

        print(f"\n[{i+1}/{len(combos)}] Config: {config}")

        train_loader, val_loader = prepare_dataloaders(
            X_train, X_val, y_train, y_val,
            batch_size=config["batch_size"]
        )

        model = BiLSTMClassifier(
            input_dim=384,
            hidden_dim=config["hidden_dim"],
            num_layers=2,
            num_classes=3,
            dropout=config["dropout"]
        ).to(device)

        save_path = os.path.join(save_dir, f"lstm_search_{i+1}.pt")
        history   = train_model(model, train_loader, val_loader,
                                config, device, save_path)

        final_val_loss = min(history["val_loss"])
        results.append({"config": config, "val_loss": final_val_loss})

        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_config   = config
            # Keep the best model as lstm_best.pt
            import shutil
            shutil.copy(save_path,
                        os.path.join(save_dir, "lstm_best.pt"))

    print(f"\n{'='*55}")
    print(f"BEST CONFIG: {best_config}")
    print(f"BEST VAL LOSS: {best_val_loss:.4f}")

    # Save search results
    with open(os.path.join(save_dir, "lstm_search_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return best_config, results


# ── Inference ─────────────────────────────────────────────────────

def predict(model, embeddings, device, batch_size=64):
    """
    Run inference on new embeddings.
    Returns class predictions and softmax probabilities.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.FloatTensor(
                embeddings[i:i+batch_size]
            ).unsqueeze(1).to(device)

            logits = model(batch)
            probs  = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    probs = np.vstack(all_probs)
    preds = probs.argmax(axis=1)
    return preds, probs


if __name__ == "__main__":
    print("lstm_model.py loaded — import and use via Colab notebook.")