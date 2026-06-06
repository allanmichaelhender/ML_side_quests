"""
PyTorch Autoencoder for anomaly detection.

Trained on normal data only — learns to reconstruct normal patterns.
Anomalies are detected by high reconstruction error (MSE).
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_MODEL_DIR = PROJECT / "results" / "autoencoder_model"


class Autoencoder(nn.Module):
    """Fully-connected autoencoder.

    Architecture:
        input_dim → 64 → 32 → bottleneck → 32 → 64 → input_dim
    """

    def __init__(self, input_dim: int, bottleneck_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.encoder(x)
        reconstructed = self.decoder(bottleneck)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project to bottleneck representation."""
        return self.encoder(x)


def train_autoencoder(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    input_dim: Optional[int] = None,
    bottleneck_dim: int = 8,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    device: Optional[torch.device] = None,
    model_dir: Optional[Path] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Autoencoder, dict]:
    """Train an autoencoder on normal data.

    Only uses normal samples (y=0) for training.

    Args:
        X_train: Training features (all samples; normal ones selected internally
                 if y not provided).
        X_val: Optional validation features.
        input_dim: Feature dimension. Inferred from X_train if None.
        bottleneck_dim: Size of the bottleneck layer.
        epochs: Maximum number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: L2 regularization.
        patience: Early stopping patience (validation loss).
        device: torch device. Auto-detects CUDA if None.
        model_dir: Directory to save the trained model.
        random_state: Random seed.
        verbose: Print progress.

    Returns:
        model: Trained Autoencoder.
        history: Dict with 'train_losses' and 'val_losses'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(random_state)
    input_dim = input_dim or X_train.shape[1]

    model = Autoencoder(input_dim, bottleneck_dim=bottleneck_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Prepare data loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and len(X_val) > 0:
        val_dataset = TensorDataset(torch.from_numpy(X_val).float())
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = {"train_losses": [], "val_losses": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            recon = model(x)
            loss = criterion(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        history["train_losses"].append(train_loss)

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    recon = model(x)
                    loss = criterion(recon, x)
                    val_loss += loss.item() * x.size(0)
            val_loss /= len(val_loader.dataset)
            history["val_losses"].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break

            if verbose and epoch % 10 == 0:
                print(
                    f"  Epoch {epoch:3d}/{epochs}  "
                    f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
                )
        else:
            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.6f}")

    # Restore best state if validation was used
    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - start_time
    if verbose:
        print(
            f"  Training complete in {elapsed:.1f}s. "
            f"Final train loss: {history['train_losses'][-1]:.6f}"
        )

    # Save model
    if model_dir is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / "autoencoder.pt")
        with open(model_dir / "args.json", "w") as f:
            import json

            json.dump(
                {
                    "input_dim": input_dim,
                    "bottleneck_dim": bottleneck_dim,
                    "epochs_trained": len(history["train_losses"]),
                },
                f,
            )
        logger.info(f"Model saved to {model_dir}")

    return model, history


@torch.no_grad()
def compute_anomaly_scores(
    model: Autoencoder,
    X: np.ndarray,
    device: Optional[torch.device] = None,
    batch_size: int = 512,
) -> np.ndarray:
    """Compute per-sample reconstruction error as anomaly score.

    Args:
        model: Trained Autoencoder.
        X: (n_samples, n_features) array.
        device: torch device.
        batch_size: Batch size for evaluation.

    Returns:
        scores: (n_samples,) array — higher = more anomalous.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_scores = []
    for batch in loader:
        x = batch[0].to(device)
        recon = model(x)
        mse = torch.mean((x - recon) ** 2, dim=1)
        all_scores.append(mse.cpu().numpy())

    return np.concatenate(all_scores)


def load_autoencoder(
    model_dir: Path,
    device: Optional[torch.device] = None,
) -> Autoencoder:
    """Load a trained autoencoder from disk."""
    import json

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(model_dir / "args.json") as f:
        args = json.load(f)

    model = Autoencoder(
        input_dim=args["input_dim"],
        bottleneck_dim=args.get("bottleneck_dim", 8),
    )
    state = torch.load(
        model_dir / "autoencoder.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info(f"Autoencoder loaded from {model_dir}")
    return model
