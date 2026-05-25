"""
models/tab_transformer.py — Classical TabTransformer for CKD binary classification.

Architecture:
  Input (B, 24) → feature projection → pos embedding → Transformer encoder
  → mean pool → LayerNorm → Linear → logit

Step 2 of the QIP 2027 pipeline.
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# ── Seeds ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """Stops training when val loss does not improve for `patience` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class TabTransformer(nn.Module):
    """
    Classical TabTransformer for tabular binary classification.

    Each feature is treated as a token; all 24 tokens are passed through
    a Transformer encoder; mean pooling gives the final representation.
    """

    def __init__(
        self,
        n_features: int = 24,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        self.feature_proj = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_features, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.feature_proj.weight)
        nn.init.zeros_(self.feature_proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, n_features) → logit (B, 1)."""
        x = x.unsqueeze(-1)                # (B, 24, 1)
        x = self.feature_proj(x)           # (B, 24, d_model)
        x = x + self.pos_embedding         # (B, 24, d_model)
        x = self.encoder(x)                # (B, 24, d_model)
        x = x.mean(dim=1)                  # (B, d_model)
        return self.head(self.norm(x))     # (B, 1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probability for the positive class."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).squeeze(-1)


def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(
        torch.FloatTensor(X), torch.FloatTensor(y)
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def train_tab_transformer(
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    patience: int = 10,
) -> TabTransformer:
    """Train the classical TabTransformer; return the best model."""
    print(f"Using device: {DEVICE}")

    # ── Load data ──────────────────────────────────────────────────────────────
    for fname in ["X_train.npy", "X_val.npy", "y_train.npy", "y_val.npy"]:
        if not (data_dir / fname).exists():
            raise FileNotFoundError(
                f"{data_dir / fname} not found. Run preprocessing.py first."
            )

    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)

    n_features = X_train.shape[1]
    config = {
        "n_features": n_features, "d_model": 32, "n_heads": 4,
        "n_layers": 2, "dim_ff": 128, "dropout": 0.1,
    }

    # ── Model ──────────────────────────────────────────────────────────────────
    model = TabTransformer(**config).to(DEVICE)
    print("\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

    # ── Optimiser & loss ───────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.BCEWithLogitsLoss()
    early_stop = EarlyStopping(patience=patience)

    best_val_loss = float("inf")
    best_state: dict = {}
    best_epoch = 1
    history: dict = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\nTraining for up to {epochs} epochs (early stopping patience={patience}):")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb).squeeze(-1)
                val_losses.append(criterion(logits, yb).item())
                preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(yb.long().cpu().numpy())

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(avg_val)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_acc"].append(val_acc)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch [{epoch:3d}/{epochs}] | "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | "
            f"Val Acc: {val_acc*100:.2f}%"
        )
        sys.stdout.flush()

        if early_stop(avg_val):
            print(f"\nEarly stopping triggered at epoch {epoch} (patience={patience})")
            break

    # ── Restore best ──────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    best_val_acc = max(history["val_acc"])
    print(f"\nBest Val Accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}")

    # ── Save ──────────────────────────────────────────────────────────────────
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "tab_transformer.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "history": history,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        },
        save_path,
    )
    print(f"Model saved to {save_path}")
    return model


def run_training() -> None:
    """Entry point when running this file directly."""
    print("=" * 60)
    print("STEP 2 — CLASSICAL TABTRANSFORMER TRAINING")
    print("=" * 60)
    train_tab_transformer()
    print("=" * 60)


if __name__ == "__main__":
    run_training()
