"""
models/hybrid_quantum_transformer.py — Hybrid Quantum-Classical TabTransformer.

The feed-forward sublayer inside each Transformer encoder layer is replaced by a
Variational Quantum Circuit (VQC) implemented with PennyLane:

  FF sublayer (classical):  x → Linear(d,d_ff) → GELU → Linear(d_ff,d)
  FF sublayer (quantum):    x → Linear(d,4) → tanh×π → VQC(4 qubits) → Linear(4,d)

VQC circuit (4 qubits, default.qubit):
  AngleEmbedding (RY rotations)
  2× variational layers: RY(θ) on each qubit + CNOT ring entanglement
  Measurement: PauliZ expectation on all 4 wires

diff_method='backprop' allows gradients to flow through the quantum simulation
via PyTorch autograd, enabling efficient training without parameter-shift rules.

Step 3 of the QIP 2027 pipeline.
"""

import math
import random
import sys
from pathlib import Path

import numpy as np
import pennylane as qml
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

# ── Quantum constants ──────────────────────────────────────────────────────────
N_QUBITS = 4
N_VQC_LAYERS = 2


def build_vqc_layer() -> qml.qnn.TorchLayer:
    """Create a PennyLane TorchLayer wrapping the 4-qubit VQC."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def vqc_circuit(inputs, weights):
        """
        inputs : (N_QUBITS,)  — per-sample; TorchLayer batches automatically
        weights: (N_VQC_LAYERS, N_QUBITS)
        """
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        for layer_idx in range(N_VQC_LAYERS):
            for i in range(N_QUBITS):
                qml.RY(weights[layer_idx, i], wires=i)
            for i in range(N_QUBITS):
                qml.CNOT(wires=[i, (i + 1) % N_QUBITS])  # ring entanglement
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    weight_shapes = {"weights": (N_VQC_LAYERS, N_QUBITS)}
    return qml.qnn.TorchLayer(vqc_circuit, weight_shapes)


def print_circuit_diagram() -> None:
    """Draw and print the VQC circuit to the terminal."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def demo_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        for li in range(N_VQC_LAYERS):
            for i in range(N_QUBITS):
                qml.RY(weights[li, i], wires=i)
            for i in range(N_QUBITS):
                qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    dummy_inputs = np.zeros(N_QUBITS)
    dummy_weights = np.zeros((N_VQC_LAYERS, N_QUBITS))
    diagram = qml.draw(demo_circuit)(dummy_inputs, dummy_weights)
    print("\nQuantum circuit diagram:")
    # Encode safely for Windows terminals that don't support Unicode box-drawing
    try:
        print(diagram)
    except UnicodeEncodeError:
        safe = diagram.encode("ascii", errors="replace").decode("ascii")
        print(safe)


class VQCFeedForward(nn.Module):
    """
    Quantum feed-forward sublayer.

    Replaces the standard Linear→GELU→Linear block with:
      Linear(d_model→4) → tanh×π → VQC(4q) → Linear(4→d_model)
    """

    def __init__(self, d_model: int = 32, dropout: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(d_model, N_QUBITS)
        self.qlayer = build_vqc_layer()
        self.up_proj = nn.Linear(N_QUBITS, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Tanh()

        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, d_model) → (B, seq_len, d_model)."""
        B, S, D = x.shape
        x_flat = x.reshape(B * S, D)
        x_down = self.down_proj(x_flat)           # (B*S, 4)
        x_scaled = self.act(x_down) * math.pi    # scale to [-π, π]
        x_q = self.qlayer(x_scaled)               # (B*S, 4)
        x_up = self.up_proj(x_q)                  # (B*S, d_model)
        return self.dropout(x_up).reshape(B, S, D)


class HybridTransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer with VQC feed-forward sublayer.

    Self-attention is identical to the classical model; only the FF block
    is replaced by VQCFeedForward.
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.vqc_ff = VQCFeedForward(d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask=None,
        src_key_padding_mask=None,
    ) -> torch.Tensor:
        """src: (B, seq_len, d_model) → (B, seq_len, d_model)."""
        # Self-attention sublayer with post-norm residual
        attn_out, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        x = self.norm1(src + self.dropout1(attn_out))

        # Quantum FF sublayer with post-norm residual
        ff_out = self.vqc_ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class HybridTabTransformer(nn.Module):
    """
    Hybrid Quantum-Classical TabTransformer.

    Identical to the classical TabTransformer except the Transformer encoder
    uses HybridTransformerEncoderLayer (with VQC feed-forward) instead of
    nn.TransformerEncoderLayer.
    """

    def __init__(
        self,
        n_features: int = 24,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        self.feature_proj = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_features, d_model))

        # Use ModuleList instead of TransformerEncoder to avoid PyTorch 2.x
        # internal checks that reject custom encoder layers.
        self.layers = nn.ModuleList([
            HybridTransformerEncoderLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

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
        """x: (B, n_features) → logit (B, 1)."""
        x = x.unsqueeze(-1)            # (B, 24, 1)
        x = self.feature_proj(x)       # (B, 24, d_model)
        x = x + self.pos_embedding     # (B, 24, d_model)
        for layer in self.layers:
            x = layer(x)              # (B, 24, d_model)
        x = x.mean(dim=1)             # (B, d_model)
        return self.head(self.norm(x))  # (B, 1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probability for the positive class."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).squeeze(-1)


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


def train_hybrid_transformer(
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
    epochs: int = 50,
    lr: float = 5e-4,
    batch_size: int = 32,
    patience: int = 10,
) -> HybridTabTransformer:
    """Train the Hybrid Quantum-Classical TabTransformer; return best model."""
    print(f"Using device: {DEVICE}")
    print(f"PennyLane version: {qml.__version__}")

    # ── Check data ─────────────────────────────────────────────────────────────
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
        "n_features": n_features, "d_model": 32,
        "n_heads": 4, "n_layers": 2, "dropout": 0.1,
    }

    # ── Circuit diagram ────────────────────────────────────────────────────────
    print_circuit_diagram()

    # ── Model ──────────────────────────────────────────────────────────────────
    model = HybridTabTransformer(**config).to(DEVICE)
    print("\nHybrid model architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Count quantum parameters separately
    quantum_params = sum(
        p.numel()
        for layer in model.layers
        for p in layer.vqc_ff.qlayer.parameters()
        if p.requires_grad
    )
    classical_params = total_params - quantum_params
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"  Classical parameters: {classical_params:,}")
    print(f"  Quantum parameters:   {quantum_params:,} ({N_VQC_LAYERS}×{N_QUBITS}×{config['n_layers']} layers)")

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

    print(f"\nTraining for up to {epochs} epochs (lr={lr}, batch={batch_size}, patience={patience}):")
    print("NOTE: VQC evaluation makes each epoch slower than classical — please wait.")
    print("-" * 75)

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
    save_path = models_dir / "hybrid_qt.pt"
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
    print(f"Hybrid model saved to {save_path}")
    return model


def run_training() -> None:
    """Entry point when running this file directly."""
    print("=" * 60)
    print("STEP 3 — HYBRID QUANTUM-CLASSICAL TRANSFORMER TRAINING")
    print("=" * 60)
    train_hybrid_transformer()
    print("=" * 60)


if __name__ == "__main__":
    run_training()
