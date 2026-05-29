"""
models/hybrid_quantum_transformer.py -- Hybrid Quantum-Classical TabTransformer.

Upgraded architecture (Q1-grade):
  - Hardware-Efficient Ansatz (HEA): 6 qubits, 3 variational layers
  - RY + RZ rotations per qubit per layer (36 variational params)
  - Data re-uploading: features encoded at every layer (not just first)
  - Ring CNOT entanglement
  - Configurable via QuantumCircuitConfig dataclass

Backward compat: QuantumCircuitConfig defaults to the canonical 6-qubit HEA.
  For the old 4-qubit model, pass QuantumCircuitConfig(n_qubits=4, n_vqc_layers=2,
  data_reuploading=False) -- used only in ablation study.

diff_method='backprop' allows gradients to flow through the quantum simulation
via PyTorch autograd without parameter-shift rules.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
RESULTS_DIR = BASE_DIR / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Quantum circuit configuration ──────────────────────────────────────────────

@dataclass
class QuantumCircuitConfig:
    """
    Configures the variational quantum circuit inside VQCFeedForward.
    Default values correspond to the canonical Q1-grade HEA architecture.
    """
    n_qubits: int = 6             # 6-qubit HEA (upgraded from 4)
    n_vqc_layers: int = 3         # 3 variational layers (upgraded from 2)
    data_reuploading: bool = True  # encode inputs at every layer
    entanglement: str = "ring"    # "ring" | "full"
    interface: str = "torch"

    @property
    def n_params(self) -> int:
        """Total variational parameters: 2 (RY+RZ) × n_qubits × n_vqc_layers."""
        return 2 * self.n_qubits * self.n_vqc_layers

    def weight_shapes(self) -> dict:
        return {"weights": (self.n_vqc_layers, 2, self.n_qubits)}


# ── Default config (canonical 6-qubit HEA) ────────────────────────────────────
DEFAULT_QC_CFG = QuantumCircuitConfig()

# ── Legacy 4-qubit config (for ablation comparison) ───────────────────────────
LEGACY_QC_CFG = QuantumCircuitConfig(
    n_qubits=4, n_vqc_layers=2, data_reuploading=False
)


# ── HEA circuit builder ───────────────────────────────────────────────────────

def build_hea_layer(cfg: QuantumCircuitConfig = DEFAULT_QC_CFG) -> qml.qnn.TorchLayer:
    """
    Build a PennyLane TorchLayer implementing the Hardware-Efficient Ansatz.

    Each variational layer applies:
      1. Data re-uploading (AngleEmbedding with RY, if data_reuploading=True)
      2. RY(theta) + RZ(phi) on every qubit
      3. Ring CNOT entanglement: CNOT(q_i, q_{i+1 mod n})

    Inputs  : (n_qubits,)       -- per-sample, scaled to [-pi, pi] by VQCFeedForward
    Weights : (n_layers, 2, n_qubits) -- trainable variational parameters
    Output  : (n_qubits,)       -- PauliZ expectation per qubit
    """
    n_q = cfg.n_qubits
    n_l = cfg.n_vqc_layers
    reupload = cfg.data_reuploading

    dev = qml.device("default.qubit", wires=n_q)

    @qml.qnode(dev, interface=cfg.interface, diff_method="backprop")
    def hea_circuit(inputs, weights):
        for layer_idx in range(n_l):
            # Data re-uploading: encode at every layer (or only layer 0)
            if layer_idx == 0 or reupload:
                qml.AngleEmbedding(inputs, wires=range(n_q), rotation="Y")
            # RY + RZ variational rotations
            for i in range(n_q):
                qml.RY(weights[layer_idx, 0, i], wires=i)
                qml.RZ(weights[layer_idx, 1, i], wires=i)
            # Entanglement
            if cfg.entanglement == "ring":
                for i in range(n_q):
                    qml.CNOT(wires=[i, (i + 1) % n_q])
            else:  # "full"
                for i in range(n_q):
                    for j in range(i + 1, n_q):
                        qml.CNOT(wires=[i, j])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

    return qml.qnn.TorchLayer(hea_circuit, cfg.weight_shapes())


# ── Legacy 4-qubit VQC builder (kept for backward compat / ablation) ──────────

def build_vqc_layer(cfg: QuantumCircuitConfig = LEGACY_QC_CFG) -> qml.qnn.TorchLayer:
    """
    Legacy 4-qubit VQC: RY angle encoding + RY variational layers + ring CNOT.
    Kept for backward compatibility with saved checkpoints.
    """
    n_q = cfg.n_qubits
    n_l = cfg.n_vqc_layers
    dev = qml.device("default.qubit", wires=n_q)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def vqc_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_q), rotation="Y")
        for layer_idx in range(n_l):
            for i in range(n_q):
                qml.RY(weights[layer_idx, i], wires=i)
            for i in range(n_q):
                qml.CNOT(wires=[i, (i + 1) % n_q])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

    weight_shapes = {"weights": (n_l, n_q)}
    return qml.qnn.TorchLayer(vqc_circuit, weight_shapes)


# ── Circuit diagram export ────────────────────────────────────────────────────

def draw_circuit(cfg: QuantumCircuitConfig = DEFAULT_QC_CFG, out_path: Optional[str] = None) -> str:
    """
    Draw the HEA circuit using qml.draw and optionally save to a text file.
    Returns the diagram string.
    """
    n_q = cfg.n_qubits
    n_l = cfg.n_vqc_layers
    dev = qml.device("default.qubit", wires=n_q)

    @qml.qnode(dev)
    def demo_circuit(inputs, weights):
        for layer_idx in range(n_l):
            if layer_idx == 0 or cfg.data_reuploading:
                qml.AngleEmbedding(inputs, wires=range(n_q), rotation="Y")
            for i in range(n_q):
                qml.RY(weights[layer_idx, 0, i], wires=i)
                qml.RZ(weights[layer_idx, 1, i], wires=i)
            for i in range(n_q):
                qml.CNOT(wires=[i, (i + 1) % n_q])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

    dummy_inputs = np.zeros(n_q)
    dummy_weights = np.zeros((n_l, 2, n_q))
    try:
        diagram = qml.draw(demo_circuit)(dummy_inputs, dummy_weights)
    except Exception as e:
        diagram = f"Circuit diagram unavailable: {e}"

    header = (
        f"Hardware-Efficient Ansatz (HEA) — {n_q} qubits, {n_l} layers\n"
        f"Data re-uploading: {cfg.data_reuploading} | "
        f"Entanglement: {cfg.entanglement} | "
        f"Variational params: {cfg.n_params}\n"
        + "=" * 60 + "\n"
    )
    full_text = header + diagram

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        # Encode safely for cross-platform text files
        safe_text = full_text.encode("ascii", errors="replace").decode("ascii")
        Path(out_path).write_text(safe_text)
        print(f"  [quantum] Circuit diagram saved: {out_path}")

    try:
        print("\n" + full_text)
    except UnicodeEncodeError:
        safe = full_text.encode("ascii", errors="replace").decode("ascii")
        print("\n" + safe)

    return full_text


# ── VQCFeedForward (HEA version) ──────────────────────────────────────────────

class VQCFeedForward(nn.Module):
    """
    Quantum feed-forward sublayer using the Hardware-Efficient Ansatz.

    x (B, S, d_model)
     -> Linear(d_model, n_qubits)
     -> tanh(.) * pi                    [scale to [-pi, pi]]
     -> HEA circuit (n_qubits outputs)
     -> Linear(n_qubits, d_model)
     -> Dropout
    """

    def __init__(
        self,
        d_model: int = 32,
        dropout: float = 0.1,
        qc_cfg: QuantumCircuitConfig = DEFAULT_QC_CFG,
    ):
        super().__init__()
        self.qc_cfg = qc_cfg
        self.down_proj = nn.Linear(d_model, qc_cfg.n_qubits)
        self.qlayer = build_hea_layer(qc_cfg)
        self.up_proj = nn.Linear(qc_cfg.n_qubits, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Tanh()

        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.reshape(B * S, D)
        x_down = self.down_proj(x_flat)            # (B*S, n_qubits)
        x_scaled = self.act(x_down) * math.pi      # scale to [-pi, pi]
        x_q = self.qlayer(x_scaled)                # (B*S, n_qubits)
        x_up = self.up_proj(x_q)                   # (B*S, d_model)
        return self.dropout(x_up).reshape(B, S, D)


# ── Transformer encoder layer ─────────────────────────────────────────────────

class HybridTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with HEA feed-forward sublayer.
    Multi-head attention is identical to the classical model.
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        dropout: float = 0.1,
        qc_cfg: QuantumCircuitConfig = DEFAULT_QC_CFG,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.vqc_ff = VQCFeedForward(d_model, dropout, qc_cfg)
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
        attn_out, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        x = self.norm1(src + self.dropout1(attn_out))
        ff_out = self.vqc_ff(x)
        return self.norm2(x + self.dropout2(ff_out))


# ── Full HybridTabTransformer ─────────────────────────────────────────────────

class HybridTabTransformer(nn.Module):
    """
    Hybrid Quantum-Classical TabTransformer with configurable VQC.

    Default: 6-qubit HEA, 3 layers, data re-uploading.
    Ablation: pass qc_cfg=LEGACY_QC_CFG for 4-qubit comparison.
    """

    def __init__(
        self,
        n_features: int = 24,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        qc_cfg: QuantumCircuitConfig = DEFAULT_QC_CFG,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.qc_cfg = qc_cfg

        self.feature_proj = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_features, d_model))

        self.layers = nn.ModuleList([
            HybridTransformerEncoderLayer(d_model, n_heads, dropout, qc_cfg)
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
        x = x.unsqueeze(-1)
        x = self.feature_proj(x)
        x = x + self.pos_embedding
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.head(self.norm(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).squeeze(-1)

    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        quantum = sum(
            p.numel()
            for layer in self.layers
            for p in layer.vqc_ff.qlayer.parameters()
            if p.requires_grad
        )
        return {"total": total, "classical": total - quantum, "quantum": quantum}


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
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


def _make_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    gen = torch.Generator()
    gen.manual_seed(SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=gen if shuffle else None)


# ── Training function ─────────────────────────────────────────────────────────

def train_hybrid_transformer(
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
    epochs: int = 50,
    lr: float = 5e-4,
    batch_size: int = 32,
    patience: int = 10,
    qc_cfg: QuantumCircuitConfig = DEFAULT_QC_CFG,
    save_name: str = "hybrid_qt.pt",
    dp_optimizer=None,          # optional DPOptimizer wrapper
) -> HybridTabTransformer:
    """Train the Hybrid Quantum-Classical TabTransformer; return best model."""
    print(f"Using device: {DEVICE}")
    print(f"PennyLane: {qml.__version__}")
    print(f"Quantum config: {qc_cfg.n_qubits}q x {qc_cfg.n_vqc_layers}L "
          f"(re-upload={qc_cfg.data_reuploading}, params={qc_cfg.n_params})")

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
        "qc_cfg": {
            "n_qubits": qc_cfg.n_qubits,
            "n_vqc_layers": qc_cfg.n_vqc_layers,
            "data_reuploading": qc_cfg.data_reuploading,
            "entanglement": qc_cfg.entanglement,
        },
    }

    # Save circuit diagram
    draw_circuit(qc_cfg, out_path=str(RESULTS_DIR / "quantum_circuit.txt"))

    model = HybridTabTransformer(n_features=n_features, qc_cfg=qc_cfg).to(DEVICE)
    counts = model.param_counts()
    print(f"\nParams: {counts['total']:,} total "
          f"({counts['classical']:,} classical + {counts['quantum']:,} quantum)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if dp_optimizer is not None:
        optimizer = dp_optimizer(optimizer)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()
    early_stop = EarlyStopping(patience=patience)

    best_val_loss = float("inf")
    best_state: dict = {}
    best_epoch = 1
    history: dict = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\nTraining up to {epochs} epochs (lr={lr}, batch={batch_size}, patience={patience}):")
    print("NOTE: VQC evaluation makes each epoch slower — please wait.")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
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
            f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | Acc: {val_acc*100:.2f}%"
        )
        sys.stdout.flush()

        if early_stop(avg_val):
            print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            break

    model.load_state_dict(best_state)
    best_val_acc = max(history["val_acc"])
    print(f"\nBest Val Acc: {best_val_acc*100:.2f}% at epoch {best_epoch}")

    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / save_name
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "qc_cfg": {
            "n_qubits": qc_cfg.n_qubits,
            "n_vqc_layers": qc_cfg.n_vqc_layers,
            "data_reuploading": qc_cfg.data_reuploading,
            "entanglement": qc_cfg.entanglement,
        },
    }, save_path)
    print(f"Hybrid model saved: {save_path}")
    return model


def run_training() -> None:
    print("=" * 60)
    print("STEP 3 -- HYBRID QUANTUM-CLASSICAL TRANSFORMER TRAINING (HEA)")
    print("=" * 60)
    train_hybrid_transformer()
    print("=" * 60)


if __name__ == "__main__":
    run_training()
