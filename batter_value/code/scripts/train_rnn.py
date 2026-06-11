#!/usr/bin/env python3
"""
Train LSTM (RNN family) to predict next-season wRC+.

Protocol (aligned with final ElasticNet):
  - Hyperparameter search on Train 2021+2022, Val 2023
  - Refit on full Train 2021+2022+2023
  - Report metrics only on Test 2024

Usage (from repo root):
    python RNN/train_rnn.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": ["Microsoft JhengHei", "sans-serif"],
    "axes.unicode_minus": False,
})
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from rnn_data import TRAIN_SEASONS, TEST_SEASON, build_arrays, load_panel, split_by_season  # noqa: E402
from rnn_model import BatterLSTM  # noqa: E402

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIER_BINS = [0, 80, 100, 130, 300]
TIER_LABELS = ["<80", "80-100", "100-130", "130+"]

HYPERPARAMS = [
    {"hidden_size": 16, "num_layers": 1, "dropout": 0.1, "lr": 1e-3},
    {"hidden_size": 32, "num_layers": 1, "dropout": 0.2, "lr": 1e-3},
    {"hidden_size": 32, "num_layers": 2, "dropout": 0.2, "lr": 5e-4},
    {"hidden_size": 64, "num_layers": 1, "dropout": 0.3, "lr": 1e-3},
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tier_mae(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tiers = np.digitize(y_true, TIER_BINS[1:-1])
    out = {}
    for i, label in enumerate(TIER_LABELS):
        mask = tiers == i
        if mask.sum() == 0:
            out[label] = float("nan")
        else:
            out[label] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    return out


def make_loader(
    X_seq: np.ndarray,
    X_age: np.ndarray,
    X_pos: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X_seq),
        torch.from_numpy(X_age),
        torch.from_numpy(X_pos),
        torch.from_numpy(y),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one(
    model: BatterLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    epochs: int = 120,
    patience: int = 15,
) -> tuple[BatterLSTM, float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    wait = 0

    for _ in range(epochs):
        model.train()
        for x_seq, x_age, x_pos, y in train_loader:
            x_seq, x_age, x_pos, y = (
                x_seq.to(DEVICE), x_age.to(DEVICE), x_pos.to(DEVICE), y.to(DEVICE)
            )
            opt.zero_grad()
            pred = model(x_seq, x_age, x_pos)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for x_seq, x_age, x_pos, y in val_loader:
                x_seq, x_age, x_pos = x_seq.to(DEVICE), x_age.to(DEVICE), x_pos.to(DEVICE)
                pred = model(x_seq, x_age, x_pos).cpu().numpy()
                val_preds.append(pred)
                val_true.append(y.numpy())
        val_mae = mean_absolute_error(np.concatenate(val_true), np.concatenate(val_preds))

        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


@torch.no_grad()
def predict(model: BatterLSTM, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, ys = [], []
    for x_seq, x_age, x_pos, y in loader:
        x_seq, x_age, x_pos = x_seq.to(DEVICE), x_age.to(DEVICE), x_pos.to(DEVICE)
        pred = model(x_seq, x_age, x_pos).cpu().numpy()
        preds.append(pred)
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds)


def plot_scatter(y: np.ndarray, pred: np.ndarray, mae: float, r2: float) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y, pred, alpha=0.55, s=45, c="#6A4C93", edgecolors="white", linewidths=0.4)
    lo = max(20, min(float(y.min()), float(pred.min())) - 10)
    hi = min(230, max(float(y.max()), float(pred.max())) + 10)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
    ax.set_xlabel("實際 wRC+（2024）", fontsize=11)
    ax.set_ylabel("預測 wRC+", fontsize=11)
    ax.set_title(f"LSTM（RNN）Test 2024\nMAE = {mae:.1f}  |  R² = {r2:.3f}", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIG / "rnn_actual_vs_pred.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    set_seed(42)
    print(f"Device: {DEVICE}")

    panel = load_panel()
    data = build_arrays(panel)
    split = split_by_season(data)

    tr_mask = split["train_seasons"] != 2023
    va_mask = split["train_seasons"] == 2023

    X_seq_tr = split["train_seq"][tr_mask]
    X_age_tr = split["train_age"][tr_mask]
    X_pos_tr = split["train_pos"][tr_mask]
    y_tr = split["train_y"][tr_mask]

    X_seq_va = split["train_seq"][va_mask]
    X_age_va = split["train_age"][va_mask]
    X_pos_va = split["train_pos"][va_mask]
    y_va = split["train_y"][va_mask]

    train_loader = make_loader(X_seq_tr, X_age_tr, X_pos_tr, y_tr)
    val_loader = make_loader(X_seq_va, X_age_va, X_pos_va, y_va, shuffle=False)

    print(f"HP search: Train {tr_mask.sum()} (2021+2022), Val {va_mask.sum()} (2023)")

    best_cfg = None
    best_val_mae = float("inf")

    for cfg in HYPERPARAMS:
        model = BatterLSTM(
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            pos_dim=split["train_pos"].shape[1],
        ).to(DEVICE)
        model, val_mae = train_one(
            model, train_loader, val_loader, lr=cfg["lr"]
        )
        print(f"  cfg={cfg}  val_MAE={val_mae:.3f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_cfg = cfg

    print(f"\nBest config: {best_cfg}  (val MAE={best_val_mae:.3f})")

    # Refit on full 2021-2023 with best hyperparameters (fixed epochs, no test leakage)
    full_loader = make_loader(
        split["train_seq"], split["train_age"], split["train_pos"], split["train_y"]
    )
    final_model = BatterLSTM(
        hidden_size=best_cfg["hidden_size"],
        num_layers=best_cfg["num_layers"],
        dropout=best_cfg["dropout"],
        pos_dim=split["train_pos"].shape[1],
    ).to(DEVICE)
    opt = torch.optim.Adam(final_model.parameters(), lr=best_cfg["lr"], weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    final_model.train()
    for _ in range(80):
        for x_seq, x_age, x_pos, y in full_loader:
            x_seq, x_age, x_pos, y = (
                x_seq.to(DEVICE), x_age.to(DEVICE), x_pos.to(DEVICE), y.to(DEVICE)
            )
            opt.zero_grad()
            loss = loss_fn(final_model(x_seq, x_age, x_pos), y)
            loss.backward()
            opt.step()

    test_loader = make_loader(
        split["test_seq"], split["test_age"], split["test_pos"], split["test_y"],
        shuffle=False,
    )
    y_te, pred_te = predict(final_model, test_loader)

    mae = float(mean_absolute_error(y_te, pred_te))
    r2 = float(r2_score(y_te, pred_te))
    tiers = tier_mae(y_te, pred_te)

    # Null model baseline
    null_pred = np.full_like(y_te, y_te.mean())
    null_mae = float(mean_absolute_error(y_te, null_pred))
    null_r2 = float(r2_score(y_te, null_pred))

    print("\n=== LSTM (RNN) Test 2024 ===")
    print(f"MAE  : {mae:.3f}")
    print(f"R2   : {r2:.3f}")
    print(f"Null MAE (guess mean): {null_mae:.3f}  |  Null R2: {null_r2:.3f}")
    for k, v in tiers.items():
        print(f"  {k}: MAE={v:.1f}")

    metrics = {
        "model": "LSTM (RNN)",
        "train_seasons": list(TRAIN_SEASONS),
        "test_season": TEST_SEASON,
        "n_train": int(len(split["train_y"])),
        "n_test": int(len(y_te)),
        "mae": round(mae, 3),
        "r2": round(r2, 3),
        "null_mae": round(null_mae, 3),
        "null_r2": round(null_r2, 3),
        "best_config": best_cfg,
        "tier_mae": {k: round(v, 2) for k, v in tiers.items()},
    }
    (RESULTS / "rnn_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_scatter(y_te, pred_te, mae, r2)
    print(f"\nSaved figures -> {FIG}")
    print(f"Saved metrics -> {RESULTS / 'rnn_metrics.json'}")


if __name__ == "__main__":
    main()
