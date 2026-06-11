"""Build 3-step sequences (S-3, S-2, S-1) for RNN / LSTM models."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ds_final"))

from bb_pipeline.eval_baselines import attach_lag_wrc  # noqa: E402

# Per-timestep process rates (+ wRC+ when available)
SEQ_FEATURES = ["K%", "BB%", "BABIP", "BIP%", "PA", "wRCplus"]
TRAIN_SEASONS = (2021, 2022, 2023)
TEST_SEASON = 2024
POSitions = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "OF", "IF", "UNK"]


def load_panel(data_root: Path | None = None) -> pd.DataFrame:
    root = data_root or (ROOT / "ds_final" / "data" / "processed")
    panel = pd.read_parquet(root / "projection_panel.parquet")
    season = pd.read_csv(root / "batting_clean.csv")
    panel = attach_lag_wrc(panel, season)

    for lag in (1, 2, 3):
        col = f"wRCplus_lag{lag}"
        if col in panel.columns:
            panel[col] = panel[col].fillna(panel[col].median())
    panel["age_t"] = panel["age_t"].fillna(panel["age_t"].median())
    panel["primary_pos"] = panel["primary_pos"].fillna("UNK")
    return panel


def _row_to_sequence(row: pd.Series) -> np.ndarray:
    """Return array shape (3, len(SEQ_FEATURES)) ordered S-3 → S-1."""
    steps = []
    for lag in (3, 2, 1):
        steps.append([
            row[f"K%_lag{lag}"],
            row[f"BB%_lag{lag}"],
            row[f"BABIP_lag{lag}"],
            row[f"BIP%_lag{lag}"],
            row[f"PA_lag{lag}"],
            row[f"wRCplus_lag{lag}"],
        ])
    return np.asarray(steps, dtype=np.float32)


def _pos_one_hot(pos: str) -> np.ndarray:
    vec = np.zeros(len(POSitions), dtype=np.float32)
    key = pos if pos in POSitions else "UNK"
    vec[POSitions.index(key)] = 1.0
    return vec


def build_arrays(
    panel: pd.DataFrame,
    *,
    fit_scaler: StandardScaler | None = None,
) -> dict[str, np.ndarray | StandardScaler]:
    """Convert panel rows to tensors-ready arrays with scaling."""
    seq_list, ages, pos_list, targets, seasons = [], [], [], [], []

    for _, row in panel.iterrows():
        seq_list.append(_row_to_sequence(row))
        ages.append(float(row["age_t"]))
        pos_list.append(_pos_one_hot(str(row["primary_pos"])))
        targets.append(float(row["wRCplus_target"]))
        seasons.append(int(row["target_season"]))

    X_seq = np.stack(seq_list)                          # (N, 3, 6)
    X_age = np.asarray(ages, dtype=np.float32).reshape(-1, 1)
    X_pos = np.stack(pos_list)                          # (N, n_pos)
    y = np.asarray(targets, dtype=np.float32)
    season_arr = np.asarray(seasons, dtype=np.int32)

    flat = X_seq.reshape(len(X_seq), -1)
    if fit_scaler is None:
        scaler = StandardScaler()
        flat_scaled = scaler.fit_transform(flat)
    else:
        scaler = fit_scaler
        flat_scaled = scaler.transform(flat)
    X_seq = flat_scaled.reshape(X_seq.shape).astype(np.float32)

    # Scale age separately (simple z-score on train)
    return {
        "X_seq": X_seq,
        "X_age": X_age,
        "X_pos": X_pos,
        "y": y,
        "seasons": season_arr,
        "scaler": scaler,
    }


def split_by_season(data: dict, train_seasons=TRAIN_SEASONS, test_season=TEST_SEASON):
    seasons = data["seasons"]
    train_mask = np.isin(seasons, train_seasons)
    test_mask = seasons == test_season
    out = {}
    for key in ("X_seq", "X_age", "X_pos", "y"):
        out[f"train_{key[2:] if key.startswith('X') else key}"] = data[key][train_mask]
        out[f"test_{key[2:] if key.startswith('X') else key}"] = data[key][test_mask]
    out["train_seasons"] = seasons[train_mask]
    out["test_seasons"] = seasons[test_mask]
    return out
