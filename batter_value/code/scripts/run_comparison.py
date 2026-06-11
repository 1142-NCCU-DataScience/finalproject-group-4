#!/usr/bin/env python3
"""
Compare LSTM (RNN) with existing project models on Test 2024.

Models:
  A. Null model (predict train-set mean)
  B. Classmate RF (Train 2yr)
  C. Classmate RF (Train 3yr)
  D. ElasticNet + 3x sample weights (final)
  E. LSTM (RNN) — from RNN/train_rnn.py

Usage:
    python RNN/run_comparison.py
"""

from __future__ import annotations

import json
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
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bb_pipeline"))
sys.path.insert(0, str(ROOT / "ds_final"))
sys.path.insert(0, str(HERE))

from bb_pipeline.eval_baselines import attach_lag_wrc  # noqa: E402
from bb_pipeline.eval_models import WRC_NUM, _prep_wrc_xy, _one_hot_encoder  # noqa: E402
from bb_pipeline.train_projection import fit_by_target_season, prepare_xy  # noqa: E402

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

TIER_BINS = [0, 80, 100, 130, 300]
TIER_LABELS = ["<80\n(弱打)", "80-100\n(中下)", "100-130\n(中上)", "130+\n(明星)"]
COLORS = ["#ADB5BD", "#E63946", "#F4A261", "#2A9D8F", "#6A4C93"]


def tier_mae(y_true, y_pred):
    tiers = pd.cut(y_true, bins=TIER_BINS, labels=TIER_LABELS)
    df = pd.DataFrame({"a": y_true, "p": y_pred, "t": tiers})
    return df.groupby("t", observed=True).apply(
        lambda g: mean_absolute_error(g["a"], g["p"])
    ).values


def build_en_weighted():
    pre = ColumnTransformer([
        ("num", StandardScaler(), WRC_NUM),
        ("cat", _one_hot_encoder(), ["primary_pos"]),
    ])
    return Pipeline([
        ("prep", pre),
        ("model", ElasticNet(alpha=0.03, l1_ratio=0.3, max_iter=5000)),
    ])


def main() -> None:
    # ── Load RNN metrics if available ─────────────────────────────────────────
    rnn_path = HERE / "results" / "rnn_metrics.json"
    if not rnn_path.exists():
        print("RNN metrics not found. Run: python RNN/train_rnn.py")
        sys.exit(1)
    rnn_metrics = json.loads(rnn_path.read_text(encoding="utf-8"))

    panel_bb = pd.read_parquet(ROOT / "data" / "processed" / "projection_panel.parquet")
    panel_ds = pd.read_parquet(ROOT / "ds_final" / "data" / "processed" / "projection_panel.parquet")
    season_df = pd.read_csv(ROOT / "ds_final" / "data" / "processed" / "batting_clean.csv")
    panel_wrc = attach_lag_wrc(panel_ds, season_df)
    for col in WRC_NUM:
        if col in panel_wrc.columns:
            panel_wrc[col] = panel_wrc[col].fillna(panel_wrc[col].median())

    train_wrc = panel_wrc[panel_wrc["target_season"].isin([2021, 2022, 2023])]
    test_wrc = panel_wrc[panel_wrc["target_season"] == 2024]
    x_tr, y_tr = _prep_wrc_xy(train_wrc)
    x_te, y_te = _prep_wrc_xy(test_wrc)

    # Null model
    null_pred = np.full_like(y_te, y_tr.mean())
    null_mae = mean_absolute_error(y_te, null_pred)
    null_r2 = r2_score(y_te, null_pred)

    # RF models
    te_bb = panel_bb[panel_bb["target_season"] == 2024]
    xte_bb, yte_bb = prepare_xy(te_bb)

    res_2yr = fit_by_target_season(panel_bb)
    pred_rf2 = res_2yr["rf_model"].predict(xte_bb)

    res_3yr = fit_by_target_season(
        panel_bb, train_targets=(2021, 2022, 2023), val_target=2023, test_target=2024
    )
    pred_rf3 = res_3yr["rf_model"].predict(xte_bb)

    # ElasticNet weighted
    sw = np.ones(len(y_tr))
    sw[(y_tr < 80) | (y_tr > 130)] = 3.0
    en = build_en_weighted()
    en.fit(x_tr, y_tr, model__sample_weight=sw)
    pred_en = en.predict(x_te)

    names = [
        "Null\n(猜平均)",
        "同學 RF\n(Train 2yr)",
        "同學 RF\n(Train 3yr)",
        "ElasticNet\n+加權3×",
        "LSTM\n(RNN)",
    ]
    maes = [
        null_mae,
        mean_absolute_error(yte_bb, pred_rf2),
        mean_absolute_error(yte_bb, pred_rf3),
        mean_absolute_error(y_te, pred_en),
        rnn_metrics["mae"],
    ]
    r2s = [
        null_r2,
        r2_score(yte_bb, pred_rf2),
        r2_score(yte_bb, pred_rf3),
        r2_score(y_te, pred_en),
        rnn_metrics["r2"],
    ]
    tiers = [
        tier_mae(y_te, null_pred),
        tier_mae(yte_bb, pred_rf2),
        tier_mae(yte_bb, pred_rf3),
        tier_mae(y_te, pred_en),
        np.array(list(rnn_metrics["tier_mae"].values())),
    ]

    print("=" * 78)
    print(f"{'方法':<22} {'MAE':>7} {'R²':>8}  {'<80':>7} {'80-100':>8} {'100-130':>9} {'130+':>7}")
    print("-" * 78)
    for name, m, r, t in zip(names, maes, r2s, tiers):
        print(f"{name.replace(chr(10),' '):<22} {m:>7.2f} {r:>8.3f}  "
              f"{t[0]:>7.1f} {t[1]:>8.1f} {t[2]:>9.1f} {t[3]:>7.1f}")

    # ── Plot MAE ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    bars = ax.bar(range(5), maes, color=COLORS, width=0.55, edgecolor="white")
    for bar, v in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2, f"{v:.2f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(5))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("MAE（wRC+ 點數）")
    ax.set_title("Test 2024 整體 MAE")
    ax.set_ylim(0, max(maes) * 1.25)

    ax = axes[1]
    x = np.arange(4)
    w = 0.15
    for i, (name, t, c) in enumerate(zip(names, tiers, COLORS)):
        ax.bar(x + (i - 2) * w, t, w, color=c, label=name.replace("\n", " "), edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(TIER_LABELS, fontsize=9)
    ax.set_ylabel("MAE（wRC+ 點數）")
    ax.set_title("Test 2024 分層 MAE")
    ax.legend(fontsize=6.5, loc="upper right")

    plt.suptitle("Null / RF / ElasticNet / LSTM(RNN) 比較", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(FIG / "rnn_vs_all_mae.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── Plot R² ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(5), r2s, color=COLORS, width=0.55, edgecolor="white")
    for bar, v in zip(bars, r2s):
        ypos = v + 0.005 if v >= 0 else v - 0.015
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(5))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.axhline(0, color="gray", lw=1)
    ax.set_ylabel("R²")
    ax.set_title("Test 2024 R²（0 = Null Model 水準）")
    plt.tight_layout()
    fig.savefig(FIG / "rnn_vs_all_r2.png", dpi=300, bbox_inches="tight")
    plt.close()

    summary = pd.DataFrame({
        "model": [n.replace("\n", " ") for n in names],
        "mae": maes,
        "r2": r2s,
    })
    summary.to_csv(HERE / "results" / "all_models_comparison.csv", index=False)
    print(f"\nSaved -> {FIG}")


if __name__ == "__main__":
    main()
