#!/usr/bin/env python3
"""
ElasticNet (歷年 wRC+ 軌跡) — Train 2021+2022+2023 → Test 2024.

產生 4 張結果圖：
  1. eval_en3yr_actual_vs_pred.png   散點圖（實際 vs 預測）
  2. eval_en3yr_error_by_tier.png    誤差分層（按 wRC+ 四段）
  3. eval_en3yr_residuals.png        殘差分布直方圖
  4. eval_en3yr_coef.png             ElasticNet 係數重要性

Usage (from ds_final/):
    python scripts/run_elasticnet_train3yr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bb_pipeline.eval_baselines import attach_lag_wrc
from bb_pipeline.eval_models import WRC_NUM, _one_hot_encoder, _prep_wrc_xy

PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures" / "evaluation"
FIG.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    "font.family": ["Microsoft JhengHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

CAT = ["primary_pos"]
TRAIN_SEASONS = [2021, 2022, 2023]
TEST_SEASON = 2024
MODEL_LABEL = "ElasticNet — 歷年 wRC+ 軌跡\n(Train 2021+22+23 → Test 2024)"
TIER_BINS = [0, 80, 100, 130, 300]
TIER_LABELS = ["< 80\n(弱打)", "80–100\n(中下)", "100–130\n(中上)", "130+\n(明星)"]


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), WRC_NUM),
        ("cat", _one_hot_encoder(), CAT),
    ])
    return Pipeline([
        ("prep", pre),
        ("model", ElasticNet(alpha=0.03, l1_ratio=0.3, random_state=42, max_iter=5000)),
    ])


def get_feature_names(pipe: Pipeline, df: pd.DataFrame) -> list[str]:
    """Get feature names after ColumnTransformer."""
    ct = pipe.named_steps["prep"]
    num_names = WRC_NUM
    try:
        cat_names = ct.named_transformers_["cat"].get_feature_names_out(CAT).tolist()
    except Exception:
        cat_names = []
    return num_names + cat_names


def plot_actual_vs_pred(y: np.ndarray, pred: np.ndarray, mae: float, r2: float) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y, pred, alpha=0.55, s=45, c="#457B9D", edgecolors="white", linewidths=0.4)
    lim_min = max(20, min(float(y.min()), float(pred.min())) - 10)
    lim_max = min(230, max(float(y.max()), float(pred.max())) + 10)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.3, label="完美預測線 (y=x)")
    ax.set_xlabel("實際 wRC+（2024）", fontsize=11)
    ax.set_ylabel("預測 wRC+", fontsize=11)
    ax.set_title(f"{MODEL_LABEL}\nMAE = {mae:.1f}  |  R² = {r2:.3f}", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = FIG / "eval_en3yr_actual_vs_pred.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.name}")


def plot_error_by_tier(y: np.ndarray, pred: np.ndarray) -> None:
    df = pd.DataFrame({"actual": y, "pred": pred})
    df["tier"] = pd.cut(df["actual"], bins=TIER_BINS, labels=TIER_LABELS, right=False)
    agg = df.groupby("tier", observed=True).apply(
        lambda g: pd.Series({
            "mae": mean_absolute_error(g["actual"], g["pred"]),
            "mean_actual": g["actual"].mean(),
            "mean_pred": g["pred"].mean(),
            "n": len(g),
        })
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: MAE by tier
    ax = axes[0]
    bars = ax.bar(agg["tier"].astype(str), agg["mae"],
                  color=["#E63946", "#F4A261", "#2A9D8F", "#457B9D"],
                  edgecolor="white", width=0.55)
    for bar, row in zip(bars, agg.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"MAE={row.mae:.1f}\n(n={row.n})",
                ha="center", fontsize=8)
    ax.set_xlabel("實際 wRC+ 區間（Test 2024）", fontsize=10)
    ax.set_ylabel("MAE（wRC+ 點數）", fontsize=10)
    ax.set_title("各打者等級的平均絕對誤差", fontsize=11)

    # Right: actual vs predicted mean per tier
    ax = axes[1]
    x = np.arange(len(agg))
    b1 = ax.bar(x - 0.22, agg["mean_actual"], 0.42, label="實際平均 wRC+", color="#2A9D8F", alpha=0.9)
    b2 = ax.bar(x + 0.22, agg["mean_pred"],   0.42, label="預測平均 wRC+", color="#E63946", alpha=0.85)
    ax.axhline(100, color="gray", ls="--", lw=1.2, label="聯盟平均 (100)")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["tier"].astype(str), fontsize=9)
    ax.set_ylabel("wRC+", fontsize=10)
    ax.set_title("系統性偏差：高估弱打、低估明星", fontsize=11)
    ax.legend(fontsize=8)

    plt.suptitle(MODEL_LABEL, fontsize=10, y=1.01)
    plt.tight_layout()
    path = FIG / "eval_en3yr_error_by_tier.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.name}")


def plot_residuals(y: np.ndarray, pred: np.ndarray) -> None:
    residuals = pred - y
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: histogram of residuals
    ax = axes[0]
    ax.hist(residuals, bins=30, color="#457B9D", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="#E63946", lw=1.5, ls="--", label="無誤差基準線")
    ax.axvline(float(np.mean(residuals)), color="#2A9D8F", lw=1.5, ls="-",
               label=f"平均殘差 = {np.mean(residuals):.1f}")
    ax.set_xlabel("殘差（預測 − 實際）", fontsize=10)
    ax.set_ylabel("球員人數", fontsize=10)
    ax.set_title("殘差分布直方圖", fontsize=11)
    ax.legend(fontsize=9)

    # Right: residuals vs actual
    ax = axes[1]
    ax.scatter(y, residuals, alpha=0.5, s=35, c="#457B9D", edgecolors="white", linewidths=0.3)
    ax.axhline(0, color="#E63946", lw=1.3, ls="--")
    ax.set_xlabel("實際 wRC+（2024）", fontsize=10)
    ax.set_ylabel("殘差（預測 − 實際）", fontsize=10)
    ax.set_title("殘差 vs 實際 wRC+", fontsize=11)

    plt.suptitle(MODEL_LABEL, fontsize=10, y=1.01)
    plt.tight_layout()
    path = FIG / "eval_en3yr_residuals.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.name}")


def plot_coefficients(pipe: Pipeline, feature_names: list[str]) -> None:
    coefs = pipe.named_steps["model"].coef_
    n = min(len(coefs), len(feature_names))
    coefs, feature_names = coefs[:n], feature_names[:n]

    # Sort by absolute value
    order = np.argsort(np.abs(coefs))[::-1]
    top_n = min(10, len(order))
    idx = order[:top_n]
    labels = [feature_names[i] for i in idx]
    vals = coefs[idx]

    # Nicer Chinese labels
    label_map = {
        "wRCplus_lag1": "上一季 wRC+ (S-1)",
        "wRCplus_lag2": "兩季前 wRC+ (S-2)",
        "wRCplus_lag3": "三季前 wRC+ (S-3)",
        "PA_sum_lag3":  "三季累計 PA",
        "age_t":        "預測當年年齡",
    }
    labels = [label_map.get(l, l) for l in labels]

    colors = ["#2A9D8F" if v >= 0 else "#E63946" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    for bar, v in zip(bars, vals[::-1]):
        offset = 0.002 if v >= 0 else -0.002
        ax.text(v + offset, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel("ElasticNet 係數（正 = 正相關）", fontsize=10)
    ax.set_title(f"特徵重要性（ElasticNet 係數）\n{MODEL_LABEL}", fontsize=10)
    plt.tight_layout()
    path = FIG / "eval_en3yr_coef.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.name}")


def main() -> None:
    panel = pd.read_parquet(PROC / "projection_panel.parquet")
    season = pd.read_csv(PROC / "batting_clean.csv")
    panel_w = attach_lag_wrc(panel, season)

    tr = panel_w[panel_w["target_season"].isin(TRAIN_SEASONS)]
    te = panel_w[panel_w["target_season"] == TEST_SEASON]

    x_tr, y_tr = _prep_wrc_xy(tr)
    x_te, y_te = _prep_wrc_xy(te)

    pipe = build_pipeline()
    pipe.fit(x_tr, y_tr)
    pred = pipe.predict(x_te)

    mae = float(mean_absolute_error(y_te, pred))
    r2 = float(r2_score(y_te, pred))

    print("=" * 55)
    print("ElasticNet 歷年 wRC+ 軌跡 — Train 3yr → Test 2024")
    print("=" * 55)
    print(f"Train 筆數 : {len(tr)}  ({', '.join(str(s) for s in TRAIN_SEASONS)})")
    print(f"Test 筆數  : {len(te)}  ({TEST_SEASON})")
    print(f"MAE        : {mae:.2f}")
    print(f"R2         : {r2:.3f}")

    feature_names = get_feature_names(pipe, x_te)

    print("\n產生圖片中...")
    plot_actual_vs_pred(y_te, pred, mae, r2)
    plot_error_by_tier(y_te, pred)
    plot_residuals(y_te, pred)
    plot_coefficients(pipe, feature_names)

    print(f"\nDone. 圖片位置 -> {FIG}")


if __name__ == "__main__":
    main()
