#!/usr/bin/env python3
"""
Classmate's Random Forest model — re-run with Train 3yr (2021+2022+2023) -> Test 2024.

Generates:
  figures/model/rf3yr_actual_vs_pred.png      scatter: actual vs predicted
  figures/model/rf3yr_perm_importance.png     permutation importance (top 15)
  figures/model/rf3yr_error_by_tier.png       MAE & bias per wRC+ tier
  figures/model/rf3yr_residuals.png           residual histogram + scatter
  results/rf3yr_metrics.csv                   overall + tier metrics

Usage (from classmate_train3yr/):
    python generate_figures.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PROC         = PROJECT_ROOT / "data" / "processed"
FIG_MODEL    = HERE / "figures" / "model"
RESULTS      = HERE / "results"
FIG_MODEL.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    "font.family": ["Microsoft JhengHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

# ── Classmate's exact feature set ────────────────────────────────────────────
NUM_FEATURES = [
    "K%_lag3", "K%_lag2", "K%_lag1",
    "BB%_lag3", "BB%_lag2", "BB%_lag1",
    "BABIP_lag3", "BABIP_lag2", "BABIP_lag1",
    "BIP%_lag3", "BIP%_lag2", "BIP%_lag1",
    "PA_lag3", "PA_lag2", "PA_lag1",
    "PA_sum_lag3", "age_t",
]
CAT_FEATURES = ["primary_pos"]

TRAIN_SEASONS = [2021, 2022, 2023]
TEST_SEASON   = 2024
TIER_BINS     = [0, 80, 100, 130, 300]
TIER_LABELS   = ["<80 (弱打)", "80-100 (中下)", "100-130 (中上)", "130+ (明星)"]


def _ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_rf() -> Pipeline:
    """Exact replica of classmate's RF (train_projection.py)."""
    pre = ColumnTransformer([
        ("num", "passthrough", NUM_FEATURES),
        ("cat", _ohe(), CAT_FEATURES),
    ])
    return Pipeline([
        ("prep", pre),
        ("model", RandomForestRegressor(
            n_estimators=400, max_depth=12,
            min_samples_leaf=8, random_state=42, n_jobs=-1,
        )),
    ])


def prep_xy(df: pd.DataFrame):
    use = [c for c in NUM_FEATURES + CAT_FEATURES if c in df.columns]
    x = df[use].copy()
    x["age_t"] = x["age_t"].fillna(x["age_t"].median())
    return x, df["wRCplus_target"].values.astype(float)


def tier_stats(y, pred) -> pd.DataFrame:
    df = pd.DataFrame({"actual": y, "pred": pred})
    df["tier"] = pd.cut(df["actual"], bins=TIER_BINS, labels=TIER_LABELS, right=False)
    return df.groupby("tier", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "mae": mean_absolute_error(g["actual"], g["pred"]),
            "mean_actual": g["actual"].mean(),
            "mean_pred": g["pred"].mean(),
        })
    ).reset_index()


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_actual_vs_pred(y, pred, mae, r2):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y, pred, alpha=0.5, s=40, c="#E63946",
               edgecolors="white", linewidths=0.4)
    lim = [max(20, float(min(y.min(), pred.min())) - 10),
           min(230, float(max(y.max(), pred.max())) + 10)]
    ax.plot(lim, lim, "k--", lw=1.3, label="完美預測線 (y=x)")
    ax.axvline(80,  color="gray", ls=":", lw=0.9, alpha=0.5)
    ax.axvline(130, color="gray", ls=":", lw=0.9, alpha=0.5)
    ax.set_xlabel("實際 wRC+ (2024)", fontsize=11)
    ax.set_ylabel("預測 wRC+", fontsize=11)
    ax.set_title(
        f"同學 Random Forest — 過程指標\nTrain {TRAIN_SEASONS} → Test {TEST_SEASON}\n"
        f"MAE = {mae:.2f}  |  R2 = {r2:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = FIG_MODEL / "rf3yr_actual_vs_pred.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {p.name}")


def plot_perm_importance(pipe, x_te, y_te):
    print("Computing permutation importance (this may take ~30s)...")
    pi = permutation_importance(pipe, x_te, y_te,
                                n_repeats=15, random_state=42, n_jobs=-1)
    # get feature names
    ct = pipe.named_steps["prep"]
    try:
        cat_names = ct.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES).tolist()
    except Exception:
        cat_names = []
    feat_names = NUM_FEATURES + cat_names
    n = min(len(feat_names), len(pi.importances_mean))
    feat_names = feat_names[:n]

    # friendly Chinese labels
    label_map = {
        "K%_lag1": "三振率 S-1", "K%_lag2": "三振率 S-2", "K%_lag3": "三振率 S-3",
        "BB%_lag1": "保送率 S-1", "BB%_lag2": "保送率 S-2", "BB%_lag3": "保送率 S-3",
        "BABIP_lag1": "BABIP S-1", "BABIP_lag2": "BABIP S-2", "BABIP_lag3": "BABIP S-3",
        "BIP%_lag1": "BIP% S-1",  "BIP%_lag2": "BIP% S-2",  "BIP%_lag3": "BIP% S-3",
        "PA_lag1": "打席數 S-1",  "PA_lag2": "打席數 S-2",  "PA_lag3": "打席數 S-3",
        "PA_sum_lag3": "三季累計打席", "age_t": "年齡",
    }
    labels = [label_map.get(f, f) for f in feat_names]

    order = np.argsort(pi.importances_mean[:n])[-15:]
    vals  = pi.importances_mean[:n][order]
    errs  = pi.importances_std[:n][order]
    lbls  = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(lbls, vals, xerr=errs, color="#E63946", alpha=0.8,
            edgecolor="white", capsize=3)
    ax.set_xlabel("Permutation Importance（MAE 增量）", fontsize=10)
    ax.set_title(
        f"同學 RF — 特徵重要性（Permutation）\nTrain {TRAIN_SEASONS} → Test {TEST_SEASON}",
        fontsize=10,
    )
    plt.tight_layout()
    p = FIG_MODEL / "rf3yr_perm_importance.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {p.name}")


def plot_error_by_tier(y, pred):
    tiers = tier_stats(y, pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    colors = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D"]
    bars = ax.bar(tiers["tier"].astype(str), tiers["mae"],
                  color=colors, edgecolor="white", width=0.55)
    for bar, row in zip(bars, tiers.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"MAE={row.mae:.1f}\n(n={row.n})",
                ha="center", fontsize=8)
    ax.set_xlabel("實際 wRC+ 區間", fontsize=10)
    ax.set_ylabel("MAE", fontsize=10)
    ax.set_title("各打者等級的平均絕對誤差", fontsize=11)

    ax = axes[1]
    x = np.arange(len(tiers))
    ax.bar(x - 0.22, tiers["mean_actual"], 0.42,
           label="實際平均 wRC+", color="#2A9D8F", alpha=0.9)
    ax.bar(x + 0.22, tiers["mean_pred"], 0.42,
           label="預測平均 wRC+", color="#E63946", alpha=0.85)
    ax.axhline(100, color="gray", ls="--", lw=1.2, label="聯盟平均 100")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers["tier"].astype(str), fontsize=9)
    ax.set_ylabel("wRC+", fontsize=10)
    ax.set_title("系統性偏差：高估弱打、低估明星", fontsize=11)
    ax.legend(fontsize=8)

    plt.suptitle(
        f"同學 RF — 誤差分層分析\nTrain {TRAIN_SEASONS} → Test {TEST_SEASON}",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    p = FIG_MODEL / "rf3yr_error_by_tier.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {p.name}")


def plot_residuals(y, pred):
    residuals = pred - y
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(residuals, bins=30, color="#E63946", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", lw=1.5, ls="--", label="無誤差基準線")
    ax.axvline(float(np.mean(residuals)), color="#2A9D8F", lw=1.5,
               label=f"平均殘差 = {np.mean(residuals):.1f}")
    ax.set_xlabel("殘差（預測 - 實際）", fontsize=10)
    ax.set_ylabel("球員人數", fontsize=10)
    ax.set_title("殘差分布直方圖", fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.scatter(y, residuals, alpha=0.5, s=35, c="#E63946",
               edgecolors="white", linewidths=0.3)
    ax.axhline(0, color="black", lw=1.3, ls="--")
    ax.set_xlabel("實際 wRC+ (2024)", fontsize=10)
    ax.set_ylabel("殘差（預測 - 實際）", fontsize=10)
    ax.set_title("殘差 vs 實際 wRC+", fontsize=11)

    plt.suptitle(
        f"同學 RF — 殘差分析\nTrain {TRAIN_SEASONS} → Test {TEST_SEASON}",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    p = FIG_MODEL / "rf3yr_residuals.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {p.name}")


def save_metrics(y, pred):
    tiers = tier_stats(y, pred)
    overall = pd.DataFrame([{
        "model": "Classmate RF (Train 3yr)",
        "train_seasons": str(TRAIN_SEASONS),
        "test_season": TEST_SEASON,
        "n_train": None,
        "n_test": len(y),
        "mae": round(float(mean_absolute_error(y, pred)), 2),
        "r2":  round(float(r2_score(y, pred)), 3),
    }])
    tier_out = tiers.rename(columns={"tier": "wrc_tier"})[
        ["wrc_tier", "n", "mae", "mean_actual", "mean_pred"]
    ].round(2)
    overall.to_csv(RESULTS / "rf3yr_overall_metrics.csv", index=False)
    tier_out.to_csv(RESULTS / "rf3yr_tier_metrics.csv", index=False)
    print("Saved rf3yr_overall_metrics.csv")
    print("Saved rf3yr_tier_metrics.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    panel = pd.read_parquet(PROC / "projection_panel.parquet")

    tr = panel[panel["target_season"].isin(TRAIN_SEASONS)]
    te = panel[panel["target_season"] == TEST_SEASON]

    x_tr, y_tr = prep_xy(tr)
    x_te, y_te = prep_xy(te)

    print(f"Training RF on {TRAIN_SEASONS}  (n={len(tr)}) ...")
    pipe = build_rf()
    pipe.fit(x_tr, y_tr)
    pred = pipe.predict(x_te)

    mae = float(mean_absolute_error(y_te, pred))
    r2  = float(r2_score(y_te, pred))

    print(f"\nTest {TEST_SEASON} Results:")
    print(f"  MAE = {mae:.2f}")
    print(f"  R2  = {r2:.3f}")
    print(f"  n_train = {len(tr)},  n_test = {len(te)}")

    print("\nGenerating figures...")
    plot_actual_vs_pred(y_te, pred, mae, r2)
    plot_perm_importance(pipe, x_te, y_te)
    plot_error_by_tier(y_te, pred)
    plot_residuals(y_te, pred)
    save_metrics(y_te, pred)

    print(f"\nDone. Model figures -> {FIG_MODEL}")
    print(f"      Results        -> {RESULTS}")


if __name__ == "__main__":
    main()
