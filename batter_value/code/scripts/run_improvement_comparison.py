#!/usr/bin/env python3
"""
Compare four models for improving predictions on extreme players (Train 3yr -> Test 2024).

Models:
  0. ElasticNet (baseline)           — wRC+ lag features only
  A. ElasticNet + Trend feature      — adds wRC+_trend = S-1 minus S-3
  B. Random Forest + wRC+ trajectory — non-linear, same features as baseline
  C. Quantile Regression (median)    — predicts median instead of mean

Outputs (ds_final/figures/evaluation/):
  improve_mae_comparison.png     — overall MAE bar chart (4 models)
  improve_error_by_tier.png      — MAE & bias per wRC+ tier (4 panels)
  improve_scatter_grid.png       — actual vs predicted scatter (2x2 grid)
  improve_quantile_band.png      — quantile band (10th / 50th / 90th) vs actual

Usage (from ds_final/):
    python scripts/run_improvement_comparison.py
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
from sklearn.linear_model import ElasticNet, QuantileRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

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

TRAIN_SEASONS = [2021, 2022, 2023]
TEST_SEASON = 2024
CAT = ["primary_pos"]
TIER_BINS   = [0, 80, 100, 130, 300]
TIER_LABELS = ["< 80\n(弱打)", "80–100\n(中下)", "100–130\n(中上)", "130+\n(明星)"]

MODEL_COLORS = {
    "Baseline\nElasticNet":    "#888888",
    "方案A\nElasticNet+Trend": "#457B9D",
    "方案B\nRandom Forest":    "#2A9D8F",
    "方案C\nQuantile(p50)":    "#E9C46A",
}


# ── feature helpers ──────────────────────────────────────────────────────────

def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Add wRC+_trend = lag1 - lag3 (recent momentum)."""
    out = df.copy()
    if "wRCplus_lag1" in out.columns and "wRCplus_lag3" in out.columns:
        out["wRCplus_trend"] = out["wRCplus_lag1"] - out["wRCplus_lag3"]
        out["wRCplus_trend"] = out["wRCplus_trend"].fillna(0.0)
    return out


TREND_NUM = WRC_NUM + ["wRCplus_trend"]


def _prep_trend_xy(df: pd.DataFrame):
    df = add_trend(df)
    use = [c for c in TREND_NUM + CAT if c in df.columns]
    x = df[use].copy()
    for c in TREND_NUM:
        if c in x.columns:
            x[c] = x[c].fillna(x[c].median())
    y = df["wRCplus_target"].values.astype(float)
    return x, y


# ── pipeline factories ────────────────────────────────────────────────────────

def _ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_baseline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), WRC_NUM),
        ("cat", _ohe(), CAT),
    ])
    return Pipeline([
        ("prep", pre),
        ("model", ElasticNet(alpha=0.03, l1_ratio=0.3, random_state=42, max_iter=5000)),
    ])


def build_elasticnet_trend() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), TREND_NUM),
        ("cat", _ohe(), CAT),
    ])
    return Pipeline([
        ("prep", pre),
        ("model", ElasticNet(alpha=0.03, l1_ratio=0.3, random_state=42, max_iter=5000)),
    ])


def build_rf() -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", WRC_NUM),
        ("cat", _ohe(), CAT),
    ])
    return Pipeline([
        ("prep", pre),
        ("model", RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_quantile(q: float = 0.5) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), WRC_NUM),
        ("cat", _ohe(), CAT),
    ])
    return Pipeline([
        ("prep", pre),
        ("model", QuantileRegressor(quantile=q, alpha=0.01, solver="highs")),
    ])


# ── tier analysis ─────────────────────────────────────────────────────────────

def tier_stats(y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"actual": y, "pred": pred})
    df["tier"] = pd.cut(df["actual"], bins=TIER_BINS, labels=TIER_LABELS, right=False)
    return df.groupby("tier", observed=True).apply(
        lambda g: pd.Series({
            "mae": mean_absolute_error(g["actual"], g["pred"]),
            "mean_actual": g["actual"].mean(),
            "mean_pred": g["pred"].mean(),
            "n": len(g),
        })
    ).reset_index()


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_mae_comparison(results: list[dict]) -> None:
    names  = [r["name"] for r in results]
    maes   = [r["mae"] for r in results]
    r2s    = [r["r2"] for r in results]
    colors = [MODEL_COLORS.get(n, "#888888") for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, maes, color=colors, edgecolor="white", width=0.55)
    best_mae = min(maes)
    for bar, mae, r2, name in zip(bars, maes, r2s, names):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"MAE={mae:.2f}\nR2={r2:.3f}",
            ha="center", fontsize=8.5,
        )
        if mae == best_mae:
            bar.set_edgecolor("#E63946")
            bar.set_linewidth(2.5)
    ax.set_ylabel("MAE（wRC+ 點數）— Test 2024", fontsize=10)
    ax.set_title("四種模型整體 MAE 比較\n(Train 2021+22+23 → Test 2024)", fontsize=11)
    ax.set_ylim(0, max(maes) * 1.25)
    plt.tight_layout()
    path = FIG / "improve_mae_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.name}")


def plot_error_by_tier_comparison(results: list[dict]) -> None:
    n_models = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    tier_colors = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D"]

    for ax, res in zip(axes, results):
        tiers = res["tiers"]
        x = np.arange(len(tiers))

        b1 = ax.bar(x - 0.22, tiers["mean_actual"], 0.4,
                    label="實際平均", color="#2A9D8F", alpha=0.85)
        b2 = ax.bar(x + 0.22, tiers["mean_pred"], 0.4,
                    label="預測平均", color="#E63946", alpha=0.85)
        ax.axhline(100, color="gray", ls="--", lw=1.1, label="聯盟平均 100")

        for i, row in tiers.iterrows():
            ax.text(i, max(row["mean_actual"], row["mean_pred"]) + 2,
                    f"MAE\n{row['mae']:.1f}", ha="center", fontsize=7.5, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(tiers["tier"].astype(str), fontsize=8.5)
        ax.set_ylabel("wRC+", fontsize=9)
        ax.set_title(f"{res['name']}  (MAE={res['mae']:.2f})", fontsize=10)
        ax.legend(fontsize=7.5)
        ax.set_ylim(0, 200)

    plt.suptitle("各打者等級的實際 vs 預測平均 wRC+（四模型比較）", fontsize=12, y=1.01)
    plt.tight_layout()
    path = FIG / "improve_error_by_tier.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.name}")


def plot_scatter_grid(results: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    axes = axes.flatten()
    colors = list(MODEL_COLORS.values())

    for ax, res, c in zip(axes, results, colors):
        y, pred = res["actual"], res["pred"]
        ax.scatter(y, pred, alpha=0.45, s=30, color=c,
                   edgecolors="white", linewidths=0.3)
        lim_min = max(20, float(min(y.min(), pred.min())) - 10)
        lim_max = min(230, float(max(y.max(), pred.max())) + 10)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.2, label="y=x")
        ax.set_xlabel("實際 wRC+（2024）", fontsize=9)
        ax.set_ylabel("預測 wRC+", fontsize=9)
        ax.set_title(f"{res['name']}\nMAE={res['mae']:.2f}  R2={res['r2']:.3f}", fontsize=9.5)
        ax.legend(fontsize=8)

    plt.suptitle("實際 vs 預測散點圖（四模型比較）", fontsize=12, y=1.01)
    plt.tight_layout()
    path = FIG / "improve_scatter_grid.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.name}")


def plot_quantile_band(
    y_te: np.ndarray,
    pred_q10: np.ndarray,
    pred_q50: np.ndarray,
    pred_q90: np.ndarray,
) -> None:
    order = np.argsort(y_te)
    y_s      = y_te[order]
    q10_s    = pred_q10[order]
    q50_s    = pred_q50[order]
    q90_s    = pred_q90[order]
    x_idx    = np.arange(len(y_s))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(x_idx, q10_s, q90_s, alpha=0.25, color="#457B9D", label="預測區間 10%–90%")
    ax.plot(x_idx, q50_s, color="#457B9D", lw=1.5, label="預測中位數（p50）")
    ax.scatter(x_idx, y_s, s=15, color="#E63946", alpha=0.6, label="實際 wRC+（2024）", zorder=3)
    ax.axhline(100, color="gray", ls="--", lw=1, label="聯盟平均 100")
    ax.set_xlabel("打者（依實際 wRC+ 由低到高排序）", fontsize=10)
    ax.set_ylabel("wRC+", fontsize=10)
    ax.set_title(
        "方案C：量化迴歸預測區間（p10 / p50 / p90）vs 實際 wRC+\n"
        "Train 2021+22+23 → Test 2024",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = FIG / "improve_quantile_band.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    panel = pd.read_parquet(PROC / "projection_panel.parquet")
    season = pd.read_csv(PROC / "batting_clean.csv")
    panel_w = attach_lag_wrc(panel, season)

    tr = panel_w[panel_w["target_season"].isin(TRAIN_SEASONS)]
    te = panel_w[panel_w["target_season"] == TEST_SEASON]

    # ── Baseline: ElasticNet ──
    print("Training Baseline (ElasticNet)...")
    x_tr, y_tr = _prep_wrc_xy(tr)
    x_te, y_te = _prep_wrc_xy(te)
    base = build_baseline()
    base.fit(x_tr, y_tr)
    pred_base = base.predict(x_te)

    # ── A: ElasticNet + Trend ──
    print("Training A (ElasticNet + Trend)...")
    x_tr_t, _ = _prep_trend_xy(tr)
    x_te_t, _ = _prep_trend_xy(te)
    en_trend = build_elasticnet_trend()
    en_trend.fit(x_tr_t, y_tr)
    pred_a = en_trend.predict(x_te_t)

    # ── B: Random Forest ──
    print("Training B (Random Forest + wRC+ trajectory)...")
    rf = build_rf()
    rf.fit(x_tr, y_tr)
    pred_b = rf.predict(x_te)

    # ── C: Quantile Regression (p10, p50, p90) ──
    print("Training C (Quantile Regression p10/p50/p90)...")
    q10 = build_quantile(0.10); q10.fit(x_tr, y_tr); pred_q10 = q10.predict(x_te)
    q50 = build_quantile(0.50); q50.fit(x_tr, y_tr); pred_q50 = q50.predict(x_te)
    q90 = build_quantile(0.90); q90.fit(x_tr, y_tr); pred_q90 = q90.predict(x_te)

    results = [
        {
            "name": "Baseline\nElasticNet",
            "mae":  float(mean_absolute_error(y_te, pred_base)),
            "r2":   float(r2_score(y_te, pred_base)),
            "pred": pred_base, "actual": y_te,
            "tiers": tier_stats(y_te, pred_base),
        },
        {
            "name": "方案A\nElasticNet+Trend",
            "mae":  float(mean_absolute_error(y_te, pred_a)),
            "r2":   float(r2_score(y_te, pred_a)),
            "pred": pred_a, "actual": y_te,
            "tiers": tier_stats(y_te, pred_a),
        },
        {
            "name": "方案B\nRandom Forest",
            "mae":  float(mean_absolute_error(y_te, pred_b)),
            "r2":   float(r2_score(y_te, pred_b)),
            "pred": pred_b, "actual": y_te,
            "tiers": tier_stats(y_te, pred_b),
        },
        {
            "name": "方案C\nQuantile(p50)",
            "mae":  float(mean_absolute_error(y_te, pred_q50)),
            "r2":   float(r2_score(y_te, pred_q50)),
            "pred": pred_q50, "actual": y_te,
            "tiers": tier_stats(y_te, pred_q50),
        },
    ]

    # ── Print summary ──
    print()
    print("=" * 60)
    print("Test 2024 Summary (lower MAE is better)")
    print("=" * 60)
    header = f"{'Model':<28} {'MAE':>7} {'R2':>8}"
    print(header)
    print("-" * 48)
    for r in results:
        name_flat = r["name"].replace("\n", " ")
        print(f"{name_flat:<28} {r['mae']:>7.2f} {r['r2']:>8.3f}")

    print()
    print("Tier-level MAE (弱打 / 中下 / 中上 / 明星)")
    print("-" * 60)
    for r in results:
        name_flat = r["name"].replace("\n", " ")
        tier_maes = " | ".join(f"{v:.1f}" for v in r["tiers"]["mae"])
        print(f"{name_flat:<28}  {tier_maes}")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_mae_comparison(results)
    plot_error_by_tier_comparison(results)
    plot_scatter_grid(results)
    plot_quantile_band(y_te, pred_q10, pred_q50, pred_q90)

    print(f"\nDone. Figures -> {FIG}")


if __name__ == "__main__":
    main()
