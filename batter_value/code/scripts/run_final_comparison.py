#!/usr/bin/env python3
"""
Final fair comparison: all models use Train 2021+2022+2023 -> Test 2024.

Models:
  A. Classmate's Random Forest (process features)   — same model, Train 3yr
  B. Our Baseline  ElasticNet  (wRC+ trajectory)    — Train 3yr
  C. Our Final     ElasticNet  (wRC+ trajectory)    — Train 3yr + sample weights x3

Outputs (ds_final/figures/evaluation/):
  final_mae_comparison.png     overall MAE bar chart
  final_tier_comparison.png    tier-level MAE + bias (3 models)
  final_scatter_grid.png       actual vs predicted scatter (1x3)
  final_summary.csv            exportable table

Usage (from ds_final/):
    python scripts/run_final_comparison.py
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
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bb_pipeline.eval_baselines import attach_lag_wrc
from bb_pipeline.eval_models import _prep_wrc_xy, WRC_NUM, _one_hot_encoder

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Classmate's model spec (from bb_pipeline/train_projection.py) ────────────
CLASSMATE_NUM = [
    "K%_lag3", "K%_lag2", "K%_lag1",
    "BB%_lag3", "BB%_lag2", "BB%_lag1",
    "BABIP_lag3", "BABIP_lag2", "BABIP_lag1",
    "BIP%_lag3", "BIP%_lag2", "BIP%_lag1",
    "PA_lag3", "PA_lag2", "PA_lag1",
    "PA_sum_lag3", "age_t",
]
CLASSMATE_CAT = ["primary_pos"]


def _ohe_passthrough():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_rf_pipeline() -> Pipeline:
    """Exact replica of classmate's Random Forest."""
    pre = ColumnTransformer([
        ("num", "passthrough", CLASSMATE_NUM),
        ("cat", _ohe_passthrough(), CLASSMATE_CAT),
    ])
    return Pipeline([
        ("prep", pre),
        ("model", RandomForestRegressor(
            n_estimators=400, max_depth=12,
            min_samples_leaf=8, random_state=42, n_jobs=-1,
        )),
    ])


def prepare_xy(df: pd.DataFrame):
    use = [c for c in CLASSMATE_NUM + CLASSMATE_CAT if c in df.columns]
    x = df[use].copy()
    x["age_t"] = x["age_t"].fillna(x["age_t"].median())
    y = df["wRCplus_target"].values.astype(float)
    return x, y

PROC = ROOT / "data" / "processed"
FIG  = ROOT / "figures" / "evaluation"
FIG.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    "font.family": ["Microsoft JhengHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

TRAIN_SEASONS = [2021, 2022, 2023]
TEST_SEASON   = 2024
CAT           = ["primary_pos"]
TIER_BINS     = [0, 80, 100, 130, 300]
TIER_LABELS   = ["<80\n(弱打)", "80-100\n(中下)", "100-130\n(中上)", "130+\n(明星)"]

COLORS = {
    "同學 RF\n(過程指標)":    "#E63946",
    "我們 Baseline\n(wRC+軌跡)": "#457B9D",
    "我們 最終版\n(wRC+軌跡+加權)": "#2A9D8F",
}


# ── pipelines ─────────────────────────────────────────────────────────────────

def build_elasticnet() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), WRC_NUM),
        ("cat", _one_hot_encoder(), CAT),
    ])
    return Pipeline([
        ("prep", pre),
        ("model", ElasticNet(alpha=0.03, l1_ratio=0.3, random_state=42, max_iter=5000)),
    ])


def sample_weights(y: np.ndarray, mult: float = 3.0) -> np.ndarray:
    w = np.ones(len(y))
    w[(y < 80) | (y > 130)] = mult
    return w


# ── tier stats ────────────────────────────────────────────────────────────────

def tier_stats(y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"actual": y, "pred": pred})
    df["tier"] = pd.cut(df["actual"], bins=TIER_BINS, labels=TIER_LABELS, right=False)
    return df.groupby("tier", observed=True).apply(
        lambda g: pd.Series({
            "mae":         mean_absolute_error(g["actual"], g["pred"]),
            "mean_actual": g["actual"].mean(),
            "mean_pred":   g["pred"].mean(),
            "n":           len(g),
        })
    ).reset_index()


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_mae_comparison(results: list[dict]) -> None:
    names  = [r["name"] for r in results]
    maes   = [r["mae"]  for r in results]
    r2s    = [r["r2"]   for r in results]
    colors = [COLORS[n] for n in names]
    best   = min(maes)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, maes, color=colors, edgecolor="white", width=0.5)
    for bar, mae, r2, name in zip(bars, maes, r2s, names):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"MAE {mae:.2f}\nR2  {r2:.3f}",
                ha="center", fontsize=9)
        if mae == best:
            bar.set_edgecolor("#000000")
            bar.set_linewidth(2.5)
    ax.set_ylabel("MAE（wRC+ 點數）", fontsize=11)
    ax.set_title("最終模型比較：同學 vs 我們\n(Train 2021+22+23 → Test 2024，相同資料切割)", fontsize=11)
    ax.set_ylim(0, max(maes) * 1.28)
    plt.tight_layout()
    p = FIG / "final_mae_comparison.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {p.name}")


def plot_tier_comparison(results: list[dict]) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: MAE per tier per model
    ax = axes[0]
    tier_labels = results[0]["tiers"]["tier"].astype(str).tolist()
    x = np.arange(len(tier_labels))
    w = 0.25
    for i, res in enumerate(results):
        offset = (i - 1) * w
        bars = ax.bar(x + offset, res["tiers"]["mae"], w,
                      color=COLORS[res["name"]], label=res["name"].replace("\n", " "),
                      edgecolor="white", alpha=0.9)
        for bar, mae in zip(bars, res["tiers"]["mae"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{mae:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels, fontsize=9)
    ax.set_ylabel("MAE（wRC+ 點數）", fontsize=10)
    ax.set_title("各打者等級 MAE 比較", fontsize=11)
    ax.legend(fontsize=8)

    # Right: bias (actual vs predicted mean) for our final model vs classmate
    ax = axes[1]
    ref = results[0]   # classmate RF
    our = results[2]   # our final weighted
    tiers = ref["tiers"]
    xp = np.arange(len(tiers))

    ax.plot(xp, tiers["mean_actual"], "o-", color="black",
            lw=2, ms=7, label="實際平均 wRC+", zorder=3)
    ax.plot(xp, ref["tiers"]["mean_pred"], "s--",
            color=COLORS[ref["name"]], lw=1.8, ms=7,
            label=ref["name"].replace("\n", " "))
    ax.plot(xp, our["tiers"]["mean_pred"], "^--",
            color=COLORS[our["name"]], lw=1.8, ms=7,
            label=our["name"].replace("\n", " "))
    ax.axhline(100, color="gray", ls=":", lw=1, label="聯盟平均 100")
    ax.set_xticks(xp)
    ax.set_xticklabels(tiers["tier"].astype(str), fontsize=9)
    ax.set_ylabel("wRC+", fontsize=10)
    ax.set_title("系統性偏差：實際 vs 預測平均\n（黑線=實際，其他=各模型預測）", fontsize=11)
    ax.legend(fontsize=8)

    plt.suptitle("各打者等級分析（Train 3yr → Test 2024）", fontsize=12, y=1.01)
    plt.tight_layout()
    p = FIG / "final_tier_comparison.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {p.name}")


def plot_scatter_grid(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, res in zip(axes, results):
        y, pred = res["actual"], res["pred"]
        c = COLORS[res["name"]]
        ax.scatter(y, pred, alpha=0.5, s=30, color=c,
                   edgecolors="white", linewidths=0.3)
        lim_min = max(20, float(min(y.min(), pred.min())) - 10)
        lim_max = min(230, float(max(y.max(), pred.max())) + 10)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.2, label="y=x")
        ax.axvline(80,  color="#E63946", ls=":", lw=0.8, alpha=0.35)
        ax.axvline(130, color="#E63946", ls=":", lw=0.8, alpha=0.35)
        ax.set_xlabel("實際 wRC+（2024）", fontsize=9)
        ax.set_ylabel("預測 wRC+", fontsize=9)
        ax.set_title(f"{res['name'].replace(chr(10), ' ')}\nMAE={res['mae']:.2f}  R2={res['r2']:.3f}",
                     fontsize=9.5)
        ax.legend(fontsize=8)
    plt.suptitle("實際 vs 預測散點（紅虛線為弱打/明星邊界）", fontsize=11, y=1.02)
    plt.tight_layout()
    p = FIG / "final_scatter_grid.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {p.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    panel  = pd.read_parquet(PROC / "projection_panel.parquet")
    season = pd.read_csv(PROC / "batting_clean.csv")
    panel_w = attach_lag_wrc(panel, season)

    tr = panel_w[panel_w["target_season"].isin(TRAIN_SEASONS)]
    te = panel_w[panel_w["target_season"] == TEST_SEASON]

    results = []

    # ── A: Classmate's Random Forest ────────────────────────────────────────
    print("Training classmate's Random Forest (process features, Train 3yr)...")
    x_tr_rf, y_tr = prepare_xy(tr)
    x_te_rf, y_te = prepare_xy(te)
    rf = build_rf_pipeline()
    rf.fit(x_tr_rf, y_tr)
    pred_rf = rf.predict(x_te_rf)
    results.append({
        "name":   "同學 RF\n(過程指標)",
        "mae":    float(mean_absolute_error(y_te, pred_rf)),
        "r2":     float(r2_score(y_te, pred_rf)),
        "pred":   pred_rf,
        "actual": y_te,
        "tiers":  tier_stats(y_te, pred_rf),
    })

    # ── B: Our Baseline ElasticNet ───────────────────────────────────────────
    print("Training our Baseline ElasticNet (wRC+ trajectory, Train 3yr)...")
    x_tr_en, y_tr_en = _prep_wrc_xy(tr)
    x_te_en, y_te_en = _prep_wrc_xy(te)
    en_base = build_elasticnet()
    en_base.fit(x_tr_en, y_tr_en)
    pred_base = en_base.predict(x_te_en)
    results.append({
        "name":   "我們 Baseline\n(wRC+軌跡)",
        "mae":    float(mean_absolute_error(y_te_en, pred_base)),
        "r2":     float(r2_score(y_te_en, pred_base)),
        "pred":   pred_base,
        "actual": y_te_en,
        "tiers":  tier_stats(y_te_en, pred_base),
    })

    # ── C: Our Final ElasticNet + Sample Weights ─────────────────────────────
    print("Training our Final ElasticNet (wRC+ trajectory + sample weights, Train 3yr)...")
    w = sample_weights(y_tr_en, mult=3.0)
    en_final = build_elasticnet()
    en_final.fit(x_tr_en, y_tr_en, model__sample_weight=w)
    pred_final = en_final.predict(x_te_en)
    results.append({
        "name":   "我們 最終版\n(wRC+軌跡+加權)",
        "mae":    float(mean_absolute_error(y_te_en, pred_final)),
        "r2":     float(r2_score(y_te_en, pred_final)),
        "pred":   pred_final,
        "actual": y_te_en,
        "tiers":  tier_stats(y_te_en, pred_final),
    })

    # ── Print ────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Final Comparison — Train 3yr -> Test 2024")
    print("=" * 70)
    print(f"{'Model':<30} {'MAE':>7} {'R2':>8}  "
          f"{'<80':>7} {'80-100':>7} {'100-130':>8} {'130+':>7}")
    print("-" * 70)
    for r in results:
        t = r["tiers"]["mae"].tolist()
        name = r["name"].replace("\n", " ")
        print(f"{name:<30} {r['mae']:>7.2f} {r['r2']:>8.3f}  "
              f"{t[0]:>7.1f} {t[1]:>7.1f} {t[2]:>8.1f} {t[3]:>7.1f}")

    # ── Save summary CSV ─────────────────────────────────────────────────────
    rows = []
    for r in results:
        t = r["tiers"]["mae"].tolist()
        rows.append({
            "model":        r["name"].replace("\n", " "),
            "overall_mae":  round(r["mae"], 2),
            "r2":           round(r["r2"], 3),
            "mae_weak":     round(t[0], 1),
            "mae_mid_low":  round(t[1], 1),
            "mae_mid_high": round(t[2], 1),
            "mae_elite":    round(t[3], 1),
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(PROC / "final_comparison_summary.csv", index=False)
    print("\nSaved final_comparison_summary.csv")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_mae_comparison(results)
    plot_tier_comparison(results)
    plot_scatter_grid(results)

    print(f"\nDone. Figures -> {FIG}")


if __name__ == "__main__":
    main()
