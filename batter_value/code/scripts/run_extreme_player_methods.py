#!/usr/bin/env python3
"""
Four strategies to improve extreme-player prediction (Train 3yr -> Test 2024).

Baseline : ElasticNet, wRC+ trajectory, standard MAE loss
Method 1 : + Sample weights (3x for weak <80 / elite >130)
Method 2 : + Consistency features (std, peak, trend over 3 yr)
Method 3 : Tiered model  (classifier -> tier-specific ElasticNet)
Method 4 : Smoothed target (train on 2-yr forward average wRC+)

Outputs (ds_final/figures/evaluation/):
  extreme_mae_comparison.png   overall MAE bar chart (5 models)
  extreme_tier_bias.png        actual vs predicted mean per tier (5 panels)
  extreme_scatter_grid.png     actual vs predicted scatter (2x3 grid)
  extreme_tier_mae_heatmap.png tier-level MAE heatmap

Usage (from ds_final/):
    python scripts/run_extreme_player_methods.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bb_pipeline.eval_baselines import attach_lag_wrc
from bb_pipeline.eval_models import WRC_NUM, _one_hot_encoder, _prep_wrc_xy

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
TIER_NAMES    = ["<80", "80-100", "100-130", "130+"]

MODEL_PALETTE = {
    "Baseline":    "#888888",
    "方法1 加權":   "#457B9D",
    "方法2 一致性": "#2A9D8F",
    "方法3 分層":   "#E9C46A",
    "方法4 平滑目標":"#E63946",
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

CONSISTENCY_NUM = WRC_NUM + ["wRCplus_std", "wRCplus_peak", "wRCplus_trend"]


def add_consistency(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lags = [c for c in ["wRCplus_lag1", "wRCplus_lag2", "wRCplus_lag3"] if c in out.columns]
    lag_mat = out[lags]
    out["wRCplus_std"]   = lag_mat.std(axis=1).fillna(0)
    out["wRCplus_peak"]  = lag_mat.max(axis=1).fillna(lag_mat.mean(axis=1))
    out["wRCplus_trend"] = (out.get("wRCplus_lag1", 0) - out.get("wRCplus_lag3", 0)).fillna(0)
    return out


def _prep_consistency_xy(df: pd.DataFrame):
    df = add_consistency(df)
    use = [c for c in CONSISTENCY_NUM + CAT if c in df.columns]
    x = df[use].copy()
    for c in CONSISTENCY_NUM:
        if c in x.columns:
            x[c] = x[c].fillna(x[c].median())
    y = df["wRCplus_target"].values.astype(float)
    return x, y


def build_smoothed_target(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Replace wRCplus_target with 2-yr forward average.
    For each row (player, season S): target = avg(wRC+_S, wRC+_S+1) if S+1 available.
    Falls back to single-year target if next season is missing.
    """
    id_col = "batter" if "batter" in panel.columns else "player_id"
    lookup = (
        panel[[id_col, "target_season", "wRCplus_target"]]
        .rename(columns={"target_season": "next_season", "wRCplus_target": "wrc_next"})
    )
    out = panel.copy()
    out["next_season"] = out["target_season"] + 1
    merged = out.merge(lookup, on=[id_col, "next_season"], how="left")
    merged["wRCplus_target"] = np.where(
        merged["wrc_next"].notna(),
        0.5 * merged["wRCplus_target"] + 0.5 * merged["wrc_next"],
        merged["wRCplus_target"],
    )
    return merged.drop(columns=["next_season", "wrc_next"])


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline factories
# ─────────────────────────────────────────────────────────────────────────────

def _ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_elasticnet(num_features: list[str] | None = None) -> Pipeline:
    num = num_features or WRC_NUM
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", _ohe(), CAT),
    ])
    return Pipeline([
        ("prep", pre),
        ("model", ElasticNet(alpha=0.03, l1_ratio=0.3, random_state=42, max_iter=5000)),
    ])


def sample_weights(y: np.ndarray, extreme_mult: float = 3.0) -> np.ndarray:
    w = np.ones(len(y))
    w[(y < 80) | (y > 130)] = extreme_mult
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: Tiered model
# ─────────────────────────────────────────────────────────────────────────────

def _assign_tier(wrc_lag1: pd.Series) -> pd.Series:
    return pd.cut(wrc_lag1, bins=[0, 80, 100, 130, 999],
                  labels=["weak", "mid_low", "mid_high", "elite"],
                  right=False).astype(str)


def fit_tiered_models(tr: pd.DataFrame):
    x_tr, y_tr = _prep_wrc_xy(tr)
    tier_col = _assign_tier(tr["wRCplus_lag1"].fillna(tr["wRCplus_lag1"].median()))
    models = {}
    for tier in ["weak", "mid_low", "mid_high", "elite"]:
        mask = tier_col == tier
        if mask.sum() < 10:
            # fall back to full-data model for tiny tiers
            models[tier] = None
            continue
        m = build_elasticnet()
        m.fit(x_tr[mask], y_tr[mask])
        models[tier] = m
    # fallback full model
    full = build_elasticnet()
    full.fit(x_tr, y_tr)
    models["_full"] = full
    return models


def predict_tiered(models: dict, te: pd.DataFrame) -> np.ndarray:
    x_te, _ = _prep_wrc_xy(te)
    tier_col = _assign_tier(te["wRCplus_lag1"].fillna(te["wRCplus_lag1"].median()))
    pred = np.empty(len(te))
    for tier in ["weak", "mid_low", "mid_high", "elite"]:
        mask = (tier_col == tier).values
        m = models.get(tier) or models["_full"]
        pred[mask] = m.predict(x_te[mask])
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def tier_stats(y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"actual": y, "pred": pred})
    df["tier"] = pd.cut(df["actual"], bins=TIER_BINS, labels=TIER_LABELS, right=False)
    return df.groupby("tier", observed=True).apply(
        lambda g: pd.Series({
            "mae": mean_absolute_error(g["actual"], g["pred"]),
            "bias": (g["pred"] - g["actual"]).mean(),
            "mean_actual": g["actual"].mean(),
            "mean_pred": g["pred"].mean(),
            "n": len(g),
        })
    ).reset_index()


def run_all_methods(panel_w: pd.DataFrame):
    tr_raw = panel_w[panel_w["target_season"].isin(TRAIN_SEASONS)]
    te     = panel_w[panel_w["target_season"] == TEST_SEASON]
    x_te, y_te = _prep_wrc_xy(te)

    results = []

    # ── Baseline ─────────────────────────────────────────────────────────────
    print("Baseline: ElasticNet...")
    x_tr, y_tr = _prep_wrc_xy(tr_raw)
    m = build_elasticnet(); m.fit(x_tr, y_tr)
    pred = m.predict(x_te)
    results.append({"name": "Baseline", "pred": pred, "actual": y_te,
                    "mae": mean_absolute_error(y_te, pred),
                    "r2":  r2_score(y_te, pred),
                    "tiers": tier_stats(y_te, pred)})

    # ── Method 1: Sample weights ──────────────────────────────────────────────
    print("Method 1: Sample weights...")
    w = sample_weights(y_tr, extreme_mult=3.0)
    m1 = build_elasticnet(); m1.fit(x_tr, y_tr, model__sample_weight=w)
    pred1 = m1.predict(x_te)
    results.append({"name": "方法1 加權", "pred": pred1, "actual": y_te,
                    "mae": mean_absolute_error(y_te, pred1),
                    "r2":  r2_score(y_te, pred1),
                    "tiers": tier_stats(y_te, pred1)})

    # ── Method 2: Consistency features ───────────────────────────────────────
    print("Method 2: Consistency features...")
    x_tr_c, y_tr_c = _prep_consistency_xy(tr_raw)
    x_te_c, _      = _prep_consistency_xy(te)
    m2 = build_elasticnet(CONSISTENCY_NUM); m2.fit(x_tr_c, y_tr_c)
    pred2 = m2.predict(x_te_c)
    results.append({"name": "方法2 一致性", "pred": pred2, "actual": y_te,
                    "mae": mean_absolute_error(y_te, pred2),
                    "r2":  r2_score(y_te, pred2),
                    "tiers": tier_stats(y_te, pred2)})

    # ── Method 3: Tiered model ────────────────────────────────────────────────
    print("Method 3: Tiered model...")
    tier_models = fit_tiered_models(tr_raw)
    pred3 = predict_tiered(tier_models, te)
    results.append({"name": "方法3 分層", "pred": pred3, "actual": y_te,
                    "mae": mean_absolute_error(y_te, pred3),
                    "r2":  r2_score(y_te, pred3),
                    "tiers": tier_stats(y_te, pred3)})

    # ── Method 4: Smoothed target ─────────────────────────────────────────────
    print("Method 4: Smoothed target...")
    tr_smooth = build_smoothed_target(tr_raw)
    x_tr_s, y_tr_s = _prep_wrc_xy(tr_smooth)
    m4 = build_elasticnet(); m4.fit(x_tr_s, y_tr_s)
    pred4 = m4.predict(x_te)
    results.append({"name": "方法4 平滑目標", "pred": pred4, "actual": y_te,
                    "mae": mean_absolute_error(y_te, pred4),
                    "r2":  r2_score(y_te, pred4),
                    "tiers": tier_stats(y_te, pred4)})

    return results, y_te


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_mae_comparison(results: list[dict]) -> None:
    names  = [r["name"] for r in results]
    maes   = [r["mae"]  for r in results]
    colors = [MODEL_PALETTE[n] for n in names]
    best   = min(maes)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(names, maes, color=colors, edgecolor="white", width=0.55)
    for bar, mae, r2, name in zip(bars, maes, [r["r2"] for r in results], names):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"MAE {mae:.2f}\nR2 {r2:.3f}",
                ha="center", fontsize=8.5)
        if mae == best:
            bar.set_edgecolor("#000000")
            bar.set_linewidth(2.5)
    ax.set_ylabel("MAE — Test 2024", fontsize=10)
    ax.set_title("五種模型整體 MAE 比較\n(Train 2021+22+23 → Test 2024)", fontsize=11)
    ax.set_ylim(0, max(maes) * 1.28)
    plt.tight_layout()
    p = FIG / "extreme_mae_comparison.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {p.name}")


def plot_tier_bias(results: list[dict]) -> None:
    """2-row x 3-col grid (5 models + 1 summary)."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes_flat = axes.flatten()

    for ax, res in zip(axes_flat[:5], results):
        tiers = res["tiers"]
        x = np.arange(len(tiers))
        ax.bar(x - 0.22, tiers["mean_actual"], 0.42,
               label="實際平均", color="#2A9D8F", alpha=0.85)
        ax.bar(x + 0.22, tiers["mean_pred"], 0.42,
               label="預測平均", color=MODEL_PALETTE[res["name"]], alpha=0.85)
        ax.axhline(100, color="gray", ls="--", lw=1)
        for i, row in tiers.iterrows():
            ax.text(i, max(row["mean_actual"], row["mean_pred"]) + 1.5,
                    f"MAE\n{row['mae']:.1f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(tiers["tier"].astype(str), fontsize=8)
        ax.set_ylim(0, 195)
        ax.set_ylabel("wRC+", fontsize=9)
        ax.set_title(f"{res['name']}  (整體 MAE={res['mae']:.2f})", fontsize=10)
        ax.legend(fontsize=7.5)

    # 6th panel: tier-level MAE comparison across models
    ax6 = axes_flat[5]
    tier_names = results[0]["tiers"]["tier"].astype(str).tolist()
    x6 = np.arange(len(tier_names))
    bar_w = 0.15
    for j, res in enumerate(results):
        tier_maes = res["tiers"]["mae"].tolist()
        ax6.bar(x6 + j * bar_w, tier_maes, bar_w,
                color=MODEL_PALETTE[res["name"]], label=res["name"].replace("\n", " "),
                edgecolor="white", alpha=0.9)
    ax6.set_xticks(x6 + bar_w * 2)
    ax6.set_xticklabels(tier_names, fontsize=8)
    ax6.set_ylabel("MAE")
    ax6.set_title("各等級 MAE 五模型對比", fontsize=10)
    ax6.legend(fontsize=6.5, ncol=1)

    plt.suptitle("弱打 / 中下 / 中上 / 明星 — 各方法實際 vs 預測偏差", fontsize=12, y=1.01)
    plt.tight_layout()
    p = FIG / "extreme_tier_bias.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {p.name}")


def plot_scatter_grid(results: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    axes_flat = axes.flatten()

    for ax, res in zip(axes_flat[:5], results):
        y, pred = res["actual"], res["pred"]
        c = MODEL_PALETTE[res["name"]]
        ax.scatter(y, pred, alpha=0.45, s=28, color=c,
                   edgecolors="white", linewidths=0.3)
        lim_min = max(20, float(min(y.min(), pred.min())) - 10)
        lim_max = min(230, float(max(y.max(), pred.max())) + 10)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.2)
        ax.axhline(100, color="gray", ls=":", lw=0.8, alpha=0.6)
        ax.axvline(80,  color="#E63946", ls=":", lw=0.8, alpha=0.4)
        ax.axvline(130, color="#E63946", ls=":", lw=0.8, alpha=0.4)
        ax.set_xlabel("實際 wRC+", fontsize=8.5)
        ax.set_ylabel("預測 wRC+", fontsize=8.5)
        ax.set_title(f"{res['name']}\nMAE={res['mae']:.2f}  R2={res['r2']:.3f}", fontsize=9.5)

    # 6th panel: residual box-plot per tier for each model
    ax6 = axes_flat[5]
    tier_data = {}
    for tl in TIER_NAMES:
        tier_data[tl] = []
    for res in results:
        y, pred = res["actual"], res["pred"]
        df_t = pd.DataFrame({"actual": y, "resid": pred - y})
        df_t["tier"] = pd.cut(df_t["actual"], bins=TIER_BINS,
                               labels=TIER_NAMES, right=False).astype(str)
        for tl in TIER_NAMES:
            tier_data[tl].append(df_t[df_t["tier"] == tl]["resid"].values)

    positions = []
    data_all  = []
    tick_pos  = []
    tick_lab  = []
    gap = len(results) + 1
    colors_box = list(MODEL_PALETTE.values())
    for i, tl in enumerate(TIER_NAMES):
        base = i * gap
        tick_pos.append(base + len(results) / 2)
        tick_lab.append(tl)
        for j, arr in enumerate(tier_data[tl]):
            positions.append(base + j)
            data_all.append(arr)
    bp = ax6.boxplot(data_all, positions=positions, widths=0.6,
                     patch_artist=True, showfliers=False,
                     medianprops={"color": "black", "lw": 1.5})
    for patch, pos in zip(bp["boxes"], positions):
        model_idx = pos % gap
        if model_idx < len(colors_box):
            patch.set_facecolor(colors_box[model_idx])
            patch.set_alpha(0.75)
    ax6.axhline(0, color="black", lw=1, ls="--")
    ax6.set_xticks(tick_pos)
    ax6.set_xticklabels(tick_lab, fontsize=8.5)
    ax6.set_ylabel("殘差（預測 - 實際）", fontsize=9)
    ax6.set_title("各打者等級殘差分布（五模型）", fontsize=9.5)

    plt.suptitle("實際 vs 預測散點＋殘差箱型圖（Train 3yr → Test 2024）",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    p = FIG / "extreme_scatter_grid.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {p.name}")


def plot_tier_mae_heatmap(results: list[dict]) -> None:
    model_names = [r["name"].replace("\n", " ") for r in results]
    tier_labels = results[0]["tiers"]["tier"].astype(str).tolist()
    mat = np.array([[row for row in r["tiers"]["mae"]] for r in results])

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(tier_labels)))
    ax.set_xticklabels(tier_labels, fontsize=9)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > mat[:, j].mean() else "black")
    plt.colorbar(im, ax=ax, label="MAE")
    ax.set_title("各打者等級 MAE 熱圖\n（越綠越好，越紅越差）", fontsize=11)
    plt.tight_layout()
    p = FIG / "extreme_tier_mae_heatmap.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    panel  = pd.read_parquet(PROC / "projection_panel.parquet")
    season = pd.read_csv(PROC / "batting_clean.csv")
    panel_w = attach_lag_wrc(panel, season)

    results, y_te = run_all_methods(panel_w)

    # ── Print summary ──────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("Test 2024 Summary")
    print("=" * 65)
    print(f"{'Model':<22} {'MAE':>7} {'R2':>8}  "
          f"{'<80':>8} {'80-100':>8} {'100-130':>8} {'130+':>8}")
    print("-" * 65)
    for r in results:
        tier_maes = r["tiers"]["mae"].tolist()
        name = r["name"].replace("\n", " ")
        tier_str = "  ".join(f"{v:6.1f}" for v in tier_maes)
        print(f"{name:<22} {r['mae']:>7.2f} {r['r2']:>8.3f}  {tier_str}")

    # ── Key insight: best method per tier ─────────────────────────────────
    print()
    print("Best method per tier:")
    tier_lbls = results[0]["tiers"]["tier"].tolist()
    for j, tl in enumerate(tier_lbls):
        best_idx = int(np.argmin([r["tiers"]["mae"].iloc[j] for r in results]))
        print(f"  {tl.replace(chr(10), ' '):<14}: "
              f"{results[best_idx]['name'].replace(chr(10), ' ')} "
              f"(MAE={results[best_idx]['tiers']['mae'].iloc[j]:.1f})")

    # ── Plots ──────────────────────────────────────────────────────────────
    print()
    print("Generating plots...")
    plot_mae_comparison(results)
    plot_tier_bias(results)
    plot_scatter_grid(results)
    plot_tier_mae_heatmap(results)

    print(f"\nDone. Figures -> {FIG}")


if __name__ == "__main__":
    main()
