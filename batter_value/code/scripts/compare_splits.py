#!/usr/bin/env python3
"""Compare Walk-forward vs Train-3yr vs Original split for ElasticNet (wRC+ trajectory)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": ["Microsoft JhengHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})
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

CAT = ["primary_pos"]


def build_elasticnet() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), WRC_NUM),
        ("cat", _one_hot_encoder(), CAT),
    ])
    return Pipeline([
        ("prep", pre),
        ("model", ElasticNet(alpha=0.03, l1_ratio=0.3, random_state=42, max_iter=5000)),
    ])


def walkforward(panel: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(panel["target_season"].unique())
    rows = []
    for i in range(1, len(seasons)):
        test_s = seasons[i]
        train_s = seasons[:i]
        tr = panel[panel["target_season"].isin(train_s)]
        te = panel[panel["target_season"] == test_s]
        x_tr, y_tr = _prep_wrc_xy(tr)
        x_te, y_te = _prep_wrc_xy(te)
        m = build_elasticnet()
        m.fit(x_tr, y_tr)
        pred = m.predict(x_te)
        rows.append({
            "test_season": test_s,
            "train_seasons": str(list(train_s)),
            "n_train": len(tr),
            "n_test": len(te),
            "mae": float(mean_absolute_error(y_te, pred)),
            "r2": float(r2_score(y_te, pred)),
        })
    return pd.DataFrame(rows)


def train3yr(panel: pd.DataFrame) -> dict:
    tr = panel[panel["target_season"].isin([2021, 2022, 2023])]
    te = panel[panel["target_season"] == 2024]
    x_tr, y_tr = _prep_wrc_xy(tr)
    x_te, y_te = _prep_wrc_xy(te)
    m = build_elasticnet()
    m.fit(x_tr, y_tr)
    pred = m.predict(x_te)
    return {
        "split": "Train 2021+22+23 → Test 2024",
        "n_train": len(tr),
        "n_test": len(te),
        "mae": float(mean_absolute_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
        "pred": pred,
        "actual": y_te,
    }


def original(panel: pd.DataFrame) -> dict:
    tr = panel[panel["target_season"].isin([2021, 2022])]
    te = panel[panel["target_season"] == 2024]
    x_tr, y_tr = _prep_wrc_xy(tr)
    x_te, y_te = _prep_wrc_xy(te)
    m = build_elasticnet()
    m.fit(x_tr, y_tr)
    pred = m.predict(x_te)
    return {
        "split": "Train 2021+22 → Test 2024（原切法）",
        "n_train": len(tr),
        "n_test": len(te),
        "mae": float(mean_absolute_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
        "pred": pred,
        "actual": y_te,
    }


def plot_walkforward(wf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wf["test_season"], wf["mae"], "o-", color="#457B9D", lw=2, markersize=9)

    max_mae = wf["mae"].max()
    for _, r in wf.iterrows():
        # 最高點標在下方，其餘標在上方，避免撞到標題
        if r["mae"] == max_mae:
            offset = (0, -38)
        else:
            offset = (0, 12)
        ax.annotate(
            f"MAE={r['mae']:.1f}\n(n_train={r['n_train']})",
            (r["test_season"], r["mae"]),
            textcoords="offset points", xytext=offset,
            ha="center", fontsize=8.5,
        )

    ax.set_xticks(wf["test_season"])
    ax.set_xlabel("測試目標球季", fontsize=11)
    ax.set_ylabel("MAE（wRC+ 點數）", fontsize=11)
    ax.set_title(
        "Walk-forward 驗證：ElasticNet（歷年 wRC+ 軌跡）",
        fontsize=12, pad=12,
    )
    ax.set_ylim(wf["mae"].min() - 4, wf["mae"].max() + 5)
    plt.tight_layout()
    fig.savefig(FIG / "eval_elasticnet_walkforward.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved eval_elasticnet_walkforward.png")


def plot_split_comparison(wf: pd.DataFrame, orig: dict, t3: dict) -> None:
    labels = [
        "原切法\nTrain 2yr",
        "方案二\nTrain 3yr",
        "Walk-forward\n平均",
    ]
    maes = [orig["mae"], t3["mae"], wf["mae"].mean()]
    colors = ["#E63946", "#2A9D8F", "#457B9D"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, maes, color=colors, edgecolor="white", width=0.5)
    for bar, mae in zip(bars, maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{mae:.2f}",
            ha="center", fontsize=10, fontweight="bold",
        )
    ax.set_ylabel("MAE（wRC+ 點數）")
    ax.set_title("三種切割策略比較\nElasticNet — 歷年 wRC+ 軌跡")
    ax.set_ylim(0, max(maes) * 1.2)
    plt.tight_layout()
    fig.savefig(FIG / "eval_elasticnet_split_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved eval_elasticnet_split_comparison.png")


def plot_scatter_compare(orig: dict, t3: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, res in zip(axes, [orig, t3]):
        y, pred = res["actual"], res["pred"]
        ax.scatter(y, pred, alpha=0.5, s=35, c="#457B9D", edgecolors="white", linewidths=0.3)
        lim = [max(20, min(y.min(), pred.min()) - 5), min(220, max(y.max(), pred.max()) + 5)]
        ax.plot(lim, lim, "k--", lw=1.2)
        ax.set_xlabel("實際 wRC+（2024）")
        ax.set_ylabel("預測 wRC+")
        ax.set_title(f"{res['split']}\nMAE={res['mae']:.2f}  R²={res['r2']:.3f}")
    plt.tight_layout()
    fig.savefig(FIG / "eval_elasticnet_scatter_compare.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved eval_elasticnet_scatter_compare.png")


def main() -> None:
    panel = pd.read_parquet(PROC / "projection_panel.parquet")
    season = pd.read_csv(PROC / "batting_clean.csv")
    panel_w = attach_lag_wrc(panel, season)

    print("=" * 55)
    print("方案一：Walk-forward（ElasticNet 歷年 wRC+ 軌跡）")
    print("=" * 55)
    wf = walkforward(panel_w)
    print(wf[["test_season", "train_seasons", "n_train", "n_test", "mae", "r2"]].to_string(index=False))
    print(f"\nWalk-forward 平均 MAE: {wf['mae'].mean():.2f}")
    print(f"Walk-forward 平均 R2 : {wf['r2'].mean():.3f}")

    print()
    print("=" * 55)
    print("方案二：Train 2021+2022+2023 → Test 2024")
    print("=" * 55)
    t3 = train3yr(panel_w)
    print(f"Train 筆數: {t3['n_train']}   Test 筆數: {t3['n_test']}")
    print(f"MAE: {t3['mae']:.2f}")
    print(f"R2 : {t3['r2']:.3f}")

    print()
    print("=" * 55)
    print("原切法：Train 2021+2022 → Test 2024（對照）")
    print("=" * 55)
    orig = original(panel_w)
    print(f"Train 筆數: {orig['n_train']}   Test 筆數: {orig['n_test']}")
    print(f"MAE: {orig['mae']:.2f}")
    print(f"R2 : {orig['r2']:.3f}")

    print()
    print("=" * 55)
    print("三種策略 Test 2024 MAE 一覽")
    print("=" * 55)
    summary = pd.DataFrame([
        {"策略": "原切法（Train 2yr）", "Test MAE": round(orig["mae"], 2), "R2": round(orig["r2"], 3)},
        {"策略": "方案二（Train 3yr）", "Test MAE": round(t3["mae"], 2), "R2": round(t3["r2"], 3)},
        {"策略": "Walk-forward 平均", "Test MAE": round(wf["mae"].mean(), 2), "R2": round(wf["r2"].mean(), 3)},
    ])
    print(summary.to_string(index=False))

    wf.to_csv(PROC / "eval_elasticnet_walkforward.csv", index=False)
    summary.to_csv(PROC / "eval_elasticnet_split_summary.csv", index=False)

    plot_walkforward(wf)
    plot_split_comparison(wf, orig, t3)
    plot_scatter_compare(orig, t3)

    print("\nDone. Figures ->", FIG)


if __name__ == "__main__":
    main()
