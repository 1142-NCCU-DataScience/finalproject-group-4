#!/usr/bin/env python3
"""
Run supervised model comparison (no rule-based naive baselines).

Usage (from ds_final/):
    python scripts/run_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bb_pipeline.eval_models import run_model_evaluation  # noqa: E402

PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures" / "evaluation"

matplotlib.rcParams.update(
    {
        "font.family": ["Microsoft JhengHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    }
)

MODEL_COLORS = {
    "process": "#457B9D",
    "wrc_history": "#E9C46A",
    "combined": "#2A9D8F",
}


def _load_season_df() -> pd.DataFrame:
    clean = PROC / "batting_clean.csv"
    if not clean.exists():
        raise FileNotFoundError(f"Missing {clean}")
    return pd.read_csv(clean)


def plot_models_mae(results: dict) -> None:
    df = results["models_comparison"]
    test = df[df["split"] == "test"].sort_values("mae")
    colors = [MODEL_COLORS.get(f, "#888888") for f in test["family"]]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(test["name"], test["mae"], color=colors, edgecolor="white", height=0.65)
    ax.set_xlabel("MAE（wRC+ 點數）— Hold-out Test 2024")
    ax.set_title("監督式模型比較（皆在 Train 2021–22 訓練、Test 2024 評估）")
    best = results["best_model_name"]
    for bar, mae, name in zip(bars, test["mae"], test["name"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, f"{mae:.1f}", va="center", fontsize=8)
        if name == best:
            bar.set_edgecolor("#E63946")
            bar.set_linewidth(2.5)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=MODEL_COLORS["process"], label="過程指標（K%、BB%、PA…）"),
            Patch(color=MODEL_COLORS["wrc_history"], label="歷年 wRC+ 軌跡（學習加權）"),
            Patch(color=MODEL_COLORS["combined"], label="兩者合併"),
        ],
        loc="lower right",
        fontsize=8,
    )
    plt.tight_layout()
    fig.savefig(FIG / "eval_models_mae_test.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved eval_models_mae_test.png")


def plot_models_val_test(results: dict) -> None:
    df = results["models_comparison"]
    order = df[df["split"] == "test"].sort_values("mae")["name"].tolist()
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(order))
    w = 0.35
    val = df[df["split"] == "validation"].set_index("name").loc[order]["mae"]
    tst = df[df["split"] == "test"].set_index("name").loc[order]["mae"]
    ax.bar(x - w / 2, val, w, label="Validation 2023", color="#2A9D8F", alpha=0.9)
    ax.bar(x + w / 2, tst, w, label="Test 2024", color="#E63946", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("MAE（wRC+ 點數）")
    ax.set_title("監督式模型：Validation vs Test")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG / "eval_models_val_test.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved eval_models_val_test.png")


def plot_error_by_tier(results: dict) -> None:
    tier = results["error_by_tier"]
    best = results["best_model_name"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.bar(tier["tier"].astype(str), tier["mae"], color="#457B9D", edgecolor="white")
    ax.set_xlabel("實際 wRC+ 區間（Test 2024）")
    ax.set_ylabel("MAE")
    ax.set_title(f"誤差分層 — {best}")

    ax = axes[1]
    x = np.arange(len(tier))
    ax.bar(x - 0.2, tier["mean_actual"], 0.4, label="實際平均", color="#2A9D8F")
    ax.bar(x + 0.2, tier["mean_pred"], 0.4, label="預測平均", color="#E63946", alpha=0.85)
    ax.axhline(100, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(tier["tier"], rotation=15, ha="right")
    ax.set_ylabel("wRC+")
    ax.set_title("系統性偏差")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG / "eval_error_by_tier.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved eval_error_by_tier.png")


def plot_best_actual_vs_pred(results: dict) -> None:
    y = results["best_test_actual"]
    pred = results["best_test_pred"]
    best = results["best_model_name"]
    mae = float(np.mean(np.abs(pred - y)))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y, pred, alpha=0.55, s=40, c="#457B9D", edgecolors="white", linewidths=0.3)
    lim_min = max(20, min(y.min(), pred.min()) - 10)
    lim_max = min(220, max(y.max(), pred.max()) + 10)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.2, label="y = x")
    ax.set_xlabel("實際 wRC+（2024）")
    ax.set_ylabel("預測 wRC+")
    ax.set_title(f"{best}\nTest 2024  MAE = {mae:.1f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG / "eval_best_actual_vs_pred.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved eval_best_actual_vs_pred.png")


def print_summary(results: dict) -> None:
    def _safe_print(text: str) -> None:
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("utf-8", errors="replace").decode("utf-8"))

    _safe_print("\n" + "=" * 60)
    _safe_print("SUPERVISED MODELS — Test 2024 (lower MAE is better)")
    _safe_print("=" * 60)
    test = results["models_comparison"][results["models_comparison"]["split"] == "test"].sort_values("mae")
    _safe_print(test[["name", "family", "mae", "rmse", "r2"]].to_string(index=False))
    _safe_print(f"\nBest on Test 2024: {results['best_model_name']}")


def main() -> None:
    panel_path = PROC / "projection_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing {panel_path}")

    FIG.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_path)
    season_df = _load_season_df()

    results = run_model_evaluation(panel, season_df)
    results["models_comparison"].to_csv(PROC / "eval_models_comparison.csv", index=False)
    results["error_by_tier"].to_csv(PROC / "eval_error_by_tier.csv", index=False)

    plot_models_mae(results)
    plot_models_val_test(results)
    plot_error_by_tier(results)
    plot_best_actual_vs_pred(results)
    print_summary(results)
    print("\nDone. Outputs:", ROOT)


if __name__ == "__main__":
    main()
