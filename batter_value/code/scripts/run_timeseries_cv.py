"""
ElasticNet with Time-Series-Aware Hyperparameter Tuning
=======================================================
更嚴謹的做法（期刊投稿等級）：

  Step 1. 在 2021–2023 訓練集內，用 Expanding Window TimeSeriesSplit
          搜索最佳超參數（alpha, l1_ratio），完全不碰 2024。

  Step 2. 用找到的最佳超參數，以 2021–2023 全量資料訓練最終模型。

  Step 3. 只在 2024 測試集報告最終指標（MAE, RMSE, R²）。

對照：ds_final/scripts/run_elasticnet_train3yr.py 用的是
       sklearn 預設的 KFold（不保證時序順序），
       這裡改成 TimeSeriesSplit，確保 CV 的每一個 fold
       都是「用過去預測未來」，不會讓未來資料幫助訓練。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({
    "font.family": ["Microsoft JhengHei", "sans-serif"],
    "axes.unicode_minus": False,
})
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 路徑設定 ─────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# ds_final 的 pipeline 模組
DS = ROOT / "ds_final"
sys.path.insert(0, str(DS))

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

from bb_pipeline.eval_baselines import attach_lag_wrc   # noqa: E402

# ── 資料載入 ─────────────────────────────────────────────────────────────────
PROC  = ROOT / "ds_final" / "data" / "processed"
panel = pd.read_parquet(PROC / "projection_panel.parquet")
season_df = pd.read_csv(PROC / "batting_clean.csv")

# 加入 wRC+ lag 欄位（S-1, S-2, S-3）
panel = attach_lag_wrc(panel, season_df)

# wRC+ lag 特徵（與 ds_final 相同）
LAG_COLS = ["wRCplus_lag1", "wRCplus_lag2", "wRCplus_lag3"]
TARGET   = "wRCplus_target"

# 只要 target 有值即保留；lag 欄位缺失後面用 median 填補（與 ds_final 相同）
df = panel.dropna(subset=[TARGET]).copy()
for col in LAG_COLS:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# ── 訓練 / 測試切割 ──────────────────────────────────────────────────────────
train = df[df["target_season"].isin([2021, 2022, 2023])].copy()
test  = df[df["target_season"] == 2024].copy()

# 訓練集按時間排序（TimeSeriesSplit 需要）
train = train.sort_values("target_season").reset_index(drop=True)

X_train = train[LAG_COLS].values
y_train = train[TARGET].values
X_test  = test[LAG_COLS].values
y_test  = test[TARGET].values

print(f"Train size : {len(X_train)}  (2021={len(train[train.target_season==2021])}, "
      f"2022={len(train[train.target_season==2022])}, "
      f"2023={len(train[train.target_season==2023])})")
print(f"Test  size : {len(X_test)}  (2024)")

# ── Step 1：Expanding Window TimeSeriesSplit 調超參數 ────────────────────────
#
#  TimeSeriesSplit(n_splits=2) 產生的 expanding folds：
#   fold 1: Train 2021       → Val 2022+2023  ← 但我們希望按年切
#
#  更精準：手動定義 expanding window，按目標年切分
#   fold 1: Train 2021(217)        → Val 2022(236)
#   fold 2: Train 2021+2022(453)   → Val 2023(234)
#
#  這樣每個 fold 都是「用過去預測未來」，完全符合時序原則。

fold1_train = train[train["target_season"] == 2021].index.tolist()
fold1_val   = train[train["target_season"] == 2022].index.tolist()
fold2_train = train[train["target_season"].isin([2021, 2022])].index.tolist()
fold2_val   = train[train["target_season"] == 2023].index.tolist()

custom_cv = [(fold1_train, fold1_val),
             (fold2_train, fold2_val)]

print("\n=== Expanding Window Folds ===")
print(f"Fold 1: Train n={len(fold1_train)} (2021) → Val n={len(fold1_val)} (2022)")
print(f"Fold 2: Train n={len(fold2_train)} (2021+2022) → Val n={len(fold2_val)} (2023)")

# 超參數搜索空間
param_grid = {
    "en__alpha":    [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    "en__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("en",     ElasticNet(max_iter=5000)),
])

gs = GridSearchCV(
    pipe,
    param_grid,
    cv=custom_cv,          # ← 時序 aware 的 expanding window
    scoring="neg_mean_absolute_error",
    refit=False,
    n_jobs=-1,
)
gs.fit(X_train, y_train)

best_params = gs.best_params_
best_cv_mae = -gs.best_score_
print(f"\n最佳超參數   : {best_params}")
print(f"CV MAE（均值）: {best_cv_mae:.3f}")

# 印出每個 fold 的詳細結果
cv_results = pd.DataFrame(gs.cv_results_)
best_row = cv_results.loc[gs.best_index_]
print(f"  Fold 1 MAE : {-best_row['split0_test_score']:.3f}")
print(f"  Fold 2 MAE : {-best_row['split1_test_score']:.3f}")

# ── Step 2：用最佳超參數，以 2021–2023 全量訓練最終模型 ───────────────────────
final_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("en",     ElasticNet(
        alpha=best_params["en__alpha"],
        l1_ratio=best_params["en__l1_ratio"],
        max_iter=5000,
    )),
])
final_pipe.fit(X_train, y_train)

# ── Step 3：只在 2024 報告最終指標 ────────────────────────────────────────────
pred = final_pipe.predict(X_test)
mae  = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2   = r2_score(y_test, pred)

print("\n=== 最終模型 — Test 2024 ===")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R2   : {r2:.3f}")

# 分層誤差（弱打 / 中等 / 強打）
tiers = pd.cut(y_test, bins=[0, 80, 115, 999],
               labels=["弱打 (<80)", "中等 (80-115)", "強打 (>115)"])
tier_df = pd.DataFrame({"actual": y_test, "pred": pred, "tier": tiers})
tier_mae = tier_df.groupby("tier", observed=True).apply(
    lambda g: mean_absolute_error(g["actual"], g["pred"])
).reset_index(name="MAE")
print("\n=== 分層 MAE ===")
print(tier_mae.to_string(index=False))

# ── 對照：ds_final 的 GridSearchCV（預設 KFold=5）結果 ───────────────────────
# 從已知結果直接寫入供比較
baseline_metrics = {
    "MAE":          24.35,
    "R2":           0.179,
    "弱打 MAE":     37.2,
    "強打 MAE":     36.9,
}

print("\n=== 對照表：KFold CV vs TimeSeriesSplit CV ===")
print(f"{'指標':<14} {'KFold（原本）':>14} {'TimeSeriesSplit（本腳本）':>22}")
print("-" * 52)
print(f"{'MAE':<14} {baseline_metrics['MAE']:>14.3f} {mae:>22.3f}")
print(f"{'R2':<14} {baseline_metrics['R2']:>14.3f} {r2:>22.3f}")
print(f"{'弱打 MAE':<14} {baseline_metrics['弱打 MAE']:>14.1f} "
      f"{tier_mae.loc[tier_mae['tier']=='弱打 (<80)', 'MAE'].values[0]:>22.1f}")
print(f"{'強打 MAE':<14} {baseline_metrics['強打 MAE']:>14.1f} "
      f"{tier_mae.loc[tier_mae['tier']=='強打 (>115)', 'MAE'].values[0]:>22.1f}")

# ─── 圖 1：CV Fold MAE 折線圖 ─────────────────────────────────────────────────
fold_names  = ["Fold 1\nTrain 2021→Val 2022", "Fold 2\nTrain 2021+22→Val 2023"]
fold_maes   = [-best_row["split0_test_score"], -best_row["split1_test_score"]]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(fold_names, fold_maes, color=["#457B9D", "#2A9D8F"], width=0.4, edgecolor="white")
for i, v in enumerate(fold_maes):
    ax.text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("MAE（wRC+ 點數）", fontsize=11)
ax.set_title("Expanding Window CV — 各 Fold MAE\n（最佳超參數）", fontsize=11)
ax.set_ylim(0, max(fold_maes) * 1.25)
plt.tight_layout()
fig.savefig(FIG / "ts_cv_fold_mae.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved ts_cv_fold_mae.png")

# ─── 圖 2：Actual vs Predicted ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(y_test, pred, alpha=0.55, s=45, c="#457B9D",
           edgecolors="white", linewidths=0.4)
lim_min = max(20,  min(float(y_test.min()), float(pred.min())) - 10)
lim_max = min(230, max(float(y_test.max()), float(pred.max())) + 10)
ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.3, label="完美預測線 (y=x)")
ax.set_xlabel("實際 wRC+（2024）", fontsize=11)
ax.set_ylabel("預測 wRC+", fontsize=11)
ax.set_title(f"ElasticNet + TimeSeriesSplit CV\nMAE = {mae:.1f}  |  R2 = {r2:.3f}", fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIG / "ts_cv_actual_vs_pred.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved ts_cv_actual_vs_pred.png")

# ─── 圖 3：KFold vs TimeSeriesSplit 比較長條圖 ───────────────────────────────
methods = ["KFold CV\n（原本做法）", "TimeSeriesSplit CV\n（本腳本）"]
maes    = [baseline_metrics["MAE"], mae]
r2s     = [baseline_metrics["R2"], r2]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
colors = ["#E9C46A", "#457B9D"]

axes[0].bar(methods, maes, color=colors, width=0.4, edgecolor="white")
for i, v in enumerate(maes):
    axes[0].text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
axes[0].set_title("Test 2024 MAE 比較", fontsize=12)
axes[0].set_ylabel("MAE（wRC+ 點數）", fontsize=11)
axes[0].set_ylim(0, max(maes) * 1.3)

axes[1].bar(methods, r2s, color=colors, width=0.4, edgecolor="white")
for i, v in enumerate(r2s):
    axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
axes[1].set_title("Test 2024 R² 比較", fontsize=12)
axes[1].set_ylabel("R²", fontsize=11)
axes[1].set_ylim(0, max(r2s) * 1.4)

plt.suptitle("ElasticNet：KFold CV vs TimeSeriesSplit CV（相同訓練資料 2021–2023）",
             fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(FIG / "ts_cv_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved ts_cv_comparison.png")

print("\nDone. 圖片 →", FIG)
