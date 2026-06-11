"""
完整四方法比較
==============
  A. 同學原始 RF         — Train 2021+2022, Val 2023, Test 2024
  B. 同學 RF 3yr         — Train 2021+2022+2023, Test 2024
  C. 我的最終版          — ElasticNet + 3× 加權 (Train 3yr)
  D. TimeSeriesSplit CV  — ElasticNet, Expanding Window CV (本資料夾)

產生：
  figures/full_comparison_mae.png      — 整體 & 分層 MAE 比較
  figures/full_comparison_scatter.png  — 四模型 2×2 散點圖
  figures/full_comparison_r2.png       — R² 比較
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bb_pipeline"))
sys.path.insert(0, str(ROOT / "ds_final"))

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

from bb_pipeline.eval_baselines import attach_lag_wrc          # noqa: E402
from bb_pipeline.eval_models    import WRC_NUM, _prep_wrc_xy   # noqa: E402
from bb_pipeline.train_projection import (                      # noqa: E402
    fit_by_target_season, prepare_xy, build_rf_pipeline,
)

# ── 共用設定 ─────────────────────────────────────────────────────────────────
TIER_BINS   = [0, 80, 100, 130, 300]
TIER_LABELS = ["<80\n(弱打)", "80-100\n(中下)", "100-130\n(中上)", "130+\n(明星)"]
COLORS      = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D"]
METHOD_NAMES = [
    "同學 RF\n(Train 2yr)",
    "同學 RF\n(Train 3yr)",
    "我的最終版\nElasticNet+加權3×",
    "TimeSeriesSplit CV\nElasticNet",
]

# ── 資料載入 ─────────────────────────────────────────────────────────────────
PROC_BB = ROOT / "data"      / "processed"
PROC_DS = ROOT / "ds_final"  / "data" / "processed"

panel_bb  = pd.read_parquet(PROC_BB / "projection_panel.parquet")
panel_ds  = pd.read_parquet(PROC_DS / "projection_panel.parquet")
season_df = pd.read_csv(PROC_DS / "batting_clean.csv")

panel_wrc = attach_lag_wrc(panel_ds, season_df)
for col in WRC_NUM:
    if col in panel_wrc.columns:
        panel_wrc[col] = panel_wrc[col].fillna(panel_wrc[col].median())

LAG_COLS = ["wRCplus_lag1", "wRCplus_lag2", "wRCplus_lag3"]
TARGET   = "wRCplus_target"


def tier_mae(y_true, y_pred):
    tiers = pd.cut(y_true, bins=TIER_BINS, labels=TIER_LABELS)
    df = pd.DataFrame({"a": y_true, "p": y_pred, "t": tiers})
    return df.groupby("t", observed=True).apply(
        lambda g: mean_absolute_error(g["a"], g["p"])
    ).values


# ── A. 同學原始 RF (Train 2yr) ────────────────────────────────────────────────
print("Training A: 同學原始 RF (Train 2021+2022)...")
res_orig = fit_by_target_season(panel_bb)
rf_orig  = res_orig["rf_model"]
te_bb    = panel_bb[panel_bb["target_season"] == 2024]
xte_bb, yte_bb = prepare_xy(te_bb)
pred_a   = rf_orig.predict(xte_bb)
mae_a    = mean_absolute_error(yte_bb, pred_a)
r2_a     = r2_score(yte_bb, pred_a)
tier_a   = tier_mae(yte_bb, pred_a)
print(f"  MAE={mae_a:.2f}  R²={r2_a:.3f}")

# ── B. 同學 RF 3yr (Train 3yr) ────────────────────────────────────────────────
print("Training B: 同學 RF 3yr (Train 2021+2022+2023)...")
res_3yr = fit_by_target_season(panel_bb, train_targets=(2021, 2022, 2023),
                                val_target=2023, test_target=2024)
rf_3yr  = res_3yr["rf_model"]
pred_b  = rf_3yr.predict(xte_bb)
mae_b   = mean_absolute_error(yte_bb, pred_b)
r2_b    = r2_score(yte_bb, pred_b)
tier_b  = tier_mae(yte_bb, pred_b)
print(f"  MAE={mae_b:.2f}  R²={r2_b:.3f}")

# ── C. 我的最終版：ElasticNet + 3× 加權 ──────────────────────────────────────
print("Training C: 我的最終版 ElasticNet + 3× 加權...")
train_wrc = panel_wrc[panel_wrc["target_season"].isin([2021, 2022, 2023])].copy()
test_wrc  = panel_wrc[panel_wrc["target_season"] == 2024].copy()

x_tr_c, y_tr_c = _prep_wrc_xy(train_wrc)
x_te_c, y_te_c = _prep_wrc_xy(test_wrc)

CAT = ["primary_pos"]

def _one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_en_pipe():
    pre = ColumnTransformer([
        ("num", StandardScaler(), WRC_NUM),
        ("cat", _one_hot(), CAT),
    ])
    return Pipeline([
        ("prep",  pre),
        ("model", ElasticNet(alpha=0.03, l1_ratio=0.3, max_iter=5000)),
    ])

sw = np.ones(len(y_tr_c))
sw[(y_tr_c < 80) | (y_tr_c > 130)] = 3.0

en_final = build_en_pipe()
en_final.fit(x_tr_c, y_tr_c, model__sample_weight=sw)
pred_c   = en_final.predict(x_te_c)
mae_c    = mean_absolute_error(y_te_c, pred_c)
r2_c     = r2_score(y_te_c, pred_c)
tier_c   = tier_mae(y_te_c, pred_c)
print(f"  MAE={mae_c:.2f}  R²={r2_c:.3f}")

# ── D. TimeSeriesSplit CV ElasticNet ──────────────────────────────────────────
print("Training D: TimeSeriesSplit CV ElasticNet...")
train_ts = panel_wrc[panel_wrc["target_season"].isin([2021, 2022, 2023])].sort_values("target_season").reset_index(drop=True)
X_tr_d   = train_ts[LAG_COLS].values
y_tr_d   = train_ts[TARGET].values
X_te_d   = test_wrc[LAG_COLS].values
y_te_d   = test_wrc[TARGET].values

fold1_tr = train_ts[train_ts["target_season"] == 2021].index.tolist()
fold1_va = train_ts[train_ts["target_season"] == 2022].index.tolist()
fold2_tr = train_ts[train_ts["target_season"].isin([2021, 2022])].index.tolist()
fold2_va = train_ts[train_ts["target_season"] == 2023].index.tolist()

gs = GridSearchCV(
    Pipeline([("scaler", StandardScaler()), ("en", ElasticNet(max_iter=5000))]),
    {"en__alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
     "en__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    cv=[(fold1_tr, fold1_va), (fold2_tr, fold2_va)],
    scoring="neg_mean_absolute_error",
    refit=False,
    n_jobs=-1,
)
gs.fit(X_tr_d, y_tr_d)
best = gs.best_params_
print(f"  Best params: {best}")

pipe_d = Pipeline([("scaler", StandardScaler()), ("en", ElasticNet(
    alpha=best["en__alpha"], l1_ratio=best["en__l1_ratio"], max_iter=5000))])
pipe_d.fit(X_tr_d, y_tr_d)
pred_d = pipe_d.predict(X_te_d)
mae_d  = mean_absolute_error(y_te_d, pred_d)
r2_d   = r2_score(y_te_d, pred_d)
tier_d = tier_mae(y_te_d, pred_d)
print(f"  MAE={mae_d:.2f}  R²={r2_d:.3f}")

# ── 彙整結果 ─────────────────────────────────────────────────────────────────
maes  = [mae_a,  mae_b,  mae_c,  mae_d]
r2s   = [r2_a,   r2_b,   r2_c,   r2_d]
tiers = [tier_a, tier_b, tier_c, tier_d]

print("\n" + "=" * 72)
print(f"{'方法':<28} {'MAE':>7} {'R²':>8}  {'<80':>7} {'80-100':>8} {'100-130':>9} {'130+':>7}")
print("-" * 72)
for name, m, r, t in zip(METHOD_NAMES, maes, r2s, tiers):
    label = name.replace("\n", " ")
    print(f"{label:<28} {m:>7.2f} {r:>8.3f}  {t[0]:>7.1f} {t[1]:>8.1f} {t[2]:>9.1f} {t[3]:>7.1f}")

# ── 圖 1：整體 MAE + 分層 MAE ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: 整體 MAE
ax = axes[0]
bars = ax.bar(range(4), maes, color=COLORS, width=0.5, edgecolor="white", alpha=0.9)
for bar, v in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
            f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(4))
ax.set_xticklabels(METHOD_NAMES, fontsize=9)
ax.set_ylabel("MAE（wRC+ 點數）", fontsize=11)
ax.set_title("Test 2024 整體 MAE", fontsize=12)
ax.set_ylim(0, max(maes) * 1.25)
ax.axhline(min(maes), color="gray", lw=0.8, ls="--", alpha=0.5)

# Right: 分層 MAE
ax = axes[1]
x = np.arange(4)
width = 0.2
for i, (name, t, c) in enumerate(zip(METHOD_NAMES, tiers, COLORS)):
    offset = (i - 1.5) * width
    bars2 = ax.bar(x + offset, t, width, color=c, label=name.replace("\n", " "),
                   edgecolor="white", alpha=0.9)
    for bar, v in zip(bars2, t):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15,
                f"{v:.0f}", ha="center", fontsize=6.5)
ax.set_xticks(x)
ax.set_xticklabels(TIER_LABELS, fontsize=9.5)
ax.set_ylabel("MAE（wRC+ 點數）", fontsize=11)
ax.set_title("Test 2024 分層 MAE（依 wRC+ 水準）", fontsize=12)
ax.legend(fontsize=7.5, loc="upper right")
ax.set_ylim(0, max(t.max() for t in tiers) * 1.35)

plt.suptitle("四種方法完整比較（Test 2024 n≈256）", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(FIG / "full_comparison_mae.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved full_comparison_mae.png")

# ── 圖 2：四模型 2×2 散點圖 ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
preds     = [pred_a, pred_b, pred_c, pred_d]
y_actuals = [yte_bb, yte_bb, y_te_c, y_te_d]

for ax, name, c, y_act, y_pred_val, m, r in zip(
        axes, METHOD_NAMES, COLORS, y_actuals, preds, maes, r2s):
    ax.scatter(y_act, y_pred_val, alpha=0.5, s=35, color=c,
               edgecolors="white", linewidths=0.3)
    lo = max(20,  min(float(y_act.min()), float(y_pred_val.min())) - 10)
    hi = min(230, max(float(y_act.max()), float(y_pred_val.max())) + 10)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
    ax.set_xlabel("實際 wRC+（2024）", fontsize=9.5)
    ax.set_ylabel("預測 wRC+", fontsize=9.5)
    ax.set_title(f"{name.replace(chr(10), ' ')}\nMAE={m:.1f}  R²={r:.3f}", fontsize=9.5)

plt.suptitle("四種方法 Actual vs Predicted（Test 2024）", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(FIG / "full_comparison_scatter.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved full_comparison_scatter.png")

# ── 圖 3：R² 比較 ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(range(4), r2s, color=COLORS, width=0.5, edgecolor="white", alpha=0.9)
for bar, v in zip(bars, r2s):
    ypos = v + 0.003 if v >= 0 else v - 0.01
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(4))
ax.set_xticklabels(METHOD_NAMES, fontsize=9)
ax.set_ylabel("R²", fontsize=11)
ax.set_title("Test 2024 R²（越高越好）", fontsize=12)
ax.axhline(0, color="gray", lw=1, ls="-", alpha=0.5)
plt.tight_layout()
fig.savefig(FIG / "full_comparison_r2.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved full_comparison_r2.png")

print(f"\nDone. 圖片 → {FIG}")
