"""
create_presentation_figures.py
執行方式：從 baseball_project/ 目錄執行
    python scripts/create_presentation_figures.py
輸出至 figures/poster/（dpi=300）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family": ["Arial Unicode MS", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

PROC_DIR = ROOT / "data" / "processed"
FIG_DIR  = ROOT / "figures" / "poster"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "train": "#457B9D",
    "val":   "#2A9D8F",
    "test":  "#E63946",
    "gray":  "#AAAAAA",
    "gold":  "#E9C46A",
}

FEAT_LABELS = {
    "num__PA_sum_lag3":  "三年 PA 總計",
    "num__BB%_lag3":     "四壞% (3yr前)",
    "num__K%_lag1":      "三振% (1yr前)",
    "num__PA_lag1":      "打席數 PA (1yr前)",
    "num__BIP%_lag1":    "場內球% (1yr前)",
    "num__age_t":        "年齡",
    "num__PA_lag2":      "打席數 PA (2yr前)",
    "num__BB%_lag2":     "四壞% (2yr前)",
    "num__BABIP_lag1":   "BABIP (1yr前)",
    "num__BIP%_lag3":    "場內球% (3yr前)",
    "num__BIP%_lag2":    "場內球% (2yr前)",
    "num__BB%_lag1":     "四壞% (1yr前)",
    "num__K%_lag2":      "三振% (2yr前)",
    "num__PA_lag3":      "打席數 PA (3yr前)",
    "num__K%_lag3":      "三振% (3yr前)",
    "num__BABIP_lag2":   "BABIP (2yr前)",
    "num__BABIP_lag3":   "BABIP (3yr前)",
    "cat__primary_pos_UNK": "守備位置 UNK",
}


def save(name: str):
    path = FIG_DIR / name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def make_pipeline_figure():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # 方塊定義：(x_center, y_center, width, height, label, sublabel, color)
    boxes = [
        (1.2, 2.0, 2.0, 1.2,
         "Statcast\n2018–2024",
         "逐球事件資料\n(pybaseball)", "#A8DADC"),
        (4.0, 2.0, 2.0, 1.2,
         "逐季打者面板",
         "K%, BB%, BABIP\nBIP%, wRC+, PA", "#457B9D"),
        (6.8, 2.0, 2.0, 1.2,
         "Lag 特徵工程",
         "S−1 / S−2 / S−3\n各指標 × 3 時間點", "#2A9D8F"),
        (9.6, 2.0, 2.0, 1.2,
         "時間切分",
         "Train: 2021–22\nVal: 2023 | Test: 2024", "#E9C46A"),
        (12.4, 2.0, 1.6, 1.2,
         "Random\nForest",
         "預測\n下季 wRC+", "#E63946"),
    ]

    for (xc, yc, w, h, label, sub, color) in boxes:
        rect = mpatches.FancyBboxPatch(
            (xc - w/2, yc - h/2), w, h,
            boxstyle="round,pad=0.08",
            linewidth=1.5, edgecolor="white",
            facecolor=color, zorder=3, alpha=0.92
        )
        ax.add_patch(rect)
        ax.text(xc, yc + 0.18, label,
                ha="center", va="center", fontsize=10,
                fontweight="bold", color="white", zorder=4)
        ax.text(xc, yc - 0.28, sub,
                ha="center", va="center", fontsize=7.5,
                color="white", alpha=0.9, zorder=4)

    # 箭頭
    arrow_xs = [(2.2, 2.9), (5.0, 5.8), (7.8, 8.6), (10.6, 11.4)]
    for (x0, x1) in arrow_xs:
        ax.annotate("", xy=(x1, 2.0), xytext=(x0, 2.0),
                    arrowprops=dict(arrowstyle="->", color="#555555",
                                   lw=2.0), zorder=5)

    ax.set_title("Player Career Projection — 資料管線", fontsize=14,
                 fontweight="bold", pad=10)
    save("fig_pipeline.png")


def make_timesplit_figure():
    panel = pd.read_parquet(PROC_DIR / "projection_panel.parquet")
    counts = panel.groupby("target_season").size()

    # 每個 target_season 的設定：(role, color, label, n)
    targets = [
        (2021, "train", PALETTE["train"], "Train",      counts[2021]),
        (2022, "train", PALETTE["train"], "Train",      counts[2022]),
        (2023, "val",   PALETTE["val"],   "Validation", counts[2023]),
        (2024, "test",  PALETTE["test"],  "Test",       counts[2024]),
    ]
    # y 位置：由上到下（2021 在最上）
    y_positions = {2021: 3.5, 2022: 2.5, 2023: 1.5, 2024: 0.5}
    row_h = 0.72   # 每列色塊高度

    all_seasons = list(range(2018, 2025))

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(2017.8, 2026.0)
    ax.set_ylim(0.0, 4.5)
    ax.set_yticks([])
    ax.set_xticks(all_seasons + [2025])
    ax.set_xlabel("球季", fontsize=11)
    ax.set_title("時間切分策略（避免未來資料洩漏）\n"
                 "每列：一個目標球季；左側色塊 = 特徵年（lag3/lag2/lag1）；深色 = 預測目標年",
                 fontsize=11, fontweight="bold")

    # 垂直格線
    for s in all_seasons:
        ax.axvline(s, color="#E0E0E0", lw=0.8, zorder=0)

    for (ts, role, color, role_lbl, n) in targets:
        yc = y_positions[ts]
        lag_years = [ts - 3, ts - 2, ts - 1]
        lag_labels = ["lag3", "lag2", "lag1"]

        # 特徵年色塊（淺色）—— 從 year 到 year+1，對齊刻度線
        for lag_s, lag_lbl in zip(lag_years, lag_labels):
            rect = mpatches.Rectangle(
                (lag_s, yc - row_h / 2), 1.0, row_h,
                facecolor=color, alpha=0.22,
                edgecolor=color, linewidth=1.0, zorder=2
            )
            ax.add_patch(rect)
            mid = lag_s + 0.5   # 色塊正中央
            ax.text(mid, yc + 0.06, lag_lbl,
                    ha="center", va="center", fontsize=8,
                    color=color, alpha=0.9, zorder=3)
            ax.text(mid, yc - 0.20, str(lag_s),
                    ha="center", va="center", fontsize=7.5,
                    color=color, alpha=0.75, zorder=3)

        # 目標年色塊（深色）—— 從 ts 到 ts+1
        rect_t = mpatches.Rectangle(
            (ts, yc - row_h / 2), 1.0, row_h,
            facecolor=color, alpha=0.85,
            edgecolor="white", linewidth=1.5, zorder=2
        )
        ax.add_patch(rect_t)
        ax.text(ts + 0.5, yc + 0.08, f"預測 {ts}",
                ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold", zorder=3)
        ax.text(ts + 0.5, yc - 0.18, f"n={n}",
                ha="center", va="center", fontsize=7.5,
                color="white", alpha=0.9, zorder=3)

        # 箭頭：lag3 起點 → 目標年左緣
        ax.annotate("",
            xy=(ts, yc),
            xytext=(ts - 3 + 0.02, yc),
            arrowprops=dict(arrowstyle="->", color=color,
                            lw=1.3, alpha=0.6),
            zorder=1)

        # 左側 role 標籤
        ax.text(2017.7, yc, f"{role_lbl}\n→{ts}",
                ha="left", va="center", fontsize=9,
                color=color, fontweight="bold")

    # 圖例
    legend_patches = [
        mpatches.Patch(color=PALETTE["train"], label="Train（目標年 2021–22）"),
        mpatches.Patch(color=PALETTE["val"],   label="Validation（目標年 2023）"),
        mpatches.Patch(color=PALETTE["test"],  label="Test（目標年 2024）"),
    ]
    ax.legend(handles=legend_patches, fontsize=9,
              loc="upper left", bbox_to_anchor=(0.01, 0.52),
              framealpha=0.9)

    # 說明文字框（右上）
    ax.text(2024.95, 4.35,
            "深色塊 = 目標年（預測標籤 wRC+）\n淺色塊 = 特徵年（lag1/2/3 輸入）\n→ 特徵年資料流向目標年",
            ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#F8F8F8", edgecolor="#CCCCCC"),
            color="#444444")

    plt.tight_layout()
    save("fig_timesplit.png")


def make_radar_stars():
    df_clean  = pd.read_csv(PROC_DIR / "batting_clean.csv")
    df_scaled = pd.read_csv(PROC_DIR / "batting_scaled.csv")

    # contact_rate 若不在 batting_clean，從 K% 計算
    if "contact_rate" not in df_clean.columns:
        df_clean["contact_rate"] = 1 - df_clean["K%"]

    # 資料中球員名稱為 Unknown_<batter_id>，用 batter ID 映射真實姓名
    BATTER_NAME_MAP = {
        592450: "Aaron Judge",
        665742: "Juan Soto",
        660271: "Shohei Ohtani",
        605141: "Mookie Betts",
        518692: "Freddie Freeman",
        545361: "Mike Trout",
        658668: "Yordan Alvarez",
        547180: "Paul Goldschmidt",
        592885: "Kris Bryant",
        663886: "Trea Turner",
    }

    # 優先選取明星球員（按 batter ID）
    preferred_ids = [592450, 665742, 660271, 605141, 518692]
    selected_ids = []
    for bid in preferred_ids:
        mask = df_clean["batter"] == bid
        if mask.any():
            selected_ids.append(bid)

    # 若有缺漏，補入高 wRC+ 球員
    if len(selected_ids) < 5:
        top_by_wrc = (
            df_clean.groupby("batter")["wRC+"]
            .max()
            .reset_index()
            .nlargest(20, "wRC+")["batter"]
            .tolist()
        )
        for bid in top_by_wrc:
            if bid not in selected_ids:
                selected_ids.append(bid)
            if len(selected_ids) == 5:
                break

    # 建立顯示名稱映射
    name_map = {}
    for bid in selected_ids:
        rows = df_clean[df_clean["batter"] == bid]
        raw_name = rows["Name"].iloc[0] if not rows.empty else f"Unknown_{bid}"
        display_name = BATTER_NAME_MAP.get(bid, raw_name)
        name_map[bid] = display_name

    radar_features = ["AVG", "OBP", "SLG", "BB%", "contact_rate", "ISO"]
    # 確認欄位都在 df_scaled
    radar_features = [f for f in radar_features if f in df_scaled.columns]

    categories_map = {
        "AVG": "打擊率\nAVG",
        "OBP": "上壘率\nOBP",
        "SLG": "長打率\nSLG",
        "BB%": "四壞%\nBB%",
        "contact_rate": "接觸率\nContact",
        "ISO": "孤立長打\nISO",
    }
    categories = [categories_map.get(f, f) for f in radar_features]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#9B5DE5"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, bid in enumerate(selected_ids):
        display_name = name_map[bid]
        # 取該球員所有球季的最大值（代表巔峰能力）
        rows_scaled = df_scaled[df_scaled["batter"] == bid]
        if rows_scaled.empty:
            continue
        vals = rows_scaled[radar_features].max().tolist()
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2,
                color=colors[i], label=display_name)
        ax.fill(angles, vals, alpha=0.08, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"],
                       fontsize=7, color="gray")
    ax.set_title("明星打者能力雷達圖\n（各指標標準化至 0–1，2018–2024）",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=10)

    plt.tight_layout()
    save("fig_radar_stars.png")


def make_perm_importance():
    from bb_pipeline.train_projection import fit_by_target_season

    panel = pd.read_parquet(PROC_DIR / "projection_panel.parquet")
    results = fit_by_target_season(panel)

    imp = pd.Series(results["rf_perm_importance_mean"]).sort_values(ascending=False)
    # 保留前 14 個，移除極度負值（表示該特徵無用）
    imp = imp[imp > -0.001].head(14)

    labels = [FEAT_LABELS.get(k, k.replace("num__", "").replace("cat__", ""))
              for k in imp.index]
    threshold = imp.quantile(0.70)
    colors = [PALETTE["test"] if v >= threshold else PALETTE["train"]
              for v in imp.values]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(imp)), imp.values[::-1],
            color=colors[::-1], edgecolor="white", height=0.7)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(labels[::-1], fontsize=10)
    ax.set_xlabel("Permutation Importance（Validation 2023）", fontsize=11)
    ax.set_title("Random Forest：哪些特徵最能預測下季 wRC+？\n"
                 "（打亂特徵後驗證誤差上升量）",
                 fontsize=12, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=PALETTE["test"],  label="關鍵特徵（Top 30%）"),
        mpatches.Patch(color=PALETTE["train"], label="次要特徵"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    ax.axvline(0, color="black", lw=0.8)

    plt.tight_layout()
    save("fig_perm_importance.png")


def make_actual_vs_pred():
    from bb_pipeline.train_projection import fit_by_target_season, prepare_xy
    from sklearn.metrics import mean_absolute_error

    # 與 make_radar_stars 相同的名稱映射字典
    BATTER_NAME_MAP = {
        592450: "Aaron Judge",
        665742: "Juan Soto",
        660271: "Shohei Ohtani",
        605141: "Mookie Betts",
        518692: "Freddie Freeman",
        545361: "Mike Trout",
        658668: "Yordan Alvarez",
        547180: "Paul Goldschmidt",
        592885: "Kris Bryant",
        663886: "Trea Turner",
    }

    panel = pd.read_parquet(PROC_DIR / "projection_panel.parquet")
    results = fit_by_target_season(panel)
    rf = results["rf_model"]

    te = panel[panel["target_season"] == 2024].copy()

    # 用 batter ID 映射真實姓名（Name 欄全為 Unknown_<id> 格式）
    te["Name"] = te.apply(
        lambda r: BATTER_NAME_MAP.get(r["batter"], r["Name"]),
        axis=1,
    )

    # 過濾異常值：wRC+ 應在合理範圍（過濾演算法異常值，不過濾 Unknown 球員）
    te = te[(te["wRCplus_target"] >= 20) & (te["wRCplus_target"] <= 220)]

    x_te, y_te = prepare_xy(te)
    pred = rf.predict(x_te)
    te = te.copy()
    te["pred"]    = pred
    te["err"]     = pred - y_te
    te["abs_err"] = te["err"].abs()

    mae = mean_absolute_error(y_te, pred)
    ss_res = ((y_te - pred) ** 2).sum()
    ss_tot = ((y_te - y_te.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    norm = plt.Normalize(0, te["abs_err"].quantile(0.95))
    cmap = plt.cm.RdYlGn_r

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(y_te, pred,
                    c=te["abs_err"], cmap=cmap, norm=norm,
                    alpha=0.65, s=45, edgecolors="white", linewidths=0.4)
    plt.colorbar(sc, ax=ax, label="|預測誤差| wRC+ 點數", shrink=0.8)

    lim_min = max(20, min(y_te.min(), pred.min()) - 5)
    lim_max = min(220, max(y_te.max(), pred.max()) + 5)
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            "k--", lw=1.2, label="完美預測線 y=x")

    # 標記誤差最大的 6 名球員（姓氏；Unknown 球員不標記）
    top_err = te.nlargest(6, "abs_err")
    for _, row in top_err.iterrows():
        raw_name = str(row["Name"])
        if raw_name.startswith("Unknown_"):
            continue
        name_disp = raw_name.split()[-1]
        ax.annotate(name_disp,
                    (row["wRCplus_target"], row["pred"]),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=7.5, color="#333333", alpha=0.85)

    # 標記知名球員（若存在於 te）
    known = ["Aaron Judge", "Juan Soto", "Shohei Ohtani"]
    for _, row in te[te["Name"].isin(known)].iterrows():
        ax.annotate(row["Name"].split()[-1],
                    (row["wRCplus_target"], row["pred"]),
                    textcoords="offset points", xytext=(6, -10),
                    fontsize=8, color=PALETTE["train"],
                    fontweight="bold", alpha=0.9)

    ax.set_xlabel("實際 wRC+（2024 球季）", fontsize=12)
    ax.set_ylabel("預測 wRC+（Random Forest）", fontsize=12)
    ax.set_title(
        f"Hold-out Test 2024  n={len(te)}\n"
        f"MAE = {mae:.1f} wRC+ 點  |  R² = {r2:.3f}",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    save("fig_actual_vs_pred.png")


if __name__ == "__main__":
    make_pipeline_figure()
    make_timesplit_figure()
    make_radar_stars()
    make_perm_importance()
    make_actual_vs_pred()
    print("Done. All figures in", FIG_DIR)
