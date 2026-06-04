# MLB 打者生涯預測分析
## Player Career Projection via Statcast (2018–2024)

**研究問題：** 能否僅用球員過去三個球季的打擊過程指標，預測其下一季 wRC+？

**核心設計：** 完全以 Baseball Savant Statcast 逐球事件為唯一資料來源，自行計算所有比率指標（K%、BB%、BABIP、BIP%、wRC+），不依賴 FanGraphs，確保可重現性。

---

## Contributors

| 組員 | 系級 | 學號 | 工作分配 |
|------|------|------|----------|
| 吳帛恩 | 統計三 | 112207433 | Statcast 資料管線、EDA、K-Means 分群、Random Forest wRC+ 預測、Statcast 熱區視覺化 |
| （待補） | — | — | — |

---

## Quick Start

### 1. 建立環境

```bash
conda create -n baseball python=3.10
conda activate baseball
pip install -r code/requirements.txt
```

> **macOS + XGBoost：** 若出現 `libomp.dylib` 錯誤，執行 `brew install libomp` 後重裝 xgboost。Random Forest 不受影響。

### 2. 資料準備

Raw Statcast 資料每季約 150 MB，共 7 季，體積過大不上傳。有兩種方式取得：

**方式 A — 使用已整理的 panel（推薦，直接跳至步驟 3）**

`data/raw/batting_by_season_2018_2024.parquet` 已提交，可直接使用。

**方式 B — 從頭下載原始資料（約需 30–60 分鐘）**

```python
# 於 code/notebooks/01_data_collection.ipynb 執行
from pybaseball import cache, statcast
cache.enable()  # 啟用本地快取
# 依年份下載 2018–2024
```

### 3. 執行順序

依序執行 notebooks：

```bash
cd code
jupyter notebook
```

| 步驟 | Notebook | 說明 |
|------|----------|------|
| 1 | `notebooks/01_data_collection.ipynb` | 建立逐季打者面板 + lag 特徵面板 |
| 2 | `notebooks/02_eda.ipynb` | 探索性分析（分布、相關係數） |
| 3 | `notebooks/03_analysis.ipynb` | **主要結果**：RF 預測、特徵重要性、K-Means 分群 |
| 4 | `notebooks/04_visualization.ipynb` | 輔助視覺化（Statcast 熱區圖） |

或以 CLI 一次重建所有 pipeline 輸出：

```bash
cd code
python scripts/build_pipeline_outputs.py
```

---

## Folder Organization

### `docs/`
- `final_report.md` — 中文期末報告（含圖表說明與結論）
- `methodology.md` — 資料處理、模型設計、圖表解讀（章節版）

### `data/`

| 路徑 | 說明 | 大小 | 是否上傳 |
|------|------|------|----------|
| `raw/batting_by_season_2018_2024.parquet` | 逐季打者面板（PA ≥ 150，958 球員-球季） | ~500 KB | ✓ |
| `processed/projection_panel.parquet` | lag 特徵面板（943 列，預測用） | ~200 KB | ✓ |
| `raw/statcast_20XX.parquet` | 原始逐球事件（各季 ~150 MB） | ~1 GB+ | ✗ 太大 |

**資料來源：**
- [Baseball Savant Statcast](https://baseballsavant.mlb.com/) via `pybaseball.statcast()`
- [Lahman Database](http://www.seanlahman.com/) via `pybaseball.lahman.fielding()`

### `code/`

```
code/
├── bb_pipeline/
│   ├── statcast_season.py      # 逐球事件 → 逐季打者面板
│   └── projection_dataset.py   # 建立 lag 特徵面板
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_analysis.ipynb       # 主要模型與結果
│   └── 04_visualization.ipynb
├── scripts/
│   ├── build_pipeline_outputs.py   # CLI 重建面板
│   ├── create_presentation_figures.py
│   └── build_word_report.py
└── requirements.txt
```

**方法：**
- 模型：Random Forest Regressor（`n_estimators=400`, `max_depth=12`, `min_samples_leaf=8`）
- 特徵：前三季 lag 特徵（K%、BB%、BABIP、BIP%、PA × lag1/2/3）+ 年齡 + 守備位置
- Null model：均值預測（predict league-mean wRC+ for all players）

**訓練/驗證/測試切分（時間切分，避免 data leakage）：**

| 集合 | 目標球季 | 樣本數 |
|------|----------|--------|
| Train | 2021–22 | 453 |
| Validation | 2023 | 234 |
| Test（hold-out） | 2024 | 256 |

### `results/`

| 路徑 | 說明 |
|------|------|
| `figures/poster/` | 海報用高解析圖（fig_pipeline, fig_timesplit, fig_perm_importance, fig_actual_vs_pred, fig_radar_stars） |
| `figures/*.png` | EDA 與模型輔助圖（相關係數熱圖、分布圖、WAR vs wRC+、熱區圖等） |

**Hold-out Test 2024 結果：**

| 指標 | 數值 | 說明 |
|------|------|------|
| MAE | **22.1 wRC+** | 接近 wRC+ 年際自然波動上限（15–25 點） |
| R² | **−0.018** | RF 均值壓縮效應主導，弱打者高估、超級球星低估 |

---

## References

- [pybaseball](https://github.com/jldbc/pybaseball) — Statcast & Lahman 資料存取
- [Baseball Savant](https://baseballsavant.mlb.com/statcast_search) — Statcast 資料來源
- [scikit-learn](https://scikit-learn.org/) — RandomForestRegressor, permutation_importance
- [Lahman Database](http://www.seanlahman.com/) — 球員守備位置
- Albert, J. & Marchi, M. (2019). *Analyzing Baseball Data with R*, 2nd ed.
- Baumer, B. et al. (2021). *Modern Data Science with R*
