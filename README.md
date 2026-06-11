# MLB 打者生涯預測分析
## Player Career Projection via Statcast (2018–2024)

**研究問題：** 能否僅用球員過去三個球季的打擊過程指標，預測其下一季 wRC+？進一步：改用歷年 wRC+ 軌跡作為特徵，能否顯著改善預測效果？

**核心設計：** 完全以 Baseball Savant Statcast 逐球事件為唯一資料來源，自行計算所有比率指標（K%、BB%、BABIP、BIP%、wRC+），不依賴 FanGraphs，確保可重現性。

---

## Contributors

| 組員 | 系級 | 學號 | 工作分配 |
|------|------|------|----------|
| 吳帛恩 | 統計三 | 112207433 | Statcast 資料 pipeline、EDA、Random Forest wRC+ 預測、Statcast 熱區視覺化 |
| 彭珮蓉 | 資計碩一 | 114753210 | 訓練資料比較（2yr vs 3yr）、特徵哲學比較（過程指標 vs wRC+ 軌跡）、ElasticNet / LSTM 建模、多模型系統比較、極端球員加權改善 |

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

#### 同學 A — Random Forest 基準模型

依序執行 notebooks：

```bash
cd code
jupyter notebook
```

| 步驟 | Notebook | 說明 |
|------|----------|------|
| 1 | `notebooks/01_data_collection.ipynb` | 建立逐季打者面板 + lag 特徵面板 |
| 2 | `notebooks/02_eda.ipynb` | 探索性分析（分布、相關係數） |
| 3 | `notebooks/03_analysis.ipynb` | **主要結果**：RF 預測、特徵重要性 |
| 4 | `notebooks/04_visualization.ipynb` | 輔助視覺化（Statcast 熱區圖） |

或以 CLI 一次重建所有 pipeline 輸出：

```bash
cd code
python scripts/build_pipeline_outputs.py
```

#### 我的部分 — 模型比較與改進

```bash
# 前提：先確認 data/processed/batting_clean.csv 存在

# 1. 訓練年數比較（2yr vs 3yr，ElasticNet 與 RF）
python code/scripts/ours/run_elasticnet_train3yr.py

# 2. 多模型特徵比較（過程指標 vs wRC+ 軌跡）
python code/scripts/ours/run_evaluation.py

# 3. 極端球員加權改善
python code/scripts/ours/run_extreme_player_methods.py

# 4. 最終模型整體比較（含 RF / ElasticNet / LSTM）
python code/scripts/ours/run_final_comparison.py

# 5. LSTM（RNN）模型訓練與比較
python code/scripts/ours/train_rnn.py
python code/scripts/ours/run_comparison.py

# 6. Walk-Forward 驗證
python code/scripts/ours/run_timeseries_cv.py
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
| `processed/batting_clean.csv` | 清理後的打者面板（2,718 列，我的模型訓練用）| ~1 MB | ✓ |
| `processed/eval_models_comparison.csv` | 各模型（過程指標 vs wRC+ 軌跡）Test 2024 比較 | 小 | ✓ |
| `processed/eval_elasticnet_split_summary.csv` | 2yr vs 3yr vs Walk-forward 切割策略比較 | 3 列 | ✓ |
| `processed/eval_elasticnet_walkforward.csv` | Walk-forward 逐 fold 結果 | 3 列 | ✓ |
| `processed/final_comparison_summary.csv` | 最終模型比較摘要 | 小 | ✓ |
| `processed/all_models_comparison.csv` | 含 LSTM 的全模型 Test 2024 MAE/R² | 5 列 | ✓ |
| `processed/rf_3yr_overall_metrics.csv` | RF 3yr 整體指標 | 1 列 | ✓ |
| `processed/rf_3yr_tier_metrics.csv` | RF 3yr 分層（弱打 / 中下 / 中上 / 明星）誤差 | 小 | ✓ |
| `processed/rnn_metrics.json` | LSTM 超參數與 Test 2024 指標 | 極小 | ✓ |
| `raw/statcast_20XX.parquet` | 原始逐球事件（各季 ~150 MB） | ~1 GB+ | ✗ 太大 |

**資料來源：**
- [Baseball Savant Statcast](https://baseballsavant.mlb.com/) via `pybaseball.statcast()`
- [Lahman Database](http://www.seanlahman.com/) via `pybaseball.lahman.fielding()`

### `code/`

```
code/
├── bb_pipeline/
│   ├── statcast_season.py      # 逐球事件 → 逐季打者面板（同學 A）
│   ├── projection_dataset.py   # 建立 lag 特徵面板（同學 A）
│   ├── train_projection.py     # RF 訓練流程（同學 A）
│   ├── eval_baselines.py       # 基準模型、walk-forward、分層誤差（我）
│   └── eval_models.py          # 多模型監督式比較（我）
├── notebooks/
│   ├── 01_data_collection.ipynb    # 同學 A
│   ├── 02_eda.ipynb                # 同學 A
│   ├── 03_analysis.ipynb           # 同學 A（主要結果）
│   └── 04_visualization.ipynb     # 同學 A
├── scripts/
│   ├── build_pipeline_outputs.py       # CLI 重建面板（同學 A）
│   ├── create_presentation_figures.py  # 海報圖生成（同學 A）
│   └── ours/                           # 我的分析腳本
│       ├── run_evaluation.py           # 多模型特徵比較主程式
│       ├── run_elasticnet_train3yr.py  # 2yr vs 3yr 訓練比較
│       ├── run_extreme_player_methods.py # 極端球員加權改善
│       ├── run_final_comparison.py     # 最終模型整體比較
│       ├── run_improvement_comparison.py
│       ├── compare_splits.py           # 切割策略比較
│       ├── run_timeseries_cv.py        # TimeSeriesSplit CV
│       ├── run_full_comparison.py
│       ├── train_rnn.py                # LSTM 訓練
│       ├── rnn_model.py                # LSTM 模型定義
│       ├── rnn_data.py                 # LSTM 資料前處理
│       ├── run_comparison.py           # LSTM vs 其他模型比較
│       └── generate_figures.py         # RF 3yr 比較圖生成
└── requirements.txt
```

**同學 A 的方法（Random Forest，過程指標）：**
- 模型：Random Forest Regressor（`n_estimators=400`, `max_depth=12`, `min_samples_leaf=8`）
- 特徵：前三季 lag 特徵（K%、BB%、BABIP、BIP%、PA × lag1/2/3）+ 年齡 + 守備位置
- Null model：均值預測（predict league-mean wRC+ for all players）

**我的方法（ElasticNet / LSTM，wRC+ 歷年軌跡）：**
- 核心洞見：改用歷年 wRC+（wRC+_lag1/2/3）取代過程指標，資訊密度更高
- ElasticNet（L1+L2 正則化）：學習出近期 wRC+ 遞減加權，類似 Marcel 投影法
- LSTM（hidden=64, 1 層, dropout=0.3）：捕捉時序動態，但受限於小樣本
- 極端球員加權：對弱打（wRC+ < 80）與超級球星（wRC+ > 130）施加 3× 樣本權重

**訓練 / 驗證 / 測試切分（時間切分，避免 data leakage）：**

| 集合 | 目標球季 | 樣本數 |
|------|----------|--------|
| Train（同學 A） | 2021–22 | 453 |
| Train（我，擴展版） | 2021–23 | 687 |
| Validation | 2023 | 234 |
| Test（hold-out） | 2024 | 256 |

### `results/`

| 路徑 | 說明 |
|------|------|
| `figures/poster/` | 海報用高解析圖（fig_pipeline, fig_timesplit, fig_perm_importance, fig_actual_vs_pred, fig_radar_stars）— 同學 A |
| `figures/eda/` | EDA 圖（相關係數熱圖、分布圖、WAR vs wRC+ 等）— 同學 A |
| `figures/model_rf/` | RF 3yr 預測散點圖、分層誤差、特徵重要性 — 我（重新執行同學 A 的模型於 3yr 設定）|
| `figures/model_ours/` | ElasticNet 係數圖、分層誤差、最終模型比較、2yr vs 3yr 比較 — 我 |
| `figures/model_ours/rnn/` | LSTM 預測散點圖、與所有模型 MAE/R² 比較 — 我 |
| `figures/walkforward/` | ElasticNet Walk-forward 逐 fold MAE — 我 |

---

## 主要結果

**Hold-out Test 2024 結果比較：**

| 模型 | 特徵 | Train | MAE | R² |
|------|------|-------|-----|-----|
| Null（猜均值） | — | — | 28.15 | −0.044 |
| Random Forest（同學 A） | 過程指標 | 2yr | 28.24 | −0.048 |
| Random Forest | 過程指標 | 3yr | 26.89 | 0.019 |
| LSTM（RNN） | wRC+ 軌跡 | 3yr | 25.43 | 0.121 |
| ElasticNet（基準） | wRC+ 軌跡 | 3yr | 24.35 | 0.179 |
| **ElasticNet + 3× 加權** | **wRC+ 軌跡** | **3yr** | **24.63** | **0.197** |

**核心發現：**
- 特徵選擇比模型選擇更重要：同一個 ElasticNet，換用 wRC+ 軌跡特徵，MAE 從 28.9 降至 25.1（−3.8 點）
- 訓練資料越多越有效：2yr → 3yr 使 ElasticNet MAE 從 25.14 → 24.35，RF 從 28.24 → 26.89
- 均值壓縮問題：所有模型對弱打者（wRC+ < 80）高估，超級球星低估；3× 加權使弱打者 MAE 從 37.2 → 33.2（改善 11%）

---

## 模型差異說明

### 為什麼 RF（過程指標）比猜均值還差？（R² = −0.048）

Random Forest 嘗試從 K%、BB%、BABIP、BIP% 等分散的過程指標中學習複雜的非線性交互關係。然而在僅 453 筆（2yr）的小訓練集上，模型容易過擬合到雜訊，學到的規律無法泛化到 Test 2024。結果 R² 為負值，代表連「永遠猜聯盟平均值（wRC+ ≈ 100）」都不如。

### 為什麼 wRC+ 歷史軌跡特徵更有效？

wRC+ 是一個**綜合指標**，已將球場效應、選球能力（BB%）、長打率（SLG）、接觸率等所有打擊面向整合成單一數字。直接使用歷年 wRC+（lag1/lag2/lag3）作為特徵，等於把職業分析師花幾十年設計的「打者能力摘要」直接餵給模型，省去模型自己從零學習如何組合原始指標的過程。

此外，wRC+ 的年度間持續性（「今年強的球員明年也傾向強」）在數值上呈現近似線性關係，這使得 ElasticNet 這類線性模型能夠有效捕捉，而不需要複雜的非線性模型。ElasticNet 學到的係數（S-2: 6.39、S-1: 5.99、S-3: 5.51）顯示三個 lag 的權重均顯著為正且相近，符合 Marcel 投影法（業界標準的打者投影方法）的加權遞減精神。

### 為什麼 ElasticNet 勝過 LSTM？

LSTM 理論上能捕捉更複雜的時序動態，但此問題的資料規模（3 步序列 × 687 訓練樣本）對深度學習而言過小，難以充分訓練。加上 wRC+ 的年際關係本質上接近線性，LSTM 的複雜度並未帶來額外優勢，反而受限於過少的樣本。

ElasticNet 的 L1+L2 正則化在小資料集上提供了良好的偏差-方差權衡，最終 MAE（24.35）優於 LSTM（25.43）。

### 為什麼 3yr 訓練優於 2yr？

訓練樣本從 453 筆（2yr）擴增至 687 筆（3yr，+51%），模型對球員能力分布的估計更為穩定，Test 2024 MAE 隨之改善：

| 模型 | 2yr MAE | 3yr MAE | 改善 |
|------|---------|---------|------|
| RF | 28.24 | 26.89 | −1.35 |
| ElasticNet | 25.14 | 24.35 | −0.79 |

Walk-forward 驗證（expanding window）的平均 MAE（27.52）反而高於 fixed 3yr split（24.35），原因是早期 fold 僅有 217 筆訓練樣本，拉高了整體平均；當訓練資料累積到 687 筆時，walk-forward 的單 fold MAE 同樣降至 24.35，確認資料量是主要驅動因素。

### 為什麼極端球員（弱打者 / 超級球星）誤差特別大？

所有模型均存在**均值壓縮效應（mean compression）**：預測值趨向聯盟平均（wRC+ ≈ 100），導致弱打者（真實 wRC+ ≈ 55）被高估、超級球星（真實 wRC+ ≈ 145）被低估。這是因為訓練集中極端球員樣本稀少，模型傾向於學習主流分布的規律。

解決方案：對弱打者（wRC+ < 80）與超級球星（wRC+ > 130）施加 **3 倍樣本權重**，強迫模型更關注極端值。結果弱打者 MAE 從 37.2 → 33.2（−11%），整體 MAE 僅上升 0.28，是一個低成本、高效益的改善策略。

---

## References

- [pybaseball](https://github.com/jldbc/pybaseball) — Statcast & Lahman 資料存取
- [Baseball Savant](https://baseballsavant.mlb.com/statcast_search) — Statcast 資料來源
- [scikit-learn](https://scikit-learn.org/) — RandomForestRegressor, ElasticNet, permutation_importance
- [PyTorch](https://pytorch.org/) — LSTM 實作
- [Lahman Database](http://www.seanlahman.com/) — 球員守備位置
- Albert, J. & Marchi, M. (2019). *Analyzing Baseball Data with R*, 2nd ed.
- Baumer, B. et al. (2021). *Modern Data Science with R*
- Tango, T. (2004). Marcel the Monkey Forecasting System — wRC+ 歷史加權投影法參考
