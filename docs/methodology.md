# MLB 打者隔年 wRC+ 預測 — 方法論說明

本文件說明本專案之資料來源、處理流程、建模方式，以及如何解讀主要產出圖表。對應程式位於 `bb_pipeline/` 與 `notebooks/03_analysis.ipynb`（Task 7）。

---

## 1. 研究目的

利用球員**過去三個完整球季**的打擊過程指標與脈絡變數，以**監督式回歸**預測該球員在**目標球季**的 **wRC+**（標準化攻擊貢獻；聯盟平均約為 100）。目的為描述「生涯／走勢」式的隔年表現，並以**依目標球季切割**的 Train / Validation / Test 評估泛化能力，避免對同一面板做隨機抽樣而造成**時間上的資訊洩漏**。

---

## 2. 資料來源

| 類別 | 來源 | 用途 |
|------|------|------|
| 逐球／逐打席事件 | Baseball Savant **Statcast**（`pybaseball.statcast` 或本地 `data/raw/statcast_YYYY.parquet`） | **唯一**計算 **K%、BB%、BABIP、BIP%、PA、wOBA、逐季 wRC+** 之依據 |
| 時間範圍 | **2018–2024**（與 notebook 設定的下載日期窗一致） | 足以組出「前三年 → 目標年」之面板 |
| 守備位置（輔助） | Chadwick register + **Lahman** Fielding（`bb_pipeline/position_lahman.py`） | MLBAM 打者 ID → 該季 **primary position**；下載或對應失敗時為 **`UNK`** |
| 姓名／隊伍顯示 | `pybaseball.playerid_reverse_lookup`、Statcast `home_team` 之眾數 | 僅供呈現 |

本專案**不以** FanGraphs 等外部打擊排行榜表作為比率資料來源（與 Statcast-only 設計一致）。

---

## 3. 資料處理流程

### 3.1 逐季打者面板

**輸出檔：** `data/raw/batting_by_season_2018_2024.parquet`  
**程式：** `bb_pipeline/statcast_season.py` → `aggregate_batting_by_season()`

1. 保留 Statcast 中 **打席結束列**（`events` ∈ **PA_EVENTS**：安打、保送、三振、各類出局等）。
2. 依 **`(batter, game_year)`** 分組聚合：
   - **AVG、OBP、SLG、OPS、ISO、BABIP**；**BB%**、**K%** 由事件計數 ÷ **PA**。
   - **BIP%**：事件 ∈ **BIP_EVENTS**（場內擊球結束型態，如安打、野選出局、犧飞等；不含保送、三振、觸身、犧短等）之列數 ÷ **PA**。詳見程式內 `BIP_EVENTS` 集合。
   - **wOBA**：該季該打者 PA 列之 `sum(woba_value) / sum(woba_denom)`。
   - **wRC+**：以該季面板內全體打者之 **PA 加權 lg wOBA**，代入固定 **wOBA_scale = 1.18**、**lg_R_PA = 0.127** 之簡化公式（與原 notebook 一致）。
   - **age_bat_median**：該季該打者所有 PA 列 **`age_bat` 的中位數**。

### 3.2 預測用面板（lag + 標籤）

**輸出檔：** `data/processed/projection_panel.parquet`  
**程式：** `bb_pipeline/projection_dataset.py` → `build_projection_panel()`

對每位打者、每個**目標球季 S**（預設為資料允許之所有 S）：

| 項目 | 說明 |
|------|------|
| **標籤 `wRCplus_target`** | 球季 **S** 之 **wRC+**（來自 3.1） |
| **必要條件** | 同時存在 **S−3、S−2、S−1** 三季列，且該三季 **PA ≥ 100**（`min_pa_per_feature_season`） |
| **比率特徵（12 欄）** | **K%、BB%、BABIP、BIP%** × **lag3 / lag2 / lag1**（對應 **S−3、S−2、S−1**） |
| **PA** | `PA_lag3`、`PA_lag2`、`PA_lag1`、`PA_sum_lag3` |
| **`age_t`** | 取自 **S−1** 之 `age_bat_median` |
| **`primary_pos`** | 取自 **S−1**；Lahman 失敗則 **UNK** |

**範例：** 預測 **2021** 年 wRC+ 時，輸入為 **2018、2019、2020** 三季之上述變數；標籤為 **2021** 季之 wRC+。  
資料上最早可當目標年者為 **2021**（需 2018–2020）；最晚為 **2024**（需 2021–2023）。**2025** 無真實 wRC+ 時無法納入有監督測試。

---

## 4. 建模方法

### 4.1 問題形式

- **任務：** 多元回歸，輸出為連續變數 **wRC+**。
- **實作：** `sklearn.pipeline.Pipeline`：**數值特徵**直通；**`primary_pos`** 經 **OneHotEncoder**（`handle_unknown="ignore"`）。

### 4.2 模型

1. **RandomForestRegressor**（主線）  
   `n_estimators=400`，`max_depth=12`，`min_samples_leaf=8`，`random_state=42`，`n_jobs=-1`。

2. **XGBRegressor**（備線）  
   於 `bb_pipeline/train_projection.py` 中採延遲載入；若環境無法載入 XGBoost（例如 macOS 缺 **OpenMP / libomp**），則略過，僅使用 RF。

### 4.3 訓練／驗證／測試切分

**依 `target_season`（預測哪一年的 wRC+）切分**，**不**對列做隨機 shuffle。

**預設**（`fit_by_target_season()`）：

| 集合 | `target_season` |
|------|-----------------|
| Train | 2021、2022 |
| Validation | 2023 |
| Test | 2024 |

- 模型僅在 **Train** 上 **fit**。
- **Permutation importance** 於 **Validation（2023）** 上計算。
- **Test（2024）** 為最終樣外誤差。

輔助函數 **`walk_forward_metrics`** 提供逐年擴張訓練窗之額外檢查，與上表並列閱讀。

### 4.4 評估指標

- **MAE / RMSE**：wRC+ 點數誤差。
- **R²**：參考用；小樣本或弱訊號時可能為負。
- **Permutation importance**：打亂單一特徵後觀察驗證誤差變化，衡量 RF 對該特徵之依賴程度（非因果）。

---

## 5. 產出圖表解讀

### 5.1 `figures/projection_rf_perm_importance.png`

- **內容：** Random Forest 在 **validation（target_season = 2023）** 上之 **permutation importance**（通常顯示前若干名特徵）。
- **解讀：** 橫條越長 → 打亂該特徵後誤差上升越多 → 模型越依賴該輸入。常見重要項含 **K%、BABIP、PA、年齡** 等。若 **`primary_pos`** 多為 **UNK**，位置 one-hot 貢獻可能偏低。
- **注意：** 為相對重要性，不代表與 wRC+ 之單調因果關係。

### 5.2 `figures/projection_rf_actual_vs_pred_2024.png`

- **內容：** **Test**（**target_season = 2024**）：橫軸 **實際 wRC+**，縱軸 **預測 wRC+**；對角線為完美預測。
- **解讀：** 點愈靠近對角線愈準；整體偏移表示系統性高估或低估。若圖上標註 **MAE**，為 2024 測試集之平均絕對誤差。

### 5.3 其他圖（`figures/fig3`–`fig10` 等）

多數來自**輔助 EDA／視覺化**或**舊版「同年指標 → wRC+」**任務，與「**隔年 wRC+ + 三年 lag**」主線**問題定義不同**。撰寫報告時建議以 **§4–§5.2** 與 **Task 7** 為主分析；若引用 fig3–fig10，請註明為補充圖或舊任務，避免與隔年預測混淆。

---

## 6. 限制與結論撰寫建議

1. **可監督目標年**僅約 **2021–2024**，受 Statcast 起訖年與連續三年 lag 限制；無法對 **2025 wRC+** 做有標籤之 test。
2. **wRC+** 為專案內簡化公式，非 FanGraphs 官方全文複製。
3. **守備位置** 依 Lahman／網路快取成敗；全 **UNK** 時模型主要仰賴數值特徵。
4. 樣本量有限時 **R²** 可能偏低或為負；宜並呈 **MAE／RMSE** 與散點圖，避免過度解讀單一指標。

---

## 7. 相關檔案索引

| 路徑 | 說明 |
|------|------|
| `bb_pipeline/statcast_season.py` | Statcast → 逐季打者列 |
| `bb_pipeline/projection_dataset.py` | 組 `projection_panel` |
| `bb_pipeline/position_lahman.py` | Lahman 守備位置 |
| `bb_pipeline/train_projection.py` | RF / XGB、時間切分、指標 |
| `scripts/build_pipeline_outputs.py` | 重建 parquet 之 CLI |
| `notebooks/01_data_collection.ipynb` | 資料蒐集與面板輸出 |
| `notebooks/03_analysis.ipynb` | Task 7：隔年 wRC+ 預測與出圖 |
