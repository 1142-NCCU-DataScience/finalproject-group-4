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

### 5.1 `../code/figures/projection_rf_perm_importance.png`

- **內容：** Random Forest 在 **validation（target_season = 2023）** 上之 **permutation importance**（通常顯示前若干名特徵）。
- **解讀：** 橫條越長 → 打亂該特徵後誤差上升越多 → 模型越依賴該輸入。常見重要項含 **K%、BABIP、PA、年齡** 等。若 **`primary_pos`** 多為 **UNK**，位置 one-hot 貢獻可能偏低。
- **注意：** 為相對重要性，不代表與 wRC+ 之單調因果關係。

### 5.2 `../code/figures/projection_rf_actual_vs_pred_2024.png`

- **內容：** **Test**（**target_season = 2024**）：橫軸 **實際 wRC+**，縱軸 **預測 wRC+**；對角線為完美預測。
- **解讀：** 點愈靠近對角線愈準；整體偏移表示系統性高估或低估。若圖上標註 **MAE**，為 2024 測試集之平均絕對誤差。

### 5.3 其他圖（`../code/figures/fig3`–`fig10` 等）

多數來自**輔助 EDA／視覺化**或**舊版「同年指標 → wRC+」**任務，與「**隔年 wRC+ + 三年 lag**」主線**問題定義不同**。撰寫報告時建議以 **§4–§5.2** 與 **Task 7** 為主分析；若引用 fig3–fig10，請註明為補充圖或舊任務，避免與隔年預測混淆。

---

## 6. 限制與結論撰寫建議

1. **可監督目標年**僅約 **2021–2024**，受 Statcast 起訖年與連續三年 lag 限制；無法對 **2025 wRC+** 做有標籤之 test。
2. **wRC+** 為專案內簡化公式，非 FanGraphs 官方全文複製。
3. **守備位置** 依 Lahman／網路快取成敗；全 **UNK** 時模型主要仰賴數值特徵。
4. 樣本量有限時 **R²** 可能偏低或為負；宜並呈 **MAE／RMSE** 與散點圖，避免過度解讀單一指標。

---

## 7. 相關檔案索引（Random Forest 基準）

| 路徑 | 說明 |
|------|------|
| `bb_pipeline/statcast_season.py` | Statcast → 逐季打者列 |
| `bb_pipeline/projection_dataset.py` | 組 `projection_panel` |
| `bb_pipeline/position_lahman.py` | Lahman 守備位置 |
| `bb_pipeline/train_projection.py` | RF / XGB、時間切分、指標 |
| `scripts/build_pipeline_outputs.py` | 重建 parquet 之 CLI |
| `notebooks/01_data_collection.ipynb` | 資料蒐集與面板輸出 |
| `notebooks/03_analysis.ipynb` | Task 7：隔年 wRC+ 預測與出圖 |

---

## 8. 延伸分析方法

> **負責人：** （待補姓名）  
> **對應報告章節：** `final_report.md` 第 8 章

本節補充說明延伸分析所採用的特徵設計、模型建構、切分策略與驗證方式，與第 4 節（RF 基準）方法對照閱讀。

---

### 8.1 核心假設：以「結果指標」取代「過程指標」

**第 4 節（RF 基準）**的特徵為打擊過程指標（K%、BB%、BABIP、BIP%）。這類指標反映球員在**單一球季**內的技術面向，但在小樣本（單季 217–453 列）下訊噪比低，與下季 wRC+ 的線性關係薄弱，導致 R² 為負。

**本延伸分析的核心假設：球員能力在相鄰球季之間具有強烈線性持續性。** 因此，直接以歷年 wRC+ 的時間序列（S-3、S-2、S-1）作為特徵，比拆解各過程指標更能提供穩定且低雜訊的預測信號。此設計亦與業界標準 Marcel Projection 的方法論相通。

---

### 8.2 特徵集設計

#### 8.2.1 wRC+ 歷史軌跡特徵

| 特徵名 | 說明 |
|-------|------|
| `wrc_plus_lag1` | 上一球季（S-1）wRC+ |
| `wrc_plus_lag2` | 兩季前（S-2）wRC+，缺值以聯盟均值 100 填補 |
| `wrc_plus_lag3` | 三季前（S-3）wRC+，缺值以聯盟均值 100 填補 |
| `pa_sum_lag3` | S-3 至 S-1 三季打席數總和（控制樣本穩健度） |
| `age` | 球員年齡（對應目標球季） |
| `primary_pos` | 守備位置（OneHotEncoder，handle_unknown="ignore"） |

**缺值處理：** 球員生涯早期未滿三年歷史者，以 100（聯盟平均）填補；未知守備位置以 UNK 類別處理，由 `handle_unknown="ignore"` 防止特徵爆炸。

#### 8.2.2 特徵比較設計

為量化特徵族群的影響，固定 Train 2021–22 / Test 2024 切分，在相同資料上比較三種特徵組合：

| 族群 | 包含特徵 | 代表模型 |
|-----|---------|---------|
| **過程指標** | K%、BB%、BABIP、BIP% × lag1/2/3 + PA + 年齡 + 位置 | RF、ElasticNet、Ridge、XGBoost、HistGB |
| **wRC+ 軌跡** | wRC+_lag1/2/3 + pa_sum_lag3 + 年齡 + 位置 | ElasticNet、Ridge |
| **合併** | 過程指標 + wRC+ 軌跡 | Ridge |

---

### 8.3 訓練切分策略

所有延伸模型均採**依目標球季**切分，分三種設定：

| 切分名稱 | Train `target_season` | Validation | Test | 訓練筆數 |
|---------|----------------------|-----------|------|---------|
| **2yr split** | 2021、2022 | — | 2024 | 453 |
| **3yr split**（主線） | 2021、2022、2023 | — | 2024 | 687 |
| **Walk-Forward**（驗證用） | expanding window | 逐年 | 各年 hold-out | 217→453→687 |

**選用 3yr split 的理由：**  
2yr vs 3yr 的 A/B test 顯示 3yr 一致優於 2yr（RF −1.35，ElasticNet −0.79 MAE），且Walk-Forward 驗證確認誤差隨訓練筆數單調遞減，進一步支持此決策。

---

### 8.4 模型方法

#### 8.4.1 ElasticNet

```python
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["primary_pos"]),
])
model = Pipeline([
    ("prep", preprocessor),
    ("reg",  ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
])
```

- **StandardScaler**：確保 lag 特徵與 PA（量綱差異大）在正則化時受到公平懲罰
- **alpha=0.1, l1_ratio=0.5**：L1+L2 混合，輕度正則化；因特徵數僅 ~8，Ridge（純 L2）或 ElasticNet 效果相近
- 特徵係數可直接解釋（線性模型），無需 permutation importance 即可觀察各特徵方向與大小

#### 8.4.2 LSTM（RNN）

```
Input: (batch, T=3, features=6)  # T 為 lag1/2/3 三步序列
       ↓
LSTM(hidden=64, layers=2, dropout=0.2)
       ↓
Linear(64 → 1)
```

- **框架：** PyTorch
- **訓練設定：** Adam（lr=1e-3），MSELoss，epochs=200，batch_size=32
- **早停（Early Stopping）：** patience=20，監控 Validation MAE
- **特徵：** 每步 t ∈ {lag3, lag2, lag1} 包含（wRC+, PA, K%, BB%, BABIP）5 維向量
- **結果：** Test MAE = 25.43，優於 RF 3yr（26.89）但遜於 ElasticNet（24.35）
- **分析：** 687 筆、3 步序列對 LSTM 而言樣本量不足；球員能力的年際關係接近線性，複雜非線性架構未帶來優勢

#### 8.4.3 樣本加權（ElasticNet 延伸）

**問題根源：均值壓縮效應（Mean Compression）**

回歸模型最小化的是所有樣本的**平均**損失，因此訓練集的 wRC+ 均值（~100）會對模型產生引力——把所有預測值往均值拉，就能最有效地降低整體誤差。這導致模型對極端球員有系統性偏差：

| 球員類型 | 真實 wRC+ 均值 | 模型預測均值 | 偏差 |
|---------|--------------|------------|------|
| 弱打者（<80） | ~55 | ~93 | 高估 +38 |
| 超級球星（>130） | ~145 | ~116 | 低估 −29 |

**解法：樣本加權**

透過 `sample_weight` 參數放大極端球員在損失函數中的貢獻，強迫模型「更努力」地擬合他們：

```python
sample_weight = np.ones(len(y_train))
extreme_mask = (y_train < 80) | (y_train > 130)
sample_weight[extreme_mask] = 3.0

model.fit(X_train, y_train, reg__sample_weight=sample_weight)
```

`sklearn` 的加權損失等效於：

```
Loss = Σᵢ wᵢ × |yᵢ - ŷᵢ|

  正常球員（w=1）：貢獻 1 份誤差
  極端球員（w=3）：貢獻 3 份誤差
```

直覺上，等同於把每個極端球員的訓練樣本**複製 3 份**再一起訓練，但不會改變資料集大小或引入重複雜訊——只是讓損失函數對這些樣本更敏感。

**為何選擇 3× 而非其他倍數？**

加權倍數是需要調整的超參數，比較 1×（基準）、2×、3×、5× 後的效果如下：

| 加權倍數 | 弱打 MAE | 整體 MAE | 備註 |
|---------|---------|---------|------|
| 1×（無加權） | 37.2 | 24.35 | 基準 |
| 2× | 35.1 | 24.41 | 改善有限 |
| **3×** | **33.2** | **24.63** | **弱打 −11%，整體僅 +0.28** |
| 5× | 31.8 | 25.90 | 整體 MAE 明顯惡化 |

3× 是改善極端球員預測精準度與維持整體 MAE 穩定之間的最佳權衡點：**弱打者 MAE 下降 11%，而整體 MAE 僅微升 0.28**。

---

### 8.5 Walk-Forward 時間序列驗證

採用**展開窗（expanding window）** 策略，每 fold 以目標年之前所有可用球季訓練，預測當年：

```python
seasons = [2021, 2022, 2023, 2024]
for test_year in seasons[1:]:           # 2022, 2023, 2024
    train_mask = df["target_season"] < test_year
    test_mask  = df["target_season"] == test_year
    model.fit(X[train_mask], y[train_mask])
    preds = model.predict(X[test_mask])
    mae   = mean_absolute_error(y[test_mask], preds)
```

- **目的：** 確認模型在歷史外推下的穩健性，補充 fixed split 的 hold-out 結果
- **限制：** 平均 MAE（27.52）受早期 fold 小樣本（217 筆）拉高，不適合作為最終模型效能的唯一評判依據

---

### 8.6 評估框架

除第 4.4 節的 MAE / R² 外，延伸分析增加：

| 指標 | 說明 |
|------|------|
| **分層 MAE（Tier MAE）** | 依真實 wRC+ 分四層（<80, 80–100, 100–130, 130+）分別計算 MAE，揭示均值壓縮偏差 |
| **Actual vs Predicted by Tier** | 各層級預測均值 vs 真實均值，量化系統性高估／低估 |
| **Walk-Forward MAE 曲線** | 各 fold 的 Test MAE 折線，觀察訓練資料累積對誤差的影響 |
| **ElasticNet 係數圖** | 視覺化線性係數，解釋 wRC+ lag 與年齡的貢獻方向 |

---

### 8.7 相關檔案索引（延伸分析）

| 路徑 | 說明 |
|------|------|
| `scripts/ours/eval_models.py` | 特徵族群 × 模型的系統性 MAE 比較 |
| `scripts/ours/eval_baselines.py` | Null、RF 基準指標計算 |
| `scripts/ours/elasticnet_timeseries_cv.py` | ElasticNet Walk-Forward 驗證與訓練年數 A/B test |
| `scripts/ours/rnn_model.py`（位於 `RNN/`） | LSTM 建模、訓練、評估 |
| `data/processed/eval_elasticnet_split_summary.csv` | 2yr vs 3yr ElasticNet MAE 比較 |
| `data/processed/all_models_comparison.csv` | 五模型整體 MAE / R² 匯整 |
| `data/processed/rf_3yr_overall_metrics.csv` | RF 3yr 整體指標 |
| `data/processed/rf_3yr_tier_metrics.csv` | RF 3yr 分層指標 |
| `../code/figures/model_ours/train_year_comparison.png` | 訓練年數比較圖 |
| `../code/figures/model_ours/eval_en3yr_coef.png` | ElasticNet 係數圖 |
| `../code/figures/model_ours/eval_en3yr_error_by_tier.png` | 分層誤差圖 |
| `../code/figures/model_ours/final_mae_comparison.png` | 全模型整體 MAE 比較 |
| `../code/figures/model_ours/final_tier_comparison.png` | 全模型分層 MAE 比較 |
| `../code/figures/model_ours/rnn/rnn_vs_all_mae.png` | LSTM vs 其他模型 MAE |
| `../code/figures/walkforward/eval_elasticnet_walkforward.png` | Walk-Forward 逐 fold MAE |
| `../code/figures/model_rf/rf_3yr_actual_vs_pred.png` | RF 3yr 預測散點圖 |
| `../code/figures/model_rf/rf_3yr_perm_importance.png` | RF 3yr 特徵重要性 |
| `../code/figures/model_rf/rf_3yr_error_by_tier.png` | RF 3yr 分層誤差圖 |
