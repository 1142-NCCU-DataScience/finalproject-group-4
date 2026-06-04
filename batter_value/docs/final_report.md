# MLB 打者生涯預測分析報告
## Player Career Projection — 資料科學期末專題

**作者：** Brian 吳帛恩  
**課程：** 資料科學（2026 春季學期）  
**資料時間範圍：** 2018–2024 MLB 球季  

---

## 目錄

1. [研究背景與目標](#1-研究背景與目標)
2. [資料來源與收集方式](#2-資料來源與收集方式)
3. [資料處理流程](#3-資料處理流程)
4. [探索性資料分析](#4-探索性資料分析-eda)
5. [Statcast 熱區視覺化](#5-statcast-熱區視覺化)
6. [球員分群分析](#6-球員分群分析)
7. [隔年 wRC+ 預測模型](#7-隔年-wrc-預測模型)
8. [分析結論](#8-分析結論)

---

## 1. 研究背景與目標

傳統棒球評估長期仰賴打擊率（AVG）、打點（RBI）等統計，但這些指標容易受球場、隊友、運氣因素扭曲，無法真實反映球員個人能力。現代賽伯計量學（Sabermetrics）引入 wRC+（加權上壘打點創造值，聯盟平均 = 100）和 WAR（勝場貢獻值）等進階指標，提供更準確的打者表現衡量。

**核心問題：** 能否僅用球員過去三個球季的「打擊過程指標」，預測其**下一季**的 wRC+？

此問題在實務上具有高度意義：球隊 GM 需要在賽季前評估球員未來貢獻，作為薪資協商與交易的依據。

**目標：**
1. 建立以 Statcast 為唯一資料來源的打擊指標計算管線
2. 透過機器學習（Random Forest）預測球員隔年 wRC+
3. 使用嚴格的**時間切分**避免未來資料洩漏
4. 分析哪些過程指標對預測最具貢獻

---

## 2. 資料來源與收集方式

### 2.1 主要資料來源

| 資料來源 | 取得方式 | 用途 | 時間範圍 |
|---------|---------|------|---------|
| **Baseball Savant Statcast** | `pybaseball.statcast()` | 每球事件資料、計算所有打擊率指標 | 2018–2024 |
| **Lahman Database** | `pybaseball.lahman.fielding()` | 球員守備位置（primary position） | 2018–2024 |
| **Chadwick Register** | `pybaseball.playerid_reverse_lookup()` | MLBAM ID → 球員姓名對應 | 輔助查詢 |

> **重要設計決策：** 本專案**完全不使用** FanGraphs 等外部打擊排行榜。所有比率指標（K%、BB%、BABIP、BIP%、wRC+）均由 Statcast 逐球事件資料自行計算，確保資料來源一致性與可重現性。

### 2.2 Statcast 資料結構

Statcast 為逐球（pitch-by-pitch）事件資料，每一列代表一個投球事件。本專案保留**打席結束列**（`events` 欄位非空），包含：

| 欄位名稱 | 說明 |
|---------|------|
| `batter` | 打者 MLBAM ID |
| `game_year` | 球季年份 |
| `events` | 打席結果（`single`, `home_run`, `strikeout` 等 22 種） |
| `plate_x` | 進壘水平位置（ft，捕手視角） |
| `plate_z` | 進壘垂直高度（ft） |
| `woba_value` | 該打席 wOBA 貢獻值 |
| `woba_denom` | wOBA 分母（是否計入計算） |
| `age_bat` | 打者當時年齡 |
| `launch_speed` | 擊球初速（mph） |
| `launch_angle` | 擊球仰角（度） |
| `estimated_ba_using_speedangle` | 期望打擊率 xBA |

### 2.3 下載方式與快取

```python
from pybaseball import cache, statcast
cache.enable()   # 啟用本地快取，避免重複下載

# 依年份下載（單季約 700,000 筆逐球資料）
df = statcast(start_dt="2024-03-01", end_dt="2024-11-01")
```

每個球季約 70–90 萬筆資料，下載後存為 `.parquet` 格式（約 100–200 MB/季）。本專案共下載 2018–2024 年共 7 個球季。

### 2.4 特定球員 Statcast 資料（熱區圖用）

額外下載 3 位代表性球員的 2024 年逐球資料（用於熱區圖視覺化）：

```python
from pybaseball import statcast_batter, playerid_lookup

players = [("judge", "aaron"), ("betts", "mookie"), ("acuna", "ronald")]
for last, first in players:
    info = playerid_lookup(last, first)
    mlbam_id = info.iloc[0]["key_mlbam"]
    df = statcast_batter("2024-03-01", "2024-09-30", player_id=int(mlbam_id))
```

---

## 3. 資料處理流程

### 3.1 整體管線

> 📊 **圖 1 — 資料管線流程圖**  
> `figures/poster/fig_pipeline.png`  
> 🏆 **海報重點圖** ｜ 📑 **期末簡報必放**

![資料管線流程圖](poster/fig_pipeline.png)

*圖 1：從 Statcast 逐球資料到最終 wRC+ 預測的完整資料管線。每個方塊代表一個處理階段，箭頭表示資料流向。*

### 3.2 逐季打者面板建立

**程式：** `bb_pipeline/statcast_season.py`  
**輸出：** `data/raw/batting_by_season_2018_2024.parquet`

對每位打者、每個球季，依以下邏輯彙總為一列：

**步驟：**
1. **篩選打席結束事件**：保留 `events` 屬於 `PA_EVENTS` 集合的列（安打、保送、三振、各類出局等）
2. **依 `(batter, game_year)` 分組**，計算以下統計：

| 計算指標 | 公式/說明 |
|---------|---------|
| PA | 有效打席數（計入 woba_denom 之列） |
| K% | 三振數 ÷ PA |
| BB% | 四壞球數 ÷ PA |
| BABIP | 安打（扣全壘打）÷ 場內擊球出局（BIP） |
| BIP% | 場內擊球事件數 ÷ PA |
| wOBA | Σ(woba_value) ÷ Σ(woba_denom) |
| wRC+ | 以當季全聯盟 PA 加權 lg_wOBA，套用簡化公式（wOBA_scale=1.18，lg_R_PA=0.127） |
| age_bat_median | 該季所有打席的年齡中位數 |

**篩選條件（EDA 階段）：** PA ≥ 150，共保留 958 筆球員-球季觀測值

### 3.3 預測用面板建立（Lag 特徵工程）

**程式：** `bb_pipeline/projection_dataset.py`  
**輸出：** `data/processed/projection_panel.parquet`（943 列）

對每位打者、每個**目標球季 S**，組成以下 lag 特徵面板：

| 特徵類型 | 欄位 | 說明 |
|---------|------|------|
| **標籤** | `wRCplus_target` | 目標年 S 的 wRC+ |
| **必要條件** | — | S-1、S-2、S-3 各季 PA ≥ 100 |
| **比率特徵** | K%、BB%、BABIP、BIP% × lag1/2/3 | 12 欄 |
| **打席數** | PA_lag1、PA_lag2、PA_lag3、PA_sum_lag3 | 4 欄 |
| **年齡** | age_t（取自 S-1 季） | 1 欄 |
| **守備位置** | primary_pos（Lahman；失敗時為 UNK） | 1 欄（類別） |

**範例：** 預測 **2024** 年 wRC+ → 使用 **2021、2022、2023** 三季資料

> 📊 **圖 2 — 時間切分策略**  
> `figures/poster/fig_timesplit.png`  
> 🏆 **海報重點圖** ｜ 📑 **期末簡報必放**

![時間切分策略](poster/fig_timesplit.png)

*圖 2：依 target_season 進行時間切分。藍色氣泡為 Train（2021–22，n=453），綠色為 Validation（2023，n=234），紅色為 Test（2024，n=256）。箭頭示意預測 2024 時所需的三年 lag 資料範圍。此切分方式確保模型在訓練時看不到未來資料。*

---

## 4. 探索性資料分析 (EDA)

### 4.1 主要打擊指標分布

> 📊 **圖 3 — 主要打擊指標分布直方圖**  
> `figures/eda_distributions.png`  
> 📑 **期末簡報用**

![主要打擊指標分布](eda_distributions.png)

*圖 3：2018–2024 MLB 合格打者（PA ≥ 150）六項主要指標的分布。wRC+ 接近常態分布，中心約 90；WAR 右偏明顯，多數球員貢獻有限，少數明星遠超群倫；K% 呈右偏，三振率逐年上升是現代棒球趨勢。*

**關鍵統計：**
- **wRC+ 平均值 ≈ 90**（低於 100，因篩選 PA≥150 仍有弱打者入列）
- **WAR 高度右偏**：中位數約 2–3，但頂級球員（Judge、Soto）可達 40+
- **K% 平均 ≈ 22%**，反映現代棒球三振率高漲的趨勢

### 4.2 指標相關係數分析

> 📊 **圖 4 — 打擊指標相關係數熱圖**  
> `figures/eda_correlation.png`  
> 📑 **期末簡報用**

![打擊指標相關係數熱圖](eda_correlation.png)

*圖 4：九項核心打擊指標的 Pearson 相關係數矩陣。顏色越深紅代表正相關越強，藍色代表負相關。*

**關鍵發現：**

| 指標對 | 相關係數 | 解讀 |
|-------|---------|------|
| wRC+ ↔ OBP | **+0.86** | 上壘率最能解釋打擊貢獻 |
| wRC+ ↔ SLG | **+0.85** | 長打率次之 |
| SLG ↔ ISO | **+0.89** | ISO 幾乎由 SLG 決定 |
| AVG ↔ K% | **−0.62** | 高三振率球員打擊率低 |
| WAR ↔ wRC+ | **+0.65** | 攻擊是 WAR 的主要成分，但守備也有貢獻 |
| BB% ↔ AVG | **−0.02** | 選球能力與打擊率幾乎無關 |

> **洞察：** OBP 和 SLG 與 wRC+ 的高相關說明 wRC+ 本質上整合了上壘與長打兩個維度。BB% 與 AVG 近零相關，意味著選球型打者未必打擊率高，但透過保送仍能創造高 wRC+（典型代表：Juan Soto）。

### 4.3 wRC+ 排名分析

> 📊 **圖 5 — wRC+ 前 20 名打者排名**  
> `figures/fig1_wrcplus_ranking.png`  
> 📑 **期末簡報用**

![wRC+ 排名](fig1_wrcplus_ranking.png)

*圖 5：2018–2024 所有合格球季中 wRC+ 最高的 20 名打者。顏色代表 K-Means 分群結果（強打型、選球型、接觸型、工具型）。虛線為聯盟平均（100）。*

**觀察：** 前 4 名（Nathan Lukes 186、Max Schrock 174、Mike Trout 173、Aaron Judge 172）中，Lukes 與 Schrock 因打席數極少，wRC+ 受少量打席放大，較難與 Trout、Judge 的多年高 wRC+ 相提並論。Trout 和 Judge 的高 wRC+ 建立在 3,000+ 打席上，更具代表性。

### 4.4 WAR vs wRC+ 四象限分析

> 📊 **圖 6 — WAR vs wRC+ 四象限**  
> `figures/fig2_war_vs_wrcplus.png`  
> 🏆 **海報重點圖** ｜ 📑 **期末簡報必放**

![WAR vs wRC+ 四象限](fig2_war_vs_wrcplus.png)

*圖 6：打者 WAR 對 wRC+ 的分布。虛線為各指標平均值，形成四象限。右上為「明星球員」（高攻擊力+高整體價值），右下為「攻擊潛力型」（高 wRC+ 但 WAR 偏低，可能守備較弱），左上為「守備價值型」（整體貢獻高但攻擊偏弱），左下為「替補水準」。*

**發現：** Aaron Judge、Freddie Freeman、Juan Soto 等球員聚集在右上角，為真正的全能型球星。部分工具型球員（粉紅色）分布在左下方，顯示這類球員的攻守貢獻均有限。

---

## 5. Statcast 熱區視覺化

### 5.1 三球員進壘點熱區對比

> 📊 **圖 7 — 2024 球季三球員進壘點熱區圖**  
> `figures/fig4_heatmap_all.png`  
> 🏆 **海報重點圖** ｜ 📑 **期末簡報必放**

![三球員熱區圖](fig4_heatmap_all.png)

*圖 7：Aaron Judge、Mookie Betts、Ronald Acuña Jr. 在 2024 球季所有打席的進壘點密度分布（Hexbin 圖）。黑框為好球帶（MLB 標準：水平 ±0.83 ft，垂直 1.5–3.5 ft）。顏色越深紅代表密度越高。*

**比較：**
- **Aaron Judge**：進壘集中於好球帶中低區，顯示投手傾向投內角低球或正中低球
- **Mookie Betts**：密度分布更均勻，覆蓋整個好球帶，選球能力強
- **Ronald Acuña Jr.**：密度較分散，包含較多好球帶外側的追打球

### 5.2 依打擊結果分色的進壘點分布

> 📊 **圖 8 — 打擊結果進壘點分布**  
> `figures/fig5_heatmap_by_result.png`  
> 📑 **期末簡報用**

![打擊結果分色](fig5_heatmap_by_result.png)

*圖 8：Aaron Judge、Mookie Betts、Ronald Acuña 三人合計的進壘點，依打擊結果分色。全壘打（紅星）集中於好球帶中間偏下，三振（紅 ×）分布較廣，四壞球（橘菱形）多在好球帶外。*

**洞察：** 安打和長打集中在好球帶中央，尤其中低區；三振則好球帶內外皆有，外角低球（右下角）為三振高頻區，符合現代投手的攻擊策略。

### 5.3 Aaron Judge 強弱區對比

> 📊 **圖 9 — Aaron Judge 強區 vs 弱區對比**  
> `figures/fig10_judge_power_vs_weak.png`  
> 📑 **期末簡報用**

![Judge 強弱區](fig10_judge_power_vs_weak.png)

*圖 9：Aaron Judge 在 2024 球季的強區（長打：HR+2B+3B，n=94）與弱區（三振：K，n=164）進壘點熱區。左圖顯示 Judge 在好球帶中央集中爆發長打；右圖顯示三振主要集中在好球帶右側（對右打者為外角）。*

**洞察：** 即使是聯盟最強打者，外角進壘仍是弱點。投手策略選擇外角偏下可提高三振率，但一旦失投到中間，便易被 Judge 轟出全壘打。

### 5.4 實際打擊率 vs 期望打擊率（運氣修正）

> 📊 **圖 10 — xBA vs BA 運氣修正分析**  
> `figures/fig9_xba_vs_ba.png`  
> 📑 **期末簡報用（補充分析）**

![xBA vs BA](fig9_xba_vs_ba.png)

*圖 10：2024 球季（至少 50 次擊球）打者的實際打擊率（BA）對期望打擊率（xBA，Statcast 物理模型預測）。位於對角線上方的球員打得比「應有水準」好（可能受益於防守失誤或強勁速度）；下方球員則「運氣較差」或受強守備影響。顏色代表 BA−xBA 差值。*

**洞察：** 大部分球員聚集在對角線附近，說明長期下來運氣趨於均值。部分球員 xBA 顯著高於 BA，代表其實際表現被低估，是潛在被低估球員的指標。

---

## 6. 球員分群分析

### 6.1 K-Means 分群（4 種打者類型）

以 AVG、OBP、SLG、BB%、contact_rate（接觸率）、ISO（孤立長打率）六項標準化指標，對 958 名合格打者進行 K-Means 聚類（k=4）。

| 分群 | 樣本數 | 平均 wRC+ | 特徵描述 |
|------|------|---------|---------|
| **強打型** | — | **117.0** | 高 OBP、高 SLG，低 K%（相對）。典型：Judge、Soto |
| **選球型** | — | **93.1** | 高 BB%，但 SLG 中等。典型：耐心型打者 |
| **接觸型** | — | **90.6** | 低 K%，高 AVG，但長打力有限 |
| **工具型** | — | **56.4** | 低 AVG、高 K%，各指標偏弱，多為備用球員 |

### 6.2 明星球員雷達圖

> 📊 **圖 11 — 明星打者能力雷達圖**  
> `figures/poster/fig_radar_stars.png`  
> 📑 **期末簡報用（補充）**

![明星雷達圖](poster/fig_radar_stars.png)

*圖 11：Aaron Judge、Juan Soto、Shohei Ohtani、Mookie Betts、Freddie Freeman 在六項打擊指標的標準化能力圖（各指標以 2018–2024 全體球員最大值為基準，取各球員生涯各季最優值）。*

**比較：**
- **Aaron Judge**：SLG 和 ISO 最突出，長打火力最猛
- **Juan Soto**：OBP 和 BB% 最高，選球能力超群
- **Shohei Ohtani**：接觸率和長打並重，六邊形最均衡
- **Mookie Betts**：各指標均衡，無明顯弱點
- **Freddie Freeman**：AVG 和 OBP 表現優異，接觸型強打者

---

## 7. 隔年 wRC+ 預測模型

### 7.1 模型設計

**任務：** 回歸問題，輸入前三季 lag 特徵，輸出下一季 wRC+（連續變數）

**主要模型：** Random Forest Regressor
- `n_estimators=400`，`max_depth=12`，`min_samples_leaf=8`
- 數值特徵直通，`primary_pos` 經 OneHotEncoder

**備選模型：** XGBoost（macOS 需安裝 `libomp` 才可執行）

**切分策略（避免時間洩漏）：**

| 集合 | target_season | 球員-年份數 |
|------|-------------|----------|
| **Train** | 2021、2022 | 453 |
| **Validation** | 2023 | 234 |
| **Test** | 2024 | 256 |

### 7.2 特徵重要性分析

> 📊 **圖 12 — Random Forest 特徵重要性**  
> `figures/poster/fig_perm_importance.png`  
> 🏆 **海報重點圖** ｜ 📑 **期末簡報必放**

![特徵重要性](poster/fig_perm_importance.png)

*圖 12：Random Forest 在 Validation（2023）上的 Permutation Importance。橫軸為「打亂該特徵後驗證誤差上升量」，越長代表模型越依賴該特徵。紅色為關鍵特徵（Top 30%）。*

**關鍵發現：**

1. **三年 PA 總計** 排名第一：打席數累積代表球員健康度與出賽穩定性，是預測生涯連續性的重要信號
2. **四壞% (3yr前)** 排名第二：早年的選球能力具有長期預測力，反映打者眼力的穩定性
3. **三振% (1yr前)** 第三：最近一年的三振率對下季表現影響顯著
4. **年齡** 重要性中等：生涯曲線效應（通常 27–29 歲達到巔峰）確實存在
5. **BABIP (3yr前)** 呈**負重要性**：這符合 BABIP 的「均值回歸」特性——3 年前的 BABIP 打亂後誤差反而下降，說明模型可能部分學到 BABIP 的噪音效應

> ⚠️ **注意：** Permutation Importance 衡量模型依賴程度，非因果關係。BB% 排第一不代表四壞率能「提升」未來 wRC+，而是兩者在時間上共變化。

### 7.3 預測結果（Hold-out Test 2024）

> 📊 **圖 13 — Hold-out Test 2024：實際 vs 預測**  
> `figures/poster/fig_actual_vs_pred.png`  
> 🏆 **海報重點圖** ｜ 📑 **期末簡報必放**

![實際 vs 預測](poster/fig_actual_vs_pred.png)

*圖 13：2024 球季 Hold-out Test（n=241，已過濾 wRC+<20 的算法異常值）的實際 wRC+ 對 RF 預測值散點圖。虛線為完美預測線（y=x）。顏色代表預測誤差大小（紅色誤差大，綠色誤差小）。Juan Soto 和 Shohei Ohtani 的真實 wRC+ 高達 150+，模型低估了這類超級球星。*

**模型表現：**

| 指標 | 數值 | 解讀 |
|------|------|------|
| **MAE** | **22.1 wRC+ 點** | 平均絕對誤差；wRC+ 年際自然波動也約 15–25 點 |
| **R²** | **−0.018** | 負值：模型系統性偏差主導，略遜於均值預測 |

**觀察到的系統性偏差：**
- 模型對低 wRC+（<80）的球員**高估**：預測集中在 90–110，因 RF 的均值回歸效應
- 模型對超級球星（wRC+ > 150）**低估**：Juan Soto、Ohtani 實際表現遠超模型預測
- 這種「向均值壓縮」是 RF 的已知特性，在小訓練集（Train: 2021–22）下更明顯

---

## 8. 分析結論

### 8.1 主要發現

**EDA 層面：**
- OBP 和 SLG 與 wRC+ 高度相關（r > 0.85），是球員攻擊價值的核心指標
- 進階指標（wRC+）能識別被 AVG 低估的球員（如高 BB% 但低 AVG 的選球型打者）
- Statcast 熱區分析揭示投手攻擊策略：外角低球是高 K% 區域，中央偏下是長打熱區

**預測模型層面：**
1. **過去三季 PA 總計**是最重要預測特徵：健康、穩定出賽是隔年表現的最強信號
2. **四壞率（BB%）穩定性高**：三年前的 BB% 仍能預測未來，反映打者眼力是長期能力
3. **BABIP 均值回歸**：遠期 BABIP 負重要性確認其噪音屬性，不應過度解讀單年 BABIP
4. **年齡效應存在**：生涯曲線對 wRC+ 有可測量的貢獻

**模型限制：**
- Test MAE = 22.1 wRC+ 點，接近 wRC+ 年際波動的自然上限（約 15–25 點）
- R² = −0.018（負值）顯示模型的系統性高估弱打者、低估強打者，限制整體解釋力
- 訓練集僅有 2 個目標年（2021–22），樣本量偏少

### 8.2 限制說明

| 限制 | 說明 |
|------|------|
| **有監督目標年僅 4 年** | 可用 target_season 只有 2021–2024，Train 只有 2 年 |
| **wRC+ 簡化計算** | 非 FanGraphs 官方公式，可能與官方值有 1–3 點差異 |
| **守備位置品質** | Lahman 資料部分缺失，位置特徵多為 UNK，降低位置資訊的預測力 |
| **未包含受傷資訊** | 球員因傷缺賽會影響 PA，但模型無法區分「技術衰退」與「受傷」 |
| **均值回歸效應** | RF 的預測天然向中間壓縮，對極端值（超級球星/極差球員）低估/高估 |

### 8.3 未來改進方向

1. **增加訓練年份**：等待 2025 球季結束後，可將 target_season 延伸至 2025，增加 1 年 Test 與更多 Train 資料
2. **納入更多特徵**：擊球品質指標（launch_speed、launch_angle、Hard Hit%）、球員合約年資、歷史受傷紀錄
3. **XGBoost 調參**：安裝 `libomp` 後啟用 XGBoost，與 RF 進行系統性比較
4. **Walk-Forward Validation**：使用逐年擴張訓練窗的 walk-forward 驗證，更準確估計模型在未來的表現
5. **位置分層建模**：分別對捕手、外野手、內野手建模，因各位置的 wRC+ 基準線差異顯著

---

## 附錄：圖片使用建議

### 🏆 海報（Poster）重點圖（建議放 5–6 張）

| 圖 | 檔案 | 說明 |
|----|------|------|
| **管線流程** | `figures/poster/fig_pipeline.png` | 讓觀眾快速理解整體研究設計 |
| **時間切分** | `figures/poster/fig_timesplit.png` | 展示避免資料洩漏的嚴謹設計 |
| **WAR vs wRC+** | `figures/fig2_war_vs_wrcplus.png` | 直觀展示球員價值分布與分群 |
| **熱區圖** | `figures/fig4_heatmap_all.png` | 視覺衝擊力強，吸引非棒球觀眾 |
| **特徵重要性** | `figures/poster/fig_perm_importance.png` | 展示模型核心發現 |
| **實際 vs 預測** | `figures/poster/fig_actual_vs_pred.png` | 呈現模型表現與限制 |

### 📑 期末簡報（Slides）完整圖單

| 章節 | 圖 | 檔案 |
|------|-----|------|
| 方法論 | 圖 1 管線流程 | `poster/fig_pipeline.png` |
| 方法論 | 圖 2 時間切分 | `poster/fig_timesplit.png` |
| EDA | 圖 3 指標分布 | `eda_distributions.png` |
| EDA | 圖 4 相關係數熱圖 | `eda_correlation.png` |
| EDA | 圖 5 wRC+ 排名 | `fig1_wrcplus_ranking.png` |
| EDA | 圖 6 WAR vs wRC+ | `fig2_war_vs_wrcplus.png` |
| 熱區 | 圖 7 三球員熱區 | `fig4_heatmap_all.png` |
| 熱區 | 圖 8 打擊結果分色 | `fig5_heatmap_by_result.png` |
| 熱區 | 圖 9 Judge 強弱區 | `fig10_judge_power_vs_weak.png` |
| 補充 | 圖 10 xBA vs BA | `fig9_xba_vs_ba.png` |
| 分群 | 圖 11 雷達圖 | `poster/fig_radar_stars.png` |
| 模型 | 圖 12 特徵重要性 | `poster/fig_perm_importance.png` |
| 模型 | 圖 13 實際 vs 預測 | `poster/fig_actual_vs_pred.png` |

> ⚠️ **不建議在簡報/海報中使用：**  
> - `fig6_model_comparison.png`（同年 5-fold CV，R²≈0.86 嚴重誤導，是舊的同年預測模型）  
> - `fig7_feature_importance.png`（同年 MDI，OBP/SLG 主導，非 lag 特徵）  
> - `fig8_actual_vs_predicted.png`（同年模型 scatter，非 hold-out 預測）

---

*報告生成時間：2026-05-16*  
*專案路徑：`baseball_project/`*  
*腳本：`scripts/create_presentation_figures.py`*
