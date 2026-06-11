# MLB 打者數據科學專題

資料科學課程期末作業 — 以 MLB Statcast 逐球資料（2018–2024）為核心，從三個互補角度分析大聯盟打者表現。

---

## 專案架構

```
finalproject-group-4/
├── batter_value/                          # 子專案一：打者隔年 wRC+ 預測
│   ├── code/                              
│   ├── data/                              
│   ├── results/                           
│   └── docs/                              
├── Batted_Ball_Prediction/                # 子專案二：擊球結果分類
│   ├── NullModel.ipynb
│   ├── RandomForest.ipynb
│   ├── Xgboost.ipynb
│   └── README.md                          
└── Batter_type_clustering/                # 子專案三：打者型態分群
    ├── src/
    │   ├── data_preprocessing.py
    │   ├── clustering.py
    │   └── app.py
    ├── data/
    └── README.md                          
```

---

## 子專案簡介

### 子專案一：打者隔年 wRC+ 預測 [`batter_value/`](batter_value/)

**研究問題：** 能否僅用球員過去三個球季的打擊指標，預測其下一季 wRC+？

以 Statcast 為唯一資料來源，自行計算 K%、BB%、BABIP、wRC+ 等指標，比較 Random Forest、ElasticNet、LSTM 三種模型，並系統性探討「過程指標」vs「wRC+ 歷年軌跡」兩種特徵設計哲學。

| 模型 | 特徵 | Test 2024 MAE | R² |
|------|------|:---:|:---:|
| Null（猜均值） | — | 28.15 | −0.044 |
| Random Forest | 過程指標（3yr） | 26.89 | 0.019 |
| LSTM | wRC+ 軌跡（3yr） | 25.43 | 0.121 |
| **ElasticNet + 3× 加權** | **wRC+ 軌跡（3yr）** | **24.63** | **0.197** |

**核心發現：** 特徵選擇的影響遠大於模型選擇；使用 wRC+ 歷年軌跡作為特徵，配合 ElasticNet 線性模型，可達到最佳預測效果。

→ 詳細說明、執行方式與完整結果請見 [batter_value/README.md](batter_value/README.md)  
→ 方法論與圖表解讀請見 [batter_value/docs/methodology.md](batter_value/docs/methodology.md)  
→ 完整期末報告請見 [batter_value/docs/final_report.md](batter_value/docs/final_report.md)

---

### 子專案二：擊球結果分類 [`Batted_Ball_Prediction/`](Batted_Ball_Prediction/)

**研究問題：** 能否只根據擊球初速（`launch_speed`）與擊球仰角（`launch_angle`），預測場內球是否形成安打？

設計二分類（安打 vs 出局）與三分類（出局 / 一壘安打 / 長打）任務，比較 Null Model、Stratified Baseline、Random Forest、XGBoost 四種方法，並延伸分析 expected batting average（xBA）與「運氣因子」。

| 模型 | 二分類 ROC-AUC | 備註 |
|------|:---:|------|
| Most-Frequent Baseline | 0.500 | 永遠預測出局 |
| Stratified Random Baseline | 0.493 | 依比例隨機預測 |
| XGBoost | 0.826 | Gradient Boosting |
| **Random Forest** | **0.865** | 最佳 |

**核心發現：** Random Forest 達最佳 ROC-AUC，XGBoost 可輸出 xBA 機率供延伸分析；僅憑兩個物理特徵即可有意義地預測擊球結果，但方向、球場、守備位置等因素仍是限制。

→ 詳細說明與結果請見 [Batted_Ball_Prediction/README.md](Batted_Ball_Prediction/README.md)

---

### 子專案三：打者型態分群查詢系統 [`Batter_type_clustering/`](Batter_type_clustering/)

**研究問題：** 如何以資料驅動的方式客觀地將打者分型，並為球隊找出同型態的替代球員選項？

以 KMeans 對 2018–2024 合格打者進行分群（K 由 Elbow Method + Silhouette Score 決定），分為四種打擊型態，並以 Bootstrap 驗證分群穩定度。提供 Streamlit 網頁介面，支援歷年型態軌跡查詢與同年度相似打者推薦。

| Cluster | 型態名稱 | 說明 |
|---------|----------|------|
| 0 | 低效能打者 | 整體進攻產出偏低 |
| 1 | 全能強棒型 | 綜合打擊能力優異 |
| 2 | 高三振長打盲砲型 | 三振率高、長打爆發力強 |
| 3 | 高接觸槍兵型 | 選球紀律佳、接觸率高 |

**線上 Demo：** https://datasciencebasebballdataanalysis-wywvlrnqe73p8mvopkqcul.streamlit.app

→ 詳細說明、安裝方式與使用教學請見 [Batter_type_clustering/README.md](Batter_type_clustering/README.md)

---

## 共用資料來源

三個子專案均以 [Baseball Savant Statcast](https://baseballsavant.mlb.com/) 為核心資料來源，透過 [`pybaseball`](https://github.com/jldbc/pybaseball) 套件取得 2018–2024 年逐球事件資料（單季約 70–90 萬筆）。

原始 Statcast parquet 檔案體積較大（各季 ~150 MB），各子專案均已提供預處理後的較小資料檔，可直接執行主要分析，無需重新下載原始資料。詳見各子專案 README 的資料準備說明。

---

## Contributors

子專案 | 組員 | 系級 | 學號 | 工作分配 |
|------|------|------|------|----------|
|batter_value | 吳帛恩 | 統計三 | 112207433 | Statcast 資料 pipeline、EDA、Random Forest wRC+ 預測、Statcast 熱區視覺化 |
|batter_value | 彭珮蓉 | 資計碩一 | 114753210 | 訓練資料比較（2yr vs 3yr）、特徵哲學比較（過程指標 vs wRC+ 軌跡）、ElasticNet / LSTM 建模、多模型系統比較、極端球員加權改善 |
Batted_Ball_Prediction |
Batted_Ball_Prediction |
Batter_type_clustering | 謝螢嘉 | 廣告四 | 111405040 | 清洗資料、模型評估、模型應用發想、Presentation |
Batter_type_clustering | 
