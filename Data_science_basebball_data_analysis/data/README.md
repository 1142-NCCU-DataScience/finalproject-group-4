# MLB 打者型態與替代球員查詢系統

基於 2018–2024 年 MLB Statcast 逐球資料，運用 KMeans 分群將打者分類為四種打擊型態，並提供 Streamlit 網頁介面，支援打者歷年型態查詢與同年度相似球員推薦。

## 功能特色

- **資料前處理**：從 Statcast parquet 逐球資料彙整打者年度打擊統計（AVG、OBP、SLG、wOBA、wRC+ 等）
- **打者分群**：以 KMeans 將打者分為 4 群，搭配 Elbow Method、Silhouette Score 選 K，並以 Bootstrap 驗證分群穩定度
- **網頁查詢**：
  - 查詢打者 2018–2024 歷年打擊型態軌跡
  - 依年度與分群，推薦前 10 位最相似的替代打者
  - 互動式雷達圖（同年度百分位比較）
  - MLB 官方球員頭像顯示

## 打者型態分類

| Cluster | 型態名稱 | 說明 |
|---------|----------|------|
| 0 | 低效能打者 | 整體進攻產出偏低 |
| 1 | 全能強棒型 | 綜合打擊能力優異 |
| 2 | 高三振長打盲砲型 | 三振率高、長打爆發力強 |
| 3 | 高接觸槍兵型 | 選球紀律佳、接觸率高 |

## 專案結構

```
Data_science_basebball_data_analysis/
├── app.py                          # Streamlit 網頁應用
├── data_preprocessing.py           # Statcast → 打擊統計 CSV
├── clustering.py                   # KMeans 分群與穩定度驗證
├── requirements.txt                # Python 依賴套件
└── data/
    ├── statcast_2018.parquet       # 原始 Statcast 逐球資料
    ├── statcast_2019.parquet
    ├── statcast_2020.parquet
    ├── statcast_2021.parquet
    ├── statcast_2022.parquet
    ├── statcast_2023.parquet
    ├── statcast_2024.parquet
    ├── clustering_batting_stats_2018_2024.csv      # 前處理產出
    └── batting_stats_2018_2024_with_clusters.csv     # 分群產出（供 app 使用）
```

## 資料流程

```
statcast_*.parquet
       │
       ▼
data_preprocessing.py  →  clustering_batting_stats_2018_2024.csv
       │
       ▼
clustering.py          →  batting_stats_2018_2024_with_clusters.csv
       │
       ▼
streamlit run app.py   →  網頁查詢介面
```

## 環境需求

- Python 3.9+
- 建議使用虛擬環境

## 安裝

```bash
# 克隆專案
git clone https://github.com/<your-username>/Data_science_basebball_data_analysis.git
cd Data_science_basebball_data_analysis

# 建立並啟用虛擬環境
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 安裝依賴
pip install -r requirements.txt
```

## 使用方式

### 1. 資料前處理（首次或需重新計算時）

將 2018–2024 年的 Statcast parquet 檔案放入 `data/` 目錄，檔名需為 `statcast_YYYY.parquet`。

```bash
python data_preprocessing.py
```

產出：`data/clustering_batting_stats_2018_2024.csv`

> 若該 CSV 已存在，腳本會直接載入快取，跳過重新計算。若要強制重算，請先刪除該檔案。

### 2. 打者分群

```bash
python clustering.py
```

產出：`data/batting_stats_2018_2024_with_clusters.csv`

此步驟會執行 KMeans 分群、繪製 Elbow / Silhouette / PCA 圖表，並進行 100 次 Bootstrap 穩定度驗證。

### 3. 啟動網頁應用

```bash
streamlit run app.py
```

瀏覽器會自動開啟本地頁面（預設 `http://localhost:8501`）。

## 分群特徵

分群使用以下 8 項比例型 / 打擊型態特徵（不含 G、PA、AB 等出賽機會指標）：

- AVG、OBP、SLG、ISO
- BB%、K%
- wOBA、wRC+

相似球員推薦則在**同一分群、同一年度**內，以標準化後的歐氏距離計算相似度。

## 主要依賴

| 套件 | 用途 |
|------|------|
| streamlit | 網頁介面 |
| pandas / numpy | 資料處理 |
| scikit-learn | KMeans、PCA、StandardScaler |
| plotly | 互動式雷達圖 |
| pybaseball | MLB 球員 ID 對照姓名 |
| pyarrow | 讀取 parquet 格式 |
| matplotlib | 分群分析視覺化 |

## 注意事項

- Statcast parquet 檔案體積較大，前處理需一定記憶體與時間
- `data_preprocessing.py` 執行時會 `os.chdir('data')`，請從專案根目錄執行
- 篩選條件：僅保留單季 PA ≥ 100 的打者
- 若缺少某年度 parquet，該年資料會被跳過並顯示警告

## 授權

本專案供學術與個人研究使用。Statcast 資料來源為 MLB Advanced Media。
