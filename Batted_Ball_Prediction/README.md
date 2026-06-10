# MLB 擊球結果預測專案

本專案是資料科學期末專案，目標是使用 MLB Statcast 擊球追蹤資料，預測球被擊進場內後的結果。專案核心問題如下：

> 能否只根據擊球初速與擊球仰角，預測擊出的球是否會形成安打？

本專案比較了簡單 baseline model 與兩種 tree-based models：Random Forest 與 XGBoost。實驗包含二分類任務與三分類任務。

---

## 專案概述

當打者把球打進場內時，兩個物理特徵與擊球結果高度相關：

* `launch_speed`：擊球初速，也就是 exit velocity
* `launch_angle`：擊球仰角，也就是球被擊出時的垂直角度

本專案使用這兩個特徵建立分類模型，並設計兩種預測任務。

---

## 任務一：二分類預測

二分類任務的目標是預測擊球結果是否為安打。

| Label | 意義       | 對應事件                                                               |
| ----- | -------- | ------------------------------------------------------------------ |
| `0`   | 出局 / 非安打 | field out、force out、double play、sacrifice、error、fielder's choice 等 |
| `1`   | 安打       | single、double、triple、home run                                      |

在二分類任務中，`catcher_interf` 被移除，因為它不是正常的擊球結果。

---

## 任務二：三分類預測

三分類任務進一步將安打拆分為一壘安打與長打。

| Label | 意義       | 對應事件                   |
| ----- | -------- | ---------------------- |
| `0`   | 出局 / 非安打 | 所有非安打結果                |
| `1`   | 一壘安打     | single                 |
| `2`   | 長打       | double、triple、home run |

---

## Repository Structure

```text
Baseball-Project/
├── NullModel.ipynb        # Baseline models：永遠預測出局、依比例隨機預測
├── RandomForest.ipynb     # Random Forest 二分類與三分類模型
├── Xgboost.ipynb          # XGBoost 二分類與三分類模型
├── README.md              # 專案說明文件
└── .gitignore
```

---

## 資料來源

本專案資料來自 MLB Statcast，並透過 `pybaseball` 套件下載。

```python
from pybaseball import statcast

df_pitch_by_pitch = statcast(start_dt="2025-03-01", end_dt="2025-11-30")
```

本專案只保留球被擊進場內的資料：

```python
df_in_play = df_pitch_by_pitch[
    df_pitch_by_pitch["description"] == "hit_into_play"
]
```

原始 in-play dataset 共有 134,759 筆擊球事件。移除 `launch_speed` 與 `launch_angle` 缺失值後，二分類資料集共有 131,637 筆有效資料。

---

## 資料分布

### 二分類資料分布

| Class | 意義       |     比例 |
| ----- | -------- | -----: |
| `0`   | 出局 / 非安打 | 67.77% |
| `1`   | 安打       | 32.23% |

### 三分類資料分布

| Class | 意義       |     筆數 |     比例 |
| ----- | -------- | -----: | -----: |
| `0`   | 出局 / 非安打 | 89,222 | 67.77% |
| `1`   | 一壘安打     | 27,574 | 20.95% |
| `2`   | 長打       | 14,852 | 11.28% |

從資料分布可以看出，本專案存在明顯的 class imbalance。出局結果佔多數，因此若只看 accuracy，模型可能會因為大量預測出局而得到看似不錯的準確率，但實際上無法有效辨識安打。

---

## 使用模型

本專案比較四種模型或預測方法。

---

### 1. Most-Frequent Null Model

Most-Frequent Null Model 是最簡單的 baseline。它永遠預測訓練集中數量最多的類別。

在本資料集中，最多的類別是出局 / 非安打。因此這個模型在二分類與三分類任務中都會永遠預測為出局。

這個模型提供一個最低標準，用來檢查其他模型是否真的學到了有意義的規則。

---

### 2. Stratified Random Baseline

Stratified Random Baseline 會根據訓練集中各類別的比例進行隨機預測。

例如在二分類任務中，若訓練集中約 68% 是出局、32% 是安打，模型就會用大約相同的比例隨機產生預測結果。

這個 baseline 可以幫助判斷模型是否只是受資料比例影響，而沒有真正學到 `launch_speed` 與 `launch_angle` 和擊球結果之間的關係。

---

### 3. Random Forest

Random Forest 是由多棵 decision trees 組成的 ensemble model。每棵樹會根據不同的 bootstrap samples 進行訓練，最後再透過投票或平均機率產生預測結果。

Random Forest 適合本專案的原因是：

* 擊球初速、擊球仰角與擊球結果之間的關係並非線性；
* Random Forest 可以捕捉 nonlinear decision boundary；
* 對 tabular data 表現穩定；
* 不需要事先假設特徵與目標之間的函數形式；
* 可以搭配 `class_weight="balanced"` 處理資料不平衡問題。

本專案對 Random Forest 使用：

* `GridSearchCV`
* `StratifiedKFold`
* `class_weight="balanced"`
* ROC-AUC 作為主要調參指標
* validation set 進行 threshold tuning

---

### 4. XGBoost

XGBoost 是 gradient boosting tree model。與 Random Forest 不同，Random Forest 的樹通常是彼此獨立訓練，而 XGBoost 則是逐步訓練多棵樹，讓後面的樹修正前面模型犯的錯誤。

本專案使用 XGBoost 的原因是：

* XGBoost 是 tabular data 上常見且強大的模型；
* 能捕捉 `launch_speed` 與 `launch_angle` 之間的非線性交互作用；
* 內建 regularization，可降低 overfitting 風險；
* 支援 class imbalance 的處理方式；
* 使用 `tree_method="hist"` 可加速訓練；
* 可以輸出機率，進一步解釋為預測安打機率。

雖然在本專案目前結果中，XGBoost 沒有超越 Random Forest，但它仍然是一個重要的比較模型，因為它代表另一種常見的 tree-based ensemble learning 方法。

---

## 實驗設計

本專案使用 stratified train / validation / test split，使各資料切分中的類別比例盡可能維持一致。

資料用途如下：

* training set：訓練模型；
* validation set：調整 threshold；
* test set：最終模型評估。

固定 random seed：42

評估指標包含：

* ROC-AUC
* Accuracy
* Precision
* Recall
* F1-score
* Macro F1-score
* Weighted F1-score
* Confusion Matrix

三分類任務中的 ROC-AUC 使用 one-vs-rest 計算：

```python
roc_auc_score(
    y_test,
    y_test_proba,
    multi_class="ovr",
    labels=[0, 1, 2]
)
```

---

## 評估指標說明

### Accuracy

Accuracy 表示模型整體預測正確的比例。

然而在本專案中，accuracy 不能單獨作為主要判斷標準。因為出局樣本佔多數，模型只要大量預測出局，就可能得到不低的 accuracy。

---

### Precision

Precision 表示模型預測為某一類時，有多少比例是真的屬於該類。

以安打類別為例：

> Precision 越高，代表模型預測為安打時，真的形成安打的比例越高。

---

### Recall

Recall 表示真實屬於某一類的樣本中，有多少比例被模型成功找出來。

以安打類別為例：

> Recall 越高，代表真實安打中有越多被模型成功辨識出來。

---

### F1-score

F1-score 是 precision 與 recall 的 harmonic mean，用來平衡模型的精確度與召回能力。

---

### Macro Average

Macro average 會先分別計算每個類別的指標，再取平均。每個類別權重相同，因此適合觀察模型是否公平地處理少數類別。

---

### Weighted Average

Weighted average 會依照每個類別的 support，也就是樣本數，加權平均各類別表現。當資料不平衡時，weighted average 會比較受到多數類別影響。

---

### ROC-AUC

ROC-AUC 衡量模型把 positive class 排在 negative class 前面的能力。它使用 predicted probability，而不是 hard prediction。

在二分類任務中，ROC-AUC 越高，代表模型越能區分安打與非安打。

在三分類任務中，本專案使用 one-vs-rest ROC-AUC，分別將每一類視為 positive class，其餘類別視為 negative class，再進行整體評估。

---

<!-- ## 實驗結果

### 二分類結果

| Model                      | ROC-AUC | Accuracy | Macro F1 | Weighted F1 | 說明               |
| -------------------------- | ------: | -------: | -------: | ----------: | ---------------- |
| Most-Frequent Baseline     |  0.5000 |     0.68 |     0.40 |        0.55 | 永遠預測出局           |
| Stratified Random Baseline |  0.4927 |     0.56 |     0.49 |        0.56 | 依類別比例隨機預測        |
| Random Forest              |  0.8647 |   0.7905 |     0.77 |        0.79 | threshold = 0.58 |
| XGBoost                    |  0.8258 |   0.7752 |     0.75 |        0.78 | threshold = 0.41 |

---

### Random Forest 二分類 Classification Report

| Class    | Precision | Recall | F1-score | Support |
| -------- | --------: | -----: | -------: | ------: |
| 出局 / 非安打 |      0.86 |   0.82 |     0.84 |  17,843 |
| 安打       |      0.66 |   0.73 |     0.69 |   8,485 |

---

### XGBoost 二分類 Classification Report

| Class    | Precision | Recall | F1-score | Support |
| -------- | --------: | -----: | -------: | ------: |
| 出局 / 非安打 |      0.84 |   0.82 |     0.83 |  17,843 |
| 安打       |      0.64 |   0.68 |     0.66 |   8,485 |

---

### 三分類結果

| Model                      | ROC-AUC OVR | Accuracy | Macro F1 | Weighted F1 |
| -------------------------- | ----------: | -------: | -------: | ----------: |
| Most-Frequent Baseline     |      0.5000 |     0.68 |     0.27 |        0.55 |
| Stratified Random Baseline |      0.5022 |     0.52 |     0.34 |        0.52 |
| Random Forest              |      0.8759 |     0.70 |     0.65 |        0.72 |
| XGBoost                    |      0.8375 |     0.66 |     0.61 |        0.67 |

---

### Random Forest 三分類 Classification Report

| Class    | Precision | Recall | F1-score | Support |
| -------- | --------: | -----: | -------: | ------: |
| 出局 / 非安打 |      0.91 |   0.68 |     0.78 |  17,843 |
| 一壘安打     |      0.50 |   0.79 |     0.61 |   5,515 |
| 長打       |      0.47 |   0.68 |     0.56 |   2,970 |

---

### XGBoost 三分類 Classification Report

| Class    | Precision | Recall | F1-score | Support |
| -------- | --------: | -----: | -------: | ------: |
| 出局 / 非安打 |      0.88 |   0.63 |     0.74 |  17,845 |
| 一壘安打     |      0.46 |   0.70 |     0.56 |   5,515 |
| 長打       |      0.41 |   0.73 |     0.53 |   2,970 |

--- -->

<!-- ## 最佳參數

### Random Forest：二分類

```python
{
    "max_depth": 10,
    "min_samples_leaf": 10,
    "min_samples_split": 2,
    "n_estimators": 500
}
```

---

### Random Forest：三分類

```python
{
    "max_depth": 10,
    "min_samples_leaf": 5,
    "min_samples_split": 2,
    "n_estimators": 500
}
```

---

### XGBoost：二分類

```python
{
    "colsample_bytree": 0.85,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_child_weight": 20,
    "n_estimators": 600,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,
    "subsample": 0.85
}
```

---

### XGBoost：三分類

```python
{
    "colsample_bytree": 0.85,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 10,
    "n_estimators": 600,
    "reg_alpha": 0.0,
    "reg_lambda": 5.0,
    "subsample": 0.85
}
``` -->

---

## 結果分析

Baseline models 顯示，accuracy 在資料不平衡的情況下可能具有誤導性。

在二分類任務中，Most-Frequent Baseline 永遠預測出局，仍然可以得到約 68% accuracy。然而，它完全無法辨識安打，因此安打類別的 recall 與 F1-score 都是 0。

Random Forest 在目前實驗中表現最好。它在二分類與三分類任務中都取得最高的 ROC-AUC，表示它能有效利用 `launch_speed` 與 `launch_angle` 區分不同擊球結果。

XGBoost 的表現也明顯優於 baseline models。雖然在目前設定下沒有超越 Random Forest，但它仍然提供了一個強而有力的 regularized boosting benchmark。XGBoost 的機率輸出也可以進一步用於 expected batting average 類型的分析。

三分類任務比二分類任務更困難。二分類只需要判斷安打或非安打；三分類則需要進一步區分一壘安打與長打。然而，單靠擊球初速與擊球仰角，仍難以完整判斷擊球結果，因為結果也受到擊球方向、防守站位、球場因素、跑者狀況與運氣影響。

---

## xBA 與 Luck Factor 分析

二分類模型會輸出每一球成為安打的預測機率。這個機率可以被視為一種簡化版的 expected batting average，也就是 `xBA`。

```python
xBA = model.predict_proba(X_test)[:, 1]
```

本專案也定義一個簡單的 luck factor：

```python
Luck_Factor = Actual_Result - xBA
```

解釋方式如下：

* `Luck_Factor > 0`：實際形成安打，但模型預測安打機率較低，代表可能是較幸運的安打。
* `Luck_Factor < 0`：實際出局，但模型預測安打機率較高，代表可能是不幸的出局。
* `Luck_Factor` 接近 0：實際結果與模型預期較接近。

這個分析可以用來找出某些擊球結果是否與其擊球品質不一致。

---

## 如何執行專案

### 1. Clone Repository

```bash
git clone https://github.com/gordon921212/Baseball-Project.git
cd Baseball-Project
```

---

### 2. 安裝套件

```bash
pip install pandas numpy matplotlib scikit-learn pybaseball xgboost jupyter
```

---

### 3. 執行 Notebook

建議執行順序如下：

```text
1. NullModel.ipynb
2. RandomForest.ipynb
3. Xgboost.ipynb
```

Statcast 資料下載可能需要較長時間。為了避免重複下載，可以先啟用 `pybaseball` cache：

```python
import pybaseball
pybaseball.cache.enable()
```

---

## Reproducibility Notes

本專案使用以下方式提升可重現性：

* 使用 `random_state=42` 固定資料切分與模型訓練結果；
* 使用 stratified splitting 維持不同資料集中的類別比例；
* ROC-AUC 使用 predicted probability 計算，而不是 hard prediction；
* validation set 用於 threshold tuning；
* test set 僅用於最終模型評估。

需要注意的是，本專案透過 `pybaseball` 即時下載 Statcast 資料。如果 Statcast 後續更新資料，實驗結果可能會有些微差異。若要完全重現結果，建議將清理後的資料另存為 CSV。

---

## 專案限制

本專案只使用兩個核心擊球特徵：`launch_speed` 與 `launch_angle`。這讓模型具有較高的可解釋性，但也限制了預測能力。

重要但未納入的因素包括：

* spray angle / 擊球方向；
* hit distance / 擊球距離；
* batter identity / 打者身分；
* pitcher identity / 投手身分；
* batter handedness / 打者慣用手；
* pitcher handedness / 投手慣用手；
* pitch type / 球種；
* pitch location / 投球位置；
* ballpark effects / 球場因素；
* defensive alignment / 防守站位；
* game situation / 比賽情境；
* weather and environmental conditions / 天氣與環境因素。

因此，本專案模型應被理解為簡化版的 batted-ball quality model，而不是完整的比賽結果預測模型。

---

## 未來改進方向

未來可以從以下方向改進：

* 加入更多 Statcast 特徵，例如 `hit_distance_sc`、`estimated_ba_using_speedangle`、`bb_type` 與 spray direction；
* 使用 time-based train/test split，更貼近真實的未來預測情境；
* 比較更多模型，例如 Logistic Regression、SVM、LightGBM 或 CatBoost；
* 使用 Platt scaling 或 isotonic regression 校準預測機率；
* 使用 SHAP values 或 partial dependence plots 解釋模型決策；
* 建立互動式 dashboard，視覺化不同擊球初速與擊球仰角下的預測安打機率。

---

## 專案總結

本專案展示了即使只使用少量 Statcast 擊球特徵，也能對擊球結果進行有意義的預測。

Baseline models 幫助我們理解 class imbalance 對模型評估的影響；Random Forest 與 XGBoost 則揭示了擊球初速、擊球仰角與安打機率之間的非線性關係。

在目前實驗中，Random Forest 取得最佳整體表現，而 XGBoost 則提供了一個具有 regularization 的 boosting benchmark，並能輸出可用於 xBA-style analysis 的預測機率。
