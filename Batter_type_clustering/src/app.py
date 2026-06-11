import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# 設定網頁標題與排版
st.set_page_config(page_title="MLB 打者型態與替代球員查詢系統", layout="wide")
st.title("⚾ MLB 打者型態與替代球員查詢系統")
st.markdown("基於 2018-2024 數據分群結果，提供打者型態查詢與相似球員推薦。")

# ==========================================
# 1. 載入資料與定義核心計算函式
# ==========================================
@st.cache_data
def load_data():
    # 讀取你的分群資料集
    df = pd.read_csv('data/batting_stats_2018_2024_with_clusters.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ 找不到 `batting_stats_2018_2024_with_clusters.csv` 檔案，請確認路徑是否正確。")
    st.stop()

# 取得球員照片的函式（利用 MLB 官方的 Headshot API）
def get_player_headshot_url(player_id):
    if pd.isna(player_id):
        return None
    # MLB 官方頭像網址格式：使用 batter 欄位的 ID
    return f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.svg/w_213,q_auto:best/v1/people/{int(player_id)}/headshot/67/current"

def find_replacement_batters(df, target_name, target_year, top_n=10):
    similarity_features = ["AVG", "OBP", "SLG", "OPS", "ISO", "BB%", "K%", "wOBA", "wRC+"]
    
    work_df = df.copy()
    X = work_df[similarity_features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    scaled_cols = [col + "_scaled" for col in similarity_features]
    for i, col in enumerate(scaled_cols):
        work_df[col] = X_scaled[:, i]
    
    target_rows = work_df[
        (work_df["Name"].str.contains(target_name, case=False, na=False)) & 
        (work_df["Year"] == target_year)
    ]
    
    if len(target_rows) == 0:
        return None, "找不到這位打者在該年份的資料，請確認姓名或年份。"
        
    target = target_rows.iloc[0]
    target_cluster = target["cluster"]
    
    candidate_pool = work_df[
        (work_df["Year"] == target_year) & 
        (work_df["cluster"] == target_cluster) & 
        (work_df["Name"] != target["Name"])
    ].copy()
    
    if len(candidate_pool) == 0:
        return None, "沒有找到可比較的同分群候選打者。"
        
    target_vector = target[scaled_cols].values.reshape(1, -1)
    candidate_vectors = candidate_pool[scaled_cols].values
    
    candidate_pool["distance"] = pairwise_distances(
        candidate_vectors,
        target_vector,
        metric="euclidean"
    ).flatten()
    
    # 這裡把 batter 欄位也留著，方便未來需要時取用
    result_cols = [
        "Name", "Year", "type", "distance", 
        "AVG", "OBP", "SLG", "OPS", "ISO", 
        "BB%", "K%", "wOBA", "wRC+", "HR", "WAR"
    ]
    available_cols = [col for col in result_cols if col in candidate_pool.columns]
    
    return candidate_pool.sort_values("distance")[available_cols].head(top_n), None

def build_radar_figure(df, target_row, radar_features):
    year_df = df[df["Year"] == target_row["Year"]].copy()
    year_stats = year_df[radar_features].replace([np.inf, -np.inf], np.nan)
    year_stats = year_stats.fillna(year_stats.median(numeric_only=True))

    percentile_df = year_stats.rank(pct=True) * 100
    target_mask = year_df.index == target_row.name
    cluster_mask = year_df["cluster"] == target_row["cluster"]

    target_values = percentile_df[target_mask].iloc[0].tolist()
    cluster_avg_values = percentile_df[cluster_mask].mean().tolist()

    theta = radar_features + [radar_features[0]]
    target_values_closed = target_values + [target_values[0]]
    cluster_avg_closed = cluster_avg_values + [cluster_avg_values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=target_values_closed,
        theta=theta,
        fill="toself",
        name=f"{target_row['Name']}（百分位）",
        line=dict(color="#1f77b4")
    ))
    fig.add_trace(go.Scatterpolar(
        r=cluster_avg_closed,
        theta=theta,
        fill="toself",
        name="同分群平均（百分位）",
        line=dict(color="#ff7f0e")
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        title="打者打擊指標雷達圖（同年度百分位）",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=30, t=70, b=30)
    )
    return fig

# ==========================================
# 2. 側邊欄：使用者輸入介面
# ==========================================
st.sidebar.header("🔍 查詢條件設定")

all_players = sorted(df["Name"].unique())
input_method = st.sidebar.radio("選擇姓名輸入方式：", ["從下拉選單選取", "自行輸入名字"])

if input_method == "從下拉選單選取":
    default_index = all_players.index("Shohei Ohtani") if "Shohei Ohtani" in all_players else 0
    player_input = st.sidebar.selectbox("選擇打者姓名：", all_players, index=default_index)
else:
    player_input = st.sidebar.text_input("輸入打者姓名（可輸入關鍵字，如 Trout）：", value="Mike Trout")

# ==========================================
# 功能 1：輸入「打者名字」輸出「年度，型態」+ 顯示照片
# ==========================================
st.header("📋 功能一：打者歷年型態軌跡")

if player_input:
    # 模糊搜尋球員
    matched_players = df[df["Name"].str.contains(player_input, case=False, na=False)]
    
    if len(matched_players) > 0:
        exact_name = matched_players["Name"].iloc[0]
        
        # 建立左右兩欄，左邊放照片，右邊放歷年型態表格
        col_img, col_table = st.columns([1, 4])
        
        with col_img:
            # 取得該球員的官方 ID 並撈取頭像
            p_id = matched_players["batter"].iloc[0]
            img_url = get_player_headshot_url(p_id)
            if img_url:
                st.image(img_url, width=150, caption=exact_name)
        
        with col_table:
            st.success(f"找到「{exact_name}」的歷年紀錄：")
            history_df = matched_players[["Name", "Year", "type"]].sort_values("Year")
            display_history = history_df.rename(columns={"Year": "年度", "type": "打擊型態", "Name": "球員姓名"})
            st.dataframe(display_history, use_container_width=True, hide_index=True)
        
        available_years = sorted(matched_players["Year"].unique(), reverse=True)
    else:
        st.warning(f"❌ 找不到包含「{player_input}」的球員，請重新輸入。")
        available_years = sorted(df["Year"].unique(), reverse=True)
else:
    available_years = sorted(df["Year"].unique(), reverse=True)

st.markdown("---")

# ==========================================
# 功能 2：輸入「打者名字、年度」輸出「打者型態、可替代的選手」
# ==========================================
st.header("🔄 功能二：尋找同年度相似替代球員")

target_year = st.sidebar.selectbox("選擇查詢年度：", available_years)

if player_input:
    player_year_data = df[
        (df["Name"].str.contains(player_input, case=False, na=False)) & 
        (df["Year"] == target_year)
    ]
    
    if len(player_year_data) > 0:
        player_row = player_year_data.iloc[0]
        p_name = player_row["Name"]
        p_type = player_row["type"]
        
        st.info(f"💡 基準打者：**{p_name}** ({target_year}年) | 打擊型態：**{p_type}**")

        radar_features = ["AVG", "OBP", "SLG", "OPS", "ISO", "BB%", "K%", "wOBA", "wRC+"]
        available_radar_features = [col for col in radar_features if col in df.columns]
        if available_radar_features:
            st.subheader("📈 打者數據雷達圖")
            radar_fig = build_radar_figure(df, player_row, available_radar_features)
            st.plotly_chart(radar_fig, use_container_width=True)

            with st.expander("📘 數據欄位解釋（點擊展開）", expanded=False):
                st.markdown("""
                - **AVG（打擊率）**：安打數 / 打數，衡量打者把球打成安打的能力。  
                - **OBP（上壘率）**：包含安打、保送、觸身球等方式上壘的比例。  
                - **SLG（長打率）**：以壘打數衡量打擊破壞力，長打越多通常越高。  
                - **OPS**：`OBP + SLG`，常用的整體打擊貢獻指標。  
                - **ISO（純長打率）**：`SLG - AVG`，反映額外壘打能力。  
                - **BB%（保送率）**：打席中選到保送的比例，代表選球與上壘紀律。  
                - **K%（三振率）**：打席中被三振的比例（通常越低越好）。  
                - **wOBA（加權上壘率）**：對不同上壘事件給不同權重，較完整衡量進攻價值。  
                - **wRC+（標準化得分創造）**：以聯盟平均 100 為基準，`120` 約代表高於聯盟平均 20%。  

                雷達圖使用的是**同年度百分位**（0-100），可直接比較不同尺度指標。
                """)
        
        recommend_df, error_msg = find_replacement_batters(df, p_name, target_year, top_n=10)
        
        if error_msg:
            st.error(error_msg)
        elif recommend_df is not None:
            st.subheader(f"🔥 前 10 位最相似的「{p_type}」替代選手推薦：")
            recommend_df = recommend_df.rename(columns={
                "Name": "球員姓名", "Year": "年度", "type": "型態", 
                "distance": "相似距離 (越小越近)", "HR": "全壘打"
            })
            st.dataframe(recommend_df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"⚠️ 該球員「{player_input}」在 {target_year} 年沒有出賽或無數據紀錄。")