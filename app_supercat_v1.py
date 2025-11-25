import streamlit as st
import pandas as pd
import pyodbc
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(layout="wide", page_title="Category Sales Dashboard")

# ---------------------------------------------
# 0) CSS: 콤보박스 크기 확대 (1.5배) + 탭 색상 + 헤더 폰트 85%
# ---------------------------------------------
st.markdown("""
<style>
/* 콤보박스 크기 확대 */
div[data-baseweb="select"] > div {
    height: 48px !important;
}
div[data-baseweb="select"] span {
    font-size: 1.2rem !important;
}

/* 탭 텍스트 색상 진한 파랑 + 굵게 */
.stTabs button[role="tab"] {
    color: #004c99 !important;
    font-weight: 600 !important;
}

/* 선택된 탭은 아래 테두리로 강조 */
.stTabs button[role="tab"][aria-selected="true"] {
    border-bottom: 3px solid #004c99 !important;
}

/* 전체 헤더(h1/h2/h3) 폰트 크기를 기존의 약 85% 수준으로 축소 */
.block-container h1,
.block-container h2,
.block-container h3 {
    font-size: 85% !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------
# 1) 로그인
# -------------------------
def login():
    # 로그인 타이틀/부제목 스타일
    st.markdown("""
    <style>
    .login-title {
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 60px;
        margin-bottom: 0.75rem;
    }
    .login-subtitle {
        text-align: center;
        font-size: 0.9rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-title">대시보드 로그인</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">사내 계정으로 로그인해주세요</div>', unsafe_allow_html=True)

    # 가운데 컬럼 안에 카드(컨테이너)를 만들고 그 안에 ID/PW/버튼 배치
    left, center, right = st.columns([1, 1, 1])
    with center:
        with st.container(border=True):
            st.write("")  # 위 여백

            user_id = st.text_input("ID", key="login_id")
            password = st.text_input("PW", type="password", key="login_pw")
            login_btn = st.button("로그인", use_container_width=True)

    if login_btn:
        try:
            server = st.secrets["db"]["host"]
            database = st.secrets["db"]["database"]
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server},{st.secrets['db']['port']};"
                f"DATABASE={database};"
                f"UID={st.secrets['db']['user']};PWD={st.secrets['db']['password']}"
            )
            conn = pyodbc.connect(conn_str)
            df_user = pd.read_sql("SELECT USERNAME, USERPASSWORD FROM USER_TABLE", conn)
            conn.close()

            if ((df_user['USERNAME'] == user_id) &
                (df_user['USERPASSWORD'] == password)).any():
                st.session_state['login'] = True
                st.rerun()
            else:
                st.error("로그인 실패: ID 또는 비밀번호가 올바르지 않습니다.")

        except Exception as e:
            st.error(f"DB 연결 오류: {e}")


if 'login' not in st.session_state:
    st.session_state['login'] = False

if not st.session_state['login']:
    login()
    st.stop()


# -------------------------
# 2) DB 연결
# -------------------------
server = st.secrets["db"]["host"]
database = st.secrets["db"]["database"]
conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={server},{st.secrets['db']['port']};"
    f"DATABASE={database};"
    f"UID={st.secrets['db']['user']};PWD={st.secrets['db']['password']}"
)
conn = pyodbc.connect(conn_str)


# -------------------------
# 3) 상단 로고 + 필터 (PROMPT 테이블 사용)
# -------------------------

# 로고 경로 탐색: 1) .py와 같은 폴더 2) .streamlit 폴더
try:
    base_dir = Path(__file__).parent
except NameError:
    base_dir = Path(".")

logo_path = None
for p in [base_dir / "marketlink_log.png",
          base_dir / ".streamlit" / "marketlink_log.png"]:
    if p.is_file():
        logo_path = str(p)
        break

if logo_path:
    st.image(logo_path, width=200)

st.subheader("데이터 필터")

# ▶ SUPER_CAT_PROCESS_CHANNEL_MANUF_PROMPT 테이블에서만 조회
category_levels = pd.read_sql(
    """
    SELECT DISTINCT CATEGORY_LEVEL
    FROM SUPER_CAT_PROCESS_CHANNEL_MANUF_PROMPT
    ORDER BY CATEGORY_LEVEL DESC
    """,
    conn
)["CATEGORY_LEVEL"].tolist()

channels = pd.read_sql(
    """
    SELECT DISTINCT MARKET_TYPE_NAME
    FROM SUPER_CAT_PROCESS_CHANNEL_MANUF_PROMPT
    ORDER BY MARKET_TYPE_NAME
    """,
    conn
)["MARKET_TYPE_NAME"].tolist()

col1, col2, col3, col4 = st.columns([1.2, 4.0, 1.2, 1.6])

with col1:
    category_level = st.selectbox("카테고리 레벨", category_levels)

# 선택된 카테고리 레벨에 해당하는 카테고리 그룹 조회 (PROMPT 테이블)
query_groups = """
SELECT DISTINCT CATEGORY_NAME
FROM SUPER_CAT_PROCESS_CHANNEL_MANUF_PROMPT
WHERE CATEGORY_LEVEL = ?
ORDER BY CATEGORY_NAME
"""
category_groups = pd.read_sql(query_groups, conn, params=[category_level])["CATEGORY_NAME"].tolist()

with col2:
    category_group = st.selectbox("카테고리 그룹", category_groups)

with col3:
    channel = st.selectbox("채널", channels)

with col4:
    top_n_label = st.selectbox("Top 제조사", ["Top 5", "Top 10", "Top 20"], index=0)
    top_n = int(top_n_label.replace("Top ", ""))

if st.button("조회"):
    st.session_state['run_query'] = True

if 'run_query' not in st.session_state or not st.session_state['run_query']:
    conn.close()
    st.stop()


# -------------------------
# 4) 데이터 조회 (카테고리별 판매 추이용)
# -------------------------
query_data = """
SELECT CATEGORY_LEVEL, CATEGORY_CODE, CATEGORY_NAME,
       MONTHLY, QUARTERLY, YEARLY,
       MARKET_TYPE_NAME, AMOUNT_P, MANUF
FROM SUPER_CAT_PROCESS_CHANNEL_MANUF
WHERE CATEGORY_LEVEL = ?
AND CATEGORY_NAME = ?
AND MARKET_TYPE_NAME = ?
"""

df = pd.read_sql(query_data, conn, params=[category_level, category_group, channel])

# -------------------------
# 4-2) 채널별 데이터 조회 (채널 필터 없음, 합계 채널 제외)
# -------------------------
query_channel_data = """
SELECT CATEGORY_LEVEL, CATEGORY_CODE, CATEGORY_NAME,
       MONTHLY, QUARTERLY, YEARLY,
       MARKET_TYPE_NAME, AMOUNT_P, MANUF
FROM SUPER_CAT_PROCESS_CHANNEL_MANUF
WHERE CATEGORY_LEVEL = ?
AND CATEGORY_NAME = ?
AND MARKET_TYPE_NAME NOT IN ('1. 온라인+오프라인', '3. 오프라인')
"""

df_channel = pd.read_sql(query_channel_data, conn, params=[category_level, category_group])

conn.close()

if df.empty:
    st.warning("선택한 조건의 데이터가 없습니다.")
    st.stop()


# -------------------------
# 5) 데이터 전처리
# -------------------------
df["MANUF"] = df["MANUF"].astype(str).str.strip()

# YEARLY / QUARTERLY 에 NULL 있을 수 있으므로 안전하게 dropna 후 집계
df_year = (
    df.dropna(subset=["YEARLY"])
      .groupby("YEARLY", as_index=False)["AMOUNT_P"]
      .sum()
      .sort_values("YEARLY")
)
df_q = (
    df.dropna(subset=["QUARTERLY"])
      .groupby("QUARTERLY", as_index=False)["AMOUNT_P"]
      .sum()
      .sort_values("QUARTERLY")
)
df_m = df.groupby("MONTHLY", as_index=False)["AMOUNT_P"].sum().sort_values("MONTHLY")

df_year_plot = df_year.copy()
df_year_plot["AMOUNT_P"] /= 1_000_000
df_q_plot = df_q.copy()
if not df_q_plot.empty:
    df_q_plot["AMOUNT_P"] /= 1_000_000
df_m_plot = df_m.copy()
df_m_plot["AMOUNT_P"] /= 1_000_000

# 제조사 집계 (월 기준)
df_manuf_month = (
    df.groupby(["MANUF", "MONTHLY"], as_index=False)["AMOUNT_P"]
      .sum()
      .sort_values("MONTHLY")
)

df_top_candidates = df_manuf_month[df_manuf_month["MANUF"] != "기타"]
top_list = (
    df_top_candidates.groupby("MANUF")["AMOUNT_P"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
)

df_top = df_manuf_month[df_manuf_month["MANUF"].isin(top_list)]
df_others = df_manuf_month[~df_manuf_month["MANUF"].isin(top_list)]
df_others_grp = (
    df_others.groupby("MONTHLY", as_index=False)["AMOUNT_P"]
            .sum()
            .sort_values("MONTHLY")
)
df_others_grp["MANUF"] = "기타"

df_manuf_final = pd.concat([df_top, df_others_grp], ignore_index=True).sort_values("MONTHLY")

# 제조사 컬러맵
unique_manufs = sorted(df_manuf_final["MANUF"].unique())
color_sequence = px.colors.qualitative.Set2
color_map = {manuf: color_sequence[i % len(color_sequence)] for i, manuf in enumerate(unique_manufs)}

# "기타"는 항상 회색으로 고정
if "기타" in color_map:
    color_map["기타"] = "#B0B0B0"


# -------------------------
# 6) 제조사 시장점유율 계산 (YEARLY/QUARTERLY는 NaN 제거)
# -------------------------
df_manuf_year = (
    df.dropna(subset=["YEARLY"])
      .groupby(["MANUF", "YEARLY"], as_index=False)["AMOUNT_P"]
      .sum()
      .sort_values("YEARLY")
)
df_manuf_quarter = (
    df.dropna(subset=["QUARTERLY"])
      .groupby(["MANUF", "QUARTERLY"], as_index=False)["AMOUNT_P"]
      .sum()
      .sort_values("QUARTERLY")
)

df_top_candidates_year = df_manuf_year[df_manuf_year["MANUF"] != "기타"]
top_list_year = (
    df_top_candidates_year.groupby("MANUF")["AMOUNT_P"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
)

df_top_candidates_quarter = df_manuf_quarter[df_manuf_quarter["MANUF"] != "기타"]
top_list_quarter = (
    df_top_candidates_quarter.groupby("MANUF")["AMOUNT_P"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
)

# 년도별 제조사 데이터
df_top_year = df_manuf_year[df_manuf_year["MANUF"].isin(top_list_year)]
df_others_year = df_manuf_year[~df_manuf_year["MANUF"].isin(top_list_year)]
df_others_year_grp = (
    df_others_year.groupby("YEARLY", as_index=False)["AMOUNT_P"]
                  .sum()
                  .sort_values("YEARLY")
)
df_others_year_grp["MANUF"] = "기타"
df_manuf_year_final = pd.concat([df_top_year, df_others_year_grp], ignore_index=True).sort_values("YEARLY")

# 분기별 제조사 데이터
df_top_quarter = df_manuf_quarter[df_manuf_quarter["MANUF"].isin(top_list_quarter)]
df_others_quarter = df_manuf_quarter[~df_manuf_quarter["MANUF"].isin(top_list_quarter)]
df_others_quarter_grp = (
    df_others_quarter.groupby("QUARTERLY", as_index=False)["AMOUNT_P"]
                     .sum()
                     .sort_values("QUARTERLY")
)
df_others_quarter_grp["MANUF"] = "기타"
df_manuf_quarter_final = pd.concat([df_top_quarter, df_others_quarter_grp], ignore_index=True).sort_values("QUARTERLY")

# 월별 제조사 데이터 (점유율)
df_share = df_manuf_final.copy()
month_total = df_share.groupby("MONTHLY")["AMOUNT_P"].transform("sum")
df_share["SHARE"] = (df_share["AMOUNT_P"] / month_total) * 100

# 년도별 시장점유율
df_share_year = df_manuf_year_final.copy()
if not df_share_year.empty:
    year_total = df_share_year.groupby("YEARLY")["AMOUNT_P"].transform("sum")
    df_share_year["SHARE"] = (df_share_year["AMOUNT_P"] / year_total) * 100
else:
    df_share_year["SHARE"] = pd.Series(dtype=float)

# 분기별 시장점유율
df_share_quarter = df_manuf_quarter_final.copy()
if not df_share_quarter.empty:
    quarter_total = df_share_quarter.groupby("QUARTERLY")["AMOUNT_P"].transform("sum")
    df_share_quarter["SHARE"] = (df_share_quarter["AMOUNT_P"] / quarter_total) * 100
else:
    df_share_quarter["SHARE"] = pd.Series(dtype=float)


# ----- 정렬 관련 함수들 -----
def move_others_to_end(order_list):
    """정렬 순서는 유지하되 '기타'만 맨 뒤로 이동"""
    order_wo_others = [x for x in order_list if x != "기타"]
    return order_wo_others + (["기타"] if "기타" in order_list else [])


# ----- 제조사 정렬 순서 (마지막 시점 기준, 이후 '기타'만 맨 뒤로) -----
# 년도 기준
if not df_share_year.empty:
    latest_year = df_share_year["YEARLY"].max()
    base_order_year = (
        df_share_year[df_share_year["YEARLY"] == latest_year]
        .groupby("MANUF")["SHARE"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
elif not df_manuf_year_final.empty:
    base_order_year = sorted(df_manuf_year_final["MANUF"].unique())
else:
    base_order_year = unique_manufs

sort_order_year = move_others_to_end(base_order_year)

# 분기 기준
if not df_share_quarter.empty:
    latest_quarter = df_share_quarter["QUARTERLY"].max()
    base_order_quarter = (
        df_share_quarter[df_share_quarter["QUARTERLY"] == latest_quarter]
        .groupby("MANUF")["SHARE"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
elif not df_manuf_quarter_final.empty:
    base_order_quarter = sorted(df_manuf_quarter_final["MANUF"].unique())
else:
    base_order_quarter = unique_manufs

sort_order_quarter = move_others_to_end(base_order_quarter)

# 월 기준
if not df_share.empty:
    latest_month = df_share["MONTHLY"].max()
    base_order_month = (
        df_share[df_share["MONTHLY"] == latest_month]
        .groupby("MANUF")["SHARE"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
else:
    base_order_month = unique_manufs

sort_order_month = move_others_to_end(base_order_month)


def reorder_columns(pivot_df, order_list):
    cols = [c for c in order_list if c in pivot_df.columns] + \
           [c for c in pivot_df.columns if c not in order_list]
    return pivot_df[cols]


# -------------------------
# 7) pivot (정렬된 100% stacked bar, 제조사용)
# -------------------------
if not df_share_year.empty:
    df_share_year_pivot = (
        df_share_year.pivot(index="YEARLY", columns="MANUF", values="SHARE")
        .fillna(0)
        .sort_index()
    )
    df_share_year_pivot = reorder_columns(df_share_year_pivot, sort_order_year)
else:
    df_share_year_pivot = pd.DataFrame()

if not df_share_quarter.empty:
    df_share_quarter_pivot = (
        df_share_quarter.pivot(index="QUARTERLY", columns="MANUF", values="SHARE")
        .fillna(0)
        .sort_index()
    )
    df_share_quarter_pivot = reorder_columns(df_share_quarter_pivot, sort_order_quarter)
else:
    df_share_quarter_pivot = pd.DataFrame()

if not df_share.empty:
    df_share_pivot = (
        df_share.pivot(index="MONTHLY", columns="MANUF", values="SHARE")
        .fillna(0)
        .sort_index()
    )
    df_share_pivot = reorder_columns(df_share_pivot, sort_order_month)
else:
    df_share_pivot = pd.DataFrame()


# -------------------------
# 8) 채널별 데이터 전처리 (YEARLY/QUARTERLY NaN 제거)
# -------------------------
df_channel["MARKET_TYPE_NAME"] = df_channel["MARKET_TYPE_NAME"].astype(str).str.strip()

df_channel_year = (
    df_channel.dropna(subset=["YEARLY"])
              .groupby(["MARKET_TYPE_NAME", "YEARLY"], as_index=False)["AMOUNT_P"]
              .sum()
              .sort_values("YEARLY")
)
df_channel_quarter = (
    df_channel.dropna(subset=["QUARTERLY"])
              .groupby(["MARKET_TYPE_NAME", "QUARTERLY"], as_index=False)["AMOUNT_P"]
              .sum()
              .sort_values("QUARTERLY")
)
df_channel_month = (
    df_channel.groupby(["MARKET_TYPE_NAME", "MONTHLY"], as_index=False)["AMOUNT_P"]
              .sum()
              .sort_values("MONTHLY")
)

# 채널명 기준 정렬 순서 (알파벳/오름차순)
channel_name_sort_order = sorted(df_channel_month["MARKET_TYPE_NAME"].unique())

color_sequence_channel = px.colors.qualitative.Plotly
channel_color_map = {ch: color_sequence_channel[i % len(color_sequence_channel)] for i, ch in enumerate(channel_name_sort_order)}


# -------------------------
# 9) 탭 구성 (3개만 사용)
# -------------------------
tab1, tab2, tab3 = st.tabs(["카테고리별 판매 추이", "카테고리/채널별 판매 추이", "제조사별 판매 추이"])


# ======================================================
# ◆ TAB1 – 카테고리별 판매 추이
# ======================================================
with tab1:
    st.header(f"{category_group} - 카테고리별 판매 추이 (백만원)")

    # ===== 년도 기준 =====
    st.subheader("년도 기준")
    if not df_year_plot.empty:
        fig_year = px.bar(
            df_year_plot, x="YEARLY", y="AMOUNT_P",
            text="AMOUNT_P", labels={"AMOUNT_P": "판매액(백만원)"}
        )
        fig_year.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
        fig_year.update_layout(yaxis_title="판매액(백만원)", xaxis_title="년도")
        fig_year.update_yaxes(tickformat=",.0f", rangemode="tozero")
        fig_year.update_xaxes(categoryorder="category ascending")
        st.plotly_chart(fig_year, use_container_width=True)
    else:
        st.info("년도(YEARLY) 정보가 없어 년도별 차트를 표시할 수 없습니다.")

    # ===== 분기 기준 =====
    st.subheader("분기 기준")
    if not df_q_plot.empty:
        fig_q = px.line(
            df_q_plot, x="QUARTERLY", y="AMOUNT_P",
            markers=True, labels={"AMOUNT_P": "판매액(백만원)"}
        )
        fig_q.update_layout(yaxis_title="판매액(백만원)", xaxis_title="분기")
        fig_q.update_yaxes(tickformat=",.0f", rangemode="tozero")
        fig_q.update_xaxes(categoryorder="category ascending")
        st.plotly_chart(fig_q, use_container_width=True)
    else:
        st.info("분기(QUARTERLY) 정보가 없어 분기별 차트를 표시할 수 없습니다.")

    # ===== 월 기준 =====
    st.subheader("월 기준")
    fig_m = px.line(
        df_m_plot, x="MONTHLY", y="AMOUNT_P",
        markers=True, labels={"AMOUNT_P": "판매액(백만원)"}
    )
    fig_m.update_layout(yaxis_title="판매액(백만원)", xaxis_title="월")
    fig_m.update_yaxes(tickformat=",.0f", rangemode="tozero")
    fig_m.update_xaxes(categoryorder="category ascending")
    st.plotly_chart(fig_m, use_container_width=True)

    # ----- TAB1 RAW DATA -----
    df_tab1_year = df_year[["YEARLY", "AMOUNT_P"]].copy()
    df_tab1_year["기간구분"] = "YEARLY"
    df_tab1_year["기간명"] = df_tab1_year["YEARLY"]
    df_tab1_year.rename(columns={"AMOUNT_P": "판매액(원)"}, inplace=True)
    df_tab1_year = df_tab1_year[["기간구분", "기간명", "판매액(원)"]]

    df_tab1_quarter = df_q[["QUARTERLY", "AMOUNT_P"]].copy()
    df_tab1_quarter["기간구분"] = "QUARTERLY"
    df_tab1_quarter["기간명"] = df_tab1_quarter["QUARTERLY"]
    df_tab1_quarter.rename(columns={"AMOUNT_P": "판매액(원)"}, inplace=True)
    df_tab1_quarter = df_tab1_quarter[["기간구분", "기간명", "판매액(원)"]]

    df_tab1_month = df_m[["MONTHLY", "AMOUNT_P"]].copy()
    df_tab1_month["기간구분"] = "MONTHLY"
    df_tab1_month["기간명"] = df_tab1_month["MONTHLY"]
    df_tab1_month.rename(columns={"AMOUNT_P": "판매액(원)"}, inplace=True)
    df_tab1_month = df_tab1_month[["기간구분", "기간명", "판매액(원)"]]

    df_tab1_raw = pd.concat([df_tab1_year, df_tab1_quarter, df_tab1_month], ignore_index=True)
    cat_order_tab1 = ["YEARLY", "QUARTERLY", "MONTHLY"]
    df_tab1_raw["기간구분"] = pd.Categorical(df_tab1_raw["기간구분"], categories=cat_order_tab1, ordered=True)
    df_tab1_raw = df_tab1_raw.sort_values(["기간구분", "기간명"])

    st.subheader("Raw Data (카테고리별 합계)")
    st.dataframe(df_tab1_raw, use_container_width=True)


# ======================================================
# ◆ TAB2 – 카테고리/채널별 판매 추이 (판매액 & 점유율 나란히)
# ======================================================
with tab2:
    st.header(f"{category_group} - 채널별 판매 및 점유율")

    # 채널 필터 (범례 대신 이걸로 100% 재계산)
    selected_channels = st.multiselect(
        "표시할 채널 선택",
        options=channel_name_sort_order,
        default=channel_name_sort_order,
        help="선택한 채널들로만 점유율을 다시 100%로 맞춰 계산합니다."
    )
    if not selected_channels:
        st.warning("최소 1개 이상의 채널을 선택해주세요.")
        selected_channels = channel_name_sort_order

    # -------------------- 1) 년도 기준 --------------------
    st.subheader("년도 기준")

    # 판매액 (선택된 채널만)
    df_channel_year_trend = df_channel_year[df_channel_year["MARKET_TYPE_NAME"].isin(selected_channels)].copy()
    if not df_channel_year_trend.empty:
        df_channel_year_trend["AMOUNT_M"] = df_channel_year_trend["AMOUNT_P"] / 1_000_000
        df_channel_year_trend["MARKET_TYPE_NAME"] = pd.Categorical(
            df_channel_year_trend["MARKET_TYPE_NAME"],
            categories=selected_channels,
            ordered=True
        )
        df_channel_year_trend = df_channel_year_trend.sort_values(["YEARLY", "MARKET_TYPE_NAME"])

    # 점유율 (선택된 채널 기준으로 다시 100% 맞추기)
    df_channel_year_sel = df_channel_year[df_channel_year["MARKET_TYPE_NAME"].isin(selected_channels)].copy()
    if not df_channel_year_sel.empty:
        year_total_sel = df_channel_year_sel.groupby("YEARLY")["AMOUNT_P"].transform("sum")
        df_channel_year_sel["SHARE_SEL"] = (df_channel_year_sel["AMOUNT_P"] / year_total_sel) * 100
        df_channel_year_share_pivot_sel = (
            df_channel_year_sel.pivot(index="YEARLY", columns="MARKET_TYPE_NAME", values="SHARE_SEL")
            .fillna(0)
            .sort_index()
        )
        year_cols = [c for c in selected_channels if c in df_channel_year_share_pivot_sel.columns]
        df_channel_year_share_pivot_sel = df_channel_year_share_pivot_sel[year_cols] if year_cols else pd.DataFrame()
    else:
        df_channel_year_share_pivot_sel = pd.DataFrame()

    col_y1, col_y2 = st.columns(2)

    with col_y1:
        st.caption("년도별 / 채널별 판매액 (백만원)")
        if not df_channel_year_trend.empty:
            fig_channel_year = px.bar(
                df_channel_year_trend,
                x="YEARLY",
                y="AMOUNT_M",
                color="MARKET_TYPE_NAME",
                color_discrete_map=channel_color_map,
                barmode="group",
                text="AMOUNT_M",
                labels={"AMOUNT_M": "판매액(백만원)", "MARKET_TYPE_NAME": "채널"},
                category_orders={"MARKET_TYPE_NAME": selected_channels}
            )
            fig_channel_year.update_traces(texttemplate='%{y:,.0f}', textposition='inside')
            fig_channel_year.update_layout(yaxis_title="판매액(백만원)", xaxis_title="년도", legend_title="채널")
            fig_channel_year.update_yaxes(tickformat=",.0f", rangemode="tozero")
            fig_channel_year.update_xaxes(categoryorder="category ascending")
            st.plotly_chart(fig_channel_year, use_container_width=True)
        else:
            st.info("년도(YEARLY) 기준 채널 데이터가 없습니다.")

    with col_y2:
        st.caption("년도별 / 채널별 시장점유율 (%) (선택 채널 기준 100%)")
        fig_channel_year_share = go.Figure()
        if not df_channel_year_share_pivot_sel.empty:
            for ch in df_channel_year_share_pivot_sel.columns:
                fig_channel_year_share.add_trace(go.Bar(
                    x=df_channel_year_share_pivot_sel.index,
                    y=df_channel_year_share_pivot_sel[ch],
                    name=ch,
                    marker_color=channel_color_map.get(ch),
                    text=df_channel_year_share_pivot_sel[ch].apply(lambda x: f"{x:.1f}%"),
                    textposition="inside"
                ))
            fig_channel_year_share.update_layout(
                barmode="stack",
                yaxis=dict(range=[0, 100], title="점유율 (%)"),
                xaxis=dict(title="년도"),
                legend_title="채널"
            )
            fig_channel_year_share.update_xaxes(categoryorder="category ascending")
            st.plotly_chart(fig_channel_year_share, use_container_width=True)
        else:
            st.info("년도(YEARLY) 기준 점유율을 계산할 데이터가 없습니다.")

    # -------------------- 2) 분기 기준 --------------------
    st.subheader("분기 기준")

    df_channel_quarter_trend = df_channel_quarter[df_channel_quarter["MARKET_TYPE_NAME"].isin(selected_channels)].copy()
    if not df_channel_quarter_trend.empty:
        df_channel_quarter_trend["AMOUNT_M"] = df_channel_quarter_trend["AMOUNT_P"] / 1_000_000

    df_channel_quarter_sel = df_channel_quarter[df_channel_quarter["MARKET_TYPE_NAME"].isin(selected_channels)].copy()
    if not df_channel_quarter_sel.empty:
        quarter_total_sel = df_channel_quarter_sel.groupby("QUARTERLY")["AMOUNT_P"].transform("sum")
        df_channel_quarter_sel["SHARE_SEL"] = (df_channel_quarter_sel["AMOUNT_P"] / quarter_total_sel) * 100
        df_channel_quarter_share_pivot_sel = (
            df_channel_quarter_sel.pivot(index="QUARTERLY", columns="MARKET_TYPE_NAME", values="SHARE_SEL")
            .fillna(0)
            .sort_index()
        )
        quarter_cols = [c for c in selected_channels if c in df_channel_quarter_share_pivot_sel.columns]
        df_channel_quarter_share_pivot_sel = df_channel_quarter_share_pivot_sel[quarter_cols] if quarter_cols else pd.DataFrame()
    else:
        df_channel_quarter_share_pivot_sel = pd.DataFrame()

    col_q1, col_q2 = st.columns(2)

    with col_q1:
        st.caption("분기별 / 채널별 판매액 (백만원)")
        if not df_channel_quarter_trend.empty:
            fig_channel_quarter = px.line(
                df_channel_quarter_trend,
                x="QUARTERLY",
                y="AMOUNT_M",
                color="MARKET_TYPE_NAME",
                color_discrete_map=channel_color_map,
                markers=True,
                labels={"AMOUNT_M": "판매액(백만원)", "MARKET_TYPE_NAME": "채널"},
                category_orders={"MARKET_TYPE_NAME": selected_channels}
            )
            fig_channel_quarter.update_layout(yaxis_title="판매액(백만원)", xaxis_title="분기", legend_title="채널")
            fig_channel_quarter.update_yaxes(tickformat=",.0f", rangemode="tozero")
            fig_channel_quarter.update_xaxes(categoryorder="category ascending")
            st.plotly_chart(fig_channel_quarter, use_container_width=True)
        else:
            st.info("분기(QUARTERLY) 기준 채널 데이터가 없습니다.")

    with col_q2:
        st.caption("분기별 / 채널별 시장점유율 (%) (선택 채널 기준 100%)")
        fig_channel_quarter_share = go.Figure()
        if not df_channel_quarter_share_pivot_sel.empty:
            for ch in df_channel_quarter_share_pivot_sel.columns:
                fig_channel_quarter_share.add_trace(go.Bar(
                    x=df_channel_quarter_share_pivot_sel.index,
                    y=df_channel_quarter_share_pivot_sel[ch],
                    name=ch,
                    marker_color=channel_color_map.get(ch),
                    text=df_channel_quarter_share_pivot_sel[ch].apply(lambda x: f"{x:.1f}%"),
                    textposition="inside"
                ))
            fig_channel_quarter_share.update_layout(
                barmode="stack",
                yaxis=dict(range=[0, 100], title="점유율 (%)"),
                xaxis=dict(title="분기"),
                legend_title="채널"
            )
            fig_channel_quarter_share.update_xaxes(categoryorder="category ascending")
            st.plotly_chart(fig_channel_quarter_share, use_container_width=True)
        else:
            st.info("분기(QUARTERLY) 기준 점유율을 계산할 데이터가 없습니다.")

    # -------------------- 3) 월 기준 --------------------
    st.subheader("월 기준")

    df_channel_month_trend = df_channel_month[df_channel_month["MARKET_TYPE_NAME"].isin(selected_channels)].copy()
    df_channel_month_trend["AMOUNT_M"] = df_channel_month_trend["AMOUNT_P"] / 1_000_000

    df_channel_month_sel = df_channel_month[df_channel_month["MARKET_TYPE_NAME"].isin(selected_channels)].copy()
    if not df_channel_month_sel.empty:
        month_total_sel = df_channel_month_sel.groupby("MONTHLY")["AMOUNT_P"].transform("sum")
        df_channel_month_sel["SHARE_SEL"] = (df_channel_month_sel["AMOUNT_P"] / month_total_sel) * 100
        df_channel_month_share_pivot_sel = (
            df_channel_month_sel.pivot(index="MONTHLY", columns="MARKET_TYPE_NAME", values="SHARE_SEL")
            .fillna(0)
            .sort_index()
        )
        month_cols = [c for c in selected_channels if c in df_channel_month_share_pivot_sel.columns]
        df_channel_month_share_pivot_sel = df_channel_month_share_pivot_sel[month_cols] if month_cols else pd.DataFrame()
    else:
        df_channel_month_share_pivot_sel = pd.DataFrame()

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.caption("월별 / 채널별 판매액 (백만원)")
        if not df_channel_month_trend.empty:
            fig_channel_month = px.line(
                df_channel_month_trend,
                x="MONTHLY",
                y="AMOUNT_M",
                color="MARKET_TYPE_NAME",
                color_discrete_map=channel_color_map,
                markers=True,
                labels={"AMOUNT_M": "판매액(백만원)", "MARKET_TYPE_NAME": "채널"},
                category_orders={"MARKET_TYPE_NAME": selected_channels}
            )
            fig_channel_month.update_layout(yaxis_title="판매액(백만원)", xaxis_title="월", legend_title="채널")
            fig_channel_month.update_yaxes(tickformat=",.0f", rangemode="tozero")
            fig_channel_month.update_xaxes(categoryorder="category ascending")
            st.plotly_chart(fig_channel_month, use_container_width=True)
        else:
            st.info("월 기준 채널 데이터가 없습니다.")

    with col_m2:
        st.caption("월별 / 채널별 시장점유율 (%) (선택 채널 기준 100%)")
        fig_channel_month_share = go.Figure()
        if not df_channel_month_share_pivot_sel.empty:
            for ch in df_channel_month_share_pivot_sel.columns:
                fig_channel_month_share.add_trace(go.Bar(
                    x=df_channel_month_share_pivot_sel.index,
                    y=df_channel_month_share_pivot_sel[ch],
                    name=ch,
                    marker_color=channel_color_map.get(ch),
                    text=df_channel_month_share_pivot_sel[ch].apply(lambda x: f"{x:.1f}%"),
                    textposition="inside"
                ))
            fig_channel_month_share.update_layout(
                barmode="stack",
                yaxis=dict(range=[0, 100], title="점유율 (%)"),
                xaxis=dict(title="월"),
                legend_title="채널"
            )
            fig_channel_month_share.update_xaxes(categoryorder="category ascending")
            st.plotly_chart(fig_channel_month_share, use_container_width=True)
        else:
            st.info("월 기준 점유율을 계산할 데이터가 없습니다.")

    # ----- TAB2 RAW DATA (현재 선택된 채널 기준 점유율) -----
    # YEARLY
    if not df_channel_year_sel.empty:
        df2_year = df_channel_year_sel.copy()
        df2_year["기간구분"] = "YEARLY"
        df2_year["기간명"] = df2_year["YEARLY"]
        df2_year.rename(columns={
            "MARKET_TYPE_NAME": "채널명",
            "AMOUNT_P": "판매액(원)",
            "SHARE_SEL": "점유율(%)"
        }, inplace=True)
        df2_year = df2_year[["기간구분", "기간명", "채널명", "판매액(원)", "점유율(%)"]]
    else:
        df2_year = pd.DataFrame(columns=["기간구분", "기간명", "채널명", "판매액(원)", "점유율(%)"])

    # QUARTERLY
    if not df_channel_quarter_sel.empty:
        df2_quarter = df_channel_quarter_sel.copy()
        df2_quarter["기간구분"] = "QUARTERLY"
        df2_quarter["기간명"] = df2_quarter["QUARTERLY"]
        df2_quarter.rename(columns={
            "MARKET_TYPE_NAME": "채널명",
            "AMOUNT_P": "판매액(원)",
            "SHARE_SEL": "점유율(%)"
        }, inplace=True)
        df2_quarter = df2_quarter[["기간구분", "기간명", "채널명", "판매액(원)", "점유율(%)"]]
    else:
        df2_quarter = pd.DataFrame(columns=["기간구분", "기간명", "채널명", "판매액(원)", "점유율(%)"])

    # MONTHLY
    if not df_channel_month_sel.empty:
        df2_month = df_channel_month_sel.copy()
        df2_month["기간구분"] = "MONTHLY"
        df2_month["기간명"] = df2_month["MONTHLY"]
        df2_month.rename(columns={
            "MARKET_TYPE_NAME": "채널명",
            "AMOUNT_P": "판매액(원)",
            "SHARE_SEL": "점유율(%)"
        }, inplace=True)
        df2_month = df2_month[["기간구분", "기간명", "채널명", "판매액(원)", "점유율(%)"]]
    else:
        df2_month = pd.DataFrame(columns=["기간구분", "기간명", "채널명", "판매액(원)", "점유율(%)"])

    df_tab2_raw = pd.concat([df2_year, df2_quarter, df2_month], ignore_index=True)
    cat_order_tab2 = ["YEARLY", "QUARTERLY", "MONTHLY"]
    if not df_tab2_raw.empty:
        df_tab2_raw["기간구분"] = pd.Categorical(df_tab2_raw["기간구분"], categories=cat_order_tab2, ordered=True)
        df_tab2_raw = df_tab2_raw.sort_values(["기간구분", "기간명", "채널명"])

    st.subheader("Raw Data (채널별 합계/점유율 - 선택 채널 기준)")
    st.dataframe(df_tab2_raw, use_container_width=True)


# ======================================================
# ◆ TAB3 – 제조사별 판매 (판매액 & 점유율 나란히)
# ======================================================
with tab3:
    st.header(f"{category_group} - 제조사별 판매 및 점유율")

    # -------------------- 1) 년도 기준 --------------------
    st.subheader("년도 기준")

    df_manuf_year_trend = df_manuf_year_final.copy()
    if not df_manuf_year_trend.empty:
        df_manuf_year_trend["AMOUNT_M"] = df_manuf_year_trend["AMOUNT_P"] / 1_000_000
        df_manuf_year_trend["MANUF"] = pd.Categorical(
            df_manuf_year_trend["MANUF"],
            categories=sort_order_year,
            ordered=True
        )
        df_manuf_year_trend = df_manuf_year_trend.sort_values(["YEARLY", "MANUF"])

    col_y1, col_y2 = st.columns(2)

    with col_y1:
        st.caption("년도별 / 제조사별 판매액 (백만원)")
        if not df_manuf_year_trend.empty:
            fig_manuf_year = px.bar(
                df_manuf_year_trend,
                x="YEARLY",
                y="AMOUNT_M",
                color="MANUF",
                color_discrete_map=color_map,
                barmode="group",
                text="AMOUNT_M",
                labels={"AMOUNT_M": "판매액(백만원)", "MANUF": "제조사"},
                category_orders={"MANUF": sort_order_year}
            )
            fig_manuf_year.update_traces(texttemplate='%{y:,.0f}', textposition='inside')
            fig_manuf_year.update_layout(yaxis_title="판매액(백만원)", xaxis_title="년도", legend_title="제조사")
            fig_manuf_year.update_yaxes(tickformat=",.0f", rangemode="tozero")
            fig_manuf_year.update_xaxes(categoryorder="category ascending")
            st.plotly_chart(fig_manuf_year, use_container_width=True)
        else:
            st.info("년도(YEARLY) 기준 제조사 데이터가 없습니다.")

    with col_y2:
        st.caption("년도별 / 제조사별 시장점유율 (%)")
        fig_year_share = go.Figure()
        if not df_share_year_pivot.empty:
            for manuf in df_share_year_pivot.columns:
                fig_year_share.add_trace(go.Bar(
                    x=df_share_year_pivot.index,
                    y=df_share_year_pivot[manuf],
                    name=manuf,
                    marker_color=color_map.get(manuf),
                    text=df_share_year_pivot[manuf].apply(lambda x: f"{x:.1f}%"),
                    textposition="inside"
                ))
            fig_year_share.update_layout(
                barmode="stack",
                yaxis=dict(range=[0, 100], title="점유율 (%)"),
                xaxis=dict(title="년도"),
                legend_title="제조사",
                legend_traceorder="reversed"  # '기타'가 범례 맨 위에
            )
            fig_year_share.update_xaxes(categoryorder="category ascending")
            st.plotly_chart(fig_year_share, use_container_width=True)
        else:
            st.info("년도(YEARLY) 기준 점유율을 계산할 데이터가 없습니다.")

    # -------------------- 2) 분기 기준 --------------------
    st.subheader("분기 기준")

    df_manuf_quarter_trend = df_manuf_quarter_final.copy()
    if not df_manuf_quarter_trend.empty:
        df_manuf_quarter_trend["AMOUNT_M"] = df_manuf_quarter_trend["AMOUNT_P"] / 1_000_000
        df_manuf_quarter_trend["MANUF"] = pd.Categorical(
            df_manuf_quarter_trend["MANUF"],
            categories=sort_order_quarter,
            ordered=True
        )
        df_manuf_quarter_trend = df_manuf_quarter_trend.sort_values(["QUARTERLY", "MANUF"])

    col_q1, col_q2 = st.columns(2)

    with col_q1:
        st.caption("분기별 / 제조사별 판매액 (백만원)")
        if not df_manuf_quarter_trend.empty:
            fig_manuf_quarter = px.line(
                df_manuf_quarter_trend,
                x="QUARTERLY",
                y="AMOUNT_M",
                color="MANUF",
                color_discrete_map=color_map,
                markers=True,
                labels={"AMOUNT_M": "판매액(백만원)", "MANUF": "제조사"},
                category_orders={"MANUF": sort_order_quarter}
            )
            fig_manuf_quarter.update_layout(yaxis_title="판매액(백만원)", xaxis_title="분기", legend_title="제조사")
            fig_manuf_quarter.update_yaxes(tickformat=",.0f", rangemode="tozero")
            fig_manuf_quarter.update_xaxes(categoryorder="category ascending")
            st.plotly_chart(fig_manuf_quarter, use_container_width=True)
        else:
            st.info("분기(QUARTERLY) 기준 제조사 데이터가 없습니다.")

    with col_q2:
        st.caption("분기별 / 제조사별 시장점유율 (%)")
        fig_quarter_share = go.Figure()
        if not df_share_quarter_pivot.empty:
            for manuf in df_share_quarter_pivot.columns:
                fig_quarter_share.add_trace(go.Bar(
                    x=df_share_quarter_pivot.index,
                    y=df_share_quarter_pivot[manuf],
                    name=manuf,
                    marker_color=color_map.get(manuf),
                    text=df_share_quarter_pivot[manuf].apply(lambda x: f"{x:.1f}%"),
                    textposition="inside"
                ))
            fig_quarter_share.update_layout(
                barmode="stack",
                yaxis=dict(range=[0, 100], title="점유율 (%)"),
                xaxis=dict(title="분기"),
                legend_title="제조사",
                legend_traceorder="reversed"
            )
            fig_quarter_share.update_xaxes(categoryorder="category ascending")
            st.plotly_chart(fig_quarter_share, use_container_width=True)
        else:
            st.info("분기(QUARTERLY) 기준 점유율을 계산할 데이터가 없습니다.")

    # -------------------- 3) 월 기준 --------------------
    st.subheader("월 기준")

    df_trend = df_manuf_final.copy()
    if not df_trend.empty:
        df_trend["AMOUNT_M"] = df_trend["AMOUNT_P"] / 1_000_000
        df_trend["MANUF"] = pd.Categorical(
            df_trend["MANUF"],
            categories=sort_order_month,
            ordered=True
        )
        df_trend = df_trend.sort_values(["MONTHLY", "MANUF"])

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.caption("월별 / 제조사별 판매액 (백만원)")
        if not df_trend.empty:
            fig_trend = px.line(
                df_trend,
                x="MONTHLY",
                y="AMOUNT_M",
                color="MANUF",
                color_discrete_map=color_map,
                markers=True,
                labels={"AMOUNT_M": "판매액(백만원)", "MANUF": "제조사"},
                category_orders={"MANUF": sort_order_month}
            )
            fig_trend.update_layout(yaxis_title="판매액(백만원)", xaxis_title="월", legend_title="제조사")
            fig_trend.update_yaxes(tickformat=",.0f", rangemode="tozero")
            fig_trend.update_xaxes(categoryorder="category ascending")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("월 기준 제조사 데이터가 없습니다.")

    with col_m2:
        st.caption("월별 / 제조사별 시장점유율 (%)")
        fig_month_share = go.Figure()
        if not df_share_pivot.empty:
            for manuf in df_share_pivot.columns:
                fig_month_share.add_trace(go.Bar(
                    x=df_share_pivot.index,
                    y=df_share_pivot[manuf],
                    name=manuf,
                    marker_color=color_map.get(manuf),
                    text=df_share_pivot[manuf].apply(lambda x: f"{x:.1f}%"),
                    textposition="inside"
                ))
            fig_month_share.update_layout(
                barmode="stack",
                yaxis=dict(range=[0, 100], title="점유율 (%)"),
                xaxis=dict(title="월"),
                legend_title="제조사",
                legend_traceorder="reversed"
            )
            fig_month_share.update_xaxes(categoryorder="category ascending")
            st.plotly_chart(fig_month_share, use_container_width=True)
        else:
            st.info("월 기준 점유율을 계산할 데이터가 없습니다.")

    # ----- TAB3 RAW DATA (채널명 포함) -----
    # YEARLY
    if not df_manuf_year_final.empty:
        df3_year = df_manuf_year_final.merge(
            df_share_year[["MANUF", "YEARLY", "SHARE"]],
            on=["MANUF", "YEARLY"],
            how="left"
        )
        df3_year["기간구분"] = "YEARLY"
        df3_year["기간명"] = df3_year["YEARLY"]
        df3_year["채널명"] = channel  # 현재 선택된 채널
        df3_year.rename(columns={
            "MANUF": "제조사명",
            "AMOUNT_P": "판매액(원)",
            "SHARE": "점유율(%)"
        }, inplace=True)
        df3_year = df3_year[["기간구분", "기간명", "채널명", "제조사명", "판매액(원)", "점유율(%)"]]
    else:
        df3_year = pd.DataFrame(columns=["기간구분", "기간명", "채널명", "제조사명", "판매액(원)", "점유율(%)"])

    # QUARTERLY
    if not df_manuf_quarter_final.empty:
        df3_quarter = df_manuf_quarter_final.merge(
            df_share_quarter[["MANUF", "QUARTERLY", "SHARE"]],
            on=["MANUF", "QUARTERLY"],
            how="left"
        )
        df3_quarter["기간구분"] = "QUARTERLY"
        df3_quarter["기간명"] = df3_quarter["QUARTERLY"]
        df3_quarter["채널명"] = channel  # 현재 선택된 채널
        df3_quarter.rename(columns={
            "MANUF": "제조사명",
            "AMOUNT_P": "판매액(원)",
            "SHARE": "점유율(%)"
        }, inplace=True)
        df3_quarter = df3_quarter[["기간구분", "기간명", "채널명", "제조사명", "판매액(원)", "점유율(%)"]]
    else:
        df3_quarter = pd.DataFrame(columns=["기간구분", "기간명", "채널명", "제조사명", "판매액(원)", "점유율(%)"])

    # MONTHLY
    if not df_manuf_final.empty:
        df3_month = df_manuf_final.merge(
            df_share[["MANUF", "MONTHLY", "SHARE"]],
            on=["MANUF", "MONTHLY"],
            how="left"
        )
        df3_month["기간구분"] = "MONTHLY"
        df3_month["기간명"] = df3_month["MONTHLY"]
        df3_month["채널명"] = channel  # 현재 선택된 채널
        df3_month.rename(columns={
            "MANUF": "제조사명",
            "AMOUNT_P": "판매액(원)",
            "SHARE": "점유율(%)"
        }, inplace=True)
        df3_month = df3_month[["기간구분", "기간명", "채널명", "제조사명", "판매액(원)", "점유율(%)"]]
    else:
        df3_month = pd.DataFrame(columns=["기간구분", "기간명", "채널명", "제조사명", "판매액(원)", "점유율(%)"])

    df_tab3_raw = pd.concat([df3_year, df3_quarter, df3_month], ignore_index=True)
    cat_order_tab3 = ["YEARLY", "QUARTERLY", "MONTHLY"]
    if not df_tab3_raw.empty:
        df_tab3_raw["기간구분"] = pd.Categorical(df_tab3_raw["기간구분"], categories=cat_order_tab3, ordered=True)
        df_tab3_raw = df_tab3_raw.sort_values(["기간구분", "기간명", "제조사명"])

    st.subheader("Raw Data (제조사별 합계/점유율)")
    st.dataframe(df_tab3_raw, use_container_width=True)
