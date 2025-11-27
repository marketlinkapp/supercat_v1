import streamlit as st
import pandas as pd
import pyodbc
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import re

st.set_page_config(layout="wide", page_title="Category Sales Dashboard")

# ---------------------------------------------
# 0) CSS: 전체 여백/폰트/카드 조정
# ---------------------------------------------
st.markdown("""
<style>
/* 전체 상단 여백 줄이기 (로고를 위로 붙이기 위함) */
.block-container {
    padding-top: 0.25rem !important;
}

/* 콤보박스 크기 확대 */
div[data-baseweb="select"] > div {
    height: 48px !important;
}
div[data-baseweb="select"] span {
    font-size: 1.2rem !important;
}

/* 탭 텍스트 더 크게 + 더 굵게 */
.stTabs [role="tab"] > div {
    color: #004c99 !important;
    font-size: 1.1rem !important;   /* ? 기존보다 1.2배 */
    font-weight: 600 !important;    /* ? 굵게 */
}

/* 선택된 탭 더 굵게 + 진한 아래 테두리 */
.stTabs [role="tab"][aria-selected="true"] > div {
    border-bottom: 4px solid #004c99 !important;  /* ? 더 두껍게 */
    padding-bottom: 6px !important;               /* ? 공간 확보 → 깔끔하게 보임 */
    font-weight: 700 !important;                  /* ? 강조 */
}

/* 탭 위·아래 여백 적용 ? 실제 렌더링 컨테이너 타겟 */
div[data-testid="stTabs"] {
    margin-top: 0.75rem !important;      /* ? 탭 위 여백 */
    margin-bottom: 0 !important;
    padding-bottom: 4rem !important;   /* ? 탭 아래 여백 */
}

/* ? 탭 아래 컨텐츠 시작 지점에 여백 추가 */
div[data-testid="stTabs"] + div[data-testid="stVerticalBlock"] {
    margin-top: 2.5rem !important;
}


/* 전체 헤더(h1/h2/h3) 폰트 크기를 기존의 약 85% 수준으로 축소 */
.block-container h1,
.block-container h2,
.block-container h3 {
    font-size: 85% !important;
}

/* 카드 내부에서 캡션과 차트 사이 간격 줄이기 */
.chart-card .stCaption {
    margin-bottom: 0.15rem !important;
}

/* "데이터 필터", "년도 기준" 등 섹션 타이틀용 (아주 약간 크게) */
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-top: 0.75rem;
    margin-bottom: 0.25rem;
}

/* TAB2 멀티셀렉트에서 선택된 채널 태그 폰트 조금 작게 */
.stMultiSelect [data-baseweb="tag"] span {
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------
# 공통 함수들
# ---------------------------------------------
def get_logo_path():
    """회사 로고 파일 경로 반환 (없으면 None)

    - app_supercat_v1.py 가 있는 폴더
    - 현재 작업 폴더( streamlit run 을 실행한 위치 )
    - 각 폴더의 .streamlit 하위 폴더
    를 순서대로 탐색
    """
    try:
        base_dir = Path(__file__).parent       # 이 .py 파일이 있는 폴더
    except NameError:
        base_dir = Path(".")

    cwd = Path.cwd()                            # streamlit run 을 실행한 폴더

    candidates = [
        cwd / "marketlink_log.png",
        cwd / ".streamlit" / "marketlink_log.png",
        base_dir / "marketlink_log.png",
        base_dir / ".streamlit" / "marketlink_log.png",
    ]

    for p in candidates:
        if p.is_file():
            return str(p)

    # 못 찾으면 None
    return None


def show_logo():
    """로그인 전/후 모두 좌측 상단에 로고 표시"""
    logo_path = get_logo_path()
    if not logo_path:
        return

    # 로그인 여부 체크
    is_logged_in = st.session_state.get("login", False)

    # ? 로그인 전 크게 / 로그인 후 기존 크기
    logo_width = 260 if not is_logged_in else 200

    # 위치 유지용 여백 (원래 있던 것 그대로)
    st.markdown("<div style='height:45px'></div>", unsafe_allow_html=True)

    # ? 컬럼 제거 → 크기 더 이상 줄어들지 않음
    st.image(logo_path, width=logo_width)


def get_conn():
    """공통 DB 커넥션 함수"""
    server = st.secrets["db"]["host"]
    database = st.secrets["db"]["database"]
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server},{st.secrets['db']['port']};"
        f"DATABASE={database};"
        f"UID={st.secrets['db']['user']};PWD={st.secrets['db']['password']}"
    )
    return pyodbc.connect(conn_str)


def aggregate_amount_by_period(df: pd.DataFrame, period_col: str) -> pd.DataFrame:
    """YEARLY / QUARTERLY / MONTHLY 기준 합계"""
    tmp = (
        df.dropna(subset=[period_col])
          .groupby(period_col, as_index=False)["AMOUNT_P"]
          .sum()
          .sort_values(period_col)
    )
    return tmp

def get_font_sizes_for_share(values, base_size):
    sizes = []
    for v in values:
        try:
            v = float(v)
        except Exception:
            v = 0.0

        if v >= 30:
            scale = 1.0      # 그대로
        elif v >= 20:
            scale = 1.0      # 그대로
        elif v >= 10:
            scale = 0.99     # 거의 그대로
        elif v >= 5:
            scale = 0.98
        elif v > 0:
            scale = 0.97     # 5% 미만도 97%
        else:
            scale = 0.97

        sizes.append(round(base_size * scale))
    return sizes


def get_scaled_font_sizes(values, base_size, min_scale=0.75, max_scale=1.0):
    vals = [float(v) for v in values]
    if not vals:
        return []

    non_zero = [v for v in vals if v > 0]
    if not non_zero:
        return [base_size] * len(vals)

    v_min = min(non_zero)
    v_max = max(non_zero)

    if v_max - v_min < 1e-9:
        return [base_size] * len(vals)

    sizes = []
    for v in vals:
        if v <= 0:
            scale = min_scale
        else:
            t = (v - v_min) / (v_max - v_min)
            scale = min_scale + (max_scale - min_scale) * t
        sizes.append(round(base_size * scale))
    return sizes



def build_tab1_raw(
    df_year: pd.DataFrame,
    df_q: pd.DataFrame,
    df_m: pd.DataFrame,
    channel_name: str,
    category_name: str,
) -> pd.DataFrame:
    """TAB1 Raw Data (카테고리/기간구분/기간명/채널명/판매액(백만원))"""
    def _build(df, period_label, col_name):
        if df.empty:
            return pd.DataFrame(
                columns=["카테고리", "기간구분", "기간명", "채널명", "판매액(백만원)"]
            )
        tmp = df[[col_name, "AMOUNT_P"]].copy()
        tmp["카테고리"] = category_name
        tmp["기간구분"] = period_label
        tmp["기간명"] = tmp[col_name]
        tmp["채널명"] = channel_name
        tmp["판매액(백만원)"] = (tmp["AMOUNT_P"] / 1_000_000).round(1)
        return tmp[["카테고리", "기간구분", "기간명", "채널명", "판매액(백만원)"]]

    df_year_raw = _build(df_year, "YEARLY", "YEARLY")
    df_q_raw = _build(df_q, "QUARTERLY", "QUARTERLY")
    df_m_raw = _build(df_m, "MONTHLY", "MONTHLY")

    df_all = pd.concat([df_year_raw, df_q_raw, df_m_raw], ignore_index=True)
    cat_order = ["YEARLY", "QUARTERLY", "MONTHLY"]
    if not df_all.empty:
        df_all["기간구분"] = pd.Categorical(df_all["기간구분"], categories=cat_order, ordered=True)
        df_all = df_all.sort_values(["기간구분", "기간명"])
    return df_all


def move_others_to_end(order_list):
    """정렬 순서는 유지하되 '기타'만 맨 뒤로 이동"""
    order_wo_others = [x for x in order_list if x != "기타"]
    return order_wo_others + (["기타"] if "기타" in order_list else [])


def reorder_columns(pivot_df, order_list):
    cols = [c for c in order_list if c in pivot_df.columns] + \
           [c for c in pivot_df.columns if c not in order_list]
    return pivot_df[cols]


def tighten_figure(fig, height=420):
    """Plotly 차트 위/아래 공백 줄이기"""
    fig.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        height=height
    )
    return fig


# ---------------------------------------------
# ★ BI_FONT_SIZE_MST: 폰트 사이즈 설정
#   - 실제 조회는 아래 DB 커넥션 이후
# ---------------------------------------------
FONT_SIZE_DEFAULTS = {
    "CHART_DESCRIPTION": 12,
    "CHART_TITLE": 18,
    "CHART_X_Y_LABEL": 12,
    "CHART_NUMERIC": 12,
    "CHART_LEGEND": 11,
}
font_size_map = FONT_SIZE_DEFAULTS.copy()


def get_font_size(loc: str) -> int:
    """BI_FONT_SIZE_MST 에서 폰트 사이즈 조회 (없으면 기본값)"""
    return font_size_map.get(loc, FONT_SIZE_DEFAULTS.get(loc, 12))


def apply_axis_legend_fonts(fig, xaxis_title=None, yaxis_title=None):
    """X/Y 축 라벨 + 눈금 + 범례 폰트를 BI_FONT_SIZE_MST 기준으로 일괄 적용"""
    axis_font_size = get_font_size("CHART_X_Y_LABEL")
    legend_font_size = get_font_size("CHART_LEGEND")

    layout_kwargs = {}
    if xaxis_title is not None:
        layout_kwargs["xaxis_title"] = xaxis_title
    if yaxis_title is not None:
        layout_kwargs["yaxis_title"] = yaxis_title

    fig.update_layout(
        **layout_kwargs,
        xaxis=dict(
            title_font=dict(size=axis_font_size),
            tickfont=dict(size=axis_font_size),
        ),
        yaxis=dict(
            title_font=dict(size=axis_font_size),
            tickfont=dict(size=axis_font_size),
        ),
        legend=dict(
            font=dict(size=legend_font_size)
        ),
    )
    return fig


def chart_card(caption_text: str, fig, category_name: str = None):
    """
    설명 캡션(연한 회색) + (선택된 카테고리명) + 차트를 네모난 테두리로 감싸는 카드
    ※ 폰트 크기: BI_FONT_SIZE_MST (CHART_DESCRIPTION / CHART_TITLE) 사용
    """
    fig = tighten_figure(fig)
    with st.container(border=True):
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)

        # 1줄: 차트 설명 (BI_FONT_SIZE_MST.CHART_DESCRIPTION)
        desc_size = get_font_size("CHART_DESCRIPTION")
        st.markdown(
            f"""
            <div style="
                font-size:{desc_size}px;
                color:rgba(0,0,0,0.6);
                margin-bottom:0.15rem;
            ">
                {caption_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2줄: 선택된 카테고리명 (가운데 정렬, BI_FONT_SIZE_MST.CHART_TITLE)
        if category_name:
            title_size = get_font_size("CHART_TITLE")
            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    font-size:{title_size}px;
                    font-weight:600;
                    color:#555555;
                    margin-top:2px;
                    margin-bottom:4px;
                ">
                    {category_name}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div style="margin-top:-6px;"></div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


def section_title(text: str):
    """'데이터 필터', '년도 기준' 같은 섹션 타이틀"""
    st.markdown(f'<p class="section-title">{text}</p>', unsafe_allow_html=True)


def clean_channel_name(raw: str) -> str:
    """
    '1. 온라인+오프라인' -> '온라인+오프라인' 처럼
    앞의 숫자+점+공백을 제거한 순수 채널명 반환
    """
    return re.sub(r'^\d+\.\s*', '', str(raw))


# -------------------------
# 1) 로그인
# -------------------------
def login():
    # 로그인 페이지에서는 전체 레이아웃 폭을 좁게 (대시보드랑 분리됨)
    st.markdown("""
    <style>
    /* 로그인 페이지에서만 적용됨 (로그인 후에는 login() 을 안 타기 때문에) */
    .block-container {
        max-width: 460px;          /* ? 화면이 아무리 커져도 가로 460px 정도만 사용 */
        padding-top: 2rem;
    }
    .login-title {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 0.4rem;
    }
    .login-subtitle {
        text-align: center;
        font-size: 0.9rem;
        color: #666666;
        margin-bottom: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # 회사 로고 (위쪽에 작게)
    show_logo()

    # 타이틀/부제목
    st.markdown('<div class="login-title">대시보드 로그인</div>', unsafe_allow_html=True)

    # 입력창과 버튼을 하나의 카드처럼
    with st.container(border=True):
        user_id = st.text_input("ID", key="login_id")
        password = st.text_input("PW", type="password", key="login_pw")
        login_btn = st.button("로그인", use_container_width=True)

    # 로그인 처리 로직 (기존과 동일하게 유지)
    if login_btn:
        try:
            with get_conn() as conn:
                df_user = pd.read_sql(
                    "SELECT USERNAME, USERPASSWORD FROM USER_TABLE",
                    conn
                )

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
conn = get_conn()

# -------------------------
# ★ BI_FONT_SIZE_MST 조회
# -------------------------
try:
    df_font = pd.read_sql(
        """
        SELECT FONT_LOC, FONT_SIZE
        FROM BI_FONT_SIZE_MST
        """,
        conn,
    )
    font_size_map = FONT_SIZE_DEFAULTS.copy()
    for _, row in df_font.iterrows():
        loc = str(row["FONT_LOC"]).strip()
        size = row["FONT_SIZE"]
        if loc and pd.notna(size):
            try:
                font_size_map[loc] = int(size)
            except Exception:
                # 숫자로 변환 안되면 해당 레코드는 무시
                pass
except Exception:
    # 조회 실패 시 기본값 그대로 사용
    font_size_map = FONT_SIZE_DEFAULTS.copy()


# -------------------------
# 3) 상단 로고 + 필터 (PROMPT 테이블 사용)
# -------------------------
# 로그인 후에도 좌측 상단에 로고 표시
show_logo()

section_title("데이터 필터")

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

# 색상 테이블 (BI_COLOR_MST) 조회
try:
    df_color = pd.read_sql(
        """
        SELECT [선택타입] AS TYPE,
               [색상순번] AS COLOR_SEQ,
               [색상코드] AS COLOR_CODE,
               COMMENT1,
               COMMENT2,
               COMMENT3
        FROM BI_COLOR_MST
        """,
        conn,
    )
    df_color_manuf = (
        df_color[df_color["TYPE"] == "MANUF_RANK"]
        .sort_values("COLOR_SEQ")
        .reset_index(drop=True)
    )
    df_color_channel = (
        df_color[df_color["TYPE"] == "CHANNEL"]
        .sort_values("COLOR_SEQ")
        .reset_index(drop=True)
    )
    manuf_palette = df_color_manuf["COLOR_CODE"].tolist()
    channel_palette = df_color_channel["COLOR_CODE"].tolist()

    # CHANNEL의 COMMENT3 = 채널명 으로 색상 매핑
    bi_channel_color_map = {}
    for _, row in df_color_channel.iterrows():
        ch_name = str(row.get("COMMENT3", "")).strip()
        color_code = str(row.get("COLOR_CODE", "")).strip()
        if ch_name and color_code:
            bi_channel_color_map[ch_name] = color_code
except Exception:
    # 색상 테이블 조회 실패 시 기존 로직을 위한 기본값
    df_color_manuf = pd.DataFrame()
    df_color_channel = pd.DataFrame()
    manuf_palette = []
    channel_palette = []
    bi_channel_color_map = {}

# 필터 영역 카드로 묶기
with st.container(border=True):
    col1, col2, col3, col4 = st.columns([1.2, 4.0, 1.2, 1.6])

    with col1:
        category_level = st.selectbox("카테고리 레벨", category_levels)

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

# 선택된 채널명 (숫자/점 제거 버전)
selected_channel_name = clean_channel_name(channel)

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

df_year = aggregate_amount_by_period(df, "YEARLY")
df_q = aggregate_amount_by_period(df, "QUARTERLY")
df_m = (
    df.groupby("MONTHLY", as_index=False)["AMOUNT_P"]
      .sum()
      .sort_values("MONTHLY")
)

# 차트용: 백만원 단위
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

# 제조사 컬러맵 (제조사 순위 기준)
color_sequence = manuf_palette

# top_list는 이미 "매출 기준 내림차순 TOP N" 순서로 만들어져 있음
ranked_manufs = [m for m in top_list if m in df_manuf_final["MANUF"].unique()]

color_map = {}

# 1) TOP 순위 제조사: 순위 1 → 첫 번째 색, 순위 2 → 두 번째 색 ...
for i, manuf in enumerate(ranked_manufs):
    color_map[manuf] = color_sequence[i % len(color_sequence)]

# 2) (혹시 있을 수 있는) 그 외 개별 제조사들: 그 뒤 색부터 순차 부여
others_manufs = [
    m for m in df_manuf_final["MANUF"].unique()
    if m not in ranked_manufs and m != "기타"
]
start_idx = len(ranked_manufs)
for j, manuf in enumerate(others_manufs):
    color_map[manuf] = color_sequence[(start_idx + j) % len(color_sequence)]

# 3) TOP 밖 제조사들이 합쳐진 "기타"는 항상 연한 회색
if "기타" in df_manuf_final["MANUF"].unique():
    color_map["기타"] = "#B0B0B0"


# -------------------------
# 6) 제조사 시장점유율 계산
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


# ----- 제조사 정렬 순서 -----
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
    base_order_year = ""
sort_order_year = move_others_to_end(base_order_year)

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
    base_order_quarter = ""
sort_order_quarter = move_others_to_end(base_order_quarter)

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
    base_order_month = ""
sort_order_month = move_others_to_end(base_order_month)


# -------------------------
# 7) 제조사 점유율 피벗
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
# 8) 채널별 데이터 전처리
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

channel_name_sort_order = sorted(df_channel_month["MARKET_TYPE_NAME"].unique())

# 채널 컬러맵 (BI_COLOR_MST - COMMENT3 기준)
default_seq_channel = px.colors.qualitative.Plotly
channel_color_map = {}
for i, ch in enumerate(channel_name_sort_order):
    if bi_channel_color_map:
        color_code = bi_channel_color_map.get(ch)
        if not color_code:
            if i < len(channel_palette):
                color_code = channel_palette[i]
            else:
                color_code = default_seq_channel[i % len(default_seq_channel)]
    else:
        color_code = default_seq_channel[i % len(default_seq_channel)]
    channel_color_map[ch] = color_code


# -------------------------
# 9) 탭 구성
# -------------------------
tab1, tab2, tab3 = st.tabs(["카테고리별 판매 추이", "카테고리/채널별 판매 추이", "제조사별 판매 추이"])

# ======================================================
# ◆ TAB1 – 카테고리별 판매 추이
# ======================================================
with tab1:
    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # ===== 년도 기준 =====
    section_title("년도 기준")
    if not df_year_plot.empty:
        df_year_plot_sorted = df_year_plot.sort_values("YEARLY")

        # 막대 그래프 (숫자는 막대 안쪽에)
        fig_year = px.bar(
            df_year_plot_sorted,
            x="YEARLY",
            y="AMOUNT_P",
            text="AMOUNT_P",
            labels={"AMOUNT_P": "판매액(백만원)"}
        )

        fig_year.update_traces(
            texttemplate='%{y:,.0f}',
            textposition='inside',
            textfont=dict(
                color="rgba(0,0,0,0.7)",
                size=get_font_size("CHART_NUMERIC"),   # ★ 막대 숫자 폰트
            ),
            hovertemplate='YEARLY=%{x}<br>판매액(백만원)=%{y:,.1f}<extra></extra>',
            marker_color=bi_channel_color_map.get(channel, "#b3d9ff"),
            width=0.7
        )

        fig_year.update_yaxes(tickformat=",.0f", rangemode="tozero")
        fig_year.update_xaxes(categoryorder="category ascending")

        # --- 성장률: 막대 사이 중앙에 화살표 + 숫자 + % ---
        years = df_year_plot_sorted["YEARLY"].tolist()
        amounts = df_year_plot_sorted["AMOUNT_P"].tolist()

        for i in range(1, len(years)):
            prev_val = amounts[i - 1]
            curr_val = amounts[i]
            if prev_val == 0:
                continue

            rate = (curr_val - prev_val) / prev_val * 100
            arrow = "▲" if rate >= 0 else "▼"
            color = "red" if rate >= 0 else "blue"

            x_pos = i - 0.5
            y_pos = min(prev_val, curr_val) * 0.55

            fig_year.add_annotation(
                x=x_pos,
                y=y_pos,
                xref="x",
                yref="y",
                text=f"{arrow} {abs(rate):.1f}%",
                showarrow=False,
                # ★ 성장률은 CHART_NUMERIC 대상이 아님 → 14 고정
                font=dict(color=color, size=14)
            )
        # ----------------------------------------------------

        fig_year = apply_axis_legend_fonts(fig_year, xaxis_title="년도", yaxis_title="판매액(백만원)")
        chart_card(f"년도별 / 채널별 판매액 (백만원, {selected_channel_name})", fig_year, category_group)
    else:
        with st.container(border=True):
            st.caption(f"년도별 / 채널별 판매액 (백만원, {selected_channel_name})")
            st.info("년도(YEARLY) 정보가 없어 년도별 차트를 표시할 수 없습니다.")

    # ===== 분기 기준 =====
    section_title("분기 기준")
    if not df_q_plot.empty:
        fig_q = px.line(
            df_q_plot, x="QUARTERLY", y="AMOUNT_P",
            markers=True, labels={"AMOUNT_P": "판매액(백만원)"}
        )

        fig_q.update_traces(
            hovertemplate='QUARTERLY=%{x}<br>판매액(백만원)=%{y:,.1f}<extra></extra>',
            line_color=bi_channel_color_map.get(channel, "#6aa8ff")
        )
        fig_q.update_yaxes(tickformat=",.0f", rangemode="tozero")
        fig_q.update_xaxes(categoryorder="category ascending")

        fig_q = apply_axis_legend_fonts(fig_q, xaxis_title="분기", yaxis_title="판매액(백만원)")
        chart_card(f"분기별 / 채널별 판매액 (백만원, {selected_channel_name})", fig_q, category_group)
    else:
        with st.container(border=True):
            st.caption(f"분기별 / 채널별 판매액 (백만원, {selected_channel_name})")
            st.info("분기(QUARTERLY) 정보가 없어 분기별 차트를 표시할 수 없습니다.")

    # ===== 월 기준 =====
    section_title("월 기준")
    fig_m = px.line(
        df_m_plot, x="MONTHLY", y="AMOUNT_P",
        markers=True, labels={"AMOUNT_P": "판매액(백만원)"}
    )

    fig_m.update_traces(
        hovertemplate='MONTHLY=%{x}<br>판매액(백만원)=%{y:,.1f}<extra></extra>',
        line_color=bi_channel_color_map.get(channel, "#6aa8ff")
    )
    fig_m.update_yaxes(tickformat=",.0f", rangemode="tozero")
    fig_m.update_xaxes(categoryorder="category ascending")

    fig_m = apply_axis_legend_fonts(fig_m, xaxis_title="월", yaxis_title="판매액(백만원)")
    chart_card(f"월별 / 채널별 판매액 (백만원, {selected_channel_name})", fig_m, category_group)

    # ----- TAB1 RAW DATA -----
    section_title(f"Raw Data (카테고리별 합계, 백만원 단위, {selected_channel_name})")
    df_tab1_raw = build_tab1_raw(df_year, df_q, df_m, selected_channel_name, category_group)
    st.dataframe(df_tab1_raw, use_container_width=True)


# ======================================================
# ◆ TAB2 – 카테고리/채널별 판매 추이
# ======================================================
with tab2:
    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <p class="section-title" style="margin-bottom:-0.6rem;">
            표시할 채널 선택 (선택된 채널로 점유율 재계산)
        </p>
        """,
        unsafe_allow_html=True,
    )
    
    # ⬇️ 여기 CSS 만 교체
    st.markdown(
        """
        <style>
        /* multiselect 기본 label + 위 여백 제거 */
        div[data-testid="stMultiSelect"] label {
            display: none !important;
            margin-bottom: 0 !important;
        }
        div[data-testid="stMultiSelect"] > div:first-child {
            padding-top: 0 !important;
            margin-top: -6px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    selected_channels = st.multiselect(
        "",
        options=channel_name_sort_order,
        default=channel_name_sort_order,
        key="channel_select_box",
    )

    if not selected_channels:
        st.warning("최소 1개 이상의 채널을 선택해주세요.")
        selected_channels = channel_name_sort_order

    # -------------------- 1) 년도 기준 -------------------->
    section_title("년도 기준")

    df_channel_year_trend = df_channel_year[df_channel_year["MARKET_TYPE_NAME"].isin(selected_channels)].copy()
    if not df_channel_year_trend.empty:
        df_channel_year_trend["AMOUNT_M"] = df_channel_year_trend["AMOUNT_P"] / 1_000_000
        df_channel_year_trend["MARKET_TYPE_NAME"] = pd.Categorical(
            df_channel_year_trend["MARKET_TYPE_NAME"],
            categories=selected_channels,
            ordered=True
        )
        df_channel_year_trend = df_channel_year_trend.sort_values(["YEARLY", "MARKET_TYPE_NAME"])

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
        if not df_channel_year_trend.empty:

            # 채널 순서 결정
            if not df_channel_year_share_pivot_sel.empty:
                channel_order_year = list(df_channel_year_share_pivot_sel.columns)
            else:
                channel_order_year = selected_channels

            # ---------------- 기본 스택 막대 차트 ----------------
            fig_channel_year = go.Figure()

            numeric_font = get_font_size("CHART_NUMERIC")

            for ch in channel_order_year:
                df_temp = df_channel_year_trend[
                    df_channel_year_trend["MARKET_TYPE_NAME"] == ch
                ]
                if df_temp.empty:
                    continue

                fig_channel_year.add_trace(go.Bar(
                    x=df_temp["YEARLY"],
                    y=df_temp["AMOUNT_M"],
                    name=ch,
                    marker_color=channel_color_map.get(ch),
                    text=df_temp["AMOUNT_M"].apply(lambda v: f"{v:,.0f}"),
                    textposition="inside",
                    textfont=dict(size=numeric_font),   # ★ 막대 숫자 폰트
                    width=0.7
                ))

            fig_channel_year.update_yaxes(tickformat=",.0f", rangemode="tozero")
            fig_channel_year.update_xaxes(categoryorder="category ascending")
            
            # ★ 누적 막대 그래프 설정 복원
            fig_channel_year.update_layout(barmode="stack")

            # ---------------- 채널별 성장률 계산 ----------------
            df_growth = df_channel_year_trend.copy()
            df_growth["MARKET_TYPE_NAME"] = pd.Categorical(
                df_growth["MARKET_TYPE_NAME"],
                categories=channel_order_year,
                ordered=True
            )
            df_growth = df_growth.sort_values(["YEARLY", "MARKET_TYPE_NAME"])

            # 연도별 스택 높이 누적 + 중간 위치 계산
            df_growth["CUM"] = df_growth.groupby("YEARLY")["AMOUNT_M"].cumsum()
            df_growth["BOTTOM"] = df_growth["CUM"] - df_growth["AMOUNT_M"]
            df_growth["MID"] = df_growth["BOTTOM"] + df_growth["AMOUNT_M"] / 2

            for ch in channel_order_year:
                sub = df_growth[df_growth["MARKET_TYPE_NAME"] == ch].sort_values("YEARLY")

                years = sub["YEARLY"].tolist()
                vals = sub["AMOUNT_M"].tolist()
                mids = sub["MID"].tolist()

                for i in range(1, len(years)):
                    prev_val = vals[i - 1]
                    curr_val = vals[i]
                    if prev_val == 0:
                        continue

                    rate = (curr_val - prev_val) / prev_val * 100
                    arrow = "▲" if rate >= 0 else "▼"
                    color = "red" if rate >= 0 else "blue"

                    x_pos = i - 0.5
                    y_pos = mids[i]

                    fig_channel_year.add_annotation(
                        x=x_pos,
                        y=y_pos,
                        xref="x",
                        yref="y",
                        text=f"{arrow} {abs(rate):.1f}%",
                        showarrow=False,
                        # ★ 성장률은 NUMERIC 대상 아님
                        font=dict(color=color, size=14),
                    )

            fig_channel_year = apply_axis_legend_fonts(
                fig_channel_year,
                xaxis_title="년도",
                yaxis_title="판매액(백만원)"
            )
            chart_card("년도별 / 채널별 판매액 (백만원)", fig_channel_year, category_group)

        else:
            with st.container(border=True):
                st.caption("년도별 / 채널별 판매액 (백만원)")
                st.info("년도(YEARLY) 기준 채널 데이터가 없습니다.")

    with col_y2:
        fig_channel_year_share = go.Figure()
        if not df_channel_year_share_pivot_sel.empty:
            numeric_font = get_font_size("CHART_NUMERIC")
            for ch in df_channel_year_share_pivot_sel.columns:
                fig_channel_year_share.add_trace(go.Bar(
                    x=df_channel_year_share_pivot_sel.index,
                    y=df_channel_year_share_pivot_sel[ch],
                    name=ch,
                    marker_color=channel_color_map.get(ch),
                    text=df_channel_year_share_pivot_sel[ch].apply(lambda x: f"{x:.1f}%"),
                    textposition="inside",
                    textfont=dict(size=numeric_font),  # ★ 점유율 숫자 폰트
                    width=0.7
                ))
            fig_channel_year_share.update_layout(
                barmode="stack",
                yaxis=dict(range=[0, 100]),
                xaxis=dict(),
                # ✅ 폰트가 자동으로 작아지는 것 방지
                #uniformtext_minsize=numeric_font,
                #uniformtext_mode="show"
            )
            fig_channel_year_share = apply_axis_legend_fonts(
                fig_channel_year_share,
                xaxis_title="년도",
                yaxis_title="점유율 (%)"
            )
            chart_card("년도별 / 채널별 시장점유율 (%)", fig_channel_year_share, category_group)
        else:
            with st.container(border=True):
                st.caption("년도별 / 채널별 시장점유율 (%)")
                st.info("년도(YEARLY) 기준 점유율을 계산할 데이터가 없습니다.")

    # -------------------- 2) 분기 기준 -------------------->
    section_title("분기 기준")

    df_channel_quarter_trend = df_channel_quarter[df_channel_quarter["MARKET_TYPE_NAME"].isin(selected_channels)].copy()
    if not df_channel_quarter_trend.empty:
        df_channel_quarter_trend["AMOUNT_M"] = df_channel_quarter_trend["AMOUNT_P"] / 1_000_000
        df_channel_quarter_trend = df_channel_quarter_trend.drop(columns=["AMOUNT_P"], errors="ignore")

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
            fig_channel_quarter.update_yaxes(tickformat=",.0f", rangemode="tozero")
            fig_channel_quarter.update_xaxes(categoryorder="category ascending")

            fig_channel_quarter = apply_axis_legend_fonts(
                fig_channel_quarter,
                xaxis_title="분기",
                yaxis_title="판매액(백만원)"
            )
            chart_card("분기별 / 채널별 판매액 (백만원)", fig_channel_quarter, category_group)
        else:
            with st.container(border=True):
                st.caption("분기별 / 채널별 판매액 (백만원)")
                st.info("분기(QUARTERLY) 기준 채널 데이터가 없습니다.")

    with col_q2:
        fig_channel_quarter_share = go.Figure()
        if not df_channel_quarter_share_pivot_sel.empty:
            numeric_font = get_font_size("CHART_NUMERIC")
            for ch in df_channel_quarter_share_pivot_sel.columns:
                fig_channel_quarter_share.add_trace(go.Bar(
                    x=df_channel_quarter_share_pivot_sel.index,
                    y=df_channel_quarter_share_pivot_sel[ch],
                    name=ch,
                    marker_color=channel_color_map.get(ch),
                    text=df_channel_quarter_share_pivot_sel[ch].apply(lambda x: f"{x:.1f}%"),
                    textposition="inside",
                    textfont=dict(size=numeric_font)
                ))
            fig_channel_quarter_share.update_layout(
                barmode="stack",
                yaxis=dict(range=[0, 100]),
                xaxis=dict(),
            )
            fig_channel_quarter_share = apply_axis_legend_fonts(
                fig_channel_quarter_share,
                xaxis_title="분기",
                yaxis_title="점유율 (%)"
            )
            chart_card("분기별 / 채널별 시장점유율 (%)", fig_channel_quarter_share, category_group)
        else:
            with st.container(border=True):
                st.caption("분기별 / 채널별 시장점유율 (%)")
                st.info("분기(QUARTERLY) 기준 점유율을 계산할 데이터가 없습니다.")

    # -------------------- 3) 월 기준 -------------------->
    section_title("월 기준")

    df_channel_month_trend = df_channel_month[df_channel_month["MARKET_TYPE_NAME"].isin(selected_channels)].copy()
    df_channel_month_trend["AMOUNT_M"] = df_channel_month_trend["AMOUNT_P"] / 1_000_000
    df_channel_month_trend = df_channel_month_trend.drop(columns=["AMOUNT_P"], errors="ignore")

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
            fig_channel_month.update_yaxes(tickformat=",.0f", rangemode="tozero")
            fig_channel_month.update_xaxes(categoryorder="category ascending")

            fig_channel_month = apply_axis_legend_fonts(
                fig_channel_month,
                xaxis_title="월",
                yaxis_title="판매액(백만원)"
            )
            chart_card("월별 / 채널별 판매액 (백만원)", fig_channel_month, category_group)
        else:
            with st.container(border=True):
                st.caption("월별 / 채널별 판매액 (백만원)")
                st.info("월 기준 채널 데이터가 없습니다.")

    with col_m2:
        fig_channel_month_share = go.Figure()
        if not df_channel_month_share_pivot_sel.empty:
            numeric_font = get_font_size("CHART_NUMERIC")
            for ch in df_channel_month_share_pivot_sel.columns:
                fig_channel_month_share.add_trace(go.Bar(
                    x=df_channel_month_share_pivot_sel.index,
                    y=df_channel_month_share_pivot_sel[ch],
                    name=ch,
                    marker_color=channel_color_map.get(ch),
                    text=df_channel_month_share_pivot_sel[ch].apply(lambda x: f"{x:.1f}%"),
                    textposition="inside",
                    textfont=dict(size=numeric_font)
                ))
            fig_channel_month_share.update_layout(
                barmode="stack",
                yaxis=dict(range=[0, 100]),
                xaxis=dict(),
            )
            fig_channel_month_share = apply_axis_legend_fonts(
                fig_channel_month_share,
                xaxis_title="월",
                yaxis_title="점유율 (%)"
            )
            chart_card("월별 / 채널별 시장점유율 (%)", fig_channel_month_share, category_group)
        else:
            with st.container(border=True):
                st.caption("월별 / 채널별 시장점유율 (%)")
                st.info("월 기준 점유율을 계산할 데이터가 없습니다.")

    # ----- TAB2 RAW DATA -----
    section_title("Raw Data (채널별 합계/점유율, 백만원 단위)")

    if not df_channel_year_sel.empty:
        df2_year = df_channel_year_sel.copy()
        df2_year["기간구분"] = "YEARLY"
        df2_year["기간명"] = df2_year["YEARLY"]
        df2_year.rename(columns={
            "MARKET_TYPE_NAME": "채널명",
            "AMOUNT_P": "판매액(원)",
            "SHARE_SEL": "점유율(%)"
        }, inplace=True)
        df2_year["카테고리"] = category_group
        df2_year["판매액(백만원)"] = (df2_year["판매액(원)"] / 1_000_000).round(1)
        df2_year = df2_year[["카테고리", "기간구분", "기간명", "채널명", "판매액(백만원)", "점유율(%)"]]
    else:
        df2_year = pd.DataFrame(columns=["카테고리", "기간구분", "기간명", "채널명", "판매액(백만원)", "점유율(%)"])

    if not df_channel_quarter_sel.empty:
        df2_quarter = df_channel_quarter_sel.copy()
        df2_quarter["기간구분"] = "QUARTERLY"
        df2_quarter["기간명"] = df2_quarter["QUARTERLY"]
        df2_quarter.rename(columns={
            "MARKET_TYPE_NAME": "채널명",
            "AMOUNT_P": "판매액(원)",
            "SHARE_SEL": "점유율(%)"
        }, inplace=True)
        df2_quarter["카테고리"] = category_group
        df2_quarter["판매액(백만원)"] = (df2_quarter["판매액(원)"] / 1_000_000).round(1)
        df2_quarter = df2_quarter[["카테고리", "기간구분", "기간명", "채널명", "판매액(백만원)", "점유율(%)"]]
    else:
        df2_quarter = pd.DataFrame(columns=["카테고리", "기간구분", "기간명", "채널명", "판매액(백만원)", "점유율(%)"])

    if not df_channel_month_sel.empty:
        df2_month = df_channel_month_sel.copy()
        df2_month["기간구분"] = "MONTHLY"
        df2_month["기간명"] = df2_month["MONTHLY"]
        df2_month.rename(columns={
            "MARKET_TYPE_NAME": "채널명",
            "AMOUNT_P": "판매액(원)",
            "SHARE_SEL": "점유율(%)"
        }, inplace=True)
        df2_month["카테고리"] = category_group
        df2_month["판매액(백만원)"] = (df2_month["판매액(원)"] / 1_000_000).round(1)
        df2_month = df2_month[["카테고리", "기간구분", "기간명", "채널명", "판매액(백만원)", "점유율(%)"]]
    else:
        df2_month = pd.DataFrame(columns=["카테고리", "기간구분", "기간명", "채널명", "판매액(백만원)", "점유율(%)"])

    df_tab2_raw = pd.concat([df2_year, df2_quarter, df2_month], ignore_index=True)
    cat_order_tab2 = ["YEARLY", "QUARTERLY", "MONTHLY"]
    if not df_tab2_raw.empty:
        df_tab2_raw["기간구분"] = pd.Categorical(df_tab2_raw["기간구분"], categories=cat_order_tab2, ordered=True)
        df_tab2_raw = df_tab2_raw.sort_values(["기간구분", "기간명", "채널명"])

    st.dataframe(df_tab2_raw, use_container_width=True)


# ======================================================
# ◆ TAB3 – 제조사별 판매
# ======================================================
with tab3:
    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # -------------------- 1) 년도 기준 -------------------->
    section_title("년도 기준")

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
        if not df_manuf_year_trend.empty:
            if not df_share_year_pivot.empty:
                manuf_order_year = list(df_share_year_pivot.columns)
            else:
                manuf_order_year = sort_order_year

            fig_manuf_year = go.Figure()
            numeric_font = get_font_size("CHART_NUMERIC")

            for manuf in manuf_order_year:
                df_temp = df_manuf_year_trend[df_manuf_year_trend["MANUF"] == manuf]
                if df_temp.empty:
                    continue
                fig_manuf_year.add_trace(go.Bar(
                    x=df_temp["YEARLY"],
                    y=df_temp["AMOUNT_M"],
                    name=manuf,
                    marker_color=color_map.get(manuf),
                    text=df_temp["AMOUNT_M"].apply(lambda v: f"{v:,.0f}"),
                    textposition="inside",
                    textfont=dict(size=numeric_font),  # ★ 막대 숫자 폰트
                    width=0.7
                ))

            fig_manuf_year.update_yaxes(tickformat=",.0f", rangemode="tozero")
            fig_manuf_year.update_xaxes(categoryorder="category ascending")
            
            # ★ 누적 막대 그래프 설정 복원
            fig_manuf_year.update_layout(barmode="stack")

            # -------- 제조사별 성장률 계산 --------
            df_growth_m = df_manuf_year_trend.copy()
            df_growth_m["MANUF"] = pd.Categorical(
                df_growth_m["MANUF"],
                categories=manuf_order_year,
                ordered=True
            )
            df_growth_m = df_growth_m.sort_values(["YEARLY", "MANUF"])

            df_growth_m["CUM"] = df_growth_m.groupby("YEARLY")["AMOUNT_M"].cumsum()
            df_growth_m["BOTTOM"] = df_growth_m["CUM"] - df_growth_m["AMOUNT_M"]
            df_growth_m["MID"] = df_growth_m["BOTTOM"] + df_growth_m["AMOUNT_M"] / 2

            for manuf in manuf_order_year:
                sub = df_growth_m[df_growth_m["MANUF"] == manuf].sort_values("YEARLY")

                years = sub["YEARLY"].tolist()
                vals = sub["AMOUNT_M"].tolist()
                mids = sub["MID"].tolist()

                for i in range(1, len(years)):
                    prev_val = vals[i - 1]
                    curr_val = vals[i]
                    if prev_val == 0:
                        continue

                    rate = (curr_val - prev_val) / prev_val * 100
                    arrow = "▲" if rate >= 0 else "▼"
                    color = "red" if rate >= 0 else "blue"

                    x_pos = i - 0.5
                    y_pos = mids[i]

                    fig_manuf_year.add_annotation(
                        x=x_pos,
                        y=y_pos,
                        xref="x",
                        yref="y",
                        text=f"{arrow} {abs(rate):.1f}%",
                        showarrow=False,
                        # ★ 성장률은 NUMERIC 대상 아님
                        font=dict(color=color, size=14),
                    )

            fig_manuf_year = apply_axis_legend_fonts(
                fig_manuf_year,
                xaxis_title="년도",
                yaxis_title="판매액(백만원)"
            )
            chart_card(f"년도별 / 제조사별 판매액 (백만원, {selected_channel_name})", fig_manuf_year, category_group)
        else:
            with st.container(border=True):
                st.caption(f"년도별 / 제조사별 판매액 (백만원, {selected_channel_name})")
                st.info("년도(YEARLY) 기준 제조사 데이터가 없습니다.")

    with col_y2:
        fig_year_share = go.Figure()
        if not df_share_year_pivot.empty:
            numeric_font = get_font_size("CHART_NUMERIC")
            
            for manuf in df_share_year_pivot.columns:
                fig_year_share.add_trace(go.Bar(
                    x=df_share_year_pivot.index,
                    y=df_share_year_pivot[manuf],
                    name=manuf,
                    marker_color=color_map.get(manuf),
                    text=df_share_year_pivot[manuf].apply(lambda x: f"{x:.1f}%"),
                    textposition="inside",
                    textfont=dict(size=numeric_font),
                    width=0.7
                ))
                
            fig_year_share.update_layout(
                barmode="stack",
                yaxis=dict(range=[0, 100]),
                xaxis=dict(),
                legend_traceorder="reversed",
                # ✅ 폰트가 자동으로 작아지는 것 방지
                #uniformtext_minsize=numeric_font,
                #uniformtext_mode="show"
            )
            fig_year_share = apply_axis_legend_fonts(
                fig_year_share,
                xaxis_title="년도",
                yaxis_title="점유율 (%)"
            )
            chart_card(f"년도별 / 제조사별 시장점유율 (%, {selected_channel_name})", fig_year_share, category_group)
        else:
            with st.container(border=True):
                st.caption(f"년도별 / 제조사별 시장점유율 (%, {selected_channel_name})")
                st.info("년도(YEARLY) 기준 점유율을 계산할 데이터가 없습니다.")

    # -------------------- 2) 분기 기준 -------------------->
    section_title("분기 기준")

    df_manuf_quarter_trend = df_manuf_quarter_final.copy()
    if not df_manuf_quarter_trend.empty:
        df_manuf_quarter_trend["AMOUNT_M"] = df_manuf_quarter_trend["AMOUNT_P"] / 1_000_000
        df_manuf_quarter_trend["MANUF"] = pd.Categorical(
            df_manuf_quarter_trend["MANUF"],
            categories=sort_order_quarter,
            ordered=True
        )
        df_manuf_quarter_trend = df_manuf_quarter_trend.sort_values(["QUARTERLY", "MANUF"])
        df_manuf_quarter_trend = df_manuf_quarter_trend.drop(columns=["AMOUNT_P"], errors="ignore")

    col_q1, col_q2 = st.columns(2)

    with col_q1:
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
            fig_manuf_quarter.update_yaxes(tickformat=",.0f", rangemode="tozero")
            fig_manuf_quarter.update_xaxes(categoryorder="category ascending")

            fig_manuf_quarter = apply_axis_legend_fonts(
                fig_manuf_quarter,
                xaxis_title="분기",
                yaxis_title="판매액(백만원)"
            )
            chart_card(f"분기별 / 제조사별 판매액 (백만원, {selected_channel_name})", fig_manuf_quarter, category_group)
        else:
            with st.container(border=True):
                st.caption(f"분기별 / 제조사별 판매액 (백만원, {selected_channel_name})")
                st.info("분기(QUARTERLY) 기준 제조사 데이터가 없습니다.")

    with col_q2:
        fig_quarter_share = go.Figure()
        if not df_share_quarter_pivot.empty:
            numeric_font = get_font_size("CHART_NUMERIC")
            for manuf in df_share_quarter_pivot.columns:
                fig_quarter_share.add_trace(go.Bar(
                    x=df_share_quarter_pivot.index,
                    y=df_share_quarter_pivot[manuf],
                    name=manuf,
                    marker_color=color_map.get(manuf),
                    text=df_share_quarter_pivot[manuf].apply(lambda x: f"{x:.1f}%"),
                    textposition="inside",
                    textfont=dict(size=numeric_font)
                ))
            fig_quarter_share.update_layout(
                barmode="stack",
                yaxis=dict(range=[0, 100]),
                xaxis=dict(),
                legend_traceorder="reversed"
            )
            fig_quarter_share = apply_axis_legend_fonts(
                fig_quarter_share,
                xaxis_title="분기",
                yaxis_title="점유율(%)"
            )
            chart_card(f"분기별 / 제조사별 시장점유율 (%, {selected_channel_name})", fig_quarter_share, category_group)
        else:
            with st.container(border=True):
                st.caption(f"분기별 / 제조사별 시장점유율 (%, {selected_channel_name})")
                st.info("분기(QUARTERLY) 기준 점유율을 계산할 데이터가 없습니다.")

    # -------------------- 3) 월 기준 -------------------->
    section_title("월 기준")

    df_trend = df_manuf_final.copy()
    if not df_trend.empty:
        df_trend["AMOUNT_M"] = df_trend["AMOUNT_P"] / 1_000_000
        df_trend["MANUF"] = pd.Categorical(
            df_trend["MANUF"],
            categories=sort_order_month,
            ordered=True
        )
        df_trend = df_trend.sort_values(["MONTHLY", "MANUF"])
        df_trend = df_trend.drop(columns=["AMOUNT_P"], errors="ignore")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
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
            fig_trend.update_yaxes(tickformat=",.0f", rangemode="tozero")
            fig_trend.update_xaxes(categoryorder="category ascending")

            fig_trend = apply_axis_legend_fonts(
                fig_trend,
                xaxis_title="월",
                yaxis_title="판매액(백만원)"
            )
            chart_card(f"월별 / 제조사별 판매액 (백만원, {selected_channel_name})", fig_trend, category_group)
        else:
            with st.container(border=True):
                st.caption(f"월별 / 제조사별 판매액 (백만원, {selected_channel_name})")
                st.info("월 기준 제조사 데이터가 없습니다.")

    with col_m2:
        fig_month_share = go.Figure()
        if not df_share_pivot.empty:
            numeric_font = get_font_size("CHART_NUMERIC")
            for manuf in df_share_pivot.columns:
                fig_month_share.add_trace(go.Bar(
                    x=df_share_pivot.index,
                    y=df_share_pivot[manuf],
                    name=manuf,
                    marker_color=color_map.get(manuf),
                    text=df_share_pivot[manuf].apply(lambda x: f"{x:.1f}%"),
                    textposition="inside",
                    textfont=dict(size=numeric_font)
                ))
            fig_month_share.update_layout(
                barmode="stack",
                yaxis=dict(range=[0, 100]),
                xaxis=dict(),
                legend_traceorder="reversed"
            )
            fig_month_share = apply_axis_legend_fonts(
                fig_month_share,
                xaxis_title="월",
                yaxis_title="점유율 (%)"
            )
            chart_card(f"월별 / 제조사별 시장점유율 (%, {selected_channel_name})", fig_month_share, category_group)
        else:
            with st.container(border=True):
                st.caption(f"월별 / 제조사별 시장점유율 (%, {selected_channel_name})")
                st.info("월 기준 점유율을 계산할 데이터가 없습니다.")

    # ----- TAB3 RAW DATA -----
    section_title(f"Raw Data (제조사별 합계/점유율, 백만원 단위, {selected_channel_name})")

    if not df_manuf_year_final.empty:
        df3_year = df_manuf_year_final.merge(
            df_share_year[["MANUF", "YEARLY", "SHARE"]],
            on=["MANUF", "YEARLY"],
            how="left"
        )
        df3_year["기간구분"] = "YEARLY"
        df3_year["기간명"] = df3_year["YEARLY"]
        df3_year["채널명"] = selected_channel_name
        df3_year["카테고리"] = category_group
        df3_year.rename(columns={
            "MANUF": "제조사명",
            "AMOUNT_P": "판매액(원)",
            "SHARE": "점유율(%)"
        }, inplace=True)
        df3_year["판매액(백만원)"] = (df3_year["판매액(원)"] / 1_000_000).round(1)
        df3_year = df3_year[["카테고리", "기간구분", "기간명", "채널명", "제조사명", "판매액(백만원)", "점유율(%)"]]
    else:
        df3_year = pd.DataFrame(columns=["카테고리", "기간구분", "기간명", "채널명", "제조사명", "판매액(백만원)", "점유율(%)"])

    if not df_manuf_quarter_final.empty:
        df3_quarter = df_manuf_quarter_final.merge(
            df_share_quarter[["MANUF", "QUARTERLY", "SHARE"]],
            on=["MANUF", "QUARTERLY"],
            how="left"
        )
        df3_quarter["기간구분"] = "QUARTERLY"
        df3_quarter["기간명"] = df3_quarter["QUARTERLY"]
        df3_quarter["채널명"] = selected_channel_name
        df3_quarter["카테고리"] = category_group
        df3_quarter.rename(columns={
            "MANUF": "제조사명",
            "AMOUNT_P": "판매액(원)",
            "SHARE": "점유율(%)"
        }, inplace=True)
        df3_quarter["판매액(백만원)"] = (df3_quarter["판매액(원)"] / 1_000_000).round(1)
        df3_quarter = df3_quarter[["카테고리", "기간구분", "기간명", "채널명", "제조사명", "판매액(백만원)", "점유율(%)"]]
    else:
        df3_quarter = pd.DataFrame(columns=["카테고리", "기간구분", "기간명", "채널명", "제조사명", "판매액(백만원)", "점유율(%)"])

    if not df_manuf_final.empty:
        df3_month = df_manuf_final.merge(
            df_share[["MANUF", "MONTHLY", "SHARE"]],
            on=["MANUF", "MONTHLY"],
            how="left"
        )
        df3_month["기간구분"] = "MONTHLY"
        df3_month["기간명"] = df3_month["MONTHLY"]
        df3_month["채널명"] = selected_channel_name
        df3_month["카테고리"] = category_group
        df3_month.rename(columns={
            "MANUF": "제조사명",
            "AMOUNT_P": "판매액(원)",
            "SHARE": "점유율(%)"
        }, inplace=True)
        df3_month["판매액(백만원)"] = (df3_month["판매액(원)"] / 1_000_000).round(1)
        df3_month = df3_month[["카테고리", "기간구분", "기간명", "채널명", "제조사명", "판매액(백만원)", "점유율(%)"]]
    else:
        df3_month = pd.DataFrame(columns=["카테고리", "기간구분", "기간명", "채널명", "제조사명", "판매액(백만원)", "점유율(%)"])

    df_tab3_raw = pd.concat([df3_year, df3_quarter, df3_month], ignore_index=True)
    cat_order_tab3 = ["YEARLY", "QUARTERLY", "MONTHLY"]
    if not df_tab3_raw.empty:
        df_tab3_raw["기간구분"] = pd.Categorical(df_tab3_raw["기간구분"], categories=cat_order_tab3, ordered=True)
        df_tab3_raw = df_tab3_raw.sort_values(["기간구분", "기간명", "제조사명"])

    st.dataframe(df_tab3_raw, use_container_width=True)
