import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import math
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import IsolationForest
import feedparser
import streamlit.components.v1 as components
from datetime import datetime, timezone

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="NETGUARD AI",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# NETGUARD AI - Matrix / Cyberpunk theme (APPEARANCE ONLY)
#
# HOW TO USE
# In app.py, delete everything from the line
#     # ---- NETGUARD visual theme for every Plotly chart (colours only) ----
# down to and including the closing  """, unsafe_allow_html=True)
# of the big <style> block (about lines 25-214), and paste ALL of this
# file in its place. Your imports at the top and everything below
# stay exactly as they are. No logic is touched. No new pip packages.
# ============================================================

# ---- NETGUARD visual theme for every Plotly chart ----
_ng_template = go.layout.Template(pio.templates["plotly_dark"])
_ng_template.layout.update(
    colorway=["#0ba0c2", "#27ae60", "#c0392b", "#d4a017", "#7f8c8d", "#bdc3c7"],
    font=dict(family="Inter, system-ui, sans-serif", color="#d4d7d4"),
    title=dict(font=dict(size=15, color="#d4d7d4", weight=500)),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="rgba(212,215,212,0.07)", zerolinecolor="rgba(212,215,212,0.12)", linecolor="rgba(212,215,212,0.10)", tickfont=dict(color="rgba(212,215,212,0.45)", size=11)),
    yaxis=dict(gridcolor="rgba(212,215,212,0.07)", zerolinecolor="rgba(212,215,212,0.12)", linecolor="rgba(212,215,212,0.10)", tickfont=dict(color="rgba(212,215,212,0.45)", size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(212,215,212,0.08)", font=dict(color="rgba(212,215,212,0.65)", size=12)),
    hoverlabel=dict(bgcolor="#191d19", bordercolor="rgba(212,215,212,0.15)", font=dict(color="#d4d7d4")),
)
pio.templates["netguard"] = _ng_template

# ==========================================
# 2. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Fira+Code:wght@400;500&display=swap');

:root {
    --ng-bg:         #0d0f0d;
    --ng-s1:         #131614;
    --ng-s2:         #191d19;
    --ng-border:     rgba(212,215,212,0.08);
    --ng-text:       #d4d7d4;
    --ng-text-2:     rgba(212,215,212,0.60);
    --ng-text-3:     rgba(212,215,212,0.35);
    --ng-accent:     #0ba0c2;
    --ng-attack:     #c0392b;
    --ng-benign:     #27ae60;
    --ng-amber:      #d4a017;
    --ng-mono:       'Fira Code', 'Courier New', monospace;
}

/* ---- App canvas ---- */
.stApp {
    background: var(--ng-bg);
    color: var(--ng-text);
    font-family: 'Inter', system-ui, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }

/* ---- Headings ---- */
h1, h2, h3, h4 { font-family: 'Inter', system-ui, sans-serif; color: var(--ng-text); }
h1 { font-weight: 600; font-size: 1.5rem; letter-spacing: -0.01em; }
h2 { font-weight: 500; font-size: 1.1rem; border-left: 2px solid var(--ng-accent); padding-left: 10px !important; }
h3 { font-weight: 500; font-size: 1rem; }

hr {
    border: none !important;
    height: 1px !important;
    background: var(--ng-border) !important;
    margin: 28px 0 !important;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: var(--ng-s1);
    border-right: 1px solid var(--ng-border);
}
[data-testid="stSidebar"] h1 {
    color: var(--ng-text);
    font-family: 'Inter', system-ui, sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0;
}

/* Nav radio items */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: transparent;
    border: none;
    border-left: 2px solid transparent;
    border-radius: 0;
    padding: 10px 12px;
    font-family: 'Inter', system-ui, sans-serif;
    font-weight: 400;
    font-size: 0.875rem;
    color: var(--ng-text-2);
    transition: background-color 150ms ease, color 150ms ease, border-color 150ms ease;
    width: 100%;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    border-left-color: rgba(11,160,194,0.4);
    background: rgba(11,160,194,0.05);
    color: var(--ng-text);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: rgba(11,160,194,0.08);
    border-left-color: var(--ng-accent);
    color: var(--ng-accent);
}

/* ---- Metric cards ---- */
.metric-card {
    background: var(--ng-s1);
    padding: 20px;
    border-radius: 10px;
    border: 1px solid var(--ng-border);
    box-shadow: 0 1px 3px rgba(0,0,0,0.5);
    transition: background-color 150ms ease, border-color 150ms ease;
}
.metric-card:hover {
    border-color: rgba(212,215,212,0.14);
    background: var(--ng-s2);
}
.metric-title {
    color: var(--ng-text-2);
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 0.75rem;
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
}
.metric-value {
    color: var(--ng-text);
    font-family: var(--ng-mono);
    font-size: 2rem;
    font-weight: 500;
    letter-spacing: -0.02em;
    margin-top: 6px;
    font-variant-numeric: tabular-nums;
}

/* ---- Native st.metric ---- */
[data-testid="stMetric"] {
    background: var(--ng-s1);
    border: 1px solid var(--ng-border);
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.5);
}
[data-testid="stMetricLabel"] {
    color: var(--ng-text-2);
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 0.75rem;
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
}
[data-testid="stMetricValue"] {
    color: var(--ng-text);
    font-family: var(--ng-mono);
    font-weight: 500;
    font-variant-numeric: tabular-nums;
}

/* ---- Charts, tables, alerts, expanders ---- */
[data-testid="stPlotlyChart"] {
    background: var(--ng-s1);
    border: 1px solid var(--ng-border);
    border-radius: 12px;
    padding: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.5);
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--ng-border);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.5);
}
[data-testid="stAlert"] {
    border-radius: 8px;
    border: 1px solid var(--ng-border);
    background: var(--ng-s1);
}
[data-testid="stExpander"] {
    border: 1px solid var(--ng-border);
    border-radius: 8px;
    background: var(--ng-s1);
}

/* ---- Buttons ---- */
.stButton > button, .stDownloadButton > button, [data-testid="stBaseLinkButton-secondary"] {
    background: var(--ng-s2);
    color: var(--ng-text);
    border: 1px solid var(--ng-border);
    border-radius: 6px;
    font-family: 'Inter', system-ui, sans-serif;
    font-weight: 500;
    font-size: 0.8125rem;
    letter-spacing: 0;
    padding: 0.4rem 1rem;
    transition: background-color 150ms ease, border-color 150ms ease, color 150ms ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stBaseLinkButton-secondary"]:hover {
    background: rgba(11,160,194,0.10);
    color: var(--ng-accent);
    border-color: rgba(11,160,194,0.35);
}

/* ---- Inputs ---- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stTextInput input, .stTextArea textarea {
    background-color: var(--ng-s2) !important;
    border-color: var(--ng-border) !important;
    border-radius: 6px !important;
    color: var(--ng-text) !important;
}
[data-baseweb="select"] > div:focus-within, .stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--ng-accent) !important;
    box-shadow: 0 0 0 1px rgba(11,160,194,0.3) !important;
}
[data-baseweb="tag"] { background-color: var(--ng-accent) !important; color: #0d0f0d !important; border-radius: 4px !important; }
[data-baseweb="tag"] span { color: #0d0f0d !important; }
[data-testid="stSlider"] [role="slider"] { background-color: var(--ng-accent) !important; }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab"] { color: var(--ng-text-2); font-family: 'Inter', system-ui, sans-serif; font-size: 0.875rem; }
.stTabs [aria-selected="true"] { color: var(--ng-text) !important; font-weight: 500; }
[data-baseweb="tab-highlight"] { background-color: var(--ng-accent) !important; }

/* ---- Captions ---- */
[data-testid="stCaptionContainer"] { color: var(--ng-text-3); font-family: 'Inter', system-ui, sans-serif; font-size: 0.75rem; }

/* ---- Scrollbars ---- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--ng-bg); }
::-webkit-scrollbar-thumb { background: rgba(212,215,212,0.15); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(212,215,212,0.30); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CYBER THREAT INTELLIGENCE NEWS FEED
# ==========================================

@st.cache_data(ttl=36000)
def get_cybersecurity_news():

    rss_feeds = {
        "The Hacker News": "https://feeds.feedburner.com/TheHackersNews",
        "BleepingComputer": "https://www.bleepingcomputer.com/feed/",
        "CISA Alerts": "https://www.cisa.gov/cybersecurity-advisories/all.xml"
    }

    news_items = []

    for source, feed_url in rss_feeds.items():

        try:
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:10]:

                published = entry.get(
                    "published",
                    entry.get("updated", "Date unavailable")
                )

                news_items.append({
                    "title": entry.get("title", "Untitled threat report"),
                    "link": entry.get("link", "#"),
                    "source": source,
                    "published": published,
                    "summary": entry.get(
                        "summary",
                        "Open the original article for more information."
                    )
                })

        except Exception:
            continue

    news_items.sort(
        key=lambda item: item["published"],
        reverse=True
    )

    return news_items[:15]

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("Netguard AI")
st.sidebar.markdown("##### Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Overview",
        "Network Traffic",
        "Threat Analysis",
        "Live Monitoring",
        "Threat Map (Demo)",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Network threat detection · CIC-IDS2017")

# ==========================================
# 4. LOAD DATASET
# ==========================================

# ==========================================
# FEATURE 2: AI ANOMALY DETECTION
# ==========================================

@st.cache_resource
def train_anomaly_detection_model():

    train_path = Path("data/processed/X_train.csv")

    if not train_path.exists():
        raise FileNotFoundError("Training dataset not found!")

    X_train = pd.read_csv(train_path)

    anomaly_model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    anomaly_model.fit(X_train)

    return anomaly_model, X_train.columns.tolist()

file_path = Path("data/processed/day6_predictions.csv")

if not file_path.exists():
    st.error("Prediction dataset not found!")
    st.stop()


df = pd.read_csv(file_path)

# ==========================================
# DAY 13: CONNECT PREDICTIONS WITH TRAFFIC
# ==========================================

test_data_path = Path("data/processed/X_test.csv")

if not test_data_path.exists():
    st.error("Original network traffic dataset not found!")
    st.stop()

test_data = pd.read_csv(test_data_path)
# ==========================================
# RUN ISOLATION FOREST ANOMALY DETECTION
# ==========================================

anomaly_model, training_features = (
    train_anomaly_detection_model()
)

if list(test_data.columns) != training_features:
    st.error("Training and testing features do not match!")
    st.stop()

# Calculate anomaly scores BEFORE adding new columns
anomaly_scores = -anomaly_model.decision_function(test_data)

anomaly_predictions = anomaly_model.predict(test_data)

test_data["Anomaly_Status"] = [
    "ANOMALY" if prediction == -1 else "NORMAL"
    for prediction in anomaly_predictions
]

test_data["Anomaly_Score"] = anomaly_scores

if len(test_data) != len(df):
    st.error("Traffic records and prediction records do not match!")
    st.stop()

# Combine original traffic features with predictions
df = pd.concat(
    [
        test_data.reset_index(drop=True),
        df.reset_index(drop=True)
    ],
    axis=1
)

st.success("Network traffic successfully connected with AI predictions!")


required_columns = [
    "Actual",
    "Predicted",
    "Attack_Probability"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

df["Attack_Probability"] = pd.to_numeric(
    df["Attack_Probability"],
    errors="coerce"
)

df = df.dropna(subset=["Attack_Probability"])

# ==========================================
# 5. CALCULATE METRICS
# ==========================================

total_traffic = len(df)

benign_traffic = int(
    (df["Predicted"] == "BENIGN").sum()
)

attack_traffic = int(
    (df["Predicted"] == "ATTACK").sum()
)

correct_predictions = int(
    (df["Actual"] == df["Predicted"]).sum()
)

incorrect_predictions = total_traffic - correct_predictions

accuracy = (
    correct_predictions / total_traffic * 100
    if total_traffic > 0 else 0
)

# ==========================================
# 6. MAIN HEADER
# ==========================================

st.title("Netguard AI")
st.caption("Network threat detection · CIC-IDS2017 · Random Forest classifier")

st.markdown("---")

# ==========================================
# 7. OVERVIEW PAGE
# ==========================================

if page == "Overview":

    st.header("Security overview")

    col1, col2, col3, col4 = st.columns(4)

    metric_data = [
        (col1, "Total Traffic", f"{total_traffic:,}"),
        (col2, "Benign Traffic", f"{benign_traffic:,}"),
        (col3, "Detected Attacks", f"{attack_traffic:,}"),
        (col4, "Model Accuracy", f"{accuracy:.2f}%")
    ]

    for column, label, value in metric_data:

        with column:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ==========================================
    # LATEST CYBERSECURITY NEWS
    # ==========================================

    st.markdown("---")
    st.subheader("Threat intelligence feed")
    st.caption(
        "External cybersecurity news from The Hacker News, BleepingComputer and CISA "
        "· Refreshes every 10 hours"
    )

    col_news, col_refresh = st.columns([4, 1])

    with col_refresh:
        if st.button("Refresh news"):
            get_cybersecurity_news.clear()
            st.rerun()

    news_items = get_cybersecurity_news()

    if news_items:

        with st.container(height=420, border=True):

            for news in news_items:

                st.markdown(f"### {news['title']}")

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.caption(
                        f"{news['source']} · {news['published']}"
                    )

                with col2:
                    st.link_button(
                        "Read article ↗",
                        news["link"],
                        use_container_width=True
                    )

                st.markdown("---")

    else:

        st.warning(
            "Unable to load cybersecurity news at the moment. "
            "Please check your internet connection and try again."
        )
    # ==========================================
    # NETWORK TRAFFIC DISTRIBUTION
    # ==========================================

    st.markdown("---")
    st.subheader("Network traffic distribution")

    traffic_data = pd.DataFrame({
        "Traffic Type": ["BENIGN", "ATTACK"],
        "Count": [benign_traffic, attack_traffic]
    })

    pie_chart = px.pie(
        traffic_data,
        names="Traffic Type",
        values="Count",
        hole=0.55,
        color="Traffic Type",
        color_discrete_map={
            "BENIGN": "#22c55e",
            "ATTACK": "#ef4444"
        }
    )

    pie_chart.update_traces(
        marker=dict(line=dict(color="#0f0f10", width=3)),
        textfont=dict(color="#ffffff")
    )

    pie_chart.update_layout(
        template="netguard",
        paper_bgcolor="#0f0f10",
        font_color="white"
    )

    st.plotly_chart(pie_chart, width="stretch")

    st.markdown("---")

    st.subheader("Recent Network Predictions")

    
    st.dataframe(
        df.head(15),
        use_container_width=True
    )


# ==========================================
# 8. NETWORK TRAFFIC PAGE
# ==========================================

elif page == "Network Traffic":

    # ==========================================
    # HIGH-TECH NETWORK COMMAND CENTRE
    # ==========================================

    st.header("Network traffic")
    st.caption("Advanced traffic analysis · Service-level topology · AI-assisted investigation")

    # NETWORK OVERVIEW METRICS

    total_network_records = len(df)

    total_attacks = (
        df["Predicted"] == "ATTACK"
    ).sum()

    total_benign = (
        df["Predicted"] == "BENIGN"
    ).sum()

    total_anomalies = (
        df["Anomaly_Status"] == "ANOMALY"
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total traffic</div>
                <div class="metric-value">{total_network_records:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Detected attacks</div>
                <div class="metric-value">{total_attacks:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Benign traffic</div>
                <div class="metric-value">{total_benign:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">AI anomalies</div>
                <div class="metric-value">{total_anomalies:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.header("🔎 Network Traffic Investigation")

        # ==========================================
    # FEATURE 9: INTERACTIVE NETWORK TOPOLOGY
    # ==========================================

    st.markdown("---")
    st.subheader("Network topology visualisation")

    st.caption(
        "Service-level traffic connections based on actual "
        "CIC-IDS2017 destination-port records."
    )

    topology_data = df.copy()

    topology_data["Destination Port"] = pd.to_numeric(
        topology_data["Destination Port"],
        errors="coerce"
    )

    topology_data = topology_data.dropna(
        subset=["Destination Port"]
    )

    if not topology_data.empty:

        # SUMMARISE TRAFFIC BY DESTINATION PORT

        port_summary = (
            topology_data.groupby(
                ["Destination Port", "Predicted"]
            )
            .size()
            .unstack(fill_value=0)
        )

        for traffic_type in ["BENIGN", "ATTACK"]:
            if traffic_type not in port_summary.columns:
                port_summary[traffic_type] = 0

        port_summary["Total"] = (
            port_summary["BENIGN"] + port_summary["ATTACK"]
        )

        port_summary = port_summary.sort_values(
            "Total",
            ascending=False
        ).head(12)

        port_count = len(port_summary)

        # CREATE NETWORK NODE POSITIONS

        edge_traces = []

        node_x = []
        node_y = []
        node_labels = []
        node_sizes = []
        node_hover = []

        for index, (port, row) in enumerate(
            port_summary.iterrows()
        ):

            angle = (
                2 * math.pi * index / max(port_count, 1)
            )

            x = 3 * math.cos(angle)
            y = 3 * math.sin(angle)

            benign = int(row["BENIGN"])
            attacks = int(row["ATTACK"])
            total = int(row["Total"])

            attack_ratio = (
                attacks / total if total > 0 else 0
            )

            edge_color = (
                "#ef4444" if attacks > benign else "#22c55e"
            )

            edge_width = min(
                2 + math.log1p(total),
                10
            )

            # CONNECTION BETWEEN CENTRAL NODE AND PORT

            edge_traces.append(
                go.Scatter(
                    x=[0, x],
                    y=[0, y],
                    mode="lines",
                    line=dict(
                        color=edge_color,
                        width=edge_width
                    ),
                    hovertemplate=(
                        f"Destination Port: {int(port)}"
                        f"<br>Total Traffic: {total:,}"
                        f"<br>Benign: {benign:,}"
                        f"<br>Attacks: {attacks:,}"
                        f"<br>Attack Ratio: {attack_ratio:.1%}"
                        "<extra></extra>"
                    ),
                    showlegend=False
                )
            )

            node_x.append(x)
            node_y.append(y)
            node_labels.append(f"PORT {int(port)}")
            node_sizes.append(
                min(18 + math.log1p(total) * 3, 45)
            )

            node_hover.append(
                f"Port: {int(port)}"
                f"<br>Total Records: {total:,}"
                f"<br>Benign: {benign:,}"
                f"<br>Attacks: {attacks:,}"
                f"<br>Attack Ratio: {attack_ratio:.1%}"
            )

        # CREATE INTERACTIVE NETWORK FIGURE

        topology_figure = go.Figure()

        for trace in edge_traces:
            topology_figure.add_trace(trace)

        # DESTINATION PORT NODES

        topology_figure.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=node_labels,
                textposition="bottom center",
                marker=dict(
                    size=node_sizes,
                    color="#ff7a1a",
                    line=dict(
                        color="#ffffff",
                        width=1
                    )
                ),
                hovertext=node_hover,
                hoverinfo="text",
                name="Destination Services"
            )
        )

        # CENTRAL NETGUARD NODE

        topology_figure.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers+text",
                text=["NETGUARD AI"],
                textposition="bottom center",
                marker=dict(
                    size=48,
                    color="#1f1f23",
                    line=dict(
                        color="#ff7a1a",
                        width=3
                    )
                ),
                hovertemplate=(
                    "NETGUARD AI<br>"
                    "Network Traffic Monitoring"
                    "<extra></extra>"
                ),
                name="Monitoring Centre"
            )
        )

        topology_figure.update_layout(
            title="NETWORK SERVICE CONNECTION MAP",
            template="netguard",
            paper_bgcolor="#0f0f10",
            plot_bgcolor="#0f0f10",
            font=dict(
                color="#f2f2f3"
            ),
            height=650,
            showlegend=False,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            ),
            xaxis=dict(
                visible=False,
                range=[-4.5, 4.5]
            ),
            yaxis=dict(
                visible=False,
                range=[-4.5, 4.5],
                scaleanchor="x",
                scaleratio=1
            )
        )

        st.plotly_chart(
            topology_figure,
            use_container_width=True
        )

        st.markdown(
            """
            <div style="
                background:#131315;
                padding:15px;
                border-radius:10px;
                border:1px solid #2a2a2e;
            ">
                <span style="color:#22c55e;">
                    ● Green
                </span>
                Benign-dominant connection
                &nbsp;&nbsp;
                <span style="color:#ef4444;">
                    ● Red
                </span>
                Attack-dominant connection
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "No valid destination-port records are available."
        )

    st.header("Network Traffic Analysis")
    # ==========================================
    # DAY 13: NETWORK TRAFFIC INVESTIGATION
    # ==========================================

    st.subheader("Traffic investigation filters")

    col1, col2, col3 = st.columns(3)

    # DESTINATION PORT FILTER
    with col1:

        available_ports = sorted(
            df["Destination Port"].dropna().unique().tolist()
        )

        selected_port = st.selectbox(
            "Destination Port",
            ["ALL"] + available_ports
        )

    # PREDICTION FILTER
    with col2:

        investigation_type = st.selectbox(
            "Traffic Classification",
            ["ALL", "BENIGN", "ATTACK"]
        )

    # ATTACK PROBABILITY FILTER
    with col3:

        min_probability = st.slider(
            "Minimum Attack Probability",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )

    # APPLY INVESTIGATION FILTERS
    investigation_df = df.copy()

    if selected_port != "ALL":

        investigation_df = investigation_df[
            investigation_df["Destination Port"] == selected_port
        ]

    if investigation_type != "ALL":

        investigation_df = investigation_df[
            investigation_df["Predicted"] == investigation_type
        ]

    investigation_df = investigation_df[
        investigation_df["Attack_Probability"] >= min_probability
    ]

    st.metric(
        "Matching Network Records",
        f"{len(investigation_df):,}"
    )

    st.markdown("---")

    st.subheader("Investigated Network Records")

    st.dataframe(
        investigation_df,
        use_container_width=True
    )
    st.markdown("---")

    # INDIVIDUAL THREAT INVESTIGATION

    st.subheader("Individual traffic investigation")

    if not investigation_df.empty:

        selected_record = st.selectbox(
            "Select a network record to investigate",
            investigation_df.index.tolist()
        )

        record = investigation_df.loc[selected_record]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Destination Port",
            str(int(record["Destination Port"]))
        )

        col2.metric(
            "Attack Probability",
            f"{record['Attack_Probability'] * 100:.2f}%"
        )

        col3.metric(
            "Predicted Classification",
            record["Predicted"]
        )

        st.markdown("---")

        st.subheader("Network Traffic Characteristics")

        details = {
            "Actual Classification": record["Actual"],
            "Flow Duration": record["Flow Duration"],
            "Forward Packets": record["Total Fwd Packets"],
            "Backward Packets": record["Total Backward Packets"],
            "Flow Bytes per Second": record["Flow Bytes/s"],
            "Flow Packets per Second": record["Flow Packets/s"],
            "Average Packet Size": record["Average Packet Size"],
            "SYN Flag Count": record["SYN Flag Count"],
            "ACK Flag Count": record["ACK Flag Count"],
            "RST Flag Count": record["RST Flag Count"]
        }

        details_df = pd.DataFrame(
            list(details.items()),
            columns=["Network Feature", "Value"]
        )

        st.dataframe(
            details_df,
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown("---")

        st.subheader("Network threat-risk profile")

        risk_features = pd.DataFrame({
            "Network Behaviour": [
                "Forward Packets",
                "Backward Packets",
                "Flow Bytes",
                "Flow Packets",
                "SYN Flags",
                "RST Flags"
            ],
            "Value": [
                record["Total Fwd Packets"],
                record["Total Backward Packets"],
                record["Flow Bytes/s"],
                record["Flow Packets/s"],
                record["SYN Flag Count"],
                record["RST Flag Count"]
            ]
        })

        # NORMALIZE VALUES FOR VISUAL COMPARISON

        risk_features["Normalized Value"] = (
            risk_features["Value"].abs()
            / max(risk_features["Value"].abs().max(), 1)
        ) * 100

        radar_chart = px.line_polar(
            risk_features,
            r="Normalized Value",
            theta="Network Behaviour",
            line_close=True,
            range_r=[0, 100],
            title="Selected Traffic Behaviour Profile"
        )

        radar_chart.update_traces(
            fill="toself",
            line=dict(color="#ff7a1a", width=3),
            fillcolor="rgba(255,122,26,0.28)"
        )

        radar_chart.update_layout(
            template="netguard",
            paper_bgcolor="#0f0f10",
            font_color="white"
        )

        st.plotly_chart(
            radar_chart,
            use_container_width=True
        )
        
        st.markdown("---")

        st.subheader("Threat investigation report")

        report_data = pd.DataFrame([{
            "Record ID": selected_record,
            "Actual Classification": record["Actual"],
            "Predicted Classification": record["Predicted"],
            "Attack Probability (%)": round(
                record["Attack_Probability"] * 100, 2
            ),
            "Destination Port": record["Destination Port"],
            "Flow Duration": record["Flow Duration"],
            "Forward Packets": record["Total Fwd Packets"],
            "Backward Packets": record["Total Backward Packets"],
            "Flow Bytes per Second": record["Flow Bytes/s"],
            "Flow Packets per Second": record["Flow Packets/s"],
            "SYN Flag Count": record["SYN Flag Count"],
            "RST Flag Count": record["RST Flag Count"]
        }])

        st.dataframe(
            report_data,
            hide_index=True,
            use_container_width=True
        )

        report_csv = report_data.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download investigation report (CSV)",
            data=report_csv,
            file_name=f"netguard_investigation_{selected_record}.csv",
            mime="text/csv"
        )



    else:

        st.warning("No network records available for investigation.")


    selected_type = st.selectbox(
        "Select Traffic Type",
        ["ALL", "BENIGN", "ATTACK"]
    )

    if selected_type == "ALL":
        filtered_df = df.copy()
    else:
        filtered_df = df[
            df["Predicted"] == selected_type
        ]

    st.metric(
        "Total Filtered Records",
        f"{len(filtered_df):,}"
    )

    traffic_counts = (
        filtered_df["Predicted"]
        .value_counts()
        .rename_axis("Traffic Type")
        .reset_index(name="Count")
    )

    if not traffic_counts.empty:

        bar_chart = px.bar(
            traffic_counts,
            x="Traffic Type",
            y="Count",
            color="Traffic Type",
            color_discrete_map={
                "BENIGN": "#22c55e",
                "ATTACK": "#ef4444"
            },
            title="Network Traffic Classification"
        )

        bar_chart.update_layout(
            template="netguard",
            paper_bgcolor="#0f0f10",
            font_color="white"
        )

        st.plotly_chart(bar_chart, width="stretch")

    st.subheader("Traffic Records")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


# ==========================================
# 9. THREAT ANALYSIS AND SOC ALERTS
# ==========================================

elif page == "Threat Analysis":
    st.header("Security threat analysis")

    st.write(
        "Identify suspicious network traffic and monitor "
        "potential security threats."
    )

    st.markdown("---")

    # CREATE ATTACK ALERT RECORDS

    alerts_df = df[
        df["Predicted"] == "ATTACK"
    ].copy()

    # ASSIGN SEVERITY USING ATTACK PROBABILITY

    def assign_severity(probability):

        if probability >= 0.99:
            return "CRITICAL"

        elif probability >= 0.95:
            return "HIGH"

        elif probability >= 0.90:
            return "MEDIUM"

        else:
            return "LOW"

    alerts_df["Severity"] = (
        alerts_df["Attack_Probability"]
        .apply(assign_severity)
    )
        # ==========================================
    # FEATURE 1: MITRE ATT&CK THREAT MAPPING
    # ==========================================

    st.markdown("---")
    st.subheader("MITRE ATT&CK threat intelligence")

    st.write(
        "Map detected network attacks to recognised "
        "MITRE ATT&CK techniques."
    )

    if not alerts_df.empty:

        alerts_df["MITRE Technique ID"] = "T1498"
        alerts_df["MITRE Technique"] = (
            "Network Denial of Service"
        )
        alerts_df["MITRE Tactic"] = "Impact"

        mitre_count = len(alerts_df)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Mapped Attack Records",
            f"{mitre_count:,}"
        )

        col2.metric(
            "MITRE Technique ID",
            "T1498"
        )

        col3.metric(
            "ATT&CK Tactic",
            "Impact"
        )

        st.markdown("### Detected Attack Technique")

        st.markdown(
            """
            **T1498 – Network Denial of Service**

            Attackers attempt to make a network resource
            unavailable to legitimate users by disrupting
            its availability.
            """
        )

        st.dataframe(
            alerts_df[
                [
                    "Predicted",
                    "Attack_Probability",
                    "Severity",
                    "MITRE Technique ID",
                    "MITRE Technique",
                    "MITRE Tactic"
                ]
            ].head(100),
            use_container_width=True
        )

        st.markdown(
            "[View official MITRE ATT&CK documentation ↗]"
            "(https://attack.mitre.org/techniques/T1498/)"
        )

    else:

        st.info("No attack records available for MITRE mapping.")

        # ==========================================
    # FEATURE 2: ANOMALY DETECTION DASHBOARD
    # ==========================================

    st.markdown("---")
    st.subheader("AI anomaly detection")

    total_records = len(df)

    anomaly_count = (
        df["Anomaly_Status"] == "ANOMALY"
    ).sum()

    normal_count = (
        df["Anomaly_Status"] == "NORMAL"
    ).sum()

    anomaly_percentage = (
        (anomaly_count / total_records) * 100
        if total_records > 0 else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Anomalies",
        f"{anomaly_count:,}"
    )

    col2.metric(
        "Normal Traffic",
        f"{normal_count:,}"
    )

    col3.metric(
        "Anomaly Percentage",
        f"{anomaly_percentage:.2f}%"
    )

    # ANOMALY DISTRIBUTION CHART

    anomaly_chart_data = pd.DataFrame({
        "Traffic Status": ["Normal", "Anomaly"],
        "Records": [normal_count, anomaly_count]
    })

    anomaly_chart = px.pie(
        anomaly_chart_data,
        names="Traffic Status",
        values="Records",
        hole=0.55,
        title="Normal vs Anomalous Traffic",
        color="Traffic Status",
        color_discrete_map={
            "Normal": "#22c55e",
            "Anomaly": "#ff7a1a"
        }
    )

    anomaly_chart.update_traces(
        marker=dict(line=dict(color="#0f0f10", width=3)),
        textfont=dict(color="#ffffff")
    )

    anomaly_chart.update_layout(
        template="netguard",
        paper_bgcolor="#0f0f10",
        font_color="white"
    )

    st.plotly_chart(
        anomaly_chart,
        use_container_width=True
    )

    # SUSPICIOUS TRAFFIC RECORDS

    st.subheader("Suspicious traffic records")

    suspicious_traffic = df[
        df["Anomaly_Status"] == "ANOMALY"
    ].sort_values(
        by="Anomaly_Score",
        ascending=False
    )

    display_columns = [
        "Predicted",
        "Attack_Probability",
        "Anomaly_Status",
        "Anomaly_Score"
    ]

    st.dataframe(
        suspicious_traffic[display_columns].head(100),
        use_container_width=True
    )

        # ==========================================
    # FEATURE 5: DYNAMIC RISK SCORING
    # ==========================================

    st.markdown("---")
    st.subheader("Dynamic threat risk assessment")

    if not alerts_df.empty:

        # Calculate risk score using two AI signals

        alerts_df["Risk_Score"] = (
            alerts_df["Attack_Probability"] * 70
            + (
                alerts_df["Anomaly_Status"] == "ANOMALY"
            ).astype(int) * 30
        )

        # Assign risk categories

        def assign_risk_level(score):

            if score >= 90:
                return "CRITICAL"

            elif score >= 75:
                return "HIGH"

            elif score >= 50:
                return "MEDIUM"

            else:
                return "LOW"

        alerts_df["Risk_Level"] = (
            alerts_df["Risk_Score"].apply(assign_risk_level)
        )

        # Risk summary

        critical_risks = (
            alerts_df["Risk_Level"] == "CRITICAL"
        ).sum()

        high_risks = (
            alerts_df["Risk_Level"] == "HIGH"
        ).sum()

        medium_risks = (
            alerts_df["Risk_Level"] == "MEDIUM"
        ).sum()

        low_risks = (
            alerts_df["Risk_Level"] == "LOW"
        ).sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Critical Risks", f"{critical_risks:,}")
        col2.metric("High Risks", f"{high_risks:,}")
        col3.metric("Medium Risks", f"{medium_risks:,}")
        col4.metric("Low Risks", f"{low_risks:,}")

        # Highest risk records

        st.markdown("### Highest risk network records")

        highest_risks = alerts_df.sort_values(
            by="Risk_Score",
            ascending=False
        )

        st.dataframe(
            highest_risks[
                [
                    "Attack_Probability",
                    "Anomaly_Status",
                    "Risk_Score",
                    "Risk_Level",
                    "Severity"
                ]
            ].head(100),
            use_container_width=True
        )

    else:

        st.info("No attack records available for risk assessment.")

    st.caption(
        "Risk scores are prototype estimates based on model confidence "
        "and anomaly flags. They are not independently validated "
        "measures of real-world attack impact."
    )

    # ALERT SUMMARY

    st.subheader("Threat Alert Summary")

    critical_count = (
        alerts_df["Severity"] == "CRITICAL"
    ).sum()

    high_count = (
        alerts_df["Severity"] == "HIGH"
    ).sum()

    medium_count = (
        alerts_df["Severity"] == "MEDIUM"
    ).sum()

    low_count = (
        alerts_df["Severity"] == "LOW"
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Critical Alerts", f"{critical_count:,}")
    col2.metric("High Alerts", f"{high_count:,}")
    col3.metric("Medium Alerts", f"{medium_count:,}")
    col4.metric("Low Alerts", f"{low_count:,}")

    st.markdown("---")

    # ATTACK PROBABILITY CHART

    st.subheader("Attack Probability Distribution")

    histogram = px.histogram(
        alerts_df,
        x="Attack_Probability",
        nbins=30,
        title="Attack Prediction Confidence",
        labels={
            "Attack_Probability": "Attack Probability",
            "count": "Number of Alerts"
        }
    )

    histogram.update_traces(
        marker=dict(
            color="#ff7a1a",
            line=dict(color="#0f0f10", width=1)
        )
    )

    histogram.update_layout(
        template="netguard",
        paper_bgcolor="#0f0f10",
        font_color="white"
    )

    st.plotly_chart(
        histogram,
        use_container_width=True
    )

    st.markdown("---")

    # FILTER ALERTS
    # ==========================================
    # ADVANCED THREAT INTELLIGENCE FILTER
    # ==========================================

    st.markdown("---")
    st.subheader("Advanced threat intelligence")

    probability_threshold = st.slider(
        "Minimum Attack Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.90,
        step=0.01,
        format="%.2f"
    )

    high_confidence_alerts = alerts_df[
        alerts_df["Attack_Probability"] >= probability_threshold
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric("Confidence Threshold", f"{probability_threshold * 100:.0f}%")
    col2.metric("High-Confidence Alerts", f"{len(high_confidence_alerts):,}")
    col3.metric(
        "Critical Confidence Alerts",
        f"{(high_confidence_alerts['Attack_Probability'] >= 0.99).sum():,}"
    )

    st.caption(
        "These figures use predicted attack probability, "
        "not independently verified attack severity."
    )


    st.subheader("Detected Security Alerts")

    selected_severity = st.selectbox(
        "Filter by Severity",
        ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
    )

    if selected_severity == "ALL":

        filtered_alerts = alerts_df.copy()

    else:

        filtered_alerts = alerts_df[
            alerts_df["Severity"] == selected_severity
        ]

    st.metric(
        "Matching Security Alerts",
        f"{len(filtered_alerts):,}"
    )

    # DISPLAY ALERT TABLE

    if not filtered_alerts.empty:

        st.dataframe(
            filtered_alerts[
                [
                    "Actual",
                    "Predicted",
                    "Attack_Probability",
                    "Severity"
                ]
            ],
            use_container_width=True
        )

        # DOWNLOAD ALERTS

        csv_data = filtered_alerts.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download security alerts (CSV)",
            data=csv_data,
            file_name="netguard_security_alerts.csv",
            mime="text/csv"
        )

    else:

        st.success(
            "No alerts found for the selected severity."
        )

    st.markdown("---")

    st.caption(
        "Severity is assigned using model attack probability "
        "thresholds for this academic prototype."
    )

        # ==========================================
    # FEATURE 10: SOC ANALYST DASHBOARD
    # ==========================================

    st.markdown("---")
    st.subheader("SOC analyst command centre")

    st.caption(
        "Monitor, investigate and manage detected security incidents."
    )

    # INITIALISE INCIDENT STORAGE

    if "incident_status" not in st.session_state:
        st.session_state.incident_status = {}

    if "incident_notes" not in st.session_state:
        st.session_state.incident_notes = {}

    # INCIDENT SUMMARY

    total_incidents = len(alerts_df)

    incident_statuses = [
        st.session_state.incident_status.get(
            str(record_id), "Open"
        )
        for record_id in alerts_df.index
    ]

    open_count = incident_statuses.count("Open")
    investigating_count = incident_statuses.count("Investigating")
    resolved_count = incident_statuses.count("Resolved")
    false_positive_count = incident_statuses.count("False Positive")

    false_positive_rate = (
        (false_positive_count / total_incidents) * 100
        if total_incidents > 0 else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Incidents", f"{total_incidents:,}")
    col2.metric("Open Incidents", f"{open_count:,}")
    col3.metric("Investigating", f"{investigating_count:,}")
    col4.metric("Resolved", f"{resolved_count:,}")

    col1, col2 = st.columns(2)

    col1.metric(
        "False Positive Records",
        f"{false_positive_count:,}"
    )

    col2.metric(
        "Analyst False Positive Rate",
        f"{false_positive_rate:.2f}%"
    )

    
    # INCIDENT ANALYTICS CHARTS

    st.markdown("---")
    st.markdown("### Incident analytics")

    chart_col1, chart_col2 = st.columns(2)

    # DONUT CHART: INCIDENT STATUS

    status_chart_df = pd.DataFrame({
        "Status": [
            "Open",
            "Investigating",
            "Resolved",
            "False Positive"
        ],
        "Incidents": [
            open_count,
            investigating_count,
            resolved_count,
            false_positive_count
        ]
    })

    with chart_col1:

        st.markdown("#### Incident Status Distribution")

        status_fig = px.pie(
            status_chart_df,
            names="Status",
            values="Incidents",
            hole=0.55,
            title="Incident Status Breakdown",
            color="Status",
            color_discrete_map={
                "Open": "#ff7a1a",
                "Investigating": "#ffc83d",
                "Resolved": "#22c55e",
                "False Positive": "#6b7280"
            }
        )

        status_fig.update_traces(
            marker=dict(line=dict(color="#0f0f10", width=3))
        )

        status_fig.update_layout(
            template="netguard",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title="Incident Status"
        )

        st.plotly_chart(
            status_fig,
            use_container_width=True
        )

    # BAR CHART: INCIDENT SEVERITY

    severity_counts = alerts_df["Severity"].value_counts().reindex(
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        fill_value=0
    )

    severity_chart_df = pd.DataFrame({
        "Severity": severity_counts.index,
        "Incidents": severity_counts.values
    })

    with chart_col2:

        st.markdown("#### Incident Severity Distribution")

        severity_fig = px.bar(
            severity_chart_df,
            x="Severity",
            y="Incidents",
            color="Severity",
            color_discrete_map={
                "CRITICAL": "#ef4444",
                "HIGH": "#ff7a1a",
                "MEDIUM": "#ffc83d",
                "LOW": "#22c55e"
            },
            title="Incidents by Severity",
            text="Incidents"
        )

        severity_fig.update_layout(
            template="netguard",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Threat Severity",
            yaxis_title="Number of Incidents",
            showlegend=False
        )

        st.plotly_chart(
            severity_fig,
            use_container_width=True
        )


    st.markdown("---")

    # INCIDENT MANAGEMENT

    st.markdown("### Incident investigation")

    if total_incidents > 0:

        selected_incident = st.selectbox(
            "Select Incident Record",
            options=alerts_df.index.tolist(),
            format_func=lambda x: f"Incident #{x}"
        )

        selected_incident_key = str(selected_incident)

        selected_record = alerts_df.loc[selected_incident]

        st.write(
            f"**Predicted Classification:** "
            f"{selected_record['Predicted']}"
        )

        st.write(
            f"**Attack Probability:** "
            f"{selected_record['Attack_Probability']:.2%}"
        )

        st.write(
            f"**Severity:** {selected_record['Severity']}"
        )

        current_status = st.session_state.incident_status.get(
            selected_incident_key, "Open"
        )

        status_options = [
            "Open",
            "Investigating",
            "Resolved",
            "False Positive"
        ]

        selected_status = st.selectbox(
            "Update Incident Status",
            options=status_options,
            index=status_options.index(current_status)
        )

        analyst_note = st.text_area(
            "Analyst Investigation Notes",
            value=st.session_state.incident_notes.get(
                selected_incident_key, ""
            ),
            placeholder="Enter investigation findings..."
        )

        if st.button("Save incident update"):

            st.session_state.incident_status[
                selected_incident_key
            ] = selected_status

            st.session_state.incident_notes[
                selected_incident_key
            ] = analyst_note

            st.success("Incident update saved successfully!")

            st.rerun()

    else:

        st.success("No security incidents available for investigation.")

    st.markdown("---")

    st.caption(
        "Incident statuses and analyst notes are stored in the "
        "current Streamlit session. They are not permanently saved."
    )

# ==========================================
# THREAT MAP: SIMULATED LOCATIONS
# ==========================================

elif page == "Threat Map (Demo)":

    st.header("Global threat map (demo)")
    st.warning(
        "DEMO MODE: Locations and incidents below are simulated. "
        "They are not geographic origins of the CIC-IDS2017 traffic."
    )

    map_data = pd.DataFrame([
        {
            "City": "New York",
            "Country": "United States",
            "Latitude": 40.7128,
            "Longitude": -74.0060,
            "Threat": "Suspicious scanning activity",
            "Severity": "HIGH",
            "Brief": "Repeated connection attempts across multiple ports.",
            "Alert Count": 18
        },
        {
            "City": "London",
            "Country": "United Kingdom",
            "Latitude": 51.5072,
            "Longitude": -0.1276,
            "Threat": "Possible brute-force activity",
            "Severity": "CRITICAL",
            "Brief": "Repeated authentication attempts in a short period.",
            "Alert Count": 25
        },
        {
            "City": "Frankfurt",
            "Country": "Germany",
            "Latitude": 50.1109,
            "Longitude": 8.6821,
            "Threat": "Unusual traffic volume",
            "Severity": "MEDIUM",
            "Brief": "Traffic volume exceeds the demo baseline.",
            "Alert Count": 12
        },
        {
            "City": "Singapore",
            "Country": "Singapore",
            "Latitude": 1.3521,
            "Longitude": 103.8198,
            "Threat": "Suspicious connection pattern",
            "Severity": "HIGH",
            "Brief": "Repeated connections to a monitored service.",
            "Alert Count": 16
        },
        {
            "City": "Mumbai",
            "Country": "India",
            "Latitude": 19.0760,
            "Longitude": 72.8777,
            "Threat": "Potential scanning activity",
            "Severity": "MEDIUM",
            "Brief": "Multiple destination ports were contacted.",
            "Alert Count": 9
        },
        {
            "City": "São Paulo",
            "Country": "Brazil",
            "Latitude": -23.5505,
            "Longitude": -46.6333,
            "Threat": "High-rate traffic anomaly",
            "Severity": "CRITICAL",
            "Brief": "A simulated burst of unusually high traffic.",
            "Alert Count": 22
        }
    ])

    severity_filter = st.multiselect(
        "Filter by severity",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    )

    filtered_map_data = map_data[
        map_data["Severity"].isin(severity_filter)
    ]

    st.metric("Simulated Threat Locations", len(filtered_map_data))

    if not filtered_map_data.empty:

        threat_map = px.scatter_geo(
            filtered_map_data,
            lat="Latitude",
            lon="Longitude",
            color="Severity",
            size="Alert Count",
            hover_name="City",
            hover_data={
                "Country": True,
                "Threat": True,
                "Severity": True,
                "Brief": True,
                "Alert Count": True,
                "Latitude": False,
                "Longitude": False
            },
            color_discrete_map={
                "CRITICAL": "#ef4444",
                "HIGH": "#ff7a1a",
                "MEDIUM": "#ffc83d",
                "LOW": "#22c55e"
            },
            projection="natural earth",
            title="Simulated Global Threat Activity"
        )

        threat_map.update_geos(
            showcountries=True,
            showland=True,
            landcolor="#1f1f23",
            showocean=True,
            oceancolor="#0a0a0b",
            bgcolor="#0f0f10",
            countrycolor="#34343a",
            coastlinecolor="#4a4a52"
        )

        threat_map.update_layout(
            template="netguard",
            paper_bgcolor="#0f0f10",
            font_color="white",
            height=650,
            margin={"l": 0, "r": 0, "t": 50, "b": 0}
        )

        st.plotly_chart(
            threat_map,
            use_container_width=True
        )

        st.caption(
            "Hover over a marker to view its simulated threat summary."
        )

    else:
        st.info("No locations match the selected severity filters.")

# ==========================================
# 10. MODEL PERFORMANCE PAGE
# ==========================================

elif page == "Model Performance":

    st.header("Model performance")

    # ACTUAL AND PREDICTED LABELS

    actual = df["Actual"].astype(str).str.strip().str.upper()
    predicted = df["Predicted"].astype(str).str.strip().str.upper()

    # CALCULATE PERFORMANCE METRICS

    accuracy = (actual == predicted).mean() * 100

    precision = precision_score(
        actual,
        predicted,
        pos_label="ATTACK",
        zero_division=0
    ) * 100

    recall = recall_score(
        actual,
        predicted,
        pos_label="ATTACK",
        zero_division=0
    ) * 100

    f1 = f1_score(
        actual,
        predicted,
        pos_label="ATTACK",
        zero_division=0
    ) * 100

    # METRIC CARDS

    st.subheader("Model Evaluation Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.2f}%")
    col2.metric("Precision", f"{precision:.2f}%")
    col3.metric("Recall", f"{recall:.2f}%")
    col4.metric("F1 Score", f"{f1:.2f}%")

    st.markdown("---")

    # CONFUSION MATRIX

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        actual,
        predicted,
        labels=["BENIGN", "ATTACK"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual BENIGN", "Actual ATTACK"],
        columns=["Predicted BENIGN", "Predicted ATTACK"]
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale=[[0, "#131614"], [0.5, "#0a6f8c"], [1, "#0ba0c2"]],
        title="Actual vs predicted network traffic",
        labels={
            "x": "Predicted class",
            "y": "Actual class",
            "color": "Records"
        }
    )

    fig_cm.update_layout(
        template="netguard",
        paper_bgcolor="#0f0f10",
        font_color="white"
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )

    st.markdown("---")

    # CLASSIFICATION REPORT

    st.subheader("Detailed Classification Report")

    report = classification_report(
        actual,
        predicted,
        labels=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

    st.markdown("---")

    # FEATURE IMPORTANCE

    st.subheader("Important Network Traffic Features")

    importance_path = Path(
        "data/processed/feature_importance.csv"
    )

    if importance_path.exists():

        importance_df = pd.read_csv(importance_path)

        top_features = importance_df.head(15).sort_values(
            "Importance",
            ascending=True
        )

        fig_features = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=[[0, "#6b3a12"], [0.5, "#ff7a1a"], [1, "#ffc83d"]],
            title="Top 15 Important Features",
            labels={
                "Importance": "Importance Score",
                "Feature": "Network Feature"
            }
        )

        fig_features.update_layout(
            coloraxis_showscale=False,
            template="netguard",
            paper_bgcolor="#0f0f10",
            font_color="white"
        )

        st.plotly_chart(
            fig_features,
            use_container_width=True
        )

    else:

        st.warning(
            "Feature importance file not found. Complete Day 8 first."
        )
        # ==========================================
    # ADVANCED ANALYTICS: FEATURE CORRELATION
    # ==========================================

    st.markdown("---")
    st.subheader("Advanced analytics: feature correlation")

    st.write(
        "Explore how selected network traffic features relate to one another."
    )

    if importance_path.exists():

        available_features = [
            feature
            for feature in importance_df["Feature"].tolist()
            if feature in df.columns
        ]

        default_features = available_features[:12]

        selected_features = st.multiselect(
            "Select features to compare",
            options=available_features,
            default=default_features
        )

        correlation_method = st.selectbox(
            "Correlation method",
            ["Pearson", "Spearman"]
        )

        if len(selected_features) >= 2:

            correlation_df = df[selected_features].corr(
                method=correlation_method.lower()
            )

            correlation_chart = px.imshow(
                correlation_df,
                text_auto=".2f",
                color_continuous_scale=[[0, "#22c55e"], [0.5, "#17171a"], [1, "#ff7a1a"]],
                zmin=-1,
                zmax=1,
                title=f"{correlation_method} Correlation Matrix",
                labels={
                    "x": "Network Features",
                    "y": "Network Features",
                    "color": "Correlation"
                }
            )

            correlation_chart.update_layout(
                template="netguard",
                paper_bgcolor="#0f0f10",
                font_color="white",
                height=650
            )

            st.plotly_chart(
                correlation_chart,
                use_container_width=True
            )

            st.caption(
                "Correlation ranges from -1 to +1. "
                "Values near +1 indicate positive association, "
                "values near -1 indicate negative association, "
                "and values near 0 indicate little linear association."
            )

        else:

            st.info("Select at least two features to display the heatmap.")

    else:

        st.warning(
            "Feature importance data is required for this analysis."
        )

    st.markdown("---")

    st.info(
        "These metrics evaluate the saved test predictions from CIC-IDS2017. "
        "They do not yet establish performance on unseen live network traffic."
    )



# ==========================================
# 11. FOOTER
# ==========================================

    st.markdown("---")

    st.caption(
    "NETGUARD AI | AI-Powered Network Threat Detection | Academic Project"
)


# ==========================================
# LIVE MONITORING PAGE
# ==========================================

elif page == "Live Monitoring":

    @st.fragment(run_every="5s")
    def live_monitoring():

        st.title("🔴 Live Network Monitoring")
        st.caption("Live traffic | AI threat detection | Auto-refresh: 5 seconds")

        live_file = Path("data/live/live_flows_v2.csv")
        model_file = Path("models/live_random_forest_model.pkl")

        if not live_file.exists():
            st.warning("Live flow file not found. Start live_flow_export_v2.py.")

        elif not model_file.exists():
            st.error("Live AI model not found. Train the model first.")

        else:
            live_df = pd.read_csv(live_file)

            if live_df.empty:
                st.info("Waiting for live network traffic...")

            else:
                try:
                    saved_model = joblib.load(model_file)

                    live_model = saved_model["model"]
                    live_features = saved_model["features"]

                    missing_features = [
                        feature for feature in live_features
                        if feature not in live_df.columns
                    ]

                    if missing_features:
                        st.error(f"Missing live features: {missing_features}")
                        return

                    X_live = live_df[live_features].copy()

                    X_live = X_live.apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    X_live = X_live.replace(
                        [float("inf"), float("-inf")],
                        0
                    ).fillna(0)

                    predictions = live_model.predict(X_live)
                    probabilities = live_model.predict_proba(X_live)

                    attack_class_index = list(
                        live_model.classes_
                    ).index(1)

                    attack_probabilities = (
                        probabilities[:, attack_class_index] * 100
                    )

                    live_df["AI Prediction"] = [
                        "ATTACK" if prediction == 1 else "BENIGN"
                        for prediction in predictions
                    ]

                    live_df["Attack Probability"] = attack_probabilities

                    # ==========================================
                    # LIVE THREAT SEVERITY CLASSIFICATION
                    # ==========================================

                    def classify_live_severity(probability):

                        if probability >= 99:
                            return "CRITICAL"

                        elif probability >= 95:
                            return "HIGH"

                        elif probability >= 90:
                            return "MEDIUM"

                        else:
                            return "LOW"

                    live_df["Threat Severity"] = live_df.apply(
                        lambda row: (
                            classify_live_severity(row["Attack Probability"])
                            if row["AI Prediction"] == "ATTACK"
                            else "BENIGN"
                        ),
                        axis=1
                    )


                    
                    # ==========================================
                    # PERSISTENT LIVE SECURITY EVENT LOGGER
                    # ==========================================

                    log_dir = Path("data/live/security_logs")
                    log_dir.mkdir(parents=True, exist_ok=True)

                    log_file = log_dir / "live_security_logs.csv"

                    attack_events = live_df[
                        live_df["AI Prediction"] == "ATTACK"
                    ].copy()

                    if not attack_events.empty:

                        # Create a unique ID for each captured flow
                        attack_events["Event ID"] = (
                            attack_events["Source IP"].astype(str) + "_" +
                            attack_events["Destination IP"].astype(str) + "_" +
                            attack_events["Source Port"].astype(str) + "_" +
                            attack_events["Destination Port"].astype(str) + "_" +
                            attack_events["Protocol"].astype(str) + "_" +
                            attack_events["Start Time"].astype(str)
                        )

                        # Check previously saved events
                        if log_file.exists():
                            existing_logs = pd.read_csv(log_file)

                            logged_ids = set(
                                existing_logs["Event ID"].astype(str)
                            )
                        else:
                            logged_ids = set()

                        # Keep only newly detected flows
                        new_events = attack_events[
                            ~attack_events["Event ID"].isin(logged_ids)
                        ].copy()

                        if not new_events.empty:

                            new_events["Detection Time"] = datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )

                            log_columns = [
                                "Event ID",
                                "Detection Time",
                                "Source IP",
                                "Destination IP",
                                "Source Port",
                                "Destination Port",
                                "Protocol",
                                "Total Packets",
                                "Attack Probability",
                                "Threat Severity"
                            ]

                            new_events[log_columns].to_csv(
                                log_file,
                                mode="a",
                                header=not log_file.exists(),
                                index=False
                            )

                            st.toast(
                                f"{len(new_events)} new security event(s) logged!"
                            )


                    
                    # ==========================================
                    # ==========================================
                    # LIVE SECURITY INCIDENT HISTORY
                    # ==========================================
                    st.subheader("📜 Live Security Incident History")

                    history_file = Path("data/live/security_logs/live_security_logs.csv")
                    required_history_columns = [
                        "Event ID", "Detection Time", "Source IP", "Destination IP",
                        "Source Port", "Destination Port", "Protocol", "Total Packets",
                        "Attack Probability", "Threat Severity"
                    ]

                    if history_file.exists() and history_file.stat().st_size > 0:
                        try:
                            history_df = pd.read_csv(history_file)
                        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
                            history_df = pd.DataFrame(columns=required_history_columns)
                    else:
                        history_df = pd.DataFrame(columns=required_history_columns)

                    for column in required_history_columns:
                        if column not in history_df.columns:
                            history_df[column] = pd.Series(dtype="object")

                    if not history_df.empty:
                        history_df["Event ID"] = history_df["Event ID"].astype(str)
                        sorted_history = history_df.sort_values(
                            "Detection Time", ascending=False, na_position="last"
                        )
                        st.metric("Total Logged Security Events", len(history_df))
                        st.dataframe(sorted_history, use_container_width=True, hide_index=True)
                        st.download_button(
                            label="📥 Download Security Incident Report",
                            data=sorted_history.to_csv(index=False).encode("utf-8"),
                            file_name="NETGUARD_AI_Security_Incident_Report.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                        selected_event = st.selectbox(
                            "Select an incident to investigate",
                            sorted_history["Event ID"].tolist(),
                            format_func=lambda event_id: (
                                str(sorted_history.loc[sorted_history["Event ID"] == event_id, "Detection Time"].iloc[0])
                                + " | " + str(event_id)[:35]
                            )
                        )
                        incident = sorted_history[sorted_history["Event ID"] == selected_event].iloc[0]
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Source IP:**", incident["Source IP"])
                            st.write("**Source Port:**", incident["Source Port"])
                            st.write("**Destination IP:**", incident["Destination IP"])
                            st.write("**Destination Port:**", incident["Destination Port"])
                            st.write("**Protocol:**", incident["Protocol"])
                        with col2:
                            st.write("**Detection Time:**", incident["Detection Time"])
                            st.write("**Total Packets:**", incident["Total Packets"])
                            st.write("**Attack Probability:**", incident["Attack Probability"])
                            st.write("**Threat Severity:**", incident["Threat Severity"])

                        st.subheader("🛠️ Incident Investigation")
                        status_file = Path("data/live/security_logs/incident_statuses.csv")
                        status_columns = ["Event ID", "Status", "Last Updated"]
                        if status_file.exists() and status_file.stat().st_size > 0:
                            try:
                                status_df = pd.read_csv(status_file)
                            except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
                                status_df = pd.DataFrame(columns=status_columns)
                        else:
                            status_df = pd.DataFrame(columns=status_columns)
                        for column in status_columns:
                            if column not in status_df.columns:
                                status_df[column] = pd.Series(dtype="object")
                        status_df["Event ID"] = status_df["Event ID"].astype(str)
                        status_options = ["Open", "Investigating", "Resolved", "False Positive"]
                        saved = status_df[status_df["Event ID"] == str(selected_event)]
                        default_status = saved.iloc[0]["Status"] if not saved.empty and saved.iloc[0]["Status"] in status_options else "Open"
                        selected_status = st.selectbox(
                            "Update incident status", status_options,
                            index=status_options.index(default_status),
                            key=f"incident_status_{selected_event}"
                        )
                        if st.button("💾 Save Incident Status", key=f"save_status_{selected_event}"):
                            status_df = status_df[status_df["Event ID"] != str(selected_event)]
                            status_df = pd.concat([status_df, pd.DataFrame([{
                                "Event ID": str(selected_event), "Status": selected_status,
                                "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }])], ignore_index=True)
                            status_df.to_csv(status_file, index=False)
                            st.success(f"Incident status saved as: {selected_status}")
                    else:
                        st.info("No security incidents have been recorded yet. New AI-detected attack flows will appear here.")

                    # LIVE ATTACK ALERT PANEL
                    # ==========================================

                    st.subheader("🚨 Live Security Alerts")

                    attack_df = live_df[
                        live_df["AI Prediction"] == "ATTACK"
                    ].copy()

                    if not attack_df.empty:

                        st.error(
                            f"⚠️ {len(attack_df)} suspected attack flow(s) detected!"
                        )

                        alert_columns = [
                            "Source IP",
                            "Destination IP",
                            "Destination Port",
                            "Protocol",
                            "Total Packets",
                            "Attack Probability",
                            "Threat Severity"
                        ]

                        alert_display = attack_df[alert_columns].copy()

                        st.dataframe(
                            alert_display.style.apply(
                                lambda row: [
                                    "background-color: #641e1e; color: white"
                                ] * len(row),
                                axis=1
                            ),
                            use_container_width=True
                        )

                    else:

                        st.success(
                            "No suspected attack flows detected in the current snapshot."
                        )

                                                # ==========================================
                        # SOC INCIDENT MANAGEMENT OVERVIEW
                        # ==========================================

                        st.divider()
                        st.subheader("📊 SOC Incident Management Overview")

                        history_file = Path(
                            "data/live/security_logs/live_security_logs.csv"
                        )

                        status_file = Path(
                            "data/live/security_logs/incident_statuses.csv"
                        )

                        # Load incident history safely
                        if history_file.exists() and history_file.stat().st_size > 0:
                            history_df = pd.read_csv(history_file)
                        else:
                            history_df = pd.DataFrame(
                                columns=["Event ID"]
                            )

                        # Load saved investigation statuses
                        if status_file.exists() and status_file.stat().st_size > 0:
                            status_df = pd.read_csv(status_file)
                        else:
                            status_df = pd.DataFrame(
                                columns=["Event ID", "Status", "Last Updated"]
                            )

                        total_incidents = len(history_df)

                        # Every incident without a saved status is Open
                        status_map = dict(
                            zip(status_df["Event ID"], status_df["Status"])
                        )

                        if total_incidents > 0:
                            incident_statuses = (
                                history_df["Event ID"]
                                .map(status_map)
                                .fillna("Open")
                            )
                        else:
                            incident_statuses = pd.Series(dtype=str)

                        open_count = (incident_statuses == "Open").sum()
                        investigating_count = (
                            incident_statuses == "Investigating"
                        ).sum()
                        resolved_count = (
                            incident_statuses == "Resolved"
                        ).sum()
                        false_positive_count = (
                            incident_statuses == "False Positive"
                        ).sum()

                        metric1, metric2, metric3, metric4, metric5 = st.columns(5)

                        metric1.metric("Total Incidents", total_incidents)
                        metric2.metric("Open", int(open_count))
                        metric3.metric("Investigating", int(investigating_count))
                        metric4.metric("Resolved", int(resolved_count))
                        metric5.metric("False Positives", int(false_positive_count))


                    st.subheader("Live Network Overview")

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric("Active Flows", len(live_df))

                    col2.metric(
                        "Total Packets",
                        int(live_df["Total Packets"].sum())
                    )

                    col3.metric(
                        "AI Attack Alerts",
                        int((live_df["AI Prediction"] == "ATTACK").sum())
                    )

                    col4.metric(
                        "Benign Flows",
                        int((live_df["AI Prediction"] == "BENIGN").sum())
                    )

                    
                    # ==========================================
                    # HISTORICAL NETWORK METRICS LOGGER
                    # ==========================================

                    metrics_dir = Path("data/live/analytics")
                    metrics_dir.mkdir(parents=True, exist_ok=True)

                    metrics_file = metrics_dir / "network_metrics_history.csv"

                    current_time = datetime.now()

                    current_metrics = pd.DataFrame([{
                        "Timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Active Flows": len(live_df),
                        "Total Packets": int(live_df["Total Packets"].sum()),
                        "Attack Flows": int(
                            (live_df["AI Prediction"] == "ATTACK").sum()
                        ),
                        "Benign Flows": int(
                            (live_df["AI Prediction"] == "BENIGN").sum()
                        )
                    }])

                    if metrics_file.exists() and metrics_file.stat().st_size > 0:

                        historical_metrics = pd.read_csv(metrics_file)

                        historical_metrics = pd.concat(
                            [historical_metrics, current_metrics],
                            ignore_index=True
                        )

                    else:
                        historical_metrics = current_metrics.copy()

                    historical_metrics["Timestamp"] = pd.to_datetime(
                        historical_metrics["Timestamp"],
                        errors="coerce"
                    )

                    # Retain the most recent 24 hours
                    cutoff_time = current_time - pd.Timedelta(hours=24)

                    historical_metrics = historical_metrics[
                        historical_metrics["Timestamp"] >= cutoff_time
                    ]

                    historical_metrics.to_csv(
                        metrics_file,
                        index=False
                    )

                    st.caption("Historical network metrics are being recorded.")

                    
                    # 2. Packet volume timeline
                    fig_packets = px.area(
                        historical_metrics,
                        x="Timestamp",
                        y="Total Packets",
                        title="Network Packet Volume Over Time"
                    )

                    fig_packets.update_layout(
                        template="plotly_dark",
                        xaxis_title="Time",
                        yaxis_title="Total Packets"
                    )

                    st.plotly_chart(
                        fig_packets,
                        use_container_width=True
                    )

                    # 3. Suspected attack vs benign flow timeline

                    traffic_history = historical_metrics.melt(
                    id_vars=["Timestamp"],
                    value_vars=["Attack Flows", "Benign Flows"],
                    var_name="Traffic Type",
                    value_name="Flow Count"
                    )

                    fig_traffic = px.line(
                        traffic_history,
                        x="Timestamp",
                        y="Flow Count",
                        color="Traffic Type",
                        markers=True,
                        title="Suspected Attack vs Benign Flows",
                        color_discrete_map={
                            "Attack Flows": "#ef4444",
                            "Benign Flows": "#22c55e"
                            }
                        )
    
                    fig_traffic.update_layout(
                        template="plotly_dark",
                        xaxis_title="Time",
                        yaxis_title="Flow Count"
                    )

                    st.plotly_chart(
                        fig_traffic,
                        use_container_width=True
                    )


                    # ==========================================
                    # LIVE NETWORK VISUALIZATION CENTER
                    # ==========================================

                    st.divider()
                    st.subheader("📈 Live Network Analytics")

                    # Prepare chart data
                    chart_df = live_df.copy()
                    chart_df["Protocol"] = chart_df["Protocol"].astype(str)
                    chart_df["AI Prediction"] = chart_df["AI Prediction"].astype(str)
                    chart_df["Threat Severity"] = chart_df["Threat Severity"].astype(str)

                    chart_col1, chart_col2 = st.columns(2)

                    # 1. Protocol distribution
                    with chart_col1:
                        protocol_counts = (
                            chart_df["Protocol"]
                            .value_counts()
                            .rename_axis("Protocol")
                            .reset_index(name="Active Flows")
                        )

                        fig_protocol = px.pie(
                            protocol_counts,
                            names="Protocol",
                            values="Active Flows",
                            hole=0.55,
                            title="Protocol Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set2
                        )

                        fig_protocol.update_layout(
                            template="plotly_dark",
                            legend_title_text="Protocol"
                        )

                        st.plotly_chart(
                            fig_protocol,
                            use_container_width=True
                        )

                        
                    # ==========================================
                    # ATTACK RATE & TRAFFIC ANOMALY ANALYTICS
                    # ==========================================

                    st.divider()
                    st.subheader("🚨 Attack Rate & Traffic Behaviour")

                    # Calculate attack percentage
                    historical_metrics["Attack Rate (%)"] = (
                        historical_metrics["Attack Flows"]
                        / historical_metrics["Active Flows"].replace(0, 1)
                    ) * 100

                    # Calculate moving average of active flows
                    historical_metrics["Flow Moving Average"] = (
                        historical_metrics["Active Flows"]
                        .rolling(window=5, min_periods=1)
                        .mean()
                    )

                    # Calculate packet-volume moving average
                    historical_metrics["Packet Moving Average"] = (
                        historical_metrics["Total Packets"]
                        .rolling(window=5, min_periods=1)
                        .mean()
                    )

                    # 1. Attack rate timeline
                    fig_attack_rate = px.area(
                        historical_metrics,
                        x="Timestamp",
                        y="Attack Rate (%)",
                        title="Suspected Attack Rate Over Time"
                    )

                    fig_attack_rate.update_layout(
                        template="plotly_dark",
                        yaxis_title="Attack Rate (%)",
                        xaxis_title="Time"
                    )

                    st.plotly_chart(
                        fig_attack_rate,
                        use_container_width=True
                    )

                    # 2. Network activity moving average
                    fig_moving = px.line(
                        historical_metrics,
                        x="Timestamp",
                        y=["Active Flows", "Flow Moving Average"],
                        title="Network Activity & Moving Average"
                    )

                    fig_moving.update_layout(
                        template="plotly_dark",
                        xaxis_title="Time",
                        yaxis_title="Active Flows",
                        legend_title="Measurement"
                    )

                    st.plotly_chart(
                        fig_moving,
                        use_container_width=True
                    )
                    # 3. Packet-volume spike analysis

                    historical_metrics["Packet Spike"] = (
                    historical_metrics["Total Packets"]
    >               historical_metrics["Packet Moving Average"] * 1.5
                    )

                    fig_spikes = px.scatter(
                    historical_metrics,
                    x="Timestamp",
                    y="Total Packets",
                    color="Packet Spike",
                    title="Network Packet Volume — Potential Spikes"
                    )

                    fig_spikes.update_layout(
                    template="plotly_dark",
                    xaxis_title="Time",
                    yaxis_title="Total Packets"
                    )

                    st.plotly_chart(fig_spikes, use_container_width=True)

                    

                    

                    # 2. Attack vs benign flows
                    with chart_col2:
                        prediction_counts = (
                            chart_df["AI Prediction"]
                            .value_counts()
                            .rename_axis("Prediction")
                            .reset_index(name="Flows")
                        )

                        fig_predictions = px.pie(
                            prediction_counts,
                            names="Prediction",
                            values="Flows",
                            hole=0.55,
                            title="AI Prediction Distribution",
                            color="Prediction",
                            color_discrete_map={
                                "ATTACK": "#ef4444",
                                "BENIGN": "#22c55e"
                            }
                        )

                        fig_predictions.update_layout(
                            template="plotly_dark",
                            legend_title_text="Prediction"
                        )

                        st.plotly_chart(
                            fig_predictions,
                            use_container_width=True
                        )

                    chart_col3, chart_col4 = st.columns(2)

                    # 3. Threat severity distribution
                    with chart_col3:
                        severity_order = [
                            "CRITICAL",
                            "HIGH",
                            "MEDIUM",
                            "LOW",
                            "BENIGN"
                        ]

                        severity_counts = (
                            chart_df["Threat Severity"]
                            .value_counts()
                            .reindex(severity_order, fill_value=0)
                            .rename_axis("Severity")
                            .reset_index(name="Flows")
                        )

                        fig_severity = px.bar(
                            severity_counts,
                            x="Severity",
                            y="Flows",
                            title="Threat Severity Distribution",
                            color="Severity",
                            category_orders={"Severity": severity_order},
                            color_discrete_map={
                                "CRITICAL": "#991b1b",
                                "HIGH": "#ef4444",
                                "MEDIUM": "#f97316",
                                "LOW": "#eab308",
                                "BENIGN": "#22c55e"
                            }
                        )

                        fig_severity.update_layout(
                            template="plotly_dark",
                            showlegend=False
                        )

                        st.plotly_chart(
                            fig_severity,
                            use_container_width=True
                        )

                    # 4. Attack probability distribution
                    with chart_col4:
                        fig_probability = px.histogram(
                            chart_df,
                            x="Attack Probability",
                            color="AI Prediction",
                            nbins=20,
                            title="AI Attack Probability Distribution",
                            labels={
                                "Attack Probability": "Attack Probability (%)",
                                "count": "Flow Count"
                            },
                            color_discrete_map={
                                "ATTACK": "#ef4444",
                                "BENIGN": "#22c55e"
                            }
                        )

                        fig_probability.update_layout(
                            template="plotly_dark",
                            barmode="overlay"
                        )

                        fig_probability.update_traces(opacity=0.75)

                        st.plotly_chart(
                            fig_probability,
                            use_container_width=True
                        )

                    # 5. Most active destination ports
                    st.subheader("🎯 Top Destination Ports")

                    top_ports = (
                        chart_df.groupby("Destination Port")
                        .size()
                        .reset_index(name="Active Flows")
                        .sort_values("Active Flows", ascending=False)
                        .head(10)
                    )

                    top_ports["Destination Port"] = (
                        top_ports["Destination Port"].astype(str)
                    )

                    fig_ports = px.bar(
                        top_ports,
                        x="Active Flows",
                        y="Destination Port",
                        orientation="h",
                        title="Top 10 Destination Ports by Active Flows",
                        color="Active Flows",
                        color_continuous_scale="Turbo"
                    )

                    fig_ports.update_layout(
                        template="plotly_dark",
                        yaxis={"categoryorder": "total ascending"}
                    )

                    st.plotly_chart(
                        fig_ports,
                        use_container_width=True
                    )

                    st.subheader("AI Threat Predictions")

                    st.dataframe(
                        live_df[
                            [
                                "Source IP",
                                "Destination IP",
                                "Destination Port",
                                "Protocol",
                                "Total Packets",
                                "AI Prediction",
                                "Attack Probability"
                            ]
                        ],
                        use_container_width=True
                    )

                    st.caption(
                        f"Last dashboard refresh: "
                        f"{datetime.now().strftime('%H:%M:%S')}"
                    )

                    st.caption(
                        "Experimental AI predictions using 41 features. "
                        "Not independently validated for live traffic."
                    )

                except Exception as error:
                    st.error(f"Live AI prediction error: {error}")

    live_monitoring()