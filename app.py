import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
import os
import numpy as np
from datetime import datetime
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Ferry Dashboard",
    layout="wide"
)

# =========================================================
# BACKGROUND IMAGE
# =========================================================
def get_base64(img_path):
    if not os.path.exists(img_path):
        return ""

    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("bg.jpeg")

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(f"""
<style>

/* MAIN PAGE */
.stApp {{
    background-image: url("data:image/jpg;base64,{img}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* DARK OVERLAY */
.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.45);
    z-index: -1;
}}

.block-container {{
    padding-top: 1rem !important;
}}

/* REMOVE WHITE BLOCKS */
header {{
    background: transparent !important;
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0) !important;
}}

[data-testid="stAppViewContainer"] {{
    background: transparent !important;
}}

/* TEXT */
h1, h2, h3, h4, h5, h6, p, label, span {{
    color: white !important;
}}

/* BIGGER GRAPH TITLES */
.js-plotly-plot .plotly .gtitle {{
    font-size: 28px !important;
    fill: white !important;
    font-weight: bold !important;
}}

/* SIDEBAR */
section[data-testid="stSidebar"] {{
    background: rgba(10,20,40,0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.1);
}}

section[data-testid="stSidebar"] * {{
    color: white !important;
}}

/* BUTTONS */
.stButton > button {{
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-size: 16px;
    font-weight: bold;
}}

.stDownloadButton > button {{
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    color: white !important;
    border-radius: 12px;
    border: none;
    padding: 0.7rem 1rem;
    font-weight: bold;
}}

/* INPUTS */
div[data-baseweb="input"] > div {{
    background-color: rgba(15,23,42,0.95) !important;
    color: white !important;
    border-radius: 10px !important;
}}

div[data-baseweb="select"] > div {{
    background-color: rgba(15,23,42,0.95) !important;
    color: white !important;
    border-radius: 10px !important;
}}

span[data-baseweb="tag"] {{
    background: #1e40af !important;
    color: white !important;
}}

/* SLIDER */
.stSlider > div > div {{
    color: white !important;
}}

/* METRICS */
div[data-testid="stMetric"] {{
    background: rgba(15,23,42,0.85);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(6px);
}}

div[data-testid="stMetricValue"] {{
    color: #38bdf8 !important;
    font-size: 34px !important;
    font-weight: bold;
}}

div[data-testid="stMetricLabel"] {{
    color: white !important;
    font-size: 18px !important;
}}

/* DATAFRAME FULL DARK FIX */
[data-testid="stDataFrame"] {{
    background-color: rgba(2,6,23,0.95) !important;
    border-radius: 15px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.1);
}}

[data-testid="stDataFrame"] table {{
    background-color: rgba(2,6,23,0.95) !important;
    color: white !important;
}}

[data-testid="stDataFrame"] th {{
    background-color: #111827 !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: bold !important;
}}

[data-testid="stDataFrame"] td {{
    background-color: rgba(2,6,23,0.95) !important;
    color: white !important;
    font-size: 17px !important;
}}

[data-testid="stDataFrame"] tr:nth-child(even) td {{
    background-color: rgba(15,23,42,0.95) !important;
}}

[data-testid="stDataFrame"] div {{
    color: white !important;
}}

/* EXPANDER */
.streamlit-expanderHeader {{
    background: rgba(15,23,42,0.85) !important;
    color: white !important;
    border-radius: 10px;
    font-size: 18px !important;
}}

/* ALERTS */
.alert-success {{
    background: rgba(34,197,94,0.2);
    border: 1px solid #22c55e;
    padding: 15px;
    border-radius: 12px;
    color: white;
}}

.alert-warning {{
    background: rgba(234,179,8,0.2);
    border: 1px solid #eab308;
    padding: 15px;
    border-radius: 12px;
    color: white;
}}

.alert-danger {{
    background: rgba(239,68,68,0.2);
    border: 1px solid #ef4444;
    padding: 15px;
    border-radius: 12px;
    color: white;
}}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.title("Ferry Ticket Analytics Dashboard")
st.subheader("Real-Time Passenger Flow Insights")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv("cleaned_ferry_data.csv")

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df['Date'] = df['Timestamp'].dt.date
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day_name()
    df['Month'] = df['Timestamp'].dt.month_name()

    df['Net Movement'] = (
        df['Sales'] - df['Redemption']
    )

    return df

df = load_data()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.header("Filters")

    date_range = st.date_input(
        "Date Range",
        [df['Date'].min(), df['Date'].max()]
    )

    selected_days = st.multiselect(
        "Select Day",
        df['Day'].unique(),
        default=df['Day'].unique()
    )

    hour_range = st.slider(
        "Select Hour Range",
        0,
        23,
        (0, 23)
    )

    st.markdown("---")

    st.header("About Project")

    st.write("""
    This dashboard analyzes ferry passenger activity,
    ticket sales, redemption trends,
    and operational insights.
    """)

    st.markdown("---")

    st.header("Insights")

    peak_hour = df.groupby('Hour')['Sales'].sum().idxmax()
    busiest_day = df.groupby('Day')['Sales'].sum().idxmax()
    busiest_month = df.groupby('Month')['Sales'].sum().idxmax()

    st.success(f"Peak Hour: {peak_hour}:00")
    st.info(f"Busiest Day: {busiest_day}")
    st.info(f" Busiest Month: {busiest_month}")

# =========================================================
# FILTERING
# =========================================================
filtered_df = df.copy()

if len(date_range) == 2:

    filtered_df = filtered_df[
        (filtered_df['Date'] >= date_range[0]) &
        (filtered_df['Date'] <= date_range[1])
    ]

filtered_df = filtered_df[
    filtered_df['Day'].isin(selected_days)
]

filtered_df = filtered_df[
    filtered_df['Hour'].between(
        hour_range[0],
        hour_range[1]
    )
]

if filtered_df.empty:
    st.warning("No data available")
    st.stop()

# =========================================================
# KPIs
# =========================================================
st.markdown("##  KPIs")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Sales",
    int(filtered_df['Sales'].sum())
)

c2.metric(
    "Redemption",
    int(filtered_df['Redemption'].sum())
)

c3.metric(
    "Net Movement",
    int(filtered_df['Net Movement'].sum())
)

# =========================================================
# ALERTS
# =========================================================
st.markdown("##  Alerts")

util = (
    filtered_df['Redemption'].sum()
    / filtered_df['Sales'].sum()
) * 100

if util > 80:
    st.markdown(
        '<div class="alert-success"> High redemption rate detected.</div>',
        unsafe_allow_html=True
    )

elif util > 50:
    st.markdown(
        '<div class="alert-warning"> Moderate redemption activity.</div>',
        unsafe_allow_html=True
    )

else:
    st.markdown(
        '<div class="alert-danger"> Low redemption activity detected.</div>',
        unsafe_allow_html=True
    )

# =========================================================
# GRAPH 1
# =========================================================
st.markdown("---")

ts = filtered_df.groupby('Timestamp')[
    ['Sales', 'Redemption']
].sum().reset_index()

fig1 = px.line(
    ts,
    x='Timestamp',
    y=['Sales', 'Redemption'],
    title="Sales vs Redemption Over Time"
)

fig1.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", size=16),
    title_font=dict(size=30)
)

st.plotly_chart(fig1, use_container_width=True)
# DATASET BELOW GRAPH 1
with st.expander(" View Data"):

    st.download_button(
        "⬇ Download Time Series Data",
        ts.to_csv(index=False),
        "timeseries.csv"
    )

    st.dataframe(
        ts,
        use_container_width=True,
        height=300
    )


# =========================================================
# GRAPH 2
# =========================================================
st.markdown("---")

hourly = filtered_df.groupby(
    'Hour'
)['Sales'].sum().reset_index()

fig2 = px.bar(
    hourly,
    x='Hour',
    y='Sales',
    title="Hourly Sales"
)

fig2.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", size=16),
    title_font=dict(size=30)
)

st.plotly_chart(fig2, use_container_width=True)

#  DATASET BELOW GRAPH 2
with st.expander(" View Data"):

    st.download_button(
        "⬇ Download Hourly Data",
        hourly.to_csv(index=False),
        "hourly.csv"
    )

    st.dataframe(
        hourly,
        use_container_width=True,
        height=300
    )
# =========================================================
# GRAPH 3
# =========================================================
st.markdown("---")

net = filtered_df.groupby(
    'Timestamp'
)['Net Movement'].sum().reset_index()

fig3 = px.area(
    net,
    x='Timestamp',
    y='Net Movement',
    title="Net Passenger Movement"
)

fig3.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", size=16),
    title_font=dict(size=30)
)

st.plotly_chart(fig3, use_container_width=True)

#  DATASET BELOW GRAPH 3
with st.expander(" View Data"):

    st.download_button(
        "⬇ Download Net Movement Data",
        net.to_csv(index=False),
        "net.csv"
    )

    st.dataframe(
        net,
        use_container_width=True,
        height=300
    )

# =========================================================
# PIE CHART
# =========================================================
st.markdown("---")

day_sales = filtered_df.groupby(
    'Day'
)['Sales'].sum().reset_index()

fig4 = px.pie(
    day_sales,
    names='Day',
    values='Sales',
    title="Sales Distribution by Day"
)

fig4.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", size=16),
    title_font=dict(size=30)
)

st.plotly_chart(fig4, use_container_width=True)
#  DATASET BELOW PIE CHART
with st.expander(" View Data"):

    st.download_button(
        "⬇ Download Day Sales Data",
        day_sales.to_csv(index=False),
        "day_sales.csv"
    )

    st.dataframe(
        day_sales,
        use_container_width=True,
        height=300
    )

# =========================================================
# ML PREDICTION
# =========================================================
st.markdown("---")
st.markdown("##  Passenger Prediction")

hourly_agg = filtered_df.groupby(
    'Hour'
)['Sales'].mean().reset_index()

X = hourly_agg[['Hour']]
y = hourly_agg['Sales']

poly = PolynomialFeatures(degree=3)

X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

future_hours = np.arange(0, 24).reshape(-1,1)

future_poly = poly.transform(
    future_hours
)

predictions = model.predict(
    future_poly
)

pred_df = pd.DataFrame({
    'Hour': future_hours.flatten(),
    'Predicted Sales': predictions
})

fig_ml = px.line(
    pred_df,
    x='Hour',
    y='Predicted Sales',
    title="Predicted Passenger Demand"
)

fig_ml.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", size=16),
    title_font=dict(size=30)
)

st.plotly_chart(fig_ml, use_container_width=True)

# ✅ DATASET BELOW ML GRAPH
with st.expander(" View Prediction Data"):

    st.download_button(
        "⬇ Download Prediction Data",
        pred_df.to_csv(index=False),
        "predictions.csv"
    )

    st.dataframe(
        pred_df,
        use_container_width=True,
        height=300
    )

# =========================================================
# REAL-TIME SIMULATION
# =========================================================
st.markdown("---")
st.markdown("##  Real-Time Simulation")

run_sim = st.button("▶ Start Simulation")

if run_sim:

    placeholder = st.empty()

    sim_data = []

    for i in range(10):

        sales = np.random.randint(100, 500)
        redemption = np.random.randint(80, 450)

        sim_data.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Sales": sales,
            "Redemption": redemption
        })

        sim_df = pd.DataFrame(sim_data)

        fig_sim = px.line(
            sim_df,
            x='Time',
            y=['Sales', 'Redemption']
        )

        fig_sim.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=16),
            title_font=dict(size=30)
        )

        placeholder.plotly_chart(
            fig_sim,
            use_container_width=True
        )

        time.sleep(0.5)

# =========================================================
# FULL DATASET
# =========================================================
st.markdown("---")
st.markdown("## Full Dataset")

st.download_button(
    "⬇ Download Full Data",
    filtered_df.to_csv(index=False),
    "filtered_data.csv"
)

st.dataframe(
    filtered_df,
    use_container_width=True,
    height=500
)
