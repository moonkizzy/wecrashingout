# app.py
# Streamlit dashboard: Credit Spreads, Yield Curve, and LEI (with danger thresholds)
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Optional dependency (installed via requirements): fredapi
from fredapi import Fred

# ---------------------------
# Helpers
# ---------------------------

@st.cache_data(ttl=300)
def fetch_fred_series(api_key: str, series_id: str) -> pd.Series:
    fred = Fred(api_key=api_key)
    s = fred.get_series(series_id)
    s = s.dropna()
    s.index = pd.to_datetime(s.index)
    s.name = series_id
    return s

def trim_window(s: pd.Series, window_label: str) -> pd.Series:
    if window_label == "Max":
        return s
    now = s.index.max()
    years = {"1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5}.get(window_label, 2)
    start = now - pd.DateOffset(years=years)
    return s.loc[s.index >= start]

def plot_indicator(s: pd.Series, title: str, units: str, threshold: float, breach_if: str):
    """
    breach_if: 'above' or 'below'
    """
    latest_ts = s.index.max()
    latest_val = float(s.loc[latest_ts])

    breach = (latest_val >= threshold) if breach_if == "above" else (latest_val <= threshold)
    status = "DANGER" if breach else "OK"

    # Build plotly chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=title))
    # Add horizontal danger line
    fig.add_hline(y=threshold, line_dash="dash", annotation_text=f"Danger @ {threshold:g} {units}", annotation_position="top left")

    # Add last-point marker
    fig.add_trace(go.Scatter(
        x=[latest_ts], y=[latest_val],
        mode="markers+text",
        text=[f"{latest_val:.2f}{units}"],
        textposition="top right",
        name="Latest"
    ))

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
        xaxis_title="Date",
        yaxis_title=f"Value ({units})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Status badge
    badge_color = "#ef4444" if breach else "#10b981"  # red / green
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="font-weight:600;">Latest: {latest_val:.2f}{units} &nbsp; • &nbsp; {latest_ts.date()}</div>
            <span style="background:{badge_color};color:white;padding:4px 10px;border-radius:999px;font-weight:700;">{status}</span>
        </div>
        """, unsafe_allow_html=True
    )
    st.plotly_chart(fig, use_container_width=True)


def compute_lei_yoy(lei_level: pd.Series) -> pd.Series:
    """YoY % change from levels (12-month percent change)."""
    lei_yoy = lei_level.pct_change(12) * 100.0
    lei_yoy.name = "LEI YoY %"
    return lei_yoy.dropna()


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Markets Stress Dashboard", layout="wide")
st.title("Markets Stress Dashboard")
st.caption("Credit Spreads • Yield Curve • Leading Economic Index (YoY)")

# API key resolution: Streamlit secrets > ENV > sidebar input
default_key = os.getenv("FRED_API_KEY", None)
if "FRED_API_KEY" in st.secrets:
    default_key = st.secrets["FRED_API_KEY"]

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("FRED API Key", value=default_key or "", type="password", help="Get one free from fred.stlouisfed.org")
    window = st.selectbox("Chart window", ["2Y", "1Y", "3Y", "5Y", "Max"], index=0)

    st.subheader("Danger Thresholds")
    credit_spread_thr = st.number_input("Credit Spread (BAA - 10Y) danger above", value=2.5, step=0.1, format="%.2f")
    yield_curve_thr = st.number_input("Yield Curve (10Y - 2Y) danger below", value=0.0, step=0.1, format="%.2f")
    lei_yoy_thr = st.number_input("LEI YoY change danger below (%)", value=-0.5, step=0.1, format="%.2f")

    st.subheader("Refresh")
    st.caption("Data is cached for 5 minutes. Click to fetch fresh data.")
    if st.button("Refresh now"):
        fetch_fred_series.clear()
        st.success("Cache cleared. Data will refresh on next run.")
        st.rerun()

if not api_key:
    st.warning("Enter your FRED API key in the sidebar to load data.")
    st.stop()

col1, col2 = st.columns(2)
col3 = st.container()

# ---------------------------
# Data fetch
# ---------------------------
try:
    # Credit spread: Moody's Baa minus 10Y Treasury (percentage points)
    baa10y = fetch_fred_series(api_key, "BAA10Y")
    baa10y = trim_window(baa10y, window)

    # Yield curve: 10Y minus 2Y Treasury (percentage points)
    t10y2y = fetch_fred_series(api_key, "T10Y2Y")
    t10y2y = trim_window(t10y2y, window)

    # LEI (proxy): Leading Index for the US (USSLIND) level -> convert to YoY % change
    lei_level = fetch_fred_series(api_key, "USSLIND")
    lei_yoy = compute_lei_yoy(lei_level)
    lei_yoy = trim_window(lei_yoy, window)

    last_update = max(baa10y.index.max(), t10y2y.index.max(), lei_yoy.index.max())
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# ---------------------------
# Render
# ---------------------------

with col1:
    st.subheader("Credit Spread: Moody's Baa − 10Y Treasury")
    plot_indicator(baa10y, "Credit Spread (BAA10Y)", " ppts", credit_spread_thr, breach_if="above")

with col2:
    st.subheader("Yield Curve: 10Y − 2Y")
    plot_indicator(t10y2y, "Yield Curve (T10Y−2Y)", " ppts", yield_curve_thr, breach_if="below")

with col3:
    st.subheader("Leading Economic Index — YoY change")
    plot_indicator(lei_yoy, "LEI YoY % (from USSLIND)", " %", lei_yoy_thr, breach_if="below")

st.caption(f"Last data point across series: {last_update.date()} • Source: FRED (BAA10Y, T10Y2Y, USSLIND)")
st.caption("Note: LEI here uses the Federal Reserve's 'Leading Index for the United States (USSLIND)' as a proxy for the Conference Board LEI.")