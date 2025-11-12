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
    # (kept for backward-compatibility; not used when custom range is selected)

    if window_label == "Max":
        return s
    now = s.index.max()
    years = {"1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5}.get(window_label, 2)
    start = now - pd.DateOffset(years=years)
    return s.loc[s.index >= start]


def filter_by_range(s: pd.Series, mode: str, start_dt=None, end_dt=None) -> pd.Series:
    """Filter a time series by either 'Past 1Y' or custom [start_dt, end_dt]."""
    if s.empty:
        return s
    if mode == "Past 1Y":
        end = s.index.max()
        start = end - pd.DateOffset(years=1)
        return s.loc[(s.index >= start) & (s.index <= end)]
    # Custom range
    if start_dt is None and end_dt is None:
        return s
    # Convert date inputs to Timestamps
    if start_dt is not None:
        start_ts = pd.Timestamp(start_dt)
    else:
        start_ts = s.index.min()
    if end_dt is not None:
        # include end date by adding 1 day then using '<'
        end_ts = pd.Timestamp(end_dt) + pd.Timedelta(days=1)
    else:
        end_ts = s.index.max() + pd.Timedelta(days=1)
    return s.loc[(s.index >= start_ts) & (s.index < end_ts)]
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
    # Add horizontal danger line (red)
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Danger @ {threshold:g} {units}",
        annotation_position="top left",
        annotation_font_color="red"
    )

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

# API key resolution:
# 1) Streamlit Secrets (hidden, preferred)
# 2) ENV var
# 3) Sidebar input (only if no key found)
with st.sidebar:
    st.header("Settings")

# Priority 1: Streamlit secrets (stay hidden; show a lock note only)
if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
    with st.sidebar:
        st.success("FRED API Key loaded from secrets 🔒")
else:
    # Priority 2: environment var
    default_key = os.getenv("FRED_API_KEY", "")
    # Only show an input if we still don't have a key
    with st.sidebar:
        api_key = st.text_input(
            "FRED API Key",
            value=default_key,
            type="password",
            help="Get one free from fred.stlouisfed.org"
        )

with st.sidebar:
    time_mode = st.radio("Time range", ["Past 1Y", "Custom range"], index=0, horizontal=False)
    if time_mode == "Custom range":
        # Initialize session state for dates
        if "from_date" not in st.session_state:
            st.session_state.from_date = (pd.Timestamp.today() - pd.DateOffset(years=1)).date()
        if "to_date" not in st.session_state:
            st.session_state.to_date = pd.Timestamp.today().date()

        from_date = st.date_input("From", value=st.session_state.from_date, key="from_date")
        cols = st.columns([3,2])
        with cols[0]:
            to_date = st.date_input("To", value=st.session_state.to_date, key="to_date")
        with cols[1]:
            if st.button("Set To Today"):
                st.session_state.to_date = pd.Timestamp.today().date()
                st.rerun()

        if from_date > to_date:
            st.warning("`From` is after `To`. Please adjust.")
    else:
        from_date, to_date = None, None

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
    st.warning("Enter your FRED API key (or add it to Streamlit Secrets) to load data.")
    st.stop()

col1, col2 = st.columns(2)
col3 = st.container()

# ---------------------------
# Data fetch
# ---------------------------
try:
    # Credit spread: Moody's Baa minus 10Y Treasury (percentage points)
    baa10y_raw = fetch_fred_series(api_key, "BAA10Y")
    # Yield curve: 10Y minus 2Y Treasury (percentage points)
    t10y2y_raw = fetch_fred_series(api_key, "T10Y2Y")
    # LEI (proxy): OECD CLI for the US (USALOLITOAASTSAM) level -> convert to YoY % change
    lei_level_raw = fetch_fred_series(api_key, "USALOLITOAASTSAM")
    lei_yoy_raw = compute_lei_yoy(lei_level_raw)

    # Apply time range
    baa10y = filter_by_range(baa10y_raw, time_mode, from_date, to_date)
    t10y2y = filter_by_range(t10y2y_raw, time_mode, from_date, to_date)
    lei_yoy = filter_by_range(lei_yoy_raw, time_mode, from_date, to_date)

    last_update = max(baa10y.index.max(), t10y2y.index.max(), lei_yoy.index.max())

# Latest values for risk summary
latest_credit = float(baa10y.iloc[-1]) if not baa10y.empty else float("nan")
latest_yc = float(t10y2y.iloc[-1]) if not t10y2y.empty else float("nan")
latest_lei = float(lei_yoy.iloc[-1]) if not lei_yoy.empty else float("nan")

breach_credit = compute_breach(latest_credit, credit_spread_thr, "above") if not np.isnan(latest_credit) else False
breach_yc = compute_breach(latest_yc, yield_curve_thr, "below") if not np.isnan(latest_yc) else False
breach_lei = compute_breach(latest_lei, lei_yoy_thr, "below") if not np.isnan(latest_lei) else False
num_breaches = int(breach_credit) + int(breach_yc) + int(breach_lei)

if num_breaches == 0:
    risk_label, risk_color = "OK", "#10b981"
elif num_breaches == 3:
    risk_label, risk_color = "DANGER", "#ef4444"
else:
    risk_label, risk_color = "RISK", "#f59e0b"

# Render a top-level button-like badge
st.markdown(
    f"""<div style=\"display:flex;justify-content:flex-start;gap:8px;align-items:center;margin-bottom:8px;\"><button style=\"background:{risk_color};color:white;border:none;padding:8px 14px;border-radius:10px;font-weight:800;cursor:default;\">Overall Risk: {risk_label}</button></div>""",
    unsafe_allow_html=True
)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# ---------------------------
# Overall Risk (computed from latest values vs thresholds)
# ---------------------------
def compute_breach(latest_val, threshold, breach_if):
    return (latest_val >= threshold) if breach_if == "above" else (latest_val <= threshold)


# Combined view plotting
def plot_combined_view(baa10y: pd.Series, t10y2y: pd.Series, lei_yoy: pd.Series,
                       thr_credit: float, thr_yc: float, thr_lei: float):
    fig = go.Figure()

    # Distinct colors for series
    color_credit = "#1f77b4"  # blue
    color_yc = "#2ca02c"      # green
    color_lei = "#ff7f0e"     # orange

    # Series traces
    fig.add_trace(go.Scatter(x=baa10y.index, y=baa10y.values, mode="lines", name="Credit Spread (BAA10Y)", line=dict(color=color_credit)))
    fig.add_trace(go.Scatter(x=t10y2y.index, y=t10y2y.values, mode="lines", name="Yield Curve (T10Y−2Y)", line=dict(color=color_yc)))
    fig.add_trace(go.Scatter(x=lei_yoy.index, y=lei_yoy.values, mode="lines", name="OECD CLI YoY %", line=dict(color=color_lei)))

    # Threshold lines with distinct colors
    fig.add_hline(y=thr_credit, line_dash="dash", line_color=color_credit, annotation_text=f"Credit danger @ {thr_credit:g}", annotation_font_color=color_credit)
    fig.add_hline(y=thr_yc, line_dash="dash", line_color=color_yc, annotation_text=f"YieldCurve danger @ {thr_yc:g}", annotation_font_color=color_yc)
    fig.add_hline(y=thr_lei, line_dash="dash", line_color=color_lei, annotation_text=f"CLI YoY danger @ {thr_lei:g}%", annotation_font_color=color_lei)

    fig.update_layout(
        title="Combined View — All Indicators",
        margin=dict(l=40, r=40, t=60, b=40),
        height=520,
        xaxis_title="Date",
        yaxis_title="Value / %",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Render
# ---------------------------

view_mode = st.radio("View", ["Panels", "Combined"], index=0, horizontal=True)
if view_mode == "Combined":
    plot_combined_view(baa10y, t10y2y, lei_yoy, credit_spread_thr, yield_curve_thr, lei_yoy_thr)
else:
    with col1:
        st.subheader("Credit Spread: Moody's Baa − 10Y Treasury")
        plot_indicator(baa10y, "Credit Spread (BAA10Y)", " ppts", credit_spread_thr, breach_if="above")
    with col2:
        st.subheader("Yield Curve: 10Y − 2Y")
        plot_indicator(t10y2y, "Yield Curve (T10Y−2Y)", " ppts", yield_curve_thr, breach_if="below")
    with col3:
        st.subheader("Leading Indicator (OECD CLI) — YoY change")
        plot_indicator(lei_yoy, "OECD CLI YoY % (USALOLITOAASTSAM)", " %", lei_yoy_thr, breach_if="below")

st.caption(f"Last data point across series: {last_update.date()} • Source: FRED (BAA10Y, T10Y2Y, USALOLITOAASTSAM)")
st.caption("Note: LEI proxy now uses OECD Composite Leading Indicator for the US (series USALOLITOAASTSAM on FRED). The Conference Board LEI is proprietary; swap it in if you have access.")