# app.py
# Markets Stress Dashboard — Credit Spreads, Yield Curve, Leading Indicator (OECD CLI), and VIX table
# Features:
# - FRED API (BAA10Y, T10Y2Y, USALOLITOAASTSAM, VIXCLS)
# - Danger threshold lines (red in panel views; distinct per-series colors in combined view)
# - Time range: default Past 1Y, or Custom range with From/To and "Set To Today" button
# - Overall Risk badge (OK / RISK / DANGER) based on the first three indicators
# - Panels and Combined view
# - Entry/Exit markers and tables
# - Shaded background during danger periods (per panel; and "ALL danger" in combined view)
# - Additional VIX table with danger entry/exit detection

import os
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
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

def filter_by_range(s: pd.Series, mode: str, start_dt=None, end_dt=None) -> pd.Series:
    """Filter a time series by either 'Past 1Y' or custom [start_dt, end_dt]."""
    if s is None or s.empty:
        return s
    if mode == "Past 1Y":
        end = s.index.max()
        start = end - pd.DateOffset(years=1)
        return s.loc[(s.index >= start) & (s.index <= end)]
    # Custom range
    if start_dt is None and end_dt is None:
        return s
    start_ts = pd.Timestamp(start_dt) if start_dt is not None else s.index.min()
    end_ts = (pd.Timestamp(end_dt) + pd.Timedelta(days=1)) if end_dt is not None else (s.index.max() + pd.Timedelta(days=1))
    return s.loc[(s.index >= start_ts) & (s.index < end_ts)]

def compute_lei_yoy(lei_level: pd.Series) -> pd.Series:
    """YoY % change from levels (12-month percent change)."""
    lei_yoy = lei_level.pct_change(12) * 100.0
    lei_yoy.name = "LEI YoY %"
    return lei_yoy.dropna()

def breach_series(s: pd.Series, threshold: float, breach_if: str) -> pd.Series:
    """Return boolean Series: True when value breaches threshold per rule."""
    if s is None or s.empty:
        return pd.Series(dtype=bool)
    if breach_if == "above":
        b = s >= threshold
    else:
        b = s <= threshold
    b.name = f"breach({s.name})"
    return b

def transitions(b: pd.Series):
    """Return (entries, exits) where entries are False->True and exits True->False."""
    if b is None or b.empty:
        return [], []
    b_aligned = b.astype(bool)
    prev = b_aligned.shift(1).fillna(False)
    entries = b_aligned & (~prev)
    exits = (~b_aligned) & prev
    entry_dates = list(entries[entries].index)
    exit_dates = list(exits[exits].index)
    return entry_dates, exit_dates

def intervals_from_boolean(b: pd.Series):
    """Return list of (start, end) timestamps for contiguous True regions in boolean series b."""
    if b is None or b.empty:
        return []
    bs = b.astype(bool)
    start_points = bs & (~bs.shift(1).fillna(False))
    end_points = (~bs) & (bs.shift(1).fillna(False))
    starts = list(start_points[start_points].index)
    ends = list(end_points[end_points].index)
    # If series ends in True, close the last interval at the last index
    if len(starts) > len(ends):
        ends.append(bs.index.max())
    # Pair up
    return list(zip(starts, ends))

def add_vrects(fig, intervals, fillcolor="rgba(239,68,68,0.12)"):
    """Add vertical shaded rectangles for each (start, end) interval."""
    for (x0, x1) in intervals:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=fillcolor, opacity=0.12, line_width=0)

def plot_indicator(s: pd.Series, title: str, units: str, threshold: float, breach_if: str):
    """
    breach_if: 'above' or 'below'
    """
    if s is None or s.empty:
        st.warning(f"No data available for {title}.")
        return

    latest_ts = s.index.max()
    latest_val = float(s.loc[latest_ts])

    breach = (latest_val >= threshold) if breach_if == "above" else (latest_val <= threshold)
    status = "DANGER" if breach else "OK"

    # Build plotly chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=title))

    # Shaded danger regions for this series
    b = breach_series(s, threshold, breach_if)
    intervals = intervals_from_boolean(b)
    add_vrects(fig, intervals, fillcolor="rgba(239,68,68,0.18)")  # red translucent

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

    # Entry/Exit markers
    entry_dates, exit_dates = transitions(b)
    if entry_dates:
        fig.add_trace(go.Scatter(
            x=entry_dates, y=[s.loc[d] for d in entry_dates if d in s.index],
            mode="markers", name="Enter danger",
            marker_symbol="triangle-up", marker_size=10
        ))
    if exit_dates:
        fig.add_trace(go.Scatter(
            x=exit_dates, y=[s.loc[d] for d in exit_dates if d in s.index],
            mode="markers", name="Exit danger",
            marker_symbol="triangle-down", marker_size=10
        ))

    st.plotly_chart(fig, use_container_width=True)

# Combined view plotting
def plot_combined_view(baa10y: pd.Series, t10y2y: pd.Series, lei_yoy: pd.Series,
                       thr_credit: float, thr_yc: float, thr_lei: float):
    fig = go.Figure()

    # Distinct colors for series
    color_credit = "#1f77b4"  # blue
    color_yc = "#2ca02c"      # green
    color_lei = "#ff7f0e"     # orange

    # Build boolean breach series (fill missing with False)
    b_credit = breach_series(baa10y, thr_credit, "above") if (baa10y is not None and not baa10y.empty) else pd.Series(dtype=bool)
    b_yc = breach_series(t10y2y, thr_yc, "below") if (t10y2y is not None and not t10y2y.empty) else pd.Series(dtype=bool)
    b_lei = breach_series(lei_yoy, thr_lei, "below") if (lei_yoy is not None and not lei_yoy.empty) else pd.Series(dtype=bool)

    # Align to a common index (union), filling missing with False
    idx = pd.Index([])
    for b in [b_credit, b_yc, b_lei]:
        if not b.empty:
            idx = idx.union(b.index)
    if len(idx) > 0:
        b_credit = b_credit.reindex(idx, fill_value=False)
        b_yc = b_yc.reindex(idx, fill_value=False)
        b_lei = b_lei.reindex(idx, fill_value=False)

    all_danger = b_credit & b_yc & b_lei
    all_intervals = intervals_from_boolean(all_danger)

    # Shade "ALL danger" periods (neutral gray)
    for (x0, x1) in all_intervals:
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(0,0,0,0.12)", opacity=0.12, line_width=0)

    # Series traces (only if non-empty)
    if baa10y is not None and not baa10y.empty:
        fig.add_trace(go.Scatter(x=baa10y.index, y=baa10y.values, mode="lines",
                                 name="Credit Spread (BAA10Y)", line=dict(color=color_credit)))
        fig.add_hline(y=thr_credit, line_dash="dash", line_color=color_credit,
                      annotation_text=f"Credit danger @ {thr_credit:g}", annotation_font_color=color_credit)

    if t10y2y is not None and not t10y2y.empty:
        fig.add_trace(go.Scatter(x=t10y2y.index, y=t10y2y.values, mode="lines",
                                 name="Yield Curve (T10Y−2Y)", line=dict(color=color_yc)))
        fig.add_hline(y=thr_yc, line_dash="dash", line_color=color_yc,
                      annotation_text=f"YieldCurve danger @ {thr_yc:g}", annotation_font_color=color_yc)

    if lei_yoy is not None and not lei_yoy.empty:
        fig.add_trace(go.Scatter(x=lei_yoy.index, y=lei_yoy.values, mode="lines",
                                 name="OECD CLI YoY %", line=dict(color=color_lei)))
        fig.add_hline(y=thr_lei, line_dash="dash", line_color=color_lei,
                      annotation_text=f"CLI YoY danger @ {thr_lei:g}%", annotation_font_color=color_lei)

    # Mark simultaneous "all-danger" entries/exits with diamonds
    entries_all, exits_all = transitions(all_danger)
    if entries_all:
        fig.add_trace(go.Scatter(
            x=entries_all,
            y=[np.nanmean([
                baa10y.loc[d] if (baa10y is not None and d in getattr(baa10y, "index", [])) else np.nan,
                t10y2y.loc[d] if (t10y2y is not None and d in getattr(t10y2y, "index", [])) else np.nan,
                lei_yoy.loc[d] if (lei_yoy is not None and d in getattr(lei_yoy, "index", [])) else np.nan,
            ]) for d in entries_all],
            mode="markers",
            name="ALL Enter danger",
            marker_symbol="diamond",
            marker_size=12
        ))
    if exits_all:
        fig.add_trace(go.Scatter(
            x=exits_all,
            y=[np.nanmean([
                baa10y.loc[d] if (baa10y is not None and d in getattr(baa10y, "index", [])) else np.nan,
                t10y2y.loc[d] if (t10y2y is not None and d in getattr(t10y2y, "index", [])) else np.nan,
                lei_yoy.loc[d] if (lei_yoy is not None and d in getattr(lei_yoy, "index", [])) else np.nan,
            ]) for d in exits_all],
            mode="markers",
            name="ALL Exit danger",
            marker_symbol="diamond-open",
            marker_size=12
        ))

    fig.update_layout(
        title="Combined View — All Indicators",
        margin=dict(l=40, r=40, t=60, b=40),
        height=520,
        xaxis_title="Date",
        yaxis_title="Value / %",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def compute_breach(latest_val, threshold, breach_if):
    return (latest_val >= threshold) if breach_if == "above" else (latest_val <= threshold)

# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Markets Stress Dashboard", layout="wide")
st.title("Markets Stress Dashboard")
st.caption("Credit Spreads • Yield Curve • Leading Indicator (OECD CLI YoY) • VIX table")

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
    # Time range: Past 1Y (default) or Custom
    time_mode = st.radio("Time range", ["Past 1Y", "Custom range"], index=0, horizontal=False)
    if time_mode == "Custom range":
        # Initialize session state for dates
        if "from_date" not in st.session_state:
            st.session_state.from_date = (pd.Timestamp.today() - pd.DateOffset(years=1)).date()
        if "to_date" not in st.session_state:
            st.session_state.to_date = pd.Timestamp.today().date()

        cols = st.columns([3,2])
        with cols[1]:
            def _set_to_today():
                st.session_state.to_date = date.today()
            st.button("Set To Today", on_click=_set_to_today)
        with cols[0]:
            to_date = st.date_input("To", value=st.session_state.to_date, key="to_date")

        from_date = st.date_input("From", value=st.session_state.from_date, key="from_date")

        if from_date > to_date:
            st.warning("`From` is after `To`. Please adjust.")
    else:
        from_date, to_date = None, None

    st.subheader("Danger Thresholds")
    credit_spread_thr = st.number_input("Credit Spread (BAA - 10Y) danger above", value=2.5, step=0.1, format="%.2f")
    yield_curve_thr = st.number_input("Yield Curve (10Y - 2Y) danger below", value=0.0, step=0.1, format="%.2f")
    lei_yoy_thr = st.number_input("LEI/CLI YoY change danger below (%)", value=-0.5, step=0.1, format="%.2f")
    vix_thr = st.number_input("VIX danger above", value=30.0, step=1.0, format="%.0f")

    st.checkbox("Include VIX in Combined view (overlay only)", value=False, key="include_vix_combined")
    vix_thr = st.number_input("VIX danger above", value=25.0, step=1.0, format="%.1f")

    st.subheader("Refresh")
    st.caption("Data is cached for 5 minutes. Click to fetch fresh data.")
    if st.button("Refresh now"):
        fetch_fred_series.clear()
        st.success("Cache cleared. Data will refresh on next run.")
        st.rerun()

if not api_key:
    st.warning("Enter your FRED API key (or add it to Streamlit Secrets) to load data.")
    st.stop()

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
    # VIX (close)
    vix_raw = fetch_fred_series(api_key, "VIXCLS")

    # Apply time range
    baa10y = filter_by_range(baa10y_raw, time_mode, from_date, to_date)
    t10y2y = filter_by_range(t10y2y_raw, time_mode, from_date, to_date)
    lei_yoy = filter_by_range(lei_yoy_raw, time_mode, from_date, to_date)
    vix = filter_by_range(vix_raw, time_mode, from_date, to_date)

    last_update = max(baa10y.index.max(), t10y2y.index.max(), lei_yoy.index.max(), vix.index.max())

    # Latest values for risk summary (based on first three only)
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
        f"""<div style="display:flex;justify-content:flex-start;gap:8px;align-items:center;margin-bottom:8px;">
        <button style="background:{risk_color};color:white;border:none;padding:8px 14px;border-radius:10px;font-weight:800;cursor:default;">
        Overall Risk: {risk_label}</button></div>""",
        unsafe_allow_html=True
    )

except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# ---------------------------
# Render
# ---------------------------
view_mode = st.radio("View", ["Panels", "Combined"], index=0, horizontal=True)

if view_mode == "Combined":
    if st.session_state.get("include_vix_combined", False):
        # render combined 3, then overlay VIX with purple and its threshold
        plot_combined_view(baa10y, t10y2y, lei_yoy, credit_spread_thr, yield_curve_thr, lei_yoy_thr)
        fig = st.session_state.get("_last_combined_fig")
        # If we had stored the fig, we could extend it; since not, re-create a lightweight overlay plot
        # Instead, draw an additional chart just for VIX overlay simplicity:
        overlay = go.Figure()
        if vix is not None and not vix.empty:
            overlay.add_trace(go.Scatter(x=vix.index, y=vix.values, mode="lines", name="VIX (overlay)", line=dict(color="#9467bd")))
            overlay.add_hline(y=vix_thr, line_dash="dash", line_color="#9467bd", annotation_text=f"VIX danger @ {vix_thr:g}", annotation_font_color="#9467bd")
        overlay.update_layout(title="VIX Overlay", height=260, margin=dict(l=40, r=40, t=40, b=30), legend=dict(orientation="h"))
        st.plotly_chart(overlay, use_container_width=True)
    else:
        plot_combined_view(baa10y, t10y2y, lei_yoy, credit_spread_thr, yield_curve_thr, lei_yoy_thr)
else:
    col1, col2 = st.columns(2)
    col3 = st.container()

    with col1:
        st.subheader("Credit Spread: Moody's Baa − 10Y Treasury")
        plot_indicator(baa10y, "Credit Spread (BAA10Y)", " ppts", credit_spread_thr, breach_if="above")
        # Entry/Exit table
        b1 = breach_series(baa10y, credit_spread_thr, "above")
        e1, x1 = transitions(b1)
        if e1 or x1:
            df1 = pd.DataFrame({"Type": ["Enter"]*len(e1) + ["Exit"]*len(x1),
                                "Date": list(e1) + list(x1)}).sort_values("Date").reset_index(drop=True)
            st.caption("Credit Spread — danger entry/exit points")
            st.dataframe(df1, use_container_width=True)

    with col2:
        st.subheader("Yield Curve: 10Y − 2Y")
        plot_indicator(t10y2y, "Yield Curve (T10Y−2Y)", " ppts", yield_curve_thr, breach_if="below")
        # Entry/Exit table
        b2 = breach_series(t10y2y, yield_curve_thr, "below")
        e2, x2 = transitions(b2)
        if e2 or x2:
            df2 = pd.DataFrame({"Type": ["Enter"]*len(e2) + ["Exit"]*len(x2),
                                "Date": list(e2) + list(x2)}).sort_values("Date").reset_index(drop=True)
            st.caption("Yield Curve — danger entry/exit points")
            st.dataframe(df2, use_container_width=True)

    with col3:
    st.subheader("Leading Indicator (OECD CLI) — YoY change")
    plot_indicator(lei_yoy, "OECD CLI YoY % (USALOLITOAASTSAM)", " %", lei_yoy_thr, breach_if="below")
    # Entry/Exit table
    b3 = breach_series(lei_yoy, lei_yoy_thr, "below")
    e3, x3 = transitions(b3)
    if e3 or x3:
        df3 = pd.DataFrame({"Type": ["Enter"]*len(e3) + ["Exit"]*len(x3),
                            "Date": list(e3) + list(x3)}).sort_values("Date").reset_index(drop=True)
        st.caption("OECD CLI YoY — danger entry/exit points")
        st.dataframe(df3, use_container_width=True)

# VIX panel
col4 = st.container()
with col4:
    st.subheader("VIX — CBOE Volatility Index")
    plot_indicator(vix, "VIX (VIXCLS)", " index", vix_thr, breach_if="above")
    b4 = breach_series(vix, vix_thr, "above")
    e4, x4 = transitions(b4)
    if e4 or x4:
        df4 = pd.DataFrame({"Type": ["Enter"]*len(e4) + ["Exit"]*len(x4),
                            "Date": list(e4) + list(x4)}).sort_values("Date").reset_index(drop=True)
        st.caption("VIX — danger entry/exit points")
        st.dataframe(df4, use_container_width=True)

# --- VIX table (no chart, as requested) ---
st.subheader("VIX — danger entry/exit points")
if vix is not None and not vix.empty:
    b_vix = breach_series(vix, vix_thr, "above")
    e_vix, x_vix = transitions(b_vix)
    latest_vix = float(vix.iloc[-1])
    st.caption(f"Latest VIX: {latest_vix:.2f}")
    if e_vix or x_vix:
        df_vix = pd.DataFrame({"Type": ["Enter"]*len(e_vix) + ["Exit"]*len(x_vix),
                               "Date": list(e_vix) + list(x_vix)}).sort_values("Date").reset_index(drop=True)
        st.dataframe(df_vix, use_container_width=True)
    else:
        st.info("No VIX danger entries/exits in the selected window.")
else:
    st.warning("No VIX data available in the selected range.")

st.caption(f"Last data point across series: {last_update.date()} • Source: FRED (BAA10Y, T10Y2Y, USALOLITOAASTSAM, VIXCLS)")
st.caption("Note: LEI proxy uses OECD Composite Leading Indicator for the US (USALOLITOAASTSAM). The Conference Board LEI is proprietary.")