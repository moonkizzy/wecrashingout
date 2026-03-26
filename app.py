import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from fredapi import Fred
from datetime import date

# Basic config
st.set_page_config(page_title="Market Stress Mon", layout="wide")

# --- SIDEBAR & AUTH ---
with st.sidebar:
    st.title("Settings")
    # Check secrets or env first
    api_key = st.secrets.get("FRED_API_KEY", os.getenv("FRED_API_KEY", ""))
    if not api_key:
        api_key = st.text_input("FRED API Key", type="password")
    
    lookback = st.selectbox("Range", ["1Y", "3Y", "5Y", "10Y"], index=0)
    
    st.divider()
    st.subheader("Thresholds")
    # Grouping thresholds into a dict for cleaner access later
    limits = {
        "BAA10Y": st.number_input("Credit Spread >", value=2.5),
        "T10Y2Y": st.number_input("Yield Curve <", value=0.0),
        "CLI": st.number_input("OECD CLI YoY <", value=-0.5),
        "VIX": st.number_input("VIX >", value=30.0)
    }

if not api_key:
    st.info("Please provide a FRED API key to continue.")
    st.stop()

fred = Fred(api_key=api_key)

@st.cache_data(ttl=600)
def wcget_data():
    # Fetching core series
    data = {
        "credit": fred.get_series("BAA10Y"),
        "yc": fred.get_series("T10Y2Y"),
        "cli_raw": fred.get_series("USALOLITOAASTSAM"),
        "vix": fred.get_series("VIXCLS")
    }
    df = pd.DataFrame(data).ffill()
    # Calc CLI YoY
    df['cli_yoy'] = df['cli_raw'].pct_change(12) * 100
    return df

try:
    df = wcget_data()
    
    # Simple date filtering based on selection
    offset = {"1Y": 365, "3Y": 1095, "5Y": 1825, "10Y": 3650}
    days = offset.get(lookback, 365)
    df = df.tail(days)
    
    st.title("Market Stress Dashboard")
    
    # Top Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    
    # Logic for status badges (moved out of functions to feel more 'script-like')
    spread = df['credit'].iloc[-1]
    yc = df['yc'].iloc[-1]
    vix = df['vix'].iloc[-1]
    cli = df['cli_yoy'].iloc[-1]

    def wcget_badge(val, limit, mode='above'):
        is_bad = val >= limit if mode == 'above' else val <= limit
        color = "red" if is_bad else "green"
        label = "ALERT" if is_bad else "OK"
        return f":{color}[{label}] ({val:.2f})"

    c1.metric("Credit Spread", f"{spread:.2f}", delta_color="inverse")
    c1.markdown(wcget_badge(spread, limits['BAA10Y']))
    
    c2.metric("Yield Curve", f"{yc:.2f}")
    c2.markdown(wcget_badge(yc, limits['T10Y2Y'], 'below'))
    
    c3.metric("OECD CLI YoY", f"{cli:.2f}%")
    c3.markdown(wcget_badge(cli, limits['CLI'], 'below'))
    
    c4.metric("VIX", f"{vix:.2f}")
    c4.markdown(wcget_badge(vix, limits['VIX']))

    st.divider()

    # Main plotting loop - less redundant than the AI's "plot_indicator" function
    plots = [
        ("Credit Spread", df['credit'], limits['BAA10Y'], "above"),
        ("Yield Curve (10Y-2Y)", df['yc'], limits['T10Y2Y'], "below"),
        ("OECD CLI (YoY %)", df['cli_yoy'], limits['CLI'], "below"),
        ("VIX Index", df['vix'], limits['VIX'], "above")
    ]

    cols = st.columns(2)
    for i, (name, series, thresh, mode) in enumerate(plots):
        target_col = cols[i % 2]
        with target_col:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series, name=name, line=dict(width=2)))
            
            # Threshold line
            fig.add_hline(y=thresh, line_dash="dot", line_color="red", opacity=0.5)
            
            # Shading logic (inline)
            mask = series >= thresh if mode == "above" else series <= thresh
            # Find contiguous blocks for shading
            diff = mask.astype(int).diff().fillna(0)
            starts = diff[diff == 1].index
            ends = diff[diff == -1].index
            
            for s, e in zip(starts, ends):
                fig.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.1, line_width=0)

            fig.update_layout(
                title=name,
                margin=dict(l=20, r=20, t=40, b=20),
                height=350,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Something went wrong: {e}")

st.caption(f"Last sync: {date.today()} | Data via St. Louis Fed")