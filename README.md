# Markets Stress Dashboard (Streamlit)

Monitor **Credit Spreads**, **Yield Curve**, and **Leading Indicator (YoY)** with configurable danger thresholds.

### Data sources
- **Credit Spread**: `BAA10Y` (Moody’s Baa – 10Y Treasury), FRED
- **Yield Curve**: `T10Y2Y` (10Y minus 2Y), FRED
- **LEI proxy**: `USALOLITOAASTSAM` (OECD Composite Leading Indicator for the US), FRED; displayed as YoY % change

## Notes
- If you previously used `USSLIND` (Philadelphia Fed Leading Index), that series was **last updated Feb 2020** and is no longer maintained.
- The **Conference Board LEI** is proprietary; if you have access, you can swap it in.
