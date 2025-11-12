# Markets Stress Dashboard (Streamlit)

Monitor **Credit Spreads**, **Yield Curve**, and **Leading Economic Index (YoY)** with configurable danger thresholds.

### Data sources
- **Credit Spread**: `BAA10Y` (Moody’s Baa – 10Y Treasury), FRED
- **Yield Curve**: `T10Y2Y` (10Y minus 2Y), FRED
- **LEI proxy**: `USSLIND` (Leading Index for the United States), FRED; displayed as YoY % change

> If you subscribe to the Conference Board's LEI, you can swap in that series easily in the code.

---

## Quick start

1. **Install** (prefer a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

2. **Get a free FRED API key** and export it as an environment variable:
   ```bash
   export FRED_API_KEY=YOUR_KEY_HERE
   ```
   *(Or paste it into the sidebar when running the app.)*

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

4. **Use the sidebar** to:
   - Set the chart window (1–5 years or max)
   - Adjust **danger thresholds** for each indicator
   - Click **Refresh now** to clear cache and pull latest data (the app caches data for 5 minutes).

---

## Notes
- Threshold defaults:
  - Credit spread danger **above** `2.5` percentage points
  - Yield curve danger **below** `0.0` percentage points
  - LEI YoY danger **below** `-0.5%`
- Change these to your own risk tolerances in the sidebar.
- The app uses FRED (free) and runs locally, so your monthly cost is **$0**.