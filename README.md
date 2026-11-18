# ⚡ Microclimate-Informed Spatio-Temporal EV Charging Forecaster (TDI + LightGBM)

An end-to-end data science pipeline and interactive Streamlit smart-city planning dashboard that ingests real municipal EV charging telemetry and localized hourly weather data via live APIs, engineers the physics-based **Thermal Drain Index (TDI)**, trains and tunes a **LightGBM regressor** on a strict chronological split, proves TDI outperforms raw temperature, and surfaces predicted municipal energy spikes.

---

## Key Empirical Results (City of Boulder, CO — 2022)

- **Data Ingested via Live Public APIs**:
  - **Municipal EV Charging Logs**: City of Boulder Open Data ArcGIS API (`37,654` raw sessions → `34,284` valid sessions across `29` stations; `342,238.0 kWh` total, `173.1 kWh` peak hour)
  - **Localized Weather Telemetry**: Open-Meteo Historical Archive API (`8,760` hourly records spanning `-29.0°C` to `+35.1°C`, capturing the Dec 2022 Arctic cold snap)
- **Strict Chronological Split (Zero Temporal Leakage)**:
  - **Train**: Months 1–9 (`6,384` hrs) · **Validate**: Month 10 (`744` hrs, early stopping & 36-config grid search) · **Test**: Months 11–12 (`1,464` hrs, `505` extreme-weather hours with `TDI ≥ 60`)

### Ablation Study (Held-Out Months 11–12 Test Set)

| Model Configuration | Test RMSE (kWh) | Test MAE (kWh) | Test $R^2$ | Peak Top-10% MAE (kWh) | Extreme-Weather RMSE (kWh) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Baseline (Temporal + Causal Lags) | `13.1815` | `9.4355` | `0.8502` | `19.6975` | `10.7552` |
| 2. + Raw Weather (`temp`, `humidity`, `wind`, `precip`) | `13.1626` | `9.3387` | `0.8506` | `20.9200` | `10.4158` |
| 3. + TDI (Default LightGBM) | `13.1059` | `9.3511` | `0.8519` | `20.6481` | `10.2823` |
| 4. **+ TDI (Hyperparameter-Tuned LightGBM)** | **`13.0142`** | `9.3876` | **`0.8540`** | **`19.3933`** | `10.3498` |

### Empirical Proof: TDI Outperforms Raw Temperature
- **LightGBM Information Gain**: TDI Feature Group (`0.934%`) vs Raw `temp_c` (`0.545%`) → **1.71× higher gain**
- **Held-Out Test Permutation Importance (RMSE Δ)**: TDI Feature Group (`0.1288 kWh`) vs Raw `temp_c` (`0.0113 kWh`) → **11.4× higher permutation importance**

![Empirical Proof: TDI vs Raw Temperature](outputs/tdi_vs_raw_temp_proof.png)

![Held-Out Test Forecast During Dec 2022 Boulder Arctic Cold Snap](outputs/forecast_vs_actual_test.png)

---

## Mathematical Formulation: Thermal Drain Index (TDI)

EVs consume disproportionate battery energy for in-cabin HVAC heating (no ICE waste heat + higher Li-ion internal resistance in freezing weather) and cooling outside the optimal $20^\circ\text{C} - 25^\circ\text{C}$ operating band:

$$\Delta T_{\text{cold}}(t) = \max(0,\; 20 - T(t)), \qquad \Delta T_{\text{hot}}(t) = \max(0,\; T(t) - 25)$$

$$\text{BaseDrain}(t) = 1.45 \cdot \Delta T_{\text{cold}}(t)^{1.15} + 0.85 \cdot \Delta T_{\text{hot}}(t)^{1.10}$$

$$\text{Severity}(t) = 1 + 0.25 \cdot \frac{\max(0, H(t) - 50)}{50} + 0.18 \cdot \frac{W(t)}{30} + 0.22 \cdot \ln(1 + P(t))$$

$$\text{TDI}(t) = \text{BaseDrain}(t) \cdot \text{Severity}(t)$$

Sustained thermal stress proxies (`tdi_roll_6h`, `tdi_roll_24h`, `tdi_delta_3h`, `tdi_x_commute`) capture multi-hour battery cold-soak and commuter range anxiety.

![Thermal Drain Index Physics Curve](outputs/tdi_physics_curve.png)

![LightGBM Feature Importance by Information Gain](outputs/feature_importance_gain.png)

---

## Project Structure

```text
ev_tdi_forecaster/
├── run_pipeline.py                # One-command pipeline orchestrator (Stages 1–4)
├── app.py                         # Interactive Streamlit smart-city planner dashboard
├── requirements.txt               # Python dependencies
├── .streamlit/
│   └── secrets.toml.example       # Template for optional Google Maps API key
├── src/
│   ├── config.py                  # API endpoints, TDI physics constants, station coordinates
│   ├── ingest.py                  # Stage 1: Boulder EV + Open-Meteo API ingestion & alignment
│   ├── features.py                # Stage 2: Temporal features (hour, dow, holidays, lags) + TDI
│   ├── train.py                   # Stage 3: Chronological split (M1-9/M10/M11-12), tuning & ablation
│   └── visualize.py               # Stage 4: Feature importance, TDI proof & winter spike plots
├── data/                          # Aligned municipal EV + weather datasets
├── models/                        # Saved tuned LightGBM model bundle
└── outputs/                       # Metrics JSON, tuning leaderboard, predictions & PNG charts
```

---

## Quick Start

```bash
# 1. Create a Python 3.11 / 3.12 virtual environment and install dependencies
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# macOS / Linux:
# source .venv/bin/activate && pip install -r requirements.txt

# 2. (Optional) Enable Google Maps Heatmap + Live Traffic in Tab 4
# Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml and add your key

# 3. Run the full pipeline (API ingestion -> feature engineering -> tuning -> charts)
.\.venv\Scripts\python.exe run_pipeline.py

# 4. Launch the interactive Smart City Planner dashboard
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Dashboard Highlights (`app.py`)
1. **Predicted Energy Spikes**: Interactive forecast vs actual timeline with configurable feeder capacity alert threshold (kWh), TDI overlay, and downloadable CSV spike schedule.
2. **What-If Microclimate Stress Simulator**: Live sliders for temperature, humidity, wind, precipitation, hour, day of week, and holidays with real-time TDI, estimated HVAC range loss %, and temperature sensitivity sweep.
3. **Feature Importance & TDI Proof**: Interactive Information Gain, Permutation Importance, and Split Frequency charts alongside the ablation benchmark and hyperparameter tuning leaderboard.
4. **Municipal Station Hotspots**: Google Maps Demand Heatmap + clickable Station InfoWindows + Live Traffic overlay (with one-click OpenStreetMap Plotly fallback).
