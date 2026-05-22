# EU-ETS Carbon Statistical Arbitrage & Cointegration

An analyzis of the European Union Emissions Trading System (EU-ETS), based on the work of M. Azzone, R.Baviera, P. Manzoni "The puzzle of Carbon Allowance spread". This project bootstraps risk-free curves, computes convenience yields (C-spreads) and corporate credit spreads (Z-spreads). The Error Correction Model is used to evaluate the mean-reverting properties of the carbon market.

## Quantitative Methodology
* **OIS Bootstrapping:** Extracts the term structure of zero-coupon discount factors from observable Overnight Indexed Swap (OIS) par rates using a 3-regime iterative bootstrap.
* **Cost-of-Carry:** Calculates the **C-spread** across Front and Next December EUA futures, implementing a dynamic rollover mechanism.
* **Credit Spreads:** Calculates a weighted **Z-index** from corporate bond prices using Brent's method.
* **Econometrics:** Deploys **KPSS** and **ADF-GLS** tests to verify stationarity.
  * Utilizes the **Johansen Test** to establish cointegration between the C-spread, Z-index and 3-month risk-free rate.
  * Estimates an **Error Correction Model (ECM)** via OLS to quantify the speed of mean reversion.
* **Volatility & Tails:** Estimates conditional variance via **GARCH(1,1)** and **EWMA** and evaluates tail risks using **Quantile Regression**.

## Repository Architecture

    eu-ets-carbon-analysis/
    ├── data/               # Input market data (not uploaded)
    ├── output/             # Exported PNG visualisations
    ├── src/                # Core modular engine
    ├── mmain.py            # Pipeline script
    └── requirements.txt    # Environment dependencies

## Quick Start
1. Clone the repository.
2. Create and activate a virtual environment:
   **Windows:**
   `python -m venv .venv`
   `.venv\Scripts\activate`

   **Mac/Linux:**
   `python3 -m venv .venv`
   `source .venv/bin/activate`

3. Install the dependencies:
   `pip install -r requirements.txt`

4. Execute the main pipeline:
   `python mmain.py`

*(Note: Visual outputs are generated headlessly for speed and saved directly to the `/output` directory. Set `show_plots = True` in `mmain.py` config for interactive UI execution).*
