"""
EU-ETS Carbon Credit Analysis - Core Execution Pipeline
Final Project 3, AY 2023-2024
"""

from src.plotting import Plotter
from src.quantile_regression import QuantileRegressionAnalysis
from src.variance import VarianceEstimation
from src.cointegration import CointegrationAnalysis
from src.timeseries import TimeSeriesAnalysis
from src.liquidity import LiquidityAnalysis
from src.zspread import ZspreadAnalysis
from src.cspread import CspreadAnalysis
from src.bootstrap import BootstrapEngine
from src.preprocessing import DataPreprocessor
from src.utils import to_datenum, from_datenum
import os
import time
import warnings
import numpy as np
import pandas as pd
from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

CONFIG = {
    'start_year': 13,
    'end_year': 20,
    'flags': {
        'rolling': 0,       # 0: Nov 15, 1: OI crossover, 2: 1m before, 3: 1w before
        'variance': 1,      # 0: GARCH(1,1), 1: EWMA(0.95)
        'stationarity': 1,   # 0: KPSS only, 1: KPSS + ADF-GLS
        'show_plots': False   # Set to True to display plots interactively
    }
}


def main():
    t_start = time.time()

    # Initialize shared objects
    preprocessor = DataPreprocessor(DATA_DIR)
    plotter = Plotter(output_dir=OUTPUT_DIR,
                      show_plots=CONFIG['flags']['show_plots'])

    # 1. DATA IMPORT & BOOTSTRAP
    print("\n[1/8] Data Import & Bootstrap")

    ois_rates, dates_matrix = preprocessor.load_ois()
    daily_fut_dates, daily_fut_close = preprocessor.load_daily_futures()
    vol_march, vol_june, vol_sept = preprocessor.load_volumes()
    dates_futures = preprocessor.load_futures_dates()

    bootstrap = BootstrapEngine()
    discounts, zero_rates = bootstrap.compute(dates_matrix, ois_rates)
    dates_40y, zero_rates_40y = BootstrapEngine.extend_to_40y(
        dates_matrix, zero_rates)

    print(f"  -> OIS rates loaded: {ois_rates.shape}")
    print(f"  -> Zero curve bootstrapped & extended to 40y.")

    # Target plot for 13-Sep-2012
    target_dn = to_datenum(date(2012, 9, 13))
    sept_idx = np.where(dates_matrix[:, 0] == target_dn)[0]
    if len(sept_idx) > 0:
        plotter.plot_discount_curve(
            dates_matrix, discounts, zero_rates, sept_idx[0])

    # 2. LIQUIDITY ANALYSIS
    print("\n[2/8] Liquidity Analysis")

    liquidity = LiquidityAnalysis(preprocessor)
    front_march = liquidity.front_volumes(
        vol_march[:, 1:], 3, dates_futures) + 1
    front_june = liquidity.front_volumes(vol_june[:, 1:], 6, dates_futures) + 1
    front_sept = liquidity.front_volumes(vol_sept[:, 1:], 9, dates_futures) + 1
    front_dec, next_dec, nextnext_dec = liquidity.volumes_dec()

    print(f"  -> Front March volumes processed: {len(front_march)}")
    print(f"  -> Front December volumes processed: {len(front_dec)}")

    plotter.plot_liquidity_boxplots(
        front_march, front_june, front_sept, front_dec, next_dec, nextnext_dec)

    # 3. C-SPREAD & ROLLOVER
    print("\n[3/8] Computing C-spreads")

    cspread_analysis = CspreadAnalysis(preprocessor)

    C_front, dates_front, maturities = cspread_analysis.compute_front(
        zero_rates, dates_matrix, daily_fut_dates, daily_fut_close, CONFIG['start_year'], CONFIG['end_year'])

    C_next, dates_next = cspread_analysis.compute_next(
        zero_rates, dates_matrix, daily_fut_dates, daily_fut_close, CONFIG['start_year'], CONFIG['end_year'])

    oi_dates, oi_values = preprocessor.load_open_interest()
    roll_dates = cspread_analysis.rolling_mechanism(
        CONFIG['start_year'], CONFIG['end_year'], maturities, oi_dates, oi_values, CONFIG['flags']['rolling'])

    C_spread, dates_Cspread_roll = cspread_analysis.compute_rollover(
        zero_rates, dates_matrix, daily_fut_dates, daily_fut_close, CONFIG['start_year'], CONFIG['end_year'], roll_dates)

    print(
        f"  -> Front observations: {len(C_front)} | Rollover observations: {len(C_spread)}")
    plotter.plot_cspread(dates_front, C_front, dates_next,
                         C_next, dates_Cspread_roll, C_spread)

    # 4. Z-SPREAD / Z-INDEX
    print("\n[4/8] Computing Z-spread / Z-index")

    valid_bonds, names = preprocessor.load_bonds()
    zspread = ZspreadAnalysis(preprocessor)
    z_index, dates_Z = zspread.compute_zindex(
        names, valid_bonds, dates_40y, zero_rates_40y)

    print(f"  -> Processed {len(names)} issuers.")
    print(f"  -> Z-index compiled with {len(z_index)} observations.")

    # 5. TIME SERIES ALIGNMENT & STATIONARITY
    print("\n[5/8] Aligning Time Series & Testing Stationarity")

    ts_analysis = TimeSeriesAnalysis()
    C_aligned, z_aligned, r3m, dates_ts = ts_analysis.align_series(
        zero_rates, dates_matrix, maturities, dates_Z, dates_Cspread_roll, C_spread, z_index)

    print(f"  -> Final aligned matrix: {len(dates_ts)} dates.")

    plotter.plot_time_series(dates_ts, C_aligned, z_aligned, r3m)
    plotter.plot_acf_pacf(C_aligned, z_aligned, r3m)
    ts_analysis.stationarity_tests(
        C_aligned, z_aligned, r3m, CONFIG['flags']['stationarity'])

    # 6. COINTEGRATION ANALYSIS
    print("\n[6/8] Cointegration Analysis (Johansen)")

    coint = CointegrationAnalysis()
    Y = np.column_stack([C_aligned, z_aligned, r3m])
    cointeg_vector, coint_result = coint.johansen_test(Y, alpha=0.1)

    plotter.plot_cointegration(
        dates_ts, C_aligned, z_aligned, r3m, cointeg_vector)

    # 7. VARIANCE & ERROR CORRECTION MODEL
    print("\n[7/8] Variance Estimation & ECM")

    dates_extra, Ret_SPX, Ret_VIX, Ret_WTI = preprocessor.load_extra_variables()
    var_est = VarianceEstimation()
    var_garch, var_ewma, log_returns = var_est.estimate(
        daily_fut_dates, daily_fut_close, dates_ts)

    plotter.plot_variance(dates_ts, var_garch, var_ewma, log_returns)

    variance = var_garch if CONFIG['flags']['variance'] == 0 else var_ewma
    ecm_result, x_ecm, delta_C, dates_lag, BIC, AIC, predicted = coint.error_correction_model(
        C_aligned, z_aligned, r3m, cointeg_vector, dates_extra, Ret_SPX, Ret_VIX, Ret_WTI, variance, dates_ts, CONFIG['flags']['variance'])

    # 8. QUANTILE REGRESSION
    print("\n[8/8] Quantile Regression Analysis")

    qr = QuantileRegressionAnalysis()
    x_qr = np.column_stack([np.ones(x_ecm.shape[0]), x_ecm])
    quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    C_qr, P_qr, SE_qr = qr.multi_quantile_regression(x_qr, delta_C, quantiles)

    print("\n  Quantile Regression Results Overview:")
    for i, q in enumerate(quantiles):
        print(f"    q={q:.1f} | Coeffs: {np.round(C_qr[:, i], 4)}")

    plotter.plot_confidence_interval(
        x_qr, delta_C, C_qr, dates_lag, quantiles, 0)

    # COMPLETION
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Pipeline completed successfully in {elapsed:.2f} seconds.")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
