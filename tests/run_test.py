"""
Quick end-to-end test of the refactored modular pipeline.
Mirrors the notebook flow but runs non-interactively.
"""
import os, sys, time, warnings
import numpy as np
from datetime import date

warnings.filterwarnings('ignore')

# Ensure we import from the local directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import to_datenum, from_datenum
from preprocessing import DataPreprocessor
from bootstrap import BootstrapEngine
from cspread import CspreadAnalysis
from zspread import ZspreadAnalysis
from liquidity import LiquidityAnalysis
from timeseries import TimeSeriesAnalysis
from cointegration import CointegrationAnalysis
from variance import VarianceEstimation
from quantile_regression import QuantileRegressionAnalysis
from plotting import Plotter

import matplotlib
matplotlib.use('Agg')  # non-interactive for testing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    t0 = time.time()
    print("=" * 70)
    print("  EU-ETS Carbon Credit Analysis -- Modular Python Version")
    print("=" * 70)

    START_YEAR, END_YEAR = 13, 20
    FLAG_ROLLING = 0
    FLAG_VARIANCE = 0
    FLAG_STATIONARITY = 0

    pre = DataPreprocessor(BASE_DIR)
    plotter = Plotter(output_dir=BASE_DIR)

    # 1. Bootstrap
    print("\n--- Point 1) Data Import & Bootstrap ---")
    ois_rates, dates_matrix = pre.load_ois()
    print(f"  OIS rates shape: {ois_rates.shape}")
    daily_fut_dates, daily_fut_close = pre.load_daily_futures()
    print(f"  Daily futures: {len(daily_fut_dates)} records")
    vol_march, vol_june, vol_sept = pre.load_volumes()
    dates_futures = pre.load_futures_dates()

    boot = BootstrapEngine()
    discounts, zero_rates = boot.compute(dates_matrix, ois_rates)
    print(f"  Discounts shape: {discounts.shape}")
    dates_40y, zr40y = BootstrapEngine.extend_to_40y(dates_matrix, zero_rates)
    print(f"  Extended to 40y: {dates_40y.shape}")

    # Verify zero rates are not flat
    target_dn = to_datenum(date(2012, 9, 13))
    idx = np.where(dates_matrix[:, 0] == target_dn)[0]
    if len(idx) > 0:
        zr_sample = zero_rates[idx[0], 2:]
        print(f"  Zero rates 13-Sep-2012 (min/max): {zr_sample.min():.6f} / {zr_sample.max():.6f}")
        assert zr_sample.max() > 0.001, "Zero rates are flat!"

    # 2. Liquidity
    print("\n--- Point 2) Liquidity Analysis ---")
    liq = LiquidityAnalysis(pre)
    fm = liq.front_volumes(vol_march[:, 1:], 3, dates_futures) + 1
    fj = liq.front_volumes(vol_june[:, 1:], 6, dates_futures) + 1
    fs = liq.front_volumes(vol_sept[:, 1:], 9, dates_futures) + 1
    fd, nd, nnd = liq.volumes_dec()
    print(f"  Front March: {len(fm)}, Front Dec: {len(fd)}, Next Dec: {len(nd)}, NextNext Dec: {len(nnd)}")

    # 3. C-spread
    print("\n--- Point 3) C-spread ---")
    cs = CspreadAnalysis(pre)
    C_front, d_front, mats = cs.compute_front(zero_rates, dates_matrix, daily_fut_dates, daily_fut_close, START_YEAR, END_YEAR)
    C_next, d_next = cs.compute_next(zero_rates, dates_matrix, daily_fut_dates, daily_fut_close, START_YEAR, END_YEAR)
    print(f"  Front: {len(C_front)}, Next: {len(C_next)}")

    oi_dates, oi_values = pre.load_open_interest()
    roll_dates = cs.rolling_mechanism(START_YEAR, END_YEAR, mats, oi_dates, oi_values, FLAG_ROLLING)
    C_roll, d_roll = cs.compute_rollover(zero_rates, dates_matrix, daily_fut_dates, daily_fut_close, START_YEAR, END_YEAR, roll_dates)
    print(f"  Rollover: {len(C_roll)}")

    # 4. Z-spread
    print("\n--- Point 4) Z-spread / Z-index ---")
    vb, names = pre.load_bonds()
    print(f"  Issuers: {names}")
    zs = ZspreadAnalysis(pre)
    z_idx, d_Z = zs.compute_zindex(names, vb, dates_40y, zr40y)
    print(f"  Z-index: {len(z_idx)}")

    # 5. Time series
    print("\n--- Point 5) Time Series Alignment ---")
    tsa = TimeSeriesAnalysis()
    Ca, Za, r3m, d_ts = tsa.align_series(zero_rates, dates_matrix, mats, d_Z, d_roll, C_roll, z_idx)
    print(f"  Aligned: {len(d_ts)}")
    tsa.stationarity_tests(Ca, Za, r3m, FLAG_STATIONARITY)

    # 6. Cointegration
    print("\n--- Point 6) Cointegration ---")
    coint = CointegrationAnalysis()
    Y = np.column_stack([Ca, Za, r3m])
    cv, _ = coint.johansen_test(Y)

    # 7. Variance + ECM
    print("\n--- Point 7) Variance & ECM ---")
    d_extra, rSPX, rVIX, rWTI = pre.load_extra_variables()
    ve = VarianceEstimation()
    vg, vw, lr = ve.estimate(daily_fut_dates, daily_fut_close, d_ts)
    variance = vg if FLAG_VARIANCE == 0 else vw

    ecm, x_ecm, dC, d_lag, BIC, AIC, pred = coint.error_correction_model(
        Ca, Za, r3m, cv, d_extra, rSPX, rVIX, rWTI, variance, d_ts, FLAG_VARIANCE)

    # 8. Quantile regression
    print("\n--- Point 8) Quantile Regression ---")
    qr = QuantileRegressionAnalysis()
    x_qr = np.column_stack([np.ones(x_ecm.shape[0]), x_ecm])
    quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    Cq, Pq, SEq = qr.multi_quantile_regression(x_qr, dC, quantiles)

    for i, q in enumerate(quantiles):
        print(f"\n  Quantile {q:.1f}: coeff={Cq[:, i]}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Completed in {elapsed:.1f} seconds")
    print(f"{'=' * 70}")

if __name__ == '__main__':
    main()
