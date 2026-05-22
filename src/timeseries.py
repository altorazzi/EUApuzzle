"""
Time-series alignment and stationarity testing.

Aligns the C-spread, Z-index, and 3-month zero rate series on their
common date grid, then runs KPSS and (optionally) ADF-GLS unit-root
tests on levels and first differences.
"""

import numpy as np
from datetime import date

from statsmodels.tsa.stattools import kpss
from arch.unitroot import DFGLS

from src.utils import to_datenum


class TimeSeriesAnalysis:
    """Time-series alignment and stationarity diagnostics."""

    @staticmethod
    def align_series(zero_rates, dates_matrix, maturities, dates_Z,
                     dates_cspread, cspread, z_index):
        """
        Align C-spread, Z-index, and 3-month zero rate on common dates.

        Args:
            zero_rates    : np.ndarray (n, 28) -- bootstrapped zero rates.
            dates_matrix  : np.ndarray (n, 28) -- date matrix from OIS.
            maturities    : np.ndarray -- maturity datenums of each Dec contract.
            dates_Z       : np.ndarray -- datenums for the Z-index.
            dates_cspread : np.ndarray -- datenums for the C-spread.
            cspread       : np.ndarray -- C-spread values.
            z_index       : np.ndarray -- Z-index values.

        Returns:
            cspread_aligned   : np.ndarray
            z_index_aligned   : np.ndarray
            zero_rates_3m     : np.ndarray
            dates_ts          : np.ndarray of datenums
        """
        rates_start = dates_matrix[:, 0]
        start_dn = to_datenum(date(2013, 1, 1))
        end_dn = maturities[-1]

        # Extract 3-month zero rate (column 7) within the analysis window
        idx_range = np.where((rates_start >= start_dn) &
                             (rates_start <= end_dn))[0]
        rates_dates_3m = rates_start[idx_range]
        zero_rates_3m = zero_rates[idx_range, 7]

        # Step 1: intersect Z-index with C-spread dates
        common_zc = np.intersect1d(dates_Z, dates_cspread)
        idx_z = np.where(np.isin(dates_Z, common_zc))[0]
        dates_Z_aligned = dates_Z[idx_z]
        z_aligned = z_index[idx_z]

        # Step 2: intersect with zero-rate dates
        common_all = np.intersect1d(dates_Z_aligned, rates_dates_3m)
        idx_rates = np.where(np.isin(rates_dates_3m, common_all))[0]
        rates_dates_3m = rates_dates_3m[idx_rates]
        zero_rates_3m = zero_rates_3m[idx_rates]

        # Step 3: intersect C-spread with final dates
        idx_c = np.where(np.isin(dates_cspread, rates_dates_3m))[0]
        cspread_aligned = cspread[idx_c]
        dates_cspread_aligned = dates_cspread[idx_c]

        # Step 4: re-align Z and rates to the final date set
        idx_z_final = np.where(
            np.isin(dates_Z_aligned, dates_cspread_aligned))[0]
        z_aligned = z_aligned[idx_z_final]

        idx_r_final = np.where(
            np.isin(rates_dates_3m, dates_cspread_aligned))[0]
        zero_rates_3m = zero_rates_3m[idx_r_final]

        dates_ts = dates_cspread_aligned

        # Remove any NaN or Inf values
        valid = (
            np.isfinite(cspread_aligned)
            & np.isfinite(z_aligned)
            & np.isfinite(zero_rates_3m)
        )
        return (
            cspread_aligned[valid],
            z_aligned[valid],
            zero_rates_3m[valid],
            dates_ts[valid],
        )

    @staticmethod
    def stationarity_tests(cspread, z_index, zero_rates_3m, flag):
        """
        Run KPSS tests on levels and first differences.
        If flag == 1, also run ADF-GLS tests.

        Args:
            flag : 0 = KPSS only, 1 = KPSS + ADF-GLS.
        """
        levels = {
            'C_spread': cspread,
            'Z_index': z_index,
            'zero_rates_3m': zero_rates_3m,
        }
        diffs = {
            'delta_C': np.diff(cspread),
            'delta_Z': np.diff(z_index),
            'delta_r': np.diff(zero_rates_3m),
        }

        print("\n--- KPSS Tests on Levels ---")
        for name, series in levels.items():
            try:
                stat, pval, lags, _ = kpss(
                    series, regression='c', nlags='auto')
                print(
                    f"  {name}: stat={stat:.4f}, p-value={pval:.4f}, lags={lags}")
            except Exception as e:
                print(f"  {name}: Error - {e}")

        print("\n--- KPSS Tests on First Differences ---")
        for name, series in diffs.items():
            try:
                stat, pval, lags, _ = kpss(
                    series, regression='c', nlags='auto')
                print(
                    f"  {name}: stat={stat:.4f}, p-value={pval:.4f}, lags={lags}")
            except Exception as e:
                print(f"  {name}: Error - {e}")

        if flag == 1:
            print("\n--- ADF-GLS Tests on Levels ---")
            for name, series in levels.items():
                try:
                    result = DFGLS(series)
                    print(f"  {name}: pValue = {result.pvalue:.4f}")
                except Exception as e:
                    print(f"  {name}: Error - {e}")

            print("\n--- ADF-GLS Tests on First Differences ---")
            for name, series in diffs.items():
                try:
                    result = DFGLS(series)
                    print(f"  {name}: pValue = {result.pvalue:.4f}")
                except Exception as e:
                    print(f"  {name}: Error - {e}")
