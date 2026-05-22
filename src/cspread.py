"""
C-spread (convenience yield spread) computation.

The C-spread measures the cost-of-carry spread implied by EUA futures:

    C = ln(F / S) / delta  -  r(t, T)

where F is the futures price, S the spot proxy (daily futures close),
delta the ACT/365 year fraction to maturity, and r(t,T) the
interpolated risk-free zero rate.

This module also implements the rollover mechanism that stitches
front and next contracts into a continuous time series.
"""

import numpy as np
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

from src.utils import to_datenum, from_datenum, yearfrac_act_365


class CspreadAnalysis:
    """Compute C-spread time series from EUA futures and OIS zero rates."""

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    # ------------------------------------------------------------------
    # Core: single-year C-spread
    # ------------------------------------------------------------------

    def compute_cspread(self, year, zero_rates, rates_dates,
                        daily_fut_dates, daily_fut_close, flag):
        """
        Compute the C-spread for one contract year.

        Args:
            year            : 2-digit year (e.g. 13 for Dec-2013 contract).
            zero_rates      : np.ndarray (n, 28).
            rates_dates     : np.ndarray (n, 28) of datenums.
            daily_fut_dates : np.ndarray of datenums (spot proxy).
            daily_fut_close : np.ndarray of close prices (spot proxy).
            flag            : 0 = FRONT contract, 1 = NEXT contract.

        Returns:
            C        : np.ndarray of C-spread values.
            dates_ts : np.ndarray of datenums.
            maturity : float (datenum of contract maturity).
        """
        # Load the three relevant futures files
        dates_prev, _, _ = self.preprocessor.load_ice_futures(year - 1 - flag)
        dates_curr, _, _ = self.preprocessor.load_ice_futures(year - flag)
        dates_main, close_main, _ = self.preprocessor.load_ice_futures(year)

        start_date = dates_prev[-1]   # expiry of previous contract
        end_date = dates_curr[-1]     # expiry of current contract

        # Maturity date
        if year != 22:
            maturity = dates_main[-1]
        else:
            mat_date = from_datenum(dates_prev[-1])
            maturity = to_datenum(mat_date + relativedelta(years=1 + flag))

        # Slice futures prices to the relevant window
        future_prices = close_main.copy()
        dates_future = dates_main.copy()

        if flag == 0:
            idx = np.where(dates_future > start_date)[0]
            if len(idx) == 0:
                return np.array([]), np.array([]), maturity
            future_prices = future_prices[idx[0]:]
            dates_future = dates_future[idx[0]:]
        else:
            idx = np.where((dates_future > start_date) &
                           (dates_future <= end_date))[0]
            if len(idx) == 0:
                return np.array([]), np.array([]), maturity
            future_prices = future_prices[idx]
            dates_future = dates_future[idx]

        # Intersect dates across OIS, futures, and daily-futures
        rates_start = rates_dates[:, 0]
        common = np.intersect1d(rates_start, dates_future)
        common = np.intersect1d(common, daily_fut_dates)

        idx_rates = np.where(np.isin(rates_start, common))[0]
        idx_futures = np.where(np.isin(dates_future, common))[0]
        idx_daily = np.where(np.isin(daily_fut_dates, common))[0]

        if len(idx_rates) == 0:
            return np.array([]), np.array([]), maturity

        spot = daily_fut_close[idx_daily]
        fut = future_prices[idx_futures]

        # Interpolate zero rate at the maturity date for each observation
        rf = np.array([
            np.interp(maturity, rates_dates[ri, :], zero_rates[ri, :])
            for ri in idx_rates
        ])

        dates_ts = common.copy()

        # Year fractions to maturity (ACT/365)
        delta = np.array([
            yearfrac_act_365(from_datenum(d), from_datenum(maturity))
            for d in dates_ts
        ])

        # C = ln(F/S) / delta - r(t,T)   (drop last point where delta -> 0)
        n = len(dates_ts) - 1
        C = np.log(fut[:n] / spot[:n]) / delta[:n] - rf[:n]
        dates_ts = dates_ts[:n]

        return C, dates_ts, maturity

    # ------------------------------------------------------------------
    # Front series
    # ------------------------------------------------------------------

    def compute_front(self, zero_rates, dates, daily_fut_dates,
                      daily_fut_close, start_year, end_year):
        """Build the complete front December C-spread series."""
        C_front = np.array([])
        dates_front = np.array([])
        maturities = np.zeros(end_year - start_year + 1)

        for yr in range(start_year, end_year + 1):
            C, dt, mat = self.compute_cspread(
                yr, zero_rates, dates, daily_fut_dates, daily_fut_close, 0
            )
            if len(C) > 0:
                C_front = np.concatenate([C_front, C])
                dates_front = np.concatenate([dates_front, dt])
            maturities[yr - start_year] = mat

        return C_front, dates_front, maturities

    # ------------------------------------------------------------------
    # Next series
    # ------------------------------------------------------------------

    def compute_next(self, zero_rates, dates, daily_fut_dates,
                     daily_fut_close, start_year, end_year):
        """Build the complete next December C-spread series."""
        C_next = np.array([])
        dates_next = np.array([])

        for yr in range(start_year + 1, end_year + 1):
            C, dt, _ = self.compute_cspread(
                yr, zero_rates, dates, daily_fut_dates, daily_fut_close, 1
            )
            if len(C) > 0:
                C_next = np.concatenate([C_next, C])
                dates_next = np.concatenate([dates_next, dt])

        return C_next, dates_next

    # ------------------------------------------------------------------
    # Rolling mechanism
    # ------------------------------------------------------------------

    def rolling_mechanism(self, start_year, end_year, maturities,
                          oi_dates, oi_values, flag):
        """
        Determine rollover dates for the continuous C-spread series.

        Args:
            flag : 0 = roll on Nov 15,
                   1 = open-interest crossover,
                   2 = 1 month before expiry,
                   3 = 1 week before expiry.

        Returns:
            roll_dates : np.ndarray of datenums (one per year).
        """
        time_lag = end_year - start_year
        roll_dates = np.zeros(time_lag + 1)
        maturities_ext = np.concatenate([[oi_dates[0]], maturities])

        if flag == 0:
            for i in range(time_lag + 1):
                yr = 2000 + start_year + i
                roll_dates[i] = to_datenum(date(yr, 11, 15))

        elif flag == 1:
            for ii in range(time_lag + 1):
                col_next = ii + 1
                col_front = ii
                if col_next >= oi_values.shape[1] or col_front >= oi_values.shape[1]:
                    roll_dates[ii] = maturities_ext[ii + 1]
                    continue

                crossover = np.where(
                    oi_values[:, col_next] > oi_values[:, col_front])[0]
                if len(crossover) == 0:
                    roll_dates[ii] = maturities_ext[ii + 1]
                else:
                    candidates = oi_dates[crossover]
                    valid = candidates[
                        (candidates < maturities_ext[ii + 1])
                        & (candidates > maturities_ext[ii])
                    ]
                    roll_dates[ii] = valid[0] if len(
                        valid) > 0 else maturities_ext[ii + 1]

        elif flag == 2:
            for i in range(time_lag + 1):
                mat = from_datenum(maturities_ext[i + 1])
                roll_dates[i] = to_datenum(mat - relativedelta(months=1))

        elif flag == 3:
            for i in range(time_lag + 1):
                mat = from_datenum(maturities_ext[i + 1])
                roll_dates[i] = to_datenum(mat - timedelta(weeks=1))

        return roll_dates

    # ------------------------------------------------------------------
    # Rollover construction
    # ------------------------------------------------------------------

    def compute_rollover(self, zero_rates, rates_dates, daily_fut_dates,
                         daily_fut_close, start_year, end_year, roll_dates):
        """
        Build the unified (rolled-over) C-spread series.

        For each year, take the FRONT series up to the roll date,
        then switch to the NEXT series after the roll date.

        Returns:
            C         : np.ndarray of C-spread values.
            dates_out : np.ndarray of datenums.
        """
        C = np.array([])
        dates_out = np.array([])

        for yr in range(start_year, end_year + 1):
            # Front portion
            C_f, d_f, _ = self.compute_cspread(
                yr, zero_rates, rates_dates, daily_fut_dates, daily_fut_close, 0
            )
            rd = roll_dates[yr - start_year]
            idx_f = np.where(d_f < rd)[0]
            if len(idx_f) > 0:
                C = np.concatenate([C, C_f[idx_f]])
                dates_out = np.concatenate([dates_out, d_f[idx_f]])

            # Next portion
            C_n, d_n, _ = self.compute_cspread(
                yr + 1, zero_rates, rates_dates, daily_fut_dates, daily_fut_close, 1
            )
            if len(d_n) > 0:
                idx_n = np.where(d_n > rd)[0]
                if len(idx_n) > 0:
                    C = np.concatenate([C, C_n[idx_n[0]:]])
                    dates_out = np.concatenate([dates_out, d_n[idx_n[0]:]])

        return C, dates_out
