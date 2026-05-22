"""
OIS discount-curve bootstrap.

Extracts the term structure of discount factors and zero rates from
quoted OIS swap rates using a three-regime bootstrap:

    Regime 1 (up to 1 year):   B = 1 / (1 + delta * r)
    Regime 2 (1 to 2 years):   iterative, uses previous discount factors
    Regime 3 (2 to 10 years):  iterative with cumulative sum of previous discounts

Also provides a helper to flat-extrapolate the curve out to 40 years,
which is needed for long-dated bond Z-spread computation.
"""

import numpy as np
from dateutil.relativedelta import relativedelta

from src.utils import from_datenum, to_datenum, yearfrac_30_360, yearfrac_act_365


class BootstrapEngine:
    """Extract discount curve from quoted OIS swap rates."""

    def compute(self, dates_matrix, ois_rates):
        """
        Run the three-regime bootstrap.

        Args:
            dates_matrix : np.ndarray (n, 28) -- start dates + 27 maturity datenums.
            ois_rates    : np.ndarray (n, 27) -- OIS swap rates (decimal).

        Returns:
            discounts  : np.ndarray (n, 28) -- discount factors.
            zero_rates : np.ndarray (n, 28) -- continuously-compounded zero rates.
        """
        n_rows = ois_rates.shape[0]

        # Number of tenors in each regime
        N_1year = 15   # indices 0..14  (1w to 12m)
        N_2year = 19   # indices 15..18 (15m, 18m, 21m, 24m)
        N_end = 27     # indices 19..26 (3y to 10y)

        # ---- Year fractions -------------------------------------------------
        start_dates_py = [from_datenum(dates_matrix[i, 0])
                          for i in range(n_rows)]

        delta = np.zeros_like(ois_rates)       # 30/360
        delta_rates = np.zeros_like(ois_rates)  # ACT/365

        for j in range(ois_rates.shape[1]):
            mat_dates_py = [from_datenum(dates_matrix[i, j + 1])
                            for i in range(n_rows)]
            delta[:, j] = yearfrac_30_360(start_dates_py, mat_dates_py)
            delta_rates[:, j] = yearfrac_act_365(start_dates_py, mat_dates_py)

        # dt intervals for the 2-10 year regime
        dt = np.zeros((n_rows, 10))
        dt[:, 0] = delta[:, N_1year - 1]
        dt[:, 1] = delta[:, N_2year - 1] - delta[:, N_1year - 1]
        for k in range(2, 10):
            col_idx = N_2year + k - 2
            if col_idx < ois_rates.shape[1]:
                dt[:, k] = delta[:, col_idx] - delta[:, col_idx - 1]

        # ---- Initialise output -----------------------------------------------
        discounts = np.zeros((n_rows, 28))
        zero_rates = np.zeros((n_rows, 28))
        discounts[:, 0] = 1.0

        # ---- Regime 1: up to 1 year ------------------------------------------
        for j in range(N_1year):
            discounts[:, j + 1] = 1.0 / (1.0 + delta[:, j] * ois_rates[:, j])
            mask = delta_rates[:, j] > 0
            zero_rates[mask, j + 1] = (
                -np.log(discounts[mask, j + 1]) / delta_rates[mask, j]
            )

        # ---- Regime 2: 1 to 2 years (15m, 18m, 21m, 24m) --------------------
        for j in range(4):
            # Indices derived from MATLAB 1-indexed code, converted to 0-indexed
            ri_num = j + N_1year - 2      # rate index for numerator
            di = 5 + 3 * j                # delta index for intermediate maturity
            disc_i = 6 + 3 * j            # discount index for intermediate maturity
            ri_den = j + N_1year - 1      # rate index for denominator

            num = 1.0 - ois_rates[:, ri_num] * \
                delta[:, di] * discounts[:, disc_i]
            den = 1.0 + (delta[:, ri_den] - delta[:, di]) * \
                ois_rates[:, ri_den]
            discounts[:, j + 1 + N_1year] = num / den

            mask = delta_rates[:, ri_den] > 0
            zero_rates[mask, j + 1 + N_1year] = (
                -np.log(discounts[mask, j + 1 + N_1year]) /
                delta_rates[mask, ri_den]
            )

        # ---- Regime 3: 2 to 10 years -----------------------------------------
        for j_m in range(N_2year + 1, N_end + 1):
            j = j_m - 1  # 0-indexed rate column
            k = j_m - N_2year

            # Accumulate sum of dt * discount for intermediate terms
            sum_term = np.zeros(n_rows)
            for col in range(1, k):
                disc_col = N_2year + col
                sum_term += dt[:, col] * discounts[:, disc_col]
            sum_term += dt[:, 0] * discounts[:, N_1year]

            num = 1.0 - ois_rates[:, j] * sum_term
            den = 1.0 + (delta[:, j] - delta[:, j - 1]) * ois_rates[:, j]
            discounts[:, j + 1] = num / den

            mask = delta_rates[:, j] > 0
            zero_rates[mask, j + 1] = (
                -np.log(discounts[mask, j + 1]) / delta_rates[mask, j]
            )

        return discounts, zero_rates

    @staticmethod
    def extend_to_40y(dates_matrix, zero_rates):
        """
        Flat-extrapolate the zero curve from 10 years out to 40 years.

        Returns:
            dates_40y      : np.ndarray (n, 58) -- original 28 + 30 extra yearly datenums.
            zero_rates_40y : np.ndarray (n, 58) -- zero rates with flat extension.
        """
        n = dates_matrix.shape[0]

        extra_dates = np.zeros((n, 30))
        for i in range(n):
            end_date = from_datenum(dates_matrix[i, -1])
            for y in range(1, 31):
                extra_dates[i, y -
                            1] = to_datenum(end_date + relativedelta(years=y))

        dates_40y = np.hstack([dates_matrix, extra_dates])

        extra_rates = np.tile(zero_rates[:, -1:], (1, 30))
        zero_rates_40y = np.hstack([zero_rates, extra_rates])

        return dates_40y, zero_rates_40y
