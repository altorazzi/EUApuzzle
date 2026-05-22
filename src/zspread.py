"""
Z-spread computation for corporate bonds.

The Z-spread is the constant spread that, when added to the risk-free
zero curve, makes the present value of a bond's cash flows equal to
its market price.  The spread is found via root-finding (Brent's method).

For each issuer the individual Z-spreads are aggregated into a
notional-weighted average.  The Z-index is the cross-issuer mean.

Approach notes:
    - Bond price data is loaded from Bonds/{ticker}.csv instead of
      Bonds/New/{ticker}.mat (MATLAB table objects are not readable).
    - Root-finding uses scipy.optimize.brentq (replaces MATLAB fzero).
"""

import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from scipy.optimize import brentq

from src.utils import to_datenum, from_datenum, yearfrac_30_360, yearfrac_act_365


class ZspreadAnalysis:
    """Z-spread computation for corporate bonds."""

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def compute_zspread(self, parent_ticker, valid_bonds, dates_rates, zero_rates):
        """
        Compute the notional-weighted Z-spread for a single issuer.

        Args:
            parent_ticker : str -- Bloomberg parent ticker.
            valid_bonds   : dict with bond characteristics.
            dates_rates   : np.ndarray (n, 58) -- datenums (40-year curve).
            zero_rates    : np.ndarray (n, 58) -- zero rates (40-year curve).

        Returns:
            Z        : np.ndarray -- weighted Z-spread per date.
            dates_ts : np.ndarray -- datenums.
        """
        # Select bonds for this issuer with notional >= 500M
        mask = (
            (valid_bonds['parent'] == parent_ticker)
            & (valid_bonds['notional'] >= 5e8)
            & (valid_bonds['ISIN'] != 'XS0877820422')
        )
        indexes = np.where(mask)[0]
        if len(indexes) == 0:
            return np.array([]), np.array([])

        # Load bond prices
        isin_codes, bond_dates, all_prices = self.preprocessor.load_bond_prices(
            parent_ticker)
        if len(isin_codes) == 0:
            return np.array([]), np.array([])

        # Map valid ISINs to CSV columns
        valid_isins = valid_bonds['ISIN'][indexes]
        isin_col_map = {vi: isin_codes.index(
            vi) for vi in valid_isins if vi in isin_codes}
        if not isin_col_map:
            return np.array([]), np.array([])

        # Restrict to the analysis window
        start_dn = to_datenum(date(2013, 1, 1))
        end_dn = to_datenum(date(2022, 10, 31))
        phase_mask = (bond_dates >= start_dn) & (bond_dates <= end_dn)
        bond_dates = bond_dates[phase_mask]
        all_prices = all_prices[phase_mask]

        # Align with zero-rate dates
        rates_start = dates_rates[:, 0]
        common = np.intersect1d(rates_start, bond_dates)
        idx_rates = np.where(np.isin(rates_start, common))[0]
        idx_bonds = np.where(np.isin(bond_dates, common))[0]
        if len(idx_rates) == 0:
            return np.array([]), np.array([])

        bond_dates = bond_dates[idx_bonds]
        all_prices = all_prices[idx_bonds]
        dates_rates_local = dates_rates[idx_rates]
        zero_rates_local = zero_rates[idx_rates]

        n_dates = len(bond_dates)
        n_bonds = len(indexes)
        z_spread = np.zeros((n_dates, n_bonds))
        bond_dates_py = [from_datenum(d) for d in bond_dates]

        # Compute Z-spread for each bond
        for jj in range(n_bonds):
            bond_idx = indexes[jj]
            isin = valid_bonds['ISIN'][bond_idx]
            if isin not in isin_col_map:
                continue

            col = isin_col_map[isin]
            prices_j = all_prices[:, col]
            coupon_rate = valid_bonds['coupon_rate'][bond_idx] / 100.0
            mat_date = pd.Timestamp(valid_bonds['maturity'][bond_idx]).date()
            freq = valid_bonds['coupon_freq'][bond_idx]

            # Build coupon schedule
            if freq == 1:
                max_years = int(
                    np.floor((mat_date - bond_dates_py[0]).days / 365.25))
                cpn_dates_base = [
                    mat_date - relativedelta(years=y) for y in range(max_years, -1, -1)
                ]
            else:
                max_years = int(
                    np.floor((mat_date - bond_dates_py[0]).days / 365.25))
                flag_extra = 1 if (
                    mat_date - bond_dates_py[0]).days / 365.25 - max_years > 0.4 else 0
                n_periods = max_years * 2 + flag_extra + 1
                cpn_dates_base = [
                    mat_date - relativedelta(months=6 * p) for p in range(n_periods - 1, -1, -1)
                ]

            cpn_dates_dn = np.array([to_datenum(d)
                                    for d in cpn_dates_base], dtype=np.float64)

            for k in range(n_dates):
                t_dn = bond_dates[k]
                t_py = bond_dates_py[k]

                # Future coupons only
                active = cpn_dates_dn[cpn_dates_dn > t_dn]
                if len(active) == 0:
                    continue

                active_py = [from_datenum(d) for d in active]
                delta_30 = yearfrac_30_360([t_py] * len(active), active_py)
                delta_365 = yearfrac_act_365([t_py] * len(active), active_py)

                pos_mask = delta_30 >= 0
                delta_30 = delta_30 * pos_mask
                delta_365 = delta_365 * pos_mask

                # Coupon intervals
                dt_arr = np.zeros(len(active))
                dt_arr[0] = delta_30[0]
                dt_arr[1:] = delta_30[1:] - delta_30[:-1]

                # Interpolated risk-free rates and discount factors
                rates_interp = np.interp(
                    active, dates_rates_local[k], zero_rates_local[k])
                rates_interp = np.nan_to_num(rates_interp, 0.0)
                discounts_k = np.exp(-delta_365 * rates_interp)
                discounts_k[~pos_mask] = 0.0

                coupon_vals = np.full(len(active), coupon_rate) * pos_mask

                # Net present value of cash flows
                NPV = coupon_vals * dt_arr * discounts_k
                NPV[-1] += 1.0 * discounts_k[-1]  # add principal at maturity

                bond_price = prices_j[k]
                if (np.sum(NPV) == 0 or bond_price == 0
                        or np.isnan(bond_price) or np.max(delta_30) < 0.075):
                    z_spread[k, jj] = 0
                else:
                    try:
                        def f(z): return np.dot(
                            NPV, np.exp(-z * delta_30)) - bond_price / 100.0
                        z_spread[k, jj] = brentq(f, -2.0, 2.0, xtol=1e-12)
                    except (ValueError, RuntimeError):
                        try:
                            z_spread[k, jj] = brentq(
                                f, -10.0, 10.0, xtol=1e-12)
                        except (ValueError, RuntimeError):
                            z_spread[k, jj] = 0

        # Notional-weighted average
        notionals = valid_bonds['notional'][indexes]
        index_nonzero = (z_spread != 0)
        matrix_notionals = np.tile(notionals, (n_dates, 1)) * index_nonzero
        denom = np.sum(matrix_notionals, axis=1)
        denom[denom == 0] = 1.0

        Z = np.dot(z_spread, notionals) / denom
        return Z, bond_dates

    def compute_zindex(self, names, valid_bonds, dates_40y, zero_rates_40y):
        """
        Compute the Z-index as the cross-issuer mean of Z-spreads.

        Returns:
            z_index : np.ndarray.
            dates_Z : np.ndarray of datenums.
        """
        z_all = None
        dates_Z = None

        for jj, name in enumerate(names):
            print(f"  Computing Z-spread for {name}...")
            Z_j, dates_j = self.compute_zspread(
                name, valid_bonds, dates_40y, zero_rates_40y)

            if len(Z_j) == 0:
                continue

            if z_all is None:
                z_all = np.zeros((len(Z_j), len(names)))
                dates_Z = dates_j

            if len(Z_j) == z_all.shape[0]:
                z_all[:, jj] = Z_j
            else:
                common = np.intersect1d(dates_Z, dates_j)
                idx_z = np.where(np.isin(dates_Z, common))[0]
                idx_j = np.where(np.isin(dates_j, common))[0]
                if z_all.shape[0] != len(common):
                    z_new = np.zeros((len(common), len(names)))
                    z_new[:, :jj] = z_all[idx_z, :jj]
                    z_all = z_new
                    dates_Z = common
                    idx_z = np.arange(len(common))
                z_all[idx_z, jj] = Z_j[idx_j]

        z_all = np.nan_to_num(z_all, 0.0)

        nonzero_mask = (z_all != 0)
        n_issuers = np.sum(nonzero_mask, axis=1)
        n_issuers[n_issuers == 0] = 1
        z_index = np.sum(z_all, axis=1) / n_issuers

        return z_index, dates_Z
