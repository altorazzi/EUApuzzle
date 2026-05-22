"""
Liquidity analysis via EUA futures volume data.

Computes front-contract volumes for March, June, September, and
December futures, as well as next and next-next December volumes,
to assess the relative liquidity of different contract maturities.
"""

import numpy as np
from datetime import date

from src.utils import to_datenum


class LiquidityAnalysis:
    """Volume-based liquidity analysis for EUA futures."""

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def front_volumes(self, data_volumes, month, dates_futures):
        """
        Extract front-contract volumes for a quarterly expiry month.

        Iterates through contract columns, collecting daily volumes
        until the contract expires (volume drops to zero), then moves
        to the next contract.

        Args:
            data_volumes  : np.ndarray -- volume matrix (rows=dates, cols=contracts).
            month         : int -- expiry month (3, 6, or 9).
            dates_futures : np.ndarray -- datenums for the volume rows.

        Returns:
            np.ndarray of daily front volumes.
        """
        volumes = []

        start_dates = {3: date(2012, 3, 27), 6: date(
            2012, 6, 26), 9: date(2012, 9, 25)}
        start_date = to_datenum(start_dates.get(month, date(2012, month, 15)))

        for ii in range(data_volumes.shape[1]):
            idx = np.where(dates_futures >= start_date)[0]
            if len(idx) == 0:
                break
            start_idx = idx[0]
            end_found = False

            for jj in range(start_idx, len(dates_futures)):
                if np.sum(data_volumes[jj:, ii]) == 0:
                    start_date = dates_futures[jj]
                    end_found = True
                    break
                volumes.append(data_volumes[jj, ii])

            if not end_found:
                break

        return np.array(volumes, dtype=float)

    def volumes_dec(self):
        """
        Compute December front, next, and next-next contract volumes.

        Returns:
            front_dec    : np.ndarray
            next_dec     : np.ndarray
            nextnext_dec : np.ndarray
        """
        start_year, end_year = 13, 22

        front_list = []
        next_list = []
        nextnext_list = []

        for year in range(start_year, end_year + 1):
            # Front volumes
            try:
                dates_main, _, vol_main = self.preprocessor.load_ice_futures(
                    year)
                dates_prev, _, _ = self.preprocessor.load_ice_futures(year - 1)

                start_dn = dates_prev[-1]
                end_dn = dates_main[-1]
                idx = np.where((dates_main >= start_dn) &
                               (dates_main < end_dn))[0]
                front_list.extend(vol_main[idx].tolist())
            except Exception:
                pass

            # Next and next-next volumes
            try:
                dates1, _, vol1 = self.preprocessor.load_ice_futures(year)
                dates2, _, _ = self.preprocessor.load_ice_futures(year - 1)
                dates3, _, _ = self.preprocessor.load_ice_futures(year - 2)
                dates4, _, _ = self.preprocessor.load_ice_futures(year - 3)

                start_nn = dates4[-1]   # start of next-next window
                start_n = dates3[-1]    # start of next window
                start_f = dates2[-1]    # start of front window

                idx1 = np.where(dates1 >= start_nn)[0]
                idx2 = np.where(dates1 >= start_n)[0]
                idx3 = np.where(dates1 >= start_f)[0]

                if len(idx1) > 0 and len(idx2) > 0:
                    nextnext_list.extend(vol1[idx1[0]:idx2[0]].tolist())
                if len(idx2) > 0 and len(idx3) > 0:
                    next_list.extend(vol1[idx2[0]:idx3[0]].tolist())
            except Exception:
                pass

        return (
            np.array(front_list, dtype=float),
            np.array(next_list, dtype=float),
            np.array(nextnext_list, dtype=float),
        )
