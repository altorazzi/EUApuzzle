"""
Data loading and preprocessing for all input files.

Handles OIS rates, daily futures, bond data, extra macro variables,
open interest, volume data, and ICE futures CSV files.

Approach notes (differences from MATLAB original):
    - OIS data: loaded from OIS_Data.csv (not OIS.mat) because MATLAB
      datetime objects in the MAT file are not readable by scipy.
    - Bond data: loaded from Bonds/*.csv (not Bonds/New/*.mat) because
      MATLAB table objects are not readable by scipy.
    - ListValidBonds: loaded from List_Valid_Bonds.csv (not .mat).
    - OpenInterest: loaded from OpenInterest.xlsx (not .mat).
    - Futures dates: derived from Volumes_extra_futures.xlsx (not
      dates_futures.mat).
    - Volume MAT files (march/june/sept): readable by scipy, used directly.
    - ICE_FUT_25.csv uses semicolon delimiter (handled automatically).
"""

import os
import numpy as np
import pandas as pd
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from scipy import io as sio

from src.utils import to_datenum, to_datenum_array
from src.eur_calendar import EurCalendar


class DataPreprocessor:
    """Load and preprocess all project data files."""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.calendar = EurCalendar()

    # ------------------------------------------------------------------
    # OIS rates
    # ------------------------------------------------------------------

    def load_ois(self):
        """
        Import and preprocess OIS swap rates.

        Returns:
            ois_rates : np.ndarray (n_dates, 27) -- OIS rates (decimal).
            dates_matrix : np.ndarray (n_dates, 28) -- column 0 is the
                start-date datenum; columns 1-27 are maturity datenums
                for the 27 standard OIS tenors.
        """
        filepath = os.path.join(self.base_dir, 'OIS_Data.csv')
        df = pd.read_csv(filepath)

        # Parse dates (DD/MM/YY; pandas auto-resolves 2-digit years)
        raw_dates = pd.to_datetime(df['Date'], format='%d/%m/%y')

        # Rates are in columns 1-27; skip the first row (all NaN)
        rates = df.iloc[:, 1:].values.astype(float)
        first_valid = 1
        raw_dates = raw_dates[first_valid:]
        rates = rates[first_valid:]

        # Forward-fill NaN values row-by-row
        for i in range(rates.shape[0]):
            for j in range(rates.shape[1]):
                if np.isnan(rates[i, j]) and i > 0:
                    rates[i, j] = rates[i - 1, j]

        # Remove duplicate dates
        dates_series = pd.Series(raw_dates.values)
        unique_mask = ~dates_series.duplicated(keep='first')
        raw_dates = raw_dates[unique_mask.values]
        rates = rates[unique_mask.values]

        # Convert from percentage to decimal
        rates = rates / 100.0

        # Build a daily grid and forward-fill rates for missing calendar days
        start_date = raw_dates.iloc[0]
        end_date = raw_dates.iloc[-1]
        all_dates = pd.date_range(start_date, end_date, freq='D')

        rates_complete = np.zeros((len(all_dates), rates.shape[1]))
        date_to_idx = {d: i for i, d in enumerate(raw_dates)}

        for i, d in enumerate(all_dates):
            if d in date_to_idx:
                rates_complete[i] = rates[date_to_idx[d]]
            else:
                rates_complete[i] = rates_complete[i - 1]

        # Build the maturity-date matrix (28 columns)
        # Column 0 = start date; columns 1-27 = maturities for the
        # standard OIS tenors: 1w, 2w, 3w, 1m..12m, 15m, 18m, 21m, 24m, 3y..10y
        offsets = (
            [timedelta(weeks=w) for w in [1, 2, 3]]
            + [relativedelta(months=m) for m in range(1, 13)]
            + [relativedelta(months=m) for m in [15, 18, 21, 24]]
            + [relativedelta(years=y) for y in range(3, 11)]
        )

        n = len(all_dates)
        dates_matrix = np.zeros((n, 28), dtype=np.float64)

        for i in range(n):
            sd = all_dates[i].date()
            dates_matrix[i, 0] = to_datenum(sd)
            for j, offset in enumerate(offsets):
                target_date = sd + offset
                adjusted = self.calendar.busdate(target_date, 'follow')
                dates_matrix[i, j + 1] = to_datenum(adjusted)

        return rates_complete, dates_matrix

    # ------------------------------------------------------------------
    # Daily futures (proxy for spot price)
    # ------------------------------------------------------------------

    def load_daily_futures(self):
        """
        Load the daily continuous-futures series used as spot proxy.

        Returns:
            daily_fut_dates : np.ndarray of datenums.
            daily_fut_close : np.ndarray of close prices.
        """
        filepath = os.path.join(self.base_dir, 'Daily_Future.csv')
        df = pd.read_csv(filepath)
        dates = pd.to_datetime(df['Date'])
        return to_datenum_array(dates), df['CLOSE'].values.astype(float)

    # ------------------------------------------------------------------
    # Bond reference data
    # ------------------------------------------------------------------

    def load_bonds(self):
        """
        Load the list of valid bonds and their characteristics.

        Returns:
            valid_bonds : dict with keys ISIN, coupon_rate, maturity,
                notional, coupon_freq, parent.
            names : sorted list of parent tickers (excluding TKAG).
        """
        filepath = os.path.join(self.base_dir, 'List_Valid_Bonds.csv')
        df = pd.read_csv(filepath)

        valid_bonds = {
            'ISIN': df['Instrument'].values,
            'coupon_rate': df['Coupon Rate'].values.astype(float),
            'maturity': pd.to_datetime(df['Maturity Date']).values,
            'notional': df['Original Amount Issued'].values.astype(float),
            'coupon_freq': df['Coupon Frequency'].values.astype(int),
            'parent': df['Parent Ticker'].values,
        }

        names = sorted(set(valid_bonds['parent']))
        if 'TKAG' in names:
            names.remove('TKAG')

        return valid_bonds, names

    # ------------------------------------------------------------------
    # Extra macro variables (SPX, VIX, WTI)
    # ------------------------------------------------------------------

    def load_extra_variables(self):
        """
        Load and compute returns for SPX, VIX, and WTI.

        Returns:
            dates_extra : np.ndarray of datenums.
            logRet_SPX  : np.ndarray -- log-returns of S&P 500.
            Ret_VIX     : np.ndarray -- simple returns of VIX.
            logRet_WTI  : np.ndarray -- log-returns of WTI crude.
        """
        filepath = os.path.join(self.base_dir, 'Extra_variables.csv')
        df = pd.read_csv(filepath)

        dates = pd.to_datetime(df['Date'], format='%A, %B %d, %Y')
        dates_extra = to_datenum_array(dates)

        SPX = df['SPX'].values.astype(float)
        VIX = df['VIX'].values.astype(float)
        WTI = df['WTI'].values.astype(float)

        # Forward-fill missing / negative values
        for k in range(1, len(WTI)):
            if np.isnan(SPX[k]) or SPX[k] < 0:
                SPX[k] = SPX[k - 1]
            if np.isnan(VIX[k]) or VIX[k] < 0:
                VIX[k] = VIX[k - 1]
            if np.isnan(WTI[k]) or WTI[k] < 0:
                WTI[k] = WTI[k - 1]

        logRet_SPX = np.zeros(len(SPX))
        Ret_VIX = np.zeros(len(VIX))
        logRet_WTI = np.zeros(len(WTI))

        logRet_SPX[1:] = np.log(SPX[1:] / SPX[:-1])
        Ret_VIX[1:] = VIX[1:] / VIX[:-1]
        logRet_WTI[1:] = np.log(WTI[1:] / WTI[:-1])

        return dates_extra, logRet_SPX, Ret_VIX, logRet_WTI

    # ------------------------------------------------------------------
    # Open interest
    # ------------------------------------------------------------------

    def load_open_interest(self):
        """
        Load open-interest data for December EUA futures.

        Returns:
            oi_dates  : np.ndarray of datenums.
            oi_values : np.ndarray, one column per Dec contract year.
        """
        filepath = os.path.join(self.base_dir, 'OpenInterest.xlsx')
        df = pd.read_excel(filepath, engine='openpyxl')
        oi_dates = to_datenum_array(pd.to_datetime(df['Name']))
        oi_values = df.iloc[:, 1:].values.astype(float)
        return oi_dates, oi_values

    # ------------------------------------------------------------------
    # Volume MAT files (March / June / September)
    # ------------------------------------------------------------------

    def load_volumes(self):
        """
        Load volume data for March, June, and September futures.
        These MAT files contain plain numeric arrays and are readable.

        Returns:
            vol_march, vol_june, vol_sept : np.ndarrays.
        """
        vol_march = sio.loadmat(
            os.path.join(self.base_dir, 'volumes_data_march.mat')
        )['volumes_data_march']
        vol_june = sio.loadmat(
            os.path.join(self.base_dir, 'volumes_data_june.mat')
        )['volumes_data_june']
        vol_sept = sio.loadmat(
            os.path.join(self.base_dir, 'volumes_data_sept.mat')
        )['volumes_data_sept']
        return vol_march, vol_june, vol_sept

    # ------------------------------------------------------------------
    # Futures dates (from Volumes_extra_futures.xlsx)
    # ------------------------------------------------------------------

    def load_futures_dates(self):
        """
        Derive futures dates from the Excel volume file.

        Returns:
            dates_futures : np.ndarray of datenums.
        """
        filepath = os.path.join(self.base_dir, 'Volumes_extra_futures.xlsx')
        df = pd.read_excel(filepath, engine='openpyxl')
        dates = pd.to_datetime(df['Name'])
        return to_datenum_array(dates)

    # ------------------------------------------------------------------
    # ICE December EUA futures
    # ------------------------------------------------------------------

    def load_ice_futures(self, year):
        """
        Load ICE EUA futures CSV for a given 2-digit contract year.

        Automatically detects semicolon-delimited files.

        Returns:
            dates        : np.ndarray of datenums.
            close_prices : np.ndarray of CLOSE prices.
            volumes      : np.ndarray of VOLUME.
        """
        filename = f'ICE_FUT_{year}.csv'
        filepath = os.path.join(self.base_dir, 'Futures', filename)

        # Try comma first; fall back to semicolon
        try:
            df = pd.read_csv(filepath)
            if len(df.columns) == 1 and ';' in df.columns[0]:
                df = pd.read_csv(filepath, sep=';')
        except Exception:
            df = pd.read_csv(filepath, sep=';')

        dates = pd.to_datetime(df['Date'])
        return (
            to_datenum_array(dates),
            df['CLOSE'].values.astype(float),
            df['VOLUME'].values.astype(float),
        )

    # ------------------------------------------------------------------
    # Bond price time series
    # ------------------------------------------------------------------

    def load_bond_prices(self, ticker):
        """
        Load bond prices for a given issuer from the CSV file.

        Returns:
            isin_codes : list of ISIN strings (column headers).
            dates      : np.ndarray of datenums.
            prices     : np.ndarray (n_dates x n_bonds), NaN where
                         data is missing or unparseable.
        """
        filepath = os.path.join(self.base_dir, 'Bonds', f'{ticker}.csv')
        if not os.path.exists(filepath):
            print(
                f"    WARNING: {filepath} not found. Skipping issuer {ticker}.")
            return [], np.array([]), np.array([])

        df = pd.read_csv(filepath)

        # Column layout: 0='Bonds' (ISIN list), 1=unused, 2='Date', 3+=prices
        isin_codes = list(df.columns[3:])

        dates_raw = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        dates = to_datenum_array(dates_raw)

        # Coerce non-numeric entries (e.g. 'NO DATA AVAILABLE') to NaN
        prices = df.iloc[:, 3:].apply(
            pd.to_numeric, errors='coerce').values.astype(float)

        return isin_codes, dates, prices
