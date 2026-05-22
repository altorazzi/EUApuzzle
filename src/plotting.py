"""
Plotting functions for all project figures.

Each static method produces one figure (saved to disk and shown
interactively).  All date arrays are expected as MATLAB datenums
and are converted to Python dates for matplotlib.
"""

from src.utils import from_datenum
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


class Plotter:
    """Generate all analysis figures."""

    def __init__(self, output_dir=None, show_plots=False):
        """
        Args:
            output_dir : directory where PNGs are saved.
                         Defaults to the working directory.
            show_plots : True for interactive pop-ups, False for silent saving.
        """
        self.output_dir = output_dir or os.getcwd()
        self.show_plots = show_plots

        if not self.show_plots:
            matplotlib.use('Agg')  # Silent background plotting
        else:
            matplotlib.use('TkAgg')  # Interactive pop-ups

        import matplotlib.pyplot as plt
        self.plt = plt

    def _savefig(self, name):
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, name), dpi=150)
        plt.show()
        if self.show_plots:
            plt.show()

        plt.close('all')

    # ------------------------------------------------------------------

    def plot_discount_curve(self, dates_matrix, discounts, zero_rates,
                            row_idx, title_date="13-Sep-2012"):
        """Plot zero-rate curve and discount curve for a single date."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        x = [from_datenum(d) for d in dates_matrix[row_idx, 2:]]

        ax1.plot(x, zero_rates[row_idx, 2:] * 100, '-o', linewidth=1.5)
        ax1.set_title(f'Zero Rates curve {title_date}')
        ax1.set_xlabel('Dates')
        ax1.set_ylabel('Zero Rates [%]')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        ax2.plot(x, discounts[row_idx, 2:], '-o', linewidth=1.5)
        ax2.set_title(f'Discounting curve {title_date}')
        ax2.set_xlabel('Dates')
        ax2.set_ylabel('Discounts')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        self._savefig('plot_discount_curve.png')

    # ------------------------------------------------------------------

    def plot_liquidity_boxplots(self, front_march, front_june, front_sept,
                                front_dec, next_dec, nextnext_dec):
        """Plot liquidity boxplots for front contracts and December contracts."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Remove NaNs so matplotlib renders all boxplots correctly
        f_mar = front_march[~np.isnan(front_march)]
        f_jun = front_june[~np.isnan(front_june)]
        f_sep = front_sept[~np.isnan(front_sept)]
        f_dec = front_dec[~np.isnan(front_dec)]
        n_dec = next_dec[~np.isnan(next_dec)]
        nn_dec = nextnext_dec[~np.isnan(nextnext_dec)]

        data1 = [np.log10(f_mar + 1), np.log10(f_jun + 1),
                 np.log10(f_sep + 1), np.log10(f_dec + 1)]
        ax1.boxplot(data1, labels=['March', 'June', 'September', 'December'])
        ax1.set_title('Front contracts liquidity')
        ax1.set_ylabel('log10(Volumes)')
        ax1.grid(True)

        data2 = [np.log10(f_dec + 1), np.log10(n_dec + 1),
                 np.log10(nn_dec + 1)]
        ax2.boxplot(data2, labels=['Front', 'Next', 'Next-Next'])
        ax2.set_title('December contracts liquidity')
        ax2.set_ylabel('log10(Volumes)')
        ax2.grid(True)

        self._savefig('plot_liquidity.png')

    # ------------------------------------------------------------------

    def plot_cspread(self, dates_front, C_front, dates_next, C_next,
                     dates_roll, C_roll):
        """Plot C-spread: front, next, and roll-over series."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        for ax, dates, vals, title in [
            (axes[0], dates_front, C_front, 'C-spread Front'),
            (axes[1], dates_next, C_next, 'C-spread Next'),
            (axes[2], dates_roll, C_roll, 'C-spread Roll-Over'),
        ]:
            py_dates = [from_datenum(d) for d in dates]
            ax.plot(py_dates, vals * 100, color='green', linewidth=1)
            ax.set_title(title)
            ax.set_ylabel('[%]')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        axes[0].set_ylim(-5, 5)
        self._savefig('plot_cspread.png')

    # ------------------------------------------------------------------

    def plot_time_series(self, dates_ts, cspread, z_index, zero_rates_3m):
        """Plot aligned time series of C-spread, Z-index, and 3m rate."""
        fig, ax = plt.subplots(figsize=(12, 6))
        py_dates = [from_datenum(d) for d in dates_ts]

        ax.plot(py_dates, zero_rates_3m * 100, 'b',
                linewidth=1, label='3m risk-free rate')
        ax.plot(py_dates, cspread * 100, 'g', linewidth=1, label='C-spread')
        ax.plot(py_dates, z_index * 100, 'r', linewidth=1, label='Z-index')
        ax.set_title('Time-series plot')
        ax.set_xlabel('Dates')
        ax.set_ylim(-1, 3.5)
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(True, alpha=0.3)

        self._savefig('plot_time_series.png')

    # ------------------------------------------------------------------

    def plot_acf_pacf(self, cspread, z_index, zero_rates_3m):
        """Plot ACF and PACF for each time series."""
        series_list = [
            ('C-spread', cspread),
            ('Z-index', z_index),
            ('3m zero rates', zero_rates_3m),
        ]

        for name, s in series_list:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

            acf_vals = acf(s, nlags=30)
            pacf_vals = pacf(s, nlags=30)
            ci = 1.96 / np.sqrt(len(s))

            ax1.bar(range(len(acf_vals)), acf_vals, width=0.3)
            ax1.set_title(f'ACF - {name}')
            ax1.axhline(y=ci, color='r', linestyle='--')
            ax1.axhline(y=-ci, color='r', linestyle='--')

            ax2.bar(range(len(pacf_vals)), pacf_vals, width=0.3)
            ax2.set_title(f'PACF - {name}')
            ax2.axhline(y=ci, color='r', linestyle='--')
            ax2.axhline(y=-ci, color='r', linestyle='--')

            self._savefig(f'plot_acf_pacf_{name.replace(" ", "_")}.png')

    # ------------------------------------------------------------------

    def plot_cointegration(self, dates_ts, cspread, z_index,
                           zero_rates_3m, cointeg_vector):
        """Plot the estimated cointegration relationship."""
        fig, ax = plt.subplots(figsize=(12, 5))
        py_dates = [from_datenum(d) for d in dates_ts]

        Y = np.column_stack([cspread, z_index, zero_rates_3m])
        psi = Y @ cointeg_vector * 100

        ax.plot(py_dates, psi, linewidth=1, color='#EDB120')
        ax.axhline(y=np.mean(psi), color='r', linewidth=1)
        ax.set_title('Cointegration relationship estimated via Johansen')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(True, alpha=0.3)

        self._savefig('plot_cointegration.png')

    # ------------------------------------------------------------------

    def plot_variance(self, dates_ts, variance_garch, variance_ewma, log_returns):
        """Plot GARCH vs EWMA conditional variance."""
        py_dates = [from_datenum(d) for d in dates_ts]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(py_dates, variance_garch, 'k', label='GARCH variance')
        ax1.plot(py_dates, variance_ewma, 'r', label='EWMA variance')
        ax1.set_title('GARCH(1,1) vs EWMA(0.95)')
        ax1.legend()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        ax2.plot(py_dates, log_returns, color='#4DBEEE',
                 linewidth=1, label='log-returns')
        ax2.plot(py_dates, np.sqrt(variance_garch),
                 color='#A2142F', linewidth=1, label='sigma(t)')
        ax2.plot(py_dates, -np.sqrt(variance_garch),
                 color='#A2142F', linewidth=1, label='-sigma(t)')
        ax2.set_ylim(-0.25, 0.25)
        ax2.legend()
        ax2.set_title('Log-returns vs standard deviation (GARCH)')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        self._savefig('plot_variance.png')

    # ------------------------------------------------------------------

    def plot_confidence_interval(self, x, y, coeff, dates, quantiles, quant_idx):
        """Plot quantile-regression confidence interval."""
        fig, ax = plt.subplots(figsize=(12, 5))
        py_dates = [from_datenum(d) for d in dates]

        lower = x @ coeff[:, quant_idx]
        upper = x @ coeff[:, -(quant_idx + 1)]

        ax.plot(py_dates, y, 'b', linewidth=0.8, label='Delta C-spread')
        ax.plot(py_dates, lower, 'r-', linewidth=1,
                label=f'{quantiles[quant_idx]:.0%} quantile')
        ax.plot(py_dates, upper, 'r-', linewidth=1,
                label=f'{quantiles[-(quant_idx + 1)]:.0%} quantile')
        ax.fill_between(py_dates, lower, upper, alpha=0.15, color='red')

        ci_pct = 100 * (1 - 2 * quantiles[quant_idx])
        ax.set_title(f'Confidence interval ({ci_pct:.0f}%)')
        ax.set_xlabel('Dates')
        ax.set_ylabel('Delta C-spread')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        self._savefig('plot_confidence_interval.png')
