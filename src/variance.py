"""
Conditional variance estimation via GARCH(1,1) and EWMA.

Provides two models for estimating the conditional variance of
EUA futures log-returns:

    - GARCH(1,1): estimated via the ``arch`` library.
    - EWMA(lambda): manual recursive formula.
"""

import numpy as np
from arch import arch_model


class VarianceEstimation:
    """GARCH and EWMA conditional variance models."""

    @staticmethod
    def garch_variance(log_returns):
        """
        Fit a GARCH(1,1) model and return the conditional variance series.

        The returns are scaled by 100 for numerical stability during
        estimation, then the variance is scaled back.
        """
        model = arch_model(log_returns * 100, vol='Garch',
                           p=1, q=1, mean='Zero')
        result = model.fit(disp='off')
        variance = result.conditional_volatility ** 2 / 10000.0
        return variance

    @staticmethod
    def ewma_variance(log_returns, lam=0.95):
        """
        Compute EWMA conditional variance.

        Recursion:  var(t+1) = lambda * var(t) + (1 - lambda) * r(t)^2
        Initialised with the variance of the squared return series.

        Args:
            log_returns : np.ndarray of log-returns.
            lam         : decay factor (default 0.95).

        Returns:
            np.ndarray of conditional variances (same length as input).
        """
        N = len(log_returns)
        var_ewma = np.zeros(N + 1)
        var_ewma[0] = np.var(log_returns ** 2)

        for i in range(N):
            var_ewma[i + 1] = lam * var_ewma[i] + \
                (1 - lam) * log_returns[i] ** 2

        return var_ewma[1:]

    @staticmethod
    def estimate(daily_fut_dates, daily_fut_close, dates_ts):
        """
        Full variance estimation pipeline.

        Computes both GARCH(1,1) and EWMA(0.95) conditional variance
        for the daily futures log-returns restricted to the analysis dates.

        Returns:
            variance_garch : np.ndarray
            variance_ewma  : np.ndarray
            log_returns    : np.ndarray
        """
        idx_daily = np.where(np.isin(daily_fut_dates, dates_ts))[0]

        log_ret = np.zeros(len(daily_fut_close))
        log_ret[1:] = np.log(daily_fut_close[1:] / daily_fut_close[:-1])
        log_ret = log_ret[idx_daily]

        variance_garch = VarianceEstimation.garch_variance(log_ret)
        variance_ewma = VarianceEstimation.ewma_variance(log_ret, lam=0.95)

        return variance_garch, variance_ewma, log_ret
