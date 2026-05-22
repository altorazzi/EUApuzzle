"""
Quantile regression with bootstrap standard errors.

Estimates conditional quantiles of the dependent variable by
minimising the asymmetric check (pinball) loss function via
Nelder-Mead optimisation, matching MATLAB's fminsearch approach.

Bootstrap standard errors are computed by resampling residuals.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as t_dist


class QuantileRegressionAnalysis:
    """Quantile regression with bootstrap inference."""

    @staticmethod
    def quantile_regression(x, y, q, n_boot=30):
        """
        Estimate a single conditional quantile.

        Args:
            x      : np.ndarray (n, p) -- design matrix (include intercept).
            y      : np.ndarray (n,) -- dependent variable.
            q      : float -- quantile level in (0, 1).
            n_boot : int -- number of bootstrap replications.

        Returns:
            coeff    : np.ndarray (p,) -- estimated coefficients.
            p_values : np.ndarray (p,) -- two-sided p-values.
            std_err  : np.ndarray (p,) -- bootstrap standard errors.
        """
        # Check (pinball) loss function
        def rho(r):
            return np.sum(np.abs(r * (q - (r <= 0).astype(float))))

        # Initialise with OLS solution
        p_mean = np.linalg.lstsq(x, y, rcond=None)[0]

        def objective(p):
            return rho(y - x @ p)

        result = minimize(
            objective, p_mean, method='Nelder-Mead',
            options={'maxfev': 50000, 'disp': False, 'adaptive': True},
        )
        coeff = result.x

        # Residual bootstrap for standard errors
        y_pred = x @ coeff
        residual = y - y_pred

        boot_coeffs = np.zeros((n_boot, len(coeff)))
        for b in range(n_boot):
            boot_resid = residual[np.random.randint(
                0, len(residual), len(residual))]
            y_boot = y_pred + boot_resid

            def obj_boot(p):
                return rho(y_boot - x @ p)

            res_b = minimize(
                obj_boot, coeff, method='Nelder-Mead',
                options={'maxfev': 50000, 'disp': False},
            )
            boot_coeffs[b] = res_b.x

        std_err = np.std(boot_coeffs, axis=0)

        # t-statistics and p-values
        with np.errstate(divide='ignore', invalid='ignore'):
            t_stat = coeff / std_err
        dof = x.shape[0] - len(coeff)
        p_values = 2 * (1 - t_dist.cdf(np.abs(t_stat), dof))

        return coeff, p_values, std_err

    @staticmethod
    def multi_quantile_regression(x, y, quantiles):
        """
        Run quantile regression for a vector of quantile levels.

        Returns:
            C  : np.ndarray (p, n_q) -- coefficient matrix.
            P  : np.ndarray (p, n_q) -- p-value matrix.
            SE : np.ndarray (p, n_q) -- standard-error matrix.
        """
        n_q = len(quantiles)
        n_p = x.shape[1]

        C = np.zeros((n_p, n_q))
        P = np.zeros((n_p, n_q))
        SE = np.zeros((n_p, n_q))

        for i, q in enumerate(quantiles):
            print(f"    Quantile {q:.1f}...")
            coeff, pvals, se = QuantileRegressionAnalysis.quantile_regression(
                x, y, q)
            C[:, i] = coeff
            P[:, i] = pvals
            SE[:, i] = se

        return C, P, SE
