"""
Johansen cointegration test and Error Correction Model (ECM).

Tests for cointegration among C-spread, Z-index, and 3-month zero
rate, then estimates an ECM with lagged differences and the
cointegration residual as error-correction term.

Approach notes:
    - statsmodels coint_johansen with det_order=-1 matches MATLAB's
      jcitest 'H2' model (no deterministic trend).
    - The ECM is estimated via OLS (statsmodels).
"""

import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class CointegrationAnalysis:
    """Johansen cointegration test and Error Correction Model."""

    @staticmethod
    def johansen_test(Y, alpha=0.1):
        """
        Run the Johansen cointegration test.

        Args:
            Y     : np.ndarray (n, 3) -- [C_spread, Z_index, zero_rates_3m].
            alpha : significance level (used for display only).

        Returns:
            cointeg_vector : np.ndarray -- normalised cointegration vector.
            result         : coint_johansen result object.
        """
        result = coint_johansen(Y, det_order=-1, k_ar_diff=1)

        print("\n--- Johansen Cointegration Test ---")
        print("Trace Statistics:", result.lr1)
        print("Critical Values (90%, 95%, 99%):\n", result.cvt)
        print("Max Eigenvalue Statistics:", result.lr2)
        print("Critical Values (90%, 95%, 99%):\n", result.cvm)

        # Normalise so the first element equals 1
        beta = result.evec[:, 0]
        cointeg_vector = beta / beta[0]
        print(f"Cointegration vector (normalised): {cointeg_vector}")

        return cointeg_vector, result

    @staticmethod
    def error_correction_model(cspread, z_index, zero_rates_3m, cointeg_vector,
                               dates_extra, Ret_SPX, Ret_VIX, Ret_WTI,
                               variance, dates_ts, flag_variance, max_lag=4):
        """
        Estimate the Error Correction Model (Model I).

        The dependent variable is the first difference of C-spread.
        Regressors: 3 lagged delta-C values and the lagged cointegration
        residual (error-correction term).

        Returns:
            model     : statsmodels OLS result.
            x         : np.ndarray -- regressor matrix (without constant).
            delta_C   : np.ndarray -- differenced C-spread.
            dates_lag : np.ndarray -- datenums for the lagged sample.
            BIC, AIC  : float -- information criteria.
            predicted : np.ndarray -- fitted values.
        """
        # Cointegration residual
        Y_mat = np.column_stack([cspread, z_index, zero_rates_3m])
        Psi = Y_mat @ cointeg_vector
        LaggedPsi = Psi[:-1]

        # First differences
        delta_C = np.diff(cspread)
        delta_Z = np.diff(z_index)
        delta_r = np.diff(zero_rates_3m)

        # Construct lagged delta-C matrix (3 lags)
        n = len(delta_C)
        LaggedDeltaC = np.zeros((n, max_lag - 1))
        for lag in range(1, max_lag):
            LaggedDeltaC[lag:, lag - 1] = delta_C[:-lag]

        # Trim to the valid range (after max_lag - 1 observations)
        trim = max_lag - 1
        delta_C_trim = delta_C[trim:]

        X_full = np.column_stack([
            LaggedDeltaC[trim:],
            delta_Z[trim:],
            delta_r[trim:],
            LaggedPsi[trim:],
        ])

        # MODEL I regressors: 3 lagged delta-C + cointegration residual
        x = X_full[:, [0, 1, 2, 5]]

        x_with_const = sm.add_constant(x)
        model = sm.OLS(delta_C_trim, x_with_const).fit()

        print("\n--- Error Correction Model (MODEL I) ---")
        print(model.summary())
        print(f"BIC: {model.bic:.4f}")
        print(f"AIC: {model.aic:.4f}")

        dates_lag = dates_ts[trim:-1]
        predicted = model.predict(x_with_const)

        return model, x, delta_C_trim, dates_lag, model.bic, model.aic, predicted
