import numpy as np

def bic(measured_data, modeled_data, k_params,
        mode='white_gaussian',
        autoregressive_coeffs=None,
        autoregressive_sigma=None):
    """
    Compute the Bayesian Information Criterion (BIC) for a model fit,
    assuming normally distributed residuals.

    Parameters
    ----------
    measured_data : array-like
        The observed data (y_exp). NaNs here are handled per mode.
    modeled_data : array-like
        The model predictions (y_mod). Must not contain NaNs.
    k_params : int
        The number of free parameters in the model (p).
    mode : str
        'white_gaussian' (default) or 'autoregressive_gaussian'.
    autoregressive_coeffs : array-like, optional
        AR coefficients [a1, a2, …, ap] for the noise. Required if mode='autoregressive_gaussian'.
    autoregressive_sigma : float, optional
        Known noise standard deviation (\sigma). If None, estimated from data.

    Returns
    -------
    float
        The BIC value.
    """
    # 0) validate model output
    y_mod = np.asarray(modeled_data)
    if np.isnan(y_mod).any():
        raise ValueError("modeled_data contains NaN values; cannot proceed.")

    # 1) white‐noise Gaussian branch
    if mode == 'white_gaussian':
        y_exp = np.asarray(measured_data)
        mask = ~np.isnan(y_exp)
        y_exp_clean = y_exp[mask]
        y_mod_clean = y_mod[mask]

        n = y_exp_clean.size
        if n == 0:
            raise ValueError("No valid data points: all measured_data are NaN.")

        rss = np.sum((y_mod_clean - y_exp_clean) ** 2)
        return n * np.log(rss / n) + k_params * np.log(n)

    # 2) autoregressive‐noise Gaussian branch
    elif mode == 'autoregressive_gaussian':
        if autoregressive_coeffs is None:
            raise ValueError("autoregressive_coeffs must be provided for AR mode.")

        a = np.asarray(autoregressive_coeffs)
        p = a.size

        # 2a) handle missing data by zero‐imputation
        y_exp = np.asarray(measured_data)

        # 2b) raw residual series n_t = y_exp - y_mod
        n_series = y_exp - y_mod


        # 2c) pad the first p lags with zeros so we keep N innovations
        pad = np.zeros(p)
        N = n_series.size
        n_padded = np.concatenate([pad, n_series])
        valid_mask = ~np.isnan(n_padded)
        # 2d) compute innovations η_t = n_t - sum(a_i * n_{t-i})
        eta = np.empty(N)
        for t in range(p, p + N):
            #evolution
            evolution = np.dot(a, n_padded[t - p:t][::-1])
            if not valid_mask[t]: # if no observation supplied use evolution
                n_padded[t] = evolution
            #innovations
            eta[t - p] = n_padded[t] - evolution

        # 3) effective sample size and RSS
        valid_mask = valid_mask[p:]  # valid mask for innovations after padding
        eta_valid = eta[valid_mask]
        # print(n_padded)
        # print(eta_valid)
        n_eff = eta_valid.shape[0]
        rss_ar = np.sum(eta_valid ** 2)

        # 4) noise variance 
        if autoregressive_sigma is not None:
            sigma2 = autoregressive_sigma ** 2
            raise NotImplementedError("BIC for AR with known sigma is not implemented yet.")
        else:
            sigma2 = rss_ar / n_eff

        # 5) BIC formula
        bic_value = n_eff * np.log(rss_ar / n_eff) + k_params * np.log(n_eff)
        return bic_value

    else:
        raise ValueError(f"Unknown mode '{mode}'.")
