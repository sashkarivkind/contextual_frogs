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


def kalman_step(s_tm1, P_tm1, z_t, F, H, Q, R):
    """
    Perform a single Kalman filter update step.

    Parameters:
    ----------
    s_tm1 : numpy.ndarray
        State estimate at time t-1 (n x 1).
    P_tm1 : numpy.ndarray
        Estimate covariance at time t-1 (n x n).
    z_t : numpy.ndarray
        Observation at time t (m x 1).
    F : numpy.ndarray
        State transition model (n x n).
    H : numpy.ndarray
        Observation model (m x n).
    Q : numpy.ndarray
        Process noise covariance (n x n).
    R : numpy.ndarray
        Observation noise covariance (m x m).

    Returns:
    -------
    z_pred : numpy.ndarray
        Predicted observation (m x 1).
    S : float
        Covariance of the predicted observation (m x m).
    log_likelihood : float
        Log-likelihood of the observation.
    s_updated : numpy.ndarray
        Updated state estimate (n x 1).
    P_updated : numpy.ndarray
        Updated estimate covariance (n x n).
    """
    # Predict state and covariance
    s_pred = F @ s_tm1
    P_pred = F @ P_tm1 @ F.T + Q

    # Predict observation
    z_pred = H @ s_pred
    S = H @ P_pred @ H.T + R
    # print(f'S: {S}, H: {H}, P_pred: {P_pred}, R: {R}')

    # Check for missing observation
    if z_t is None or (isinstance(z_t, np.ndarray) and np.isnan(z_t).all()):
        # No measurement update
        return z_pred, S, None, s_pred, P_pred

    # Innovation (residual)
    y = z_t - z_pred
    S_inv = np.linalg.inv(S)

    # Kalman gain
    K = P_pred @ H.T @ S_inv

    # Update state estimate
    s_updated = s_pred + K @ y
    I = np.eye(P_pred.shape[0])
    P_updated = (I - K @ H) @ P_pred

    # Compute log-likelihood
    m = z_t.shape[0]
    _sign, log_det_S = np.linalg.slogdet(S)
    log_likelihood = -0.5 * (m * np.log(2 * np.pi) + log_det_S + (y.T @ S_inv @ y))

    return z_pred, S, float(log_likelihood), s_updated, P_updated


def run_trial(filt_params, sim_data=None, sim_params=None, Tmax=100, missing_prob=0.1, discard_1st_step_stats=False):
    """

    filt_params: dict with F, H, Q, R used inside the Kalman filter.
                 Each may be constant (ndarray) or a list/tuple of ndarrays of length Tmax.
    sim_params: optional dict with F, H, Q, R used to generate ground truth & observations.
                Each may be constant (ndarray) or a list/tuple of ndarrays of length Tmax.
    sim_data: optional, if provided, use this as the simulation data instead of generating it.
    ONLY one of sim_data or sim_params should be provided.
    Tmax: int, number of time steps to simulate.
    missing_prob: float, probability of missing an observation at each time step.
    discard_1st_step_stats: bool, if True, discard the first step from the pooled statistics.
    """
    if sim_data is not None and sim_params is not None:
        raise ValueError("Provide either sim_data or sim_params, not both.")
    if sim_data is None and sim_params is None:
        raise ValueError("Provide either sim_data or sim_params.")

    # Helper to fetch parameter at time t (constant or time-varying)
    def _get(param, t):
        if isinstance(param, (list, tuple)):
            return param[t]
        return param

    # Validate lengths for time-varying sim_params
    if sim_params is not None:
        for key in ('F', 'H', 'Q', 'R'):
            p = sim_params.get(key)
            if isinstance(p, (list, tuple)) and len(p) != Tmax:
                raise ValueError(f"sim_params['{key}'] must have length Tmax={Tmax}")

    # Determine dimensions from filt_params at t=last
    # using the first step is not safe as it F might be not defined.
    Flast = _get(filt_params['F'], -1)
    Hlast = _get(filt_params['H'], -1)
    n = Flast.shape[0]
    m = Hlast.shape[0]

    # 1) Simulate observations if not provided
    if sim_data is not None:
        z_obs = sim_data
        s_true = None
    else:
        # Simulate true state trajectory
        s_true = np.zeros((n, Tmax))
        for t in range(1, Tmax):
            F_t = _get(sim_params['F'], t)
            Q_t = _get(sim_params['Q'], t)
            w = np.random.multivariate_normal(np.zeros(n), Q_t)
            s_true[:, t] = F_t @ s_true[:, t-1] + w

        # Simulate observations with missing values
        z_obs = np.zeros((m, Tmax))
        for t in range(Tmax):
            H_t = _get(sim_params['H'], t)
            R_t = _get(sim_params['R'], t)
            v = np.random.multivariate_normal(np.zeros(m), R_t)
            z = H_t @ s_true[:, t] + v
            if np.random.rand() < missing_prob:
                z = np.full(m, np.nan)
            z_obs[:, t] = z

    # 2) Run Kalman filter
    s_est = np.zeros((n, Tmax))
    z_pred = np.zeros((m, Tmax))
    P = np.eye(n)
    # total_ll = 0.0
    # count_ll = 0
    lls = []
    Ps = []

    #a dummy first step - here we assume that the 


    for t in range(Tmax):
        F_t = _get(filt_params['F'], t)
        H_t = _get(filt_params['H'], t)
        Q_t = _get(filt_params['Q'], t)
        R_t = _get(filt_params['R'], t)

        # Current observation
        z = z_obs[:, t]
        z_t = z.reshape(m, 1)
        if np.isnan(z).all():
            z_t = None

        # Previous estimate
        s_prev = s_est[:, t-1].reshape(n,1) if t > 0 else np.zeros((n,1))

        if  t == 0:
            # TODO: revise the follwoing simplification
            # First step, no previous estimate, unknown transition from the past, 
            # all the noise is attributed to the observation
            if F_t is None:
                F_t = np.eye(n)
            if H_t is None:
                raise ValueError("H_t must be provided for the first step.")
            if Q_t is None:
                Q_t = np.zeros((n, n))
            if R_t is None:
                raise ValueError("R_t must be provided for the first step.")
            # TODO: check why this sham step was added at some point
            # z_pred_t, sigma_z, ll, s_upd, P = kalman_step(
            #     np.zeros((n, 1)), np.eye(n), z_t,
            #     F_t, H_t, Q_t, R_t
            # )

        z_pred_t, sigma_z, ll, s_upd, P = kalman_step(
                s_prev, P, z_t,
                F_t, H_t, Q_t, R_t
            )
        Ps.append(P)
        s_est[:, t] = s_upd.flatten()
        z_pred[:, t] = z_pred_t.flatten()
        lls.append(ll if ll is not None else np.nan)
        # if ll is not None:
        #     total_ll += ll
        #     count_ll += 1

    # 3) Compute metrics

    if discard_1st_step_stats:
        ii = 1
    else:
        ii = 0

    if sim_data is None:
        state_rms = np.sqrt(np.mean((s_est[:, ii:] - s_true[:, ii:])**2))
    else:
        state_rms = float('nan')
    obs_rms = np.sqrt(np.nanmean((z_obs[:, ii:] - z_pred[:, ii:])**2))
    # avg_ll = total_ll / count_ll if count_ll > 0 else float('nan')
    avg_ll = np.nanmean(lls[ii:]) if lls else np.nan

    # return state_rms, obs_rms, avg_ll , lls
    return dict(
        state_rms=state_rms,
        obs_rms=obs_rms,
        avg_ll=avg_ll,
        lls=np.array(lls),
        s_est=s_est,
        z_obs=z_obs,
        z_pred=z_pred,
        sigma_z_pred=sigma_z,
        Ps=Ps
    )


def sensitivity_test(param0, Nmax=20, M=10, delta=0.2, data=None):
    """
    param0: baseline parameters for both sim and reference filter
    Nmax: number of Monte Carlo trials per evaluation
    M: number of perturbed filter parameter sets
    delta: max fractional perturbation (±delta)
    data: optional, if provided, use this as the simulation data instead of generating it
    """
    reference = {'sim_params': param0, 'sim_data': None} if data is None else {'sim_params': None, 'sim_data': data}
    # Baseline filter on baseline model
    baseline_metrics = [run_trial(filt_params=param0, sim_params=param0) for _ in range(Nmax)]
    # base_state_rms = np.sqrt(np.mean([r**2 for r,_,_ in baseline_metrics]))
    # base_obs_rms = np.sqrt(np.mean([r**2 for _,r,_ in baseline_metrics]))
    # base_ll  = np.mean([ll for _, _,ll in baseline_metrics])
    base_state_rms = np.sqrt(np.mean([r['state_rms']**2 for r in baseline_metrics]))
    base_obs_rms = np.sqrt(np.mean([r['obs_rms']**2 for r in baseline_metrics]))
    base_ll  = np.mean([r['avg_ll'] for r in baseline_metrics])

    print(f"Baseline state RMS: {base_state_rms:.4f},  Baseline obs RMS: {base_obs_rms:.4f},  Avg Log‐L: {base_ll:.4f}\n")

    # Perturbed filters (simulation always uses param0)
    for i in range(1, M+1):
        # create perturbed filter params
        pert = {}
        for key, mat in param0.items():
            noise = np.random.uniform(-delta, delta, size=mat.shape)
            pert[key] = mat * (1 + noise)

        # run trials
        trials = [run_trial(filt_params=pert, sim_params=param0) for _ in range(Nmax)]
        # mean_state_rms = np.sqrt(np.mean([r**2 for r,_,_ in trials]))
        # mean_obs_rms = np.sqrt(np.mean([r**2 for _,r,_ in trials]))
        # mean_ll  = np.mean([ll for _, _,ll in trials])
        mean_state_rms = np.sqrt(np.mean([r['state_rms']**2 for r in trials]))
        mean_obs_rms = np.sqrt(np.mean([r['obs_rms']**2 for r in trials]))
        mean_ll  = np.mean([r['avg_ll'] for r in trials])

        dstate_rms = mean_state_rms - base_state_rms
        dobs_rms = mean_obs_rms - base_obs_rms
        dll  = mean_ll - base_ll

        print(f"Perturb {i:2d}: state RMS={mean_state_rms:.4f} (Δ (negative is better) {dstate_rms:+.4f}), "
              f"obs RMS={mean_obs_rms:.4f} (Δ  {dobs_rms:+.4f}), "
              f"AvgLL={mean_ll:.4f} (Δ (positive is better) {dll:+.4f})")


if __name__ == "__main__":
    # Example default parameters
    np.random.seed(0)

    F = np.array([[0.8, 1], [0, 1]])
    H = np.array([[1, 0.4]])
    Q = np.eye(2) * 0.1
    R = np.eye(1) * 0.5
    param0 = {'F': F, 'H': H, 'Q': Q, 'R': R}

    sensitivity_test(param0)