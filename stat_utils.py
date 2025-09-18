import numpy as np
import argparse
import pandas as pd


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
    info = dict(S=S.copy(), H=H.copy(), P_pred=P_pred.copy(), R=R.copy())
    # Check for missing observation
    if z_t is None or (isinstance(z_t, np.ndarray) and np.isnan(z_t).all()):
        # No measurement update
        return z_pred, S, None, s_pred, s_pred, P_pred, info

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
    info.update(dict(
        s_pred=s_pred.copy(),
        P_pred=P_pred.copy(),
        s_updated=s_updated.copy(),
        P_updated=P_updated.copy(),
        K=K.copy(),
        y=y.copy(),
        log_likelihood=log_likelihood.copy()
    ))

    return z_pred, S, float(log_likelihood), s_updated, s_pred, P_updated , info


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
    s_pred = np.zeros((n, Tmax))
    z_pred = np.zeros((m, Tmax))
    P = np.eye(n)
    # total_ll = 0.0
    # count_ll = 0
    lls = []
    Ps = []
    var_zs = []
    infos = []

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
            # z_pred_t, var_z, ll, s_upd, P = kalman_step(
            #     np.zeros((n, 1)), np.eye(n), z_t,
            #     F_t, H_t, Q_t, R_t
            # )

        z_pred_t, var_z, ll, s_upd, s_pred_, P , info = kalman_step(
                s_prev, P
                , z_t,
                F_t, H_t, Q_t, R_t
            )
        Ps.append(P)
        var_zs.append(var_z)
        s_est[:, t] = s_upd.flatten()
        s_pred[:, t] = s_pred_.flatten()
        z_pred[:, t] = z_pred_t.flatten()
        lls.append(ll if ll is not None else np.nan)
        infos.append(info)

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
    total_ll = np.nansum(lls[ii:]) if lls else np.nan
    avg_ll = np.nanmean(lls[ii:]) if lls else np.nan
    # return state_rms, obs_rms, avg_ll , lls
    return dict(
        state_rms=state_rms,
        obs_rms=obs_rms,
        total_ll=total_ll,
        avg_ll=avg_ll,
        lls=np.array(lls),
        s_est=s_est,
        s_pred=s_pred,
        z_obs=z_obs,
        z_pred=z_pred,
        var_z_pred=var_zs,
        Ps=Ps,
        infos=infos
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


def multivar_entropy(cov: np.ndarray, regularization: float = 1e-55) -> float:
    """
    Compute the differential entropy of an m-dimensional Gaussian with covariance `cov`:
        H = 1/2 * ln((2*pi*e)^m * det(cov)).
    """
    m = cov.shape[0]
    cov += np.eye(m) * regularization  # Regularize covariance matrix
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError(f"Covariance matrix must be positive-definite. logdet: {logdet}, sign: {sign}")
    # H = 0.5 * (m * ln(2*pi*e) + ln(det(cov)))
    print(f"m: {m}, logdet: {logdet}")
    return 0.5 * (m * np.log(2 * np.pi * np.e) + logdet)


# def run_covariance_test(filt_params,
#                         trials: int = 100,
#                         Tmax: int = 117,
#                         bootstrap_samples: int = 100,
#                         seed: int = None):
#     """
#     Compare total filter log-likelihood vs. Gaussian entropy from sample covariance.
#     """
#     if seed is not None:
#         np.random.seed(seed)

#     all_innovs = []
#     filter_lls = []

#     # collect innovations and filter LLs
#     for _ in range(trials):
#         trial = run_trial(filt_params=filt_params,
#                           sim_params=filt_params,
#                           Tmax=Tmax)
#         filter_lls.append(trial['total_ll'])
#         for info in trial['infos']:
#             if 'y' in info:
#                 all_innovs.append(info['y'].flatten())

#     innovs = np.vstack(all_innovs)
#     N, m = innovs.shape

#     # sample covariance
#     Sigma_emp = (innovs.T @ innovs) / N

#     # 1) empirical LL under Σ_emp
#     inv_emp = np.linalg.inv(Sigma_emp)
#     sign_emp, logdet_emp = np.linalg.slogdet(Sigma_emp)
#     quad = np.sum([y @ inv_emp @ y for y in innovs])
#     LL_emp = -0.5 * (N * m * np.log(2 * np.pi) + N * logdet_emp + quad)

#     # 2) filter's own total LL
#     LL_filt = np.sum(filter_lls)

#     # 3) entropy-based total neg-log-likelihood
#     H_full = multivar_entropy(Sigma_emp)
#     LL_entropy = -N * H_full

#     # 4) bootstrap tolerance
#     LL_bs = []
#     for _ in range(bootstrap_samples):
#         idx = np.random.choice(N, size=N, replace=True)
#         samp = innovs[idx]
#         S_b = (samp.T @ samp) / N
#         inv_b = np.linalg.inv(S_b)
#         sign_b, logdet_b = np.linalg.slogdet(S_b)
#         quad_b = np.sum([y @ inv_b @ y for y in samp])
#         LL_bs.append(-0.5 * (N * m * np.log(2 * np.pi) + N * logdet_b + quad_b))
#     sigma_LL = float(np.std(LL_bs, ddof=1))
#     tol = 3 * sigma_LL

#     # report
#     print("\n=== Covariance-based LL test ===")
#     print(f" Empirical-cov log-lik: {LL_emp:.4f}")
#     print(f" Filter’s total  log-lik: {LL_filt:.4f}")
#     print(f" Entropy-based negLL:   {LL_entropy:.4f}")
#     print(f" Bootstrap σ_LL:         {sigma_LL:.4f}")
#     print(f" Tolerance (3 σ):        ±{tol:.4f}\n")

#     # assertion against filter LL
#     assert abs(LL_emp - LL_filt) <= tol, (
#         f"|ΔLL_emp - LL_filt| = {abs(LL_emp - LL_filt):.4f} exceeds 3σ ({tol:.4f})"
#     )
#     # assertion entropy vs filter
#     assert abs(LL_entropy - LL_filt) <= tol, (
#         f"|LL_entropy - LL_filt| = {abs(LL_entropy - LL_filt):.4f} exceeds 3σ ({tol:.4f})"
#     )
#     print("Covariance-LL and entropy tests passed.\n")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Run sensitivity and covariance-vs-filter log-likelihood tests."
#     )
#     parser.add_argument("--trials", "-t", type=int, default=100,
#                         help="Number of trials for covariance test (default: 100)")
#     parser.add_argument("--bootstrap-samples", "-b", type=int, default=100,
#                         help="Bootstrap draws (default: 100)")
#     parser.add_argument("--seed", "-s", type=int, default=None,
#                         help="Random seed (default: None)")
#     args = parser.parse_args()

#     # default filter parameters
#     np.random.seed(0)
#     F = np.array([[0.8, 1], [0, 1]])
#     H = np.array([[1, 0.4]])
#     Q = np.eye(2) * 0.1
#     R = np.eye(1) * 0.5
#     param0 = {'F': F, 'H': H, 'Q': Q, 'R': R}

#     sensitivity_test(param0)
#     run_covariance_test(param0,
#                         trials=args.trials,
#                         Tmax=117,
#                         bootstrap_samples=args.bootstrap_samples,
#                         seed=args.seed)
    

def run_covariance_test(filt_params,
                        trials: int = 100,
                        Tmax: int = 45,
                        bootstrap_samples: int = 100,
                        seed: int = None,
                        missing_prob: float = 0.1):
    """
    Compare total filter log-likelihood vs. Gaussian entropy from sample covariance of observations.
    """
    if seed is not None:
        np.random.seed(seed)

    all_innovs = []
    all_obs = []  # collect observations
    filter_lls = []

    # 1) run trials and collect innovations, observations, and filter LLs
    for _ in range(trials):
        trial_obs = []
        trial = run_trial(filt_params=filt_params,
                          sim_params=filt_params,
                          Tmax=Tmax, 
                          missing_prob=missing_prob)
        # collect innovations
        for info in trial['infos']:
            if 'y' in info:
                all_innovs.append(info['y'].flatten())
        # collect observations (disable skip missing)
        z_obs = trial['z_obs'].T  # shape (Tmax, m)
        for z in z_obs:
            if not np.isnan(z).all():
                trial_obs.append(z.flatten())
        # collect filter log-likelihood
        filter_lls.append(trial['total_ll'])
        all_obs.append(np.array(trial_obs))

    innovs = np.vstack(all_innovs)    # shape (N_i, m)
    N_i, m = innovs.shape
    obs = np.array(all_obs)          # shape (N_o, Tmax, m)
    N_o, Tmax, _ = obs.shape
    #transpose to make the T dimension first
    obs = obs.transpose(1, 2, 0)     # shape (Tmax, m, N_o)
    #flatten the last two dimensions
    obs = obs.reshape(obs.shape[0], -1)  # shape (Tmax, m*N_o)

    # 2) empirical covariance of innovations Σ_emp and corresponding LL
    Sigma_emp = (innovs.T @ innovs) / N_i
    inv_emp = np.linalg.inv(Sigma_emp)
    sign_emp, logdet_emp = np.linalg.slogdet(Sigma_emp)
    quad_i = np.sum([y @ inv_emp @ y for y in innovs])
    LL_emp = -0.5 * (N_i * m * np.log(2 * np.pi) + N_i * logdet_emp + quad_i)

    # 3) empirical covariance of observations Σ_z and entropy-based negLL (Tmax *  Tmax)
    Sigma_z = (obs @ obs.T) / N_o
    # compute full entropy H = 1/2 * ln((2*pi*e)^m * det(Sigma_z))
    H_obs = multivar_entropy(Sigma_z)
    LL_entropy =  -N_o*H_obs  

    # 4) filter's own total LL
    LL_filt = np.sum(filter_lls)

    # 5) bootstrap tolerance on innovation LL
    LL_bs = []
    for _ in range(bootstrap_samples):
        idx = np.random.choice(N_i, size=N_i, replace=True)
        samp = innovs[idx]
        S_b = (samp.T @ samp) / N_i
        inv_b = np.linalg.inv(S_b)
        sign_b, logdet_b = np.linalg.slogdet(S_b)
        quad_b = np.sum([y @ inv_b @ y for y in samp])
        LL_bs.append(-0.5 * (N_i * m * np.log(2 * np.pi) + N_i * logdet_b + quad_b))
    sigma_LL = float(np.std(LL_bs, ddof=1))
    tol = 3 * sigma_LL

    # 6) report & assert
    print("=== Covariance-based LL test ===")
    print(f" Empirical-cov log-lik (innovations): {LL_emp:.4f}")
    print(f" Filter’s  total   log-lik:           {LL_filt:.4f}")
    print(f" Entropy-based negLL (observations): {LL_entropy:.4f}")
    print(f" Bootstrap σ_LL:                     {sigma_LL:.4f}")
    print(f" Tolerance (3 σ):                    ±{tol:.4f}")

    # assertion against filter LL
    # assert abs(LL_emp - LL_filt) <= tol, (
    #     f"|LL_emp - LL_filt| = {abs(LL_emp - LL_filt):.4f} exceeds 3σ ({tol:.4f})"
    # )
    # # assertion entropy vs filter
    # assert abs(LL_entropy - LL_filt) <= tol, (
    #     f"|LL_entropy - LL_filt| = {abs(LL_entropy - LL_filt):.4f} exceeds 3σ ({tol:.4f})"
    # )
    # print("Covariance-LL and entropy tests passed.")
    return {
                # 'iter': idx,
                'LL_emp': LL_emp,
                'LL_filt': LL_filt,
                'LL_entropy': LL_entropy,
                'sigma_LL': sigma_LL
            }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run sensitivity and covariance-vs-filter log-likelihood tests."
    )
    parser.add_argument("--trials", "-t", type=int, default=100,
                        help="Number of trials for covariance test (default: 100)")
    parser.add_argument("--bootstrap-samples", "-b", type=int, default=100,
                        help="Bootstrap draws (default: 100)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Random seed (default: None)")
    parser.add_argument("--Tmax", "-T", type=int, default=103,
                        help="Number of time steps for simulation (default: 103)")
    parser.add_argument("--do-sensitivity-test", "-sensitivity", action='store_true',
                        help="Run sensitivity test (default: False)")
    parser.add_argument("--do-covariance-test", "-covariance", action='store_true',
                        help="Run covariance-based log-likelihood test (default: False)")
    parser.add_argument("--iterate-covariance-test", "-iter", action='store_true',
                        help="Run covariance-based log-likelihood test for multiple iterations (default: False)")
    parser.add_argument("--covariance-test-iter", "-cov_iter", type=int, default=10,
                        help="Number of iterations for covariance test (default: 10)")
    parser.add_argument("--cov-test-filename", "-cov_file", type=str, default='cov_test_results.csv',
                        help="Filename to save covariance test results (default: 'cov_test_results.csv')")
    parser.add_argument("--cov-missing-prob", "-missing", type=float, default=0.0,
                        help="Probability of missing an observation in covariance test (default: 0.0)")
    args = parser.parse_args()

    if args.cov_missing_prob > 0:
        raise NotImplementedError("Covariance test with missing observations is not implemented yet.")
    # default filter parameters
    np.random.seed(args.seed)
    # F = np.array([[0.8, 1, 0.3], [0, 1, -0.7], [-0.3, 0.8, 0.1]])
    # H = np.array([[1, 0.4, 0.6]])
    # Q = np.eye(3) * 0.1
    # R = np.eye(1) * 0.5

    F = np.array([[0.8, 1], [0.2, -0.5]])
    H = np.array([[1, 0.4]])
    Q = np.eye(2) * 0.1
    R = np.eye(1) * 0.5
    param0 = {'F': F, 'H': H, 'Q': Q, 'R': R}

    print(f'spectrum of F: {np.linalg.eigvals(F)}')

    if args.do_sensitivity_test:
        sensitivity_test(param0)

    if args.do_covariance_test:
        _ = run_covariance_test(param0,
                        trials=args.trials,
                        Tmax=args.Tmax,
                        bootstrap_samples=args.bootstrap_samples,
                        seed=args.seed,
                        missing_prob=args.cov_missing_prob)
    if args.iterate_covariance_test:
        all_results = []
        for idx in range(args.covariance_test_iter):
            print(f"\n=== Iteration {idx + 1} ===")
            #generate new random parameters per iteration
            F = np.random.uniform(0.5, 1.5, size=(2, 2))
            H = np.random.uniform(0.1, 0.5, size=(1, 2))
            Q = np.eye(2) * np.random.uniform(0.01, 2)
            R = np.eye(1) * np.random.uniform(0.1, 2)
            spectral_r = np.random.uniform(0.2, 0.99)
            #check spectrum of F if unstable, rescale
            eigs = np.linalg.eigvals(F)
            F = F / np.max(np.abs(eigs)) * spectral_r
            params = {'F': F, 'H': H, 'Q': Q, 'R': R}
            res = run_covariance_test(params,
                        trials=args.trials,
                        Tmax=args.Tmax,
                        bootstrap_samples=args.bootstrap_samples,
                        seed=args.seed + 17 + idx,
                        missing_prob=args.cov_missing_prob)
            print(f"Results: {res}")
            all_results.append(res)
        #save results to a csv file
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv(args.cov_test_filename, index=False)


# Example output:
# $ python stat_utils.py --seed 42
# Baseline state RMS: 0.4664,  Baseline obs RMS: 1.1405,  Avg Log‐L: -1.5552

# Perturb  1: state RMS=1.6674 (Δ (negative is better) +1.2010), obs RMS=1.2161 (Δ  +0.0756), AvgLL=-1.6059 (Δ (positive is better) -0.0508)
# Perturb  2: state RMS=1.2823 (Δ (negative is better) +0.8159), obs RMS=1.8612 (Δ  +0.7207), AvgLL=-2.2446 (Δ (positive is better) -0.6894)
# Perturb  3: state RMS=1.3055 (Δ (negative is better) +0.8391), obs RMS=1.9708 (Δ  +0.8304), AvgLL=-2.5008 (Δ (positive is better) -0.9456)
# Perturb  4: state RMS=0.8998 (Δ (negative is better) +0.4334), obs RMS=1.1865 (Δ  +0.0460), AvgLL=-1.5914 (Δ (positive is better) -0.0363)
# Perturb  5: state RMS=1.0058 (Δ (negative is better) +0.5394), obs RMS=1.5378 (Δ  +0.3973), AvgLL=-1.9116 (Δ (positive is better) -0.3564)
# Perturb  6: state RMS=1.2610 (Δ (negative is better) +0.7946), obs RMS=1.2130 (Δ  +0.0725), AvgLL=-1.6055 (Δ (positive is better) -0.0503)
# Perturb  7: state RMS=2.2441 (Δ (negative is better) +1.7777), obs RMS=1.8071 (Δ  +0.6666), AvgLL=-2.0981 (Δ (positive is better) -0.5429)
# Perturb  8: state RMS=0.8586 (Δ (negative is better) +0.3922), obs RMS=1.1965 (Δ  +0.0560), AvgLL=-1.5952 (Δ (positive is better) -0.0400)
# Perturb  9: state RMS=1.1232 (Δ (negative is better) +0.6568), obs RMS=1.3182 (Δ  +0.1778), AvgLL=-1.6907 (Δ (positive is better) -0.1356)
# Perturb 10: state RMS=0.9858 (Δ (negative is better) +0.5194), obs RMS=1.1501 (Δ  +0.0096), AvgLL=-1.5595 (Δ (positive is better) -0.0043)

# === Covariance-based LL test ===
#  Empirical-cov log-lik: -16434.7396
#  Filter’s total  log-lik: -16394.4392
#  Entropy-based negLL:   -16434.7396
#  Bootstrap σ_LL:         65.7621
#  Tolerance (3 σ):        ±197.2864