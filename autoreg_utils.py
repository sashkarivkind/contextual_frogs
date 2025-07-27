import numpy as np

def compute_interpolation_coeffs(x,xgrid):
    '''
    Computes the interpolation coefficients for a linear interpolation.
    Implemented for 1D only. TODO: extend to higher dimensions.
    if the point is out of the grid then extrapolate using the two nearest points.
    Args:
        x (np.ndarray): points where the interpolation is computed
        xgrid (np.ndarray): grid of points used for interpolation
    Returns:
        np.ndarray: interpolation coefficients for each point in x, shape (len(x), len(xgrid)-1)
    '''
    x = np.asarray(x)
    xgrid = np.asarray(xgrid)
    if x.ndim != 1 or xgrid.ndim != 1:
        raise ValueError("Both x and xgrid must be 1D arrays.")
    if len(xgrid) == 1:
        return np.ones((len(x), 1))  # No intervals to interpolate
    coeffs = np.zeros((len(x), len(xgrid)))
    for i, xi in enumerate(x):
        # print(f"Computing coefficients for x[{i}] = {xi}")
        for j in range(len(xgrid) - 1): #TODO: vecotorise the for loop
            if xgrid[j] <= xi <= xgrid[j + 1] or (j == 0 and xi < xgrid[j]) or (j == len(xgrid) - 2 and xi > xgrid[j + 1]):
                coeffs[i, j] = (xgrid[j + 1] - xi) / (xgrid[j + 1] - xgrid[j])
                coeffs[i, j + 1] = (xi - xgrid[j]) / (xgrid[j + 1] - xgrid[j])
                break
    return coeffs




def clnn_kalman_matrix_prep(data, timestep=None, opts=None, data_format='dict'):
    """
    Prepares Kalman filter matrices for a closed-loop neural network model.

    Expects data in dict format:
      data['fixed']:
        subkeys that can appear here or in 'time_varying' (but not both):
        - sigma_u, sigma_y, sigma_x, sigma_a : noise std deviations
        - alpha                             : learning rate scalar
        - K                                 : (M x M) neural tangent kernel matrix 
        - H                                 : (M x M) Hessian* matrix (*matrix of derivatives d/x_j K_ij)
      data['time_varying']:
        subkeys that can appear here or in 'fixed' (but not both):
        - x             : length-T array of inputs
        - Jx           : length-T array of Jx(t) scalars
        - bbar_e         : length-T array of baseline errors (double bar e)
        - trial_type    : length-T boolean array (True=adaptation, False=channel)
        - lambda        : optional precomputed coefficients if compute_lambda=False

    Args:
        timestep (int, optional): If specified, returns matrices only at step timestep; otherwise returns lists of length T.
        opts (dict, optional): Options:
            - compute_lambda (bool): whether to recompute interpolation coeffs on the fly (default True)
            - x_grid (np.ndarray): grid for interpolation
        data_format (str): Must be 'dict' for current implementation.

    Returns:
        dict: Contains keys 'F', 'W', 'Q', 'H', 'R'. Each is either a list of T matrices or a single
              matrix at time t if t is provided.
    """

    if data_format != 'dict':
        raise NotImplementedError("Only 'dict' format is currently supported.")
    
    # Default options
    if opts is None:
        opts = {}
    compute_lambda = opts.get('compute_lambda', True)

    # Unpack fixed and time-varying data
    fixed = data.get('fixed', {})
    tv    = data.get('time_varying', {})

    # Any parameter may be provided in either 'fixed' or 'time_varying' (not both)
    sigma_u = fixed.get('sigma_u', tv.get('sigma_u', None))
    sigma_y = fixed.get('sigma_y', tv.get('sigma_y', None))
    sigma_x = fixed.get('sigma_x', tv.get('sigma_x', None))
    sigma_a = fixed.get('sigma_a', tv.get('sigma_a', None))
    alpha   = fixed.get('alpha',    tv.get('alpha',    None))
    K       = np.asarray(fixed.get('K',    tv.get('K',    [])))
    H_mat   = np.asarray(fixed.get('H',    tv.get('H',    [])))
    xgrid   = np.asarray(opts.get('x_grid', fixed.get('x_grid', tv.get('x_grid', []))))


    x_tv        = np.asarray(tv.get('x', []))
    Jx_tv       = np.asarray(tv.get('Jx', []))
    bbar_e_tv    = np.asarray(tv.get('bbar_e', []))
    trial_type  = np.asarray(tv.get('trial_type', []), dtype=bool)
    # print(f'x_tv: {x_tv}, Jx_tv: {Jx_tv}, bbar_e_tv: {bbar_e_tv}, trial_type: {trial_type}')
    T = x_tv.shape[0]

    # Compute or retrieve interpolation weights
    if compute_lambda:
        lambdas = compute_interpolation_coeffs(x_tv, xgrid)
    else:
        lambdas = np.asarray(tv.get('lambda', []))

    M = K.shape[0]
    nullMxM = np.zeros((M, M))
    null1xM = np.zeros((1, M))
    nullMx1 = np.zeros((M, 1))
    null1x1 = np.zeros((1, 1))

    # Observation and noise matrices (constant across time)
    H_obs = np.hstack(([[0]], 
                       null1xM, 
                       [[1]], 
                       [[0]]))
    R     = np.array([[sigma_a**2]])
    sigmas   = np.diag([sigma_u**2, sigma_y**2, sigma_x**2])

        # Prepare storage
    if timestep is None:
        F_list, W_list, Q_list, R_list, H_list = [None]*T, [None]*T, [None]*T, [R]*T, [H_obs]*T

    # Build matrices for each time step
    # TODO: decide whether to start from step 1, where step 0 is the initial state 
    # or start from step 0 with initial state being provided separately



    for t in range(1, T):
        if timestep is not None and timestep != t:
            continue
        tm1 = t - 1
        I_a       = 1 if trial_type[tm1] else 0
        I_c       = 1 - I_a
        Jx_tm1   = Jx_tv[tm1]
        lam_tm1  = lambdas[tm1]
        lam_t  = lambdas[t]
        bbar_e_tm1= bbar_e_tv[tm1] #TODO: doublecheck consistency with x: why here bbar and there no bbar?

        # Interpolate kernel and Hessian columns
        K_col = K.dot(lam_t)
        H_col = H_mat.dot(lam_t)

        # F' block components
        Fp11 = (I_c * Jx_tm1).reshape(1, 1)
        Fp12 = (I_c * lam_tm1).reshape(1, M)
        Fp21 = (I_a * alpha * (bbar_e_tm1 * H_col - Jx_tm1 * K_col)).reshape(M, 1)
        Fp22 = np.eye(M) - I_a * alpha * np.outer(K_col, lam_tm1)
        assert Fp22.shape == (M, M), f"Fp22 shape mismatch: {Fp22.shape} != {(M, M)}"

        # W' block components
        Wp11, Wp12, Wp13 = np.reshape(I_c,(1, 1)), np.reshape(I_a,(1, 1)), np.ones((1,1))
        Wp21           = -I_a * alpha * K_col
        Wp22           =  (I_a * alpha * K_col).reshape(M, 1)
        Wp23           =  nullMx1

        # Assemble F_t
        row1 = np.hstack((Fp11, Fp12, null1x1, Wp11))
        row2 = np.hstack((Fp21, Fp22, np.zeros((M,1)), Wp21.reshape(M,1)))
        Jx_t = Jx_tv[t]
        r3_1    = Jx_t * Fp11 + lam_t.T.dot(Fp21)
        r3_2    = Jx_t * Fp12 + lam_t.dot(Fp22)
        r3_4    = Jx_t * Wp11 + lam_t.dot(Wp21)
        # print(f'shape of r3_1: {r3_1.shape},shape of r3_2: {r3_2.shape}, shape of r3_4: {r3_4.shape}')
        row3    = np.hstack((r3_1, r3_2, null1x1, r3_4))
        row4    = np.zeros((1, 2 + M + 1))
        F_t     = np.vstack((row1, row2, row3, row4))

        # Assemble W_t
        row1W = np.hstack((null1x1, Wp12, Wp13))
        row2W = np.hstack((nullMx1, Wp22, Wp23))
        w3_2  = Jx_t * Wp12 + lam_t.dot(Wp22)
        w3_3  = Jx_t * Wp13 + lam_t.dot(Wp23)
        row3W = np.hstack([np.ones((1, 1)), w3_2, w3_3])
        row4W = np.hstack([np.ones((1, 1)), null1x1, null1x1])
        W_t   = np.vstack((row1W, row2W, row3W, row4W))

        # Process noise covariance
        Q_t = W_t.dot(sigmas).dot(W_t.T)

        if timestep is None:
            F_list[t], W_list[t], Q_list[t] = F_t, W_t, Q_t 

    # #for the initial step t=0 apply the same parameters as for t=1
    # if timestep is None:
    #     F_list[0], W_list[0], Q_list[0], R_list[0], H_list[0] = F_list[1], W_list[1], Q_list[1], R, H_list[1]

    # Return results
    if timestep is None:
        return {'F': F_list, 'W': W_list, 'Q': Q_list, 'H': H_list, 'R': R_list}
    else:
        return {'F': F_t, 'W': W_t, 'Q': Q_t, 'H': H_obs, 'R': R}


if __name__ == "__main__":
    # Example usage for compute_interpolation_coeffs
    print("Example usage of compute_interpolation_coeffs:")
    x = np.array([-0.7, 0.5, 2.0, 2.5, 5.9])
    xgrid = np.array([0, 1, 2, 2.3, 3])
    coeffs = compute_interpolation_coeffs(x, xgrid)
    print(f'Grid points: {xgrid}')
    print(f'data points: {x}')
    print("Interpolation coefficients:\n", coeffs)
    #example usage for clnn_kalman_matrix_prep
    print("Example usage of clnn_kalman_matrix_prep:")
    data = {
        'fixed': {
            'sigma_u': 0.1,
            'sigma_y': 0.1,
            'sigma_x': 0.1,
            'sigma_a': 0.1,
            'alpha': 0.01,
            'K': np.random.rand(5, 5),
            'H': np.random.rand(5, 5),
            'x_grid': np.linspace(0, 5, 5)
        },
        'time_varying': {
            'x': np.random.rand(10),
            'Jx': np.random.rand(10),
            'bbar_e': np.random.rand(10),
            'trial_type': np.random.choice([True, False], size=10)
        }
    }
    opts = {'compute_lambda': True, 'x_grid': np.linspace(-6, 6, 5)}
    matrices = clnn_kalman_matrix_prep(data, opts=opts)
    print("Output martices:")
    for key, value in matrices.items():
        print('--------------------------------------------')
        print(f"{key}: {value[-1]}, shape: {np.shape(value[-1])}")    

    from stat_utils import kalman_step

    #evaluate kalman_steps
    stm1 = np.random.rand(8, 1)  # Example state estimate at time t-1
    P_tm1 = np.eye(8)  # Example estimate covariance at time t-
    for t in range(1, 10):
        z_t = np.random.rand(1, 1)
        o = kalman_step(stm1, P_tm1, z_t, matrices['F'][t], matrices['H'][t], matrices['Q'][t], matrices['R'][t])
        print(f"Step {t}: z_pred = {o[0]}, log_likelihood = {o[1]}, s_updated = {o[2]}, P_updated = {o[3]}")
        stm1, P_tm1 = o[2], o[3]  # Update state and covariance for next iteration
        print(f"Updated state estimate: {stm1.flatten()}")

