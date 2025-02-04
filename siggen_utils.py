import numpy as np

def flip_sequence(p_stay=None, Tmax=None, tau=1, P1=0, P2=1):
    """
    Generate a sequence of binary flips based on specified probability and constraints.
    
    Parameters:
        p_stay (float): Probability of remaining at the same value as in the previous timestep.
        Tmax (int): Total number of timesteps.
        tau (int): Number of sub-steps during which a flip is not allowed.
                   For example, if tau=5, flips are only allowed every five timesteps.
        P1 (float): Value corresponding to polarity "1".
        P2 (float): Value corresponding to polarity "0".
    
    Returns:
        np.ndarray: Array of flipped values following the given constraints.
    """
    if Tmax % tau != 0:
        raise ValueError('Tmax must be an integer multiple of tau')
    
    flips = np.random.uniform(size=Tmax // tau) < (1 - p_stay)
    
    polarity = np.logical_xor.accumulate(flips)
    polarity = np.repeat(polarity, tau)
    
    # Map polarity to P1 and P2
    return P1 * polarity + P2 * (1 - polarity)

def herzfeld_block(z, T_pert=30, T_wash=10, P1=None,P2=None,P0=None, tau=1):
    hz = np.concatenate([flip_sequence(p_stay=z, Tmax=T_pert, tau=1, P1=P1, P2=P2),
                    [np.nan,np.nan],
                    P0*np.ones(T_wash),
                    [np.nan,P1,np.nan]]),
    return np.repeat(hz,tau)