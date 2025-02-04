import numpy as np

class LPF():
    def __init__(self, tau=1, ic=0, silent=True):
        self.tau = tau
        self.reset(ic, silent=True)
                
    def step(self,x, silent=False):
        self.state = 1./self.tau*x + (1-1./self.tau)*self.state 
        if not silent:
            return self.state
        
    def reset(self,ic=0,silent=False):
        self.state = ic
        if not silent:
            return self.state



def smooth_signal(input_array, window_size, axis=-1):
    """
    Smooth a signal along the specified axis using a square window while keeping original dimensions.

    Parameters:
    - input_array: numpy array, the signal to smooth.
    - window_size: int, the size of the square window for smoothing.
    - axis: int, the axis along which to apply the smoothing (default: -1, last axis).

    Returns:
    - smoothed_array: numpy array, the smoothed signal with the same shape as the input.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    # Create a square window (moving average filter)
    window = np.ones(window_size) / window_size

    # Apply convolution along the specified axis
    smoothed_array = np.apply_along_axis(
        lambda m: np.convolve(m, window, mode='same'), axis=axis, arr=input_array
    )

    return smoothed_array

def lpf(x, x_state, tau):
    return x_state + (x - x_state) / tau

def compute_steady_state_covariance(f=None, q=None, r=None, n_iter=1000, tol=1e-6, mu=None):
    """
    Computes the steady-state error covariance P for a Kalman filter with given parameters.
    
    Args:
        f (float): State transition coefficient.
        q (float): Process noise variance.
        r (float): Observation noise variance.
        n_iter (int): Maximum number of iterations for convergence.
        tol (float): Convergence tolerance.
    
    Returns:
        float: Steady-state error covariance P.
    """
    P = q  # Initial guess for P
    
    for _ in range(n_iter):
        P_next = f**2 * P + q - (f**2 * P + q)**2 / (f**2 * P + q + r)
        
        # Check for convergence
        if abs(P_next - P) < tol:
            return P_next
        
        P = P_next
    
    return P  # Return the last value if convergence is not reached

def parse_samples(segments):
    """
    Parse a list of segments into a vector of samples.

    Parameters:
    - segments: list of tuples, where each tuple is of the form:
        (val, n_samples): constant value over the segment, or
        ((val1, val2), n_samples): linear interpolation between val1 and val2 over the segment.
        (arr/lst, n_samples): np array or list of values over the segment.
    Returns:
    - np.ndarray: Vector of samples.
    """
    samples = []
    
    for segment in segments:
        if isinstance(segment[0], tuple):
            # Linear interpolation case
            (val1, val2), n_samples = segment
            interpolated = np.linspace(val1, val2, n_samples, endpoint=False)
            samples.extend(interpolated)
        elif isinstance(segment[0], np.ndarray):
            # Array case
            arr, n_samples = segment
            # raise error if n_samples is not equal to the length of the array
            if len(arr) != n_samples:
                raise ValueError
            samples.extend(arr.tolist())
        elif isinstance(segment[0], list):
            # List case
            lst, n_samples = segment
            # raise error if n_samples is not equal to the length of the list
            if len(lst) != n_samples:
                raise ValueError
            samples.extend(lst)
        else:
            # Constant value case
            val, n_samples = segment
            constant = [val] * n_samples
            samples.extend(constant)

    return np.array(samples)

def seg_time(segment_list,segment_num,side='end'):
    '''
    returns the commulative time till the start/end of a segment 
    '''
    if side == 'end':
        return np.sum([segment_list[i][1] for i in range(segment_num+1)]) - 1
    elif side == 'start':
        return np.sum([segment_list[i][1] for i in range(segment_num)]) if segment_num > 0 else 0
    else:
        raise ValueError