import numpy as np
import re

def parse_log_file(filename):
    """
    Parse a log file with lines from optimizer logs, like:
      at minimum <value> accepted <flag> with params [<param1> <param2> ...]
    
    Returns a dictionary with:
      'value': 1D np.array of floats,
      'accepted': 1D np.array of booleans (converted from int, where nonzero is True),
      'x': 2D np.array of floats (each row is the params array)
    """
    values = []
    accepted_flags = []
    params_list = []
    
    # Regex pattern: capture the minimum value, accepted flag, and the list of parameters
    pattern = re.compile(
        r"at minimum\s+([^\s]+)\s+accepted\s+([^\s]+)\s+with params\s+\[([^\]]+)\]"
    )
    
    with open(filename, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # Extract groups
                value_str = match.group(1)
                accepted_str = match.group(2)
                params_str = match.group(3)
                
                # Convert the extracted strings to appropriate types
                value = float(value_str)
                accepted = bool(int(accepted_str))  # 0 -> False, non-zero -> True
                # Convert params string (e.g., "-4.27372773  0.940294 ...") to a list of floats
                params = [float(x) for x in params_str.split()]
                
                values.append(value)
                accepted_flags.append(accepted)
                params_list.append(params)
    
    # Convert lists to numpy arrays
    return {
        'value': np.array(values),
        'accepted': np.array(accepted_flags, dtype=bool),
        'x': np.array(params_list)
    }
