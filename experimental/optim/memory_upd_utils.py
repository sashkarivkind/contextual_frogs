import numpy as np


def polarity(ys=None, qs=None):
    '''
    Determine the main polarity of a run of trials with the same force field direction.
    Args:
        ys: applied forces (numpy array)
        qs: sensory cues (numpy array)
    Returns:
        main_polarity: +1 or -1
    Raises:
        ValueError: if mixed polarities are found in the same run
    '''
    iys_blocks = np.concatenate([~np.isnan(ys[:-1]) & ~np.isnan(ys[1:]), [~np.isnan(ys[-2]) & ~np.isnan(ys[-1])]])
    polarities = qs[iys_blocks] * ys[iys_blocks]
    main_polarity = np.sign(np.sum(polarities))
    if np.any(np.isclose(polarities, -main_polarity)):
            raise ValueError('Mixed polarities in same run!')
    return main_polarity

def extract_mem_updates(ys, qs, aa, range_of_triplets):
    '''
    Extracts memory updates from the output traces, according to COIN paper Fig.3.
    Args:
        ys: applied forces (numpy array)
        qs: sensory cues (numpy array)
        aa: experimental/modeled adaptation levels (numpy array)
        range_of_triplets: list of start indexes of triplets
            if a negative integer is given, it is taken as the number of triplets to extract from the end of data
            if a positive integer is given, it is taken as the number of triplets to extract from the start of data
            if list is given - applies to the given indexes
    
    Returns:
        outs: 4 D numpy array with outputs corresponding to averages over a3-a1
        where a1, a2, a3 are adaptation levels for the triplet trials computed for each of the following combinations of (y2,q2): (1,1) (1,-1) (-1,1) (-1,-1)
    '''

    #align all the polarities according to qs (so that q=+1 corresponds to positive y)
    polarity_main = polarity(ys, qs)

    ys = ys * polarity_main
    # qs = qs * polarity_main
    aa = aa * polarity_main
    #create list of all the start indexes of triplets of (np.nan, some non-nan, np.nan) in ys: 
    start_indexes = []
    for i in range(len(ys) - 2):
        if np.isnan(ys[i]) and not np.isnan(ys[i + 1]) and np.isnan(ys[i + 2]):
            start_indexes.append(i)
    if isinstance(range_of_triplets, int):
        if range_of_triplets < 0:
            start_indexes = start_indexes[range_of_triplets:]
        else:
            start_indexes = start_indexes[:range_of_triplets]
    elif isinstance(range_of_triplets, list):
        start_indexes = [start_indexes[i] for i in range_of_triplets]

    #if we understood the paper correctly then all the q1 and q3 should be equal
    if not np.isclose(qs[start_indexes], qs[start_indexes[0]]).all() or not np.isclose(qs[np.array(start_indexes)+2], qs[start_indexes[0]]).all():
        raise ValueError('q1 and q3 values in triplets are not as expected! They should all be equal.')


    #align all the polarities according to qs (so that q=+1 corresponds to positive y)
    yq_polarity = polarity(ys, qs)
    q_sandwitch = qs[start_indexes[0]]
    qs = qs * q_sandwitch

    ys = ys * yq_polarity * q_sandwitch
    aa = aa  * yq_polarity * q_sandwitch



    # print(f'using start indexes: {start_indexes}')
    #create 4 D array to hold outputs
    outs_ = [[] for _ in range(4)]
    for idx, start_idx in enumerate(start_indexes):
        #all triplets should have q=+1 at the beginning and end
        # if not (np.isclose(qs[start_idx], 1) and np.isclose(qs[start_idx + 2], 1)):
        #     raise ValueError(f'q values in triplet are not as expected! Found: q1={qs[start_idx]}, q3={qs[start_idx + 2]}, for triplet starting at index {start_idx}')
        #get y2 and q2
        y2 = ys[start_idx + 1]
        q2 = qs[start_idx + 1]
        #determine which of the 4 combinations it is
        if not np.isclose(qs[start_idx], qs[start_idx + 2]):
            raise ValueError(f'q values in triplet are not as expected! Found: q1={qs[start_idx]}, q3={qs[start_idx + 2]}, for triplet starting at index {start_idx}')
        q1 = qs[start_idx] #polarity_main # or =1, depends on how we understand the polarity alignment
        if np.isclose(y2, 1) and np.isclose(q2, 1*q1):
            comb_idx = 0
        elif np.isclose(y2, 1) and np.isclose(q2, -1*q1):
            comb_idx = 1
        elif np.isclose(y2, -1) and np.isclose(q2, 1*q1):
            comb_idx = 2
        elif np.isclose(y2, -1) and np.isclose(q2, -1*q1):
            comb_idx = 3
        else:
            raise ValueError('y2 and q2 values in triplet are not binary!')
        #store the corresponding a3 - a1 in the list
        a1 = aa[start_idx]
        a3 = aa[start_idx + 2]
        outs_[comb_idx].append(a3 - a1)
    # outs_ = [np.array(o) for o in outs_]
    return np.array(outs_)
        
    
def extract_adaptation_measurements(ys, qs, aa,
                                     align_polarity=True, 
                                     post_exposure=True,
                                     what_to_return='adaptation_vs_q', # 'all_data_dict
                            ):
    '''
    Extracts memory updates from the output traces, according to COIN paper Ext data Fig.7a; postExposure.
    Args:
        ys: applied forces (numpy array)
        qs: sensory cues (numpy array)
        aa: experimental/modeled adaptation levels (numpy array)
        what_to_return: 'adaptation_vs_q' (default) - returns a dict with keys q1,q2 and values being lists of absolute adaptation levels
                        'all_data_dict' - returns a numpy arrays with the following columns: [y[t-1:t+2],q[t-1:t+2],a[t-1:t+2]]
    
    Returns:
        outs: a dictionary with keys q1,q2 and values being lsts of absolute (as opposed to sandwitch referenced) adaptation 
    '''
    if not post_exposure:
        raise NotImplementedError('Only post-exposure adaptation measurements extraction is implemented!')

    if align_polarity:
    #align all the polarities according to qs (so that q=+1 corresponds to positive y)
        polarity_main = polarity(ys, qs)
        ys = ys * polarity_main
        aa = aa * polarity_main
    #create list of all the start indexes of triplets of (np.nan, np.nan) in ys, such that the preciding y is not nan and not zero; 
    #throw an exception if: (i) in any of these pairs has the same q for both trials (ii) if in fact the doublet is more than dublet
    start_indexes = []
    if what_to_return == 'adaptation_vs_q':
        outs = {1: [], -1: []}
    elif what_to_return == 'all_data_dict':
        outs = []
    else:
        raise ValueError(f'Unknown what_to_return value: {what_to_return}')
    
    for i in range(1,len(ys) - 2):
        if np.isclose(np.abs(ys[i-1]), 1.) and np.isnan(ys[i]) and np.isnan(ys[i + 1]) and not np.isnan(ys[i+2]):
            if np.isclose(qs[i], qs[i + 1]):
                raise ValueError(f'Found a doublet with same q values at indexes {i} and {i+1}!')
            if i+2 < len(ys) and np.isnan(ys[i + 2]):
                raise ValueError(f'Found a triplet or longer sequence of NaNs starting at index {i}!')
            
            if what_to_return == 'adaptation_vs_q':
                for j in range(i, i + 2):
                    cnt = 0
                    for k in outs.keys():
                        if np.isclose(qs[j], k):
                            outs[k].append(aa[j])
                            cnt += 1
                    if cnt != 1:
                        raise ValueError(f'Matched q value {qs[j]} to {cnt} keys in outs dict, while it should match exactly one!')
            elif what_to_return == 'all_data_dict':
                data_row = np.concatenate([ys[i-1:i+2], qs[i-1:i+2], aa[i-1:i+2]])
                outs.append(data_row)
            else:
                raise ValueError(f'Unknown what_to_return value: {what_to_return}')
    return outs
        



