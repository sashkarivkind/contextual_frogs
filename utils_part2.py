import numpy as np
import torch
import sys
sys.path.append('/homes/ar2342/one_more_dir/contextual_frogs/experimental/optim/')
sys.path.append('/homes/ar2342/one_more_dir/contextual_frogs/')

from optimise_clnn import load_subject_data

def get_single_subject_data(subject_id, mode, lerner_normalization = 1./30, **kwargs):
    if mode in ['ERSR', 'MU']:
        paradigm_ = {k: 'evoked' if k <= 8 else 'spontaneous' for k in range(1, 17)}
        if mode == 'ERSR':
            csv_path = f'/homes/ar2342/frogs_project/data/COIN_data/trial_data_{paradigm_[subject_id]}_recovery_participant{(subject_id-1)%8+1}.csv'
        elif mode == 'MU':
            csv_path = f'/homes/ar2342/frogs_project/data/COIN_data/trial_data_memory_updating_participant{subject_id}.csv'
        experimental_data = load_subject_data(csv_path)

        a_exp = np.asarray(experimental_data[0], dtype=np.float32)  # target (your a_exp)
        ys    = np.asarray(experimental_data[1], dtype=np.float32)  # input ys (your ys)
        if mode == 'MU':
            qs = np.asarray(experimental_data[2], dtype=np.float32)  # input qs (your qs)

        return ys, a_exp, qs if mode == 'MU' else None
    if 'Lerner' in mode:
        #last character of mode is the group index. 
        #files containing stimulus (y) and response (a_exp) are text files with comma separated values, subjects are columns 
        # pathes are 
        # /homes/ar2342/frogs_project/data/lerner_data/lerner_anin_adaptation_group<group_id>.txt
        # and
        # /homes/ar2342/frogs_project/data/lerner_data/lerner_anin_stimulus_group<group_id>.txt 
        # (subjects should be counted from 1, not from 0).
        #if called with a subject_id that is 1 above the number of columns, function must return the average of all columns.
        
        group_id = int(mode[-1])
        stimulus_path = f'/homes/ar2342/frogs_project/data/lerner_data/lerner_anin_stimulus_group{group_id}.txt'
        response_path = f'/homes/ar2342/frogs_project/data/lerner_data/lerner_anin_adaptation_group{group_id}.txt'
        stimulus_data = np.loadtxt(stimulus_path, delimiter=',')
        response_data = np.loadtxt(response_path, delimiter=',')
        if subject_id == stimulus_data.shape[1]+1:
            ys = np.mean(stimulus_data, axis=1)
            a_exp = np.mean(response_data, axis=1)
        elif subject_id < stimulus_data.shape[1]+1:
            ys = stimulus_data[:, subject_id-1]
            a_exp = response_data[:, subject_id-1]
        else:
            raise ValueError(f"Subject ID {subject_id} is out of range for group {group_id}.")
        return ys * lerner_normalization, a_exp * lerner_normalization, None

def load_data_to_batch(n_subjects, n_seeds, mode, **kwargs):
    all_ys = []
    all_a_exp = []
    all_qs = []
    lengths = []
    paradigm_ = {k: 'evoked' if k <= 8 else 'spontaneous' for k in range(1, 17)}
    for k in range(1, n_subjects + 1):
        ys, a_exp, qs = get_single_subject_data(k, mode)

        for _ in range(n_seeds):
            all_a_exp.append(a_exp)
            all_ys.append(ys)
            if mode == 'MU':
                all_qs.append(qs)
            lengths.append(len(a_exp))
    return all_ys, all_a_exp, all_qs, lengths

