import sys
sys.path.append('/homes/ar2342/one_more_dir/contextual_frogs/experimental/optim/')
sys.path.append('/homes/ar2342/one_more_dir/contextual_frogs/')

import numpy as np
import torch
from types import SimpleNamespace
from models_part2 import BatchedElboGenerativeModelTopMulti
import os
from optimise_clnn import load_subject_data
from utils_part2 import load_data_to_batch
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_tryMU5opt3/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_tryMulti104trySchV2_m2u/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_LRmin_basicBwdCapatRMSprop/'

# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_LRmin_basicBwdCapatRMSprop_NEWinj0/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_LRmin_basicBwdCapatRMSprop_NEWveryInjTuned/'
result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_LRmin_basicBwdCapatRMSprop_NEWveryFudgeDisabled_LRrecover/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/lerner_LRmin_basicBwdCapatRMSprop_NEW/'

# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_LRmin_basicBwdCapatRMSprop_LRflr1em1/'

# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_LRmin_basicBwdCapatRMSprop_M2noAnoLRD/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_LRmin_basicBwdCapatRMSprop_M2noAnoLRD/'

# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_LRmin_basicBwdCapatRMSprop_M2noAnoLRD_injLim0p45/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_LRmin_basicBwdCapatRMSprop_M2noAnoLRD_inj/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_LRmin_basicBwdCapatRMSprop_M2noAnoLRD_inj_decCap0p3/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_LRmin_basicBwdCapatRMSprop_M2noAnoLRD_inj_REDO_NoiNewICSWeights/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_LRmin_basicBwdCapatAdam_M2noAnoLRD_inj_REDO_NoiNewICSWeights/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_LRmin_basicBwdCapatRMSprop_M2noAnoLRD_injMU_AmpliBeginEndX20/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_LRmin_basicBwdCapatRMSprop_M2noAnoLRD_injMU/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/zz1_deleteme/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/lerner_group1_CLNNm2U/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/lerner_group1_CLNNm1Xconsolidation/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/experimental_uMUm2/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/experimental_um2a_resc_sig/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_tryMU5opt3withSaves/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_tryMU5opt3withSavesRMSprop/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_tryMU5opt3withSavesRMSprop_LRflr1em1/'

# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_LRmin_XLsigmoidFRMSprop/'

# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/state_space_2ratesBound/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/state_space_2ratesXconsolidation/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/state_space_3ratesNLdecays/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/state_space_1ratesNLdecaysBound/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/state_space_1ratesNLdecaysNLerror/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/clnn_2ratesU_Bound_v3LNG/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/clnn_2ratesU_Bound_NLerror/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_rich_scsp_x8_v2/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_rich_scsp_x8_v2_NOa/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_rich_scsp_x8_v2_Ufb/'
# result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_rich_scsp_x8_v2_Ufb_injopt2/'

os.makedirs(result_dir, exist_ok=True)
# -----------------------
# 1) Setup (match your routine)
# -----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mode =  'Lerner1' #'ERSR' #'MU' 
mode =  'ERSR' #'MU' 
model_specific_seed_factor = 1
# paradigm_ = {k: 'evoked' if k <= 8 else 'spontaneous' for k in range(1, 17)}

priority_factor = 1 #10
# priority_up_to = 205
priority_intervals = [(None, 300), (1900, None)] # example: prioritize early and late time points

n_subjects_LUT = {
    'ERSR': 16,
    'MU': 24,
    'Lerner1': 16, # for these paradigms we code last paritcipant to be the average.
    'Lerner2': 21,
    'Lerner3': 20,
    'Lerner4': 19,
    'Lerner5': 21,
}

n_seeds_baseline_LUT = { #this can be modified according to the model of interest
    'ERSR': 128//2,
    'MU': 32,
    'Lerner1': 128,
    'Lerner2': 128,
    'Lerner3': 128,
    'Lerner4': 128,
    'Lerner5': 128,
}


n_subjects = n_subjects_LUT[mode]   
n_seeds = n_seeds_baseline_LUT[mode] * model_specific_seed_factor #72  
# n_subjects = 16 if mode == 'ERSR' else 24
n_epochs = 1500
template ='lr_reduct' #'rich'#,'multirate'#'state-space'#, 'state-space', #'multirate'#'state-space'#'multirate' #'state-space' #'lr_reduct' #
lr = 1e-2# 3e-3 #1e-2
class Scheduler:
    '''
    Docstring for Scheduler
    generic scheduler that reduces a value by multiplying by gamma every step_size epochs
    '''
    def __init__(self, initial_value, step_size, gamma):
        self.initial_value = initial_value
        self.step_size = step_size
        self.gamma = gamma
        self.current_epoch = 0
    def step(self):
        self.current_epoch += 1
    def get_value(self):
        return self.initial_value * (self.gamma ** (self.current_epoch // self.step_size))

class NoiseScheduler(Scheduler):
    '''
    provides scheduled scale for input and output noise injection
    default schedule is to reduce the noise by 10**(-0.5) every 250 epochs, starting from 1.0
    noise supplied as tensors of shape [T,B] located on the device. Initial amplitudes are 0.3 for both input and output
    '''
    def __init__(self, initial_scale=0.3, step_size=250, gamma=0.31622776601683794):
        super().__init__(initial_scale, step_size, gamma)
    def get_noise(self, T, B, device):
        scale = self.get_value()
        return torch.randn(T, B, device=device) * scale

if template == 'lr_reduct':
        args = SimpleNamespace(
        model='default',
        enable_q_scale_tuning= mode == 'MU',
        assume_opt_output_noise=True,
        enable_qlpf=False,
        enable_ylpf=False,
        enable_elpf=False,
        multirate_m=1,          # 
        apply_lr_decay=True, #False,
        noise_injection_node='a',
        model_tie_lr_weight_decay=False,
        bs=n_subjects * n_seeds,                      # IMPORTANT: one batch entry per subject
        zzz_legacy_init=False,
        enable_output_scale_tuning= True, #False,# mode == 'MU',
        enable_u_feedback_scale_tuning=False, #True,
        enable_direct_injection= False , #mode == 'MU',
        injection_opt=3,            # you’re using opt=2 in the model code
        skip_gain=0.0,
        channel_trial_extra_error=0.0,
        lr_min_mult = 1e-3,
        weight_decay_mode='softplus', #'sigmoid', #
        # weight_decay_mode='sigmoid',
        nl_activation='relu',
        n=128*8 if mode == 'ERSR' else 256,
        disable_lpfs=True,
        optimizer_alg='RMSprop',
        n_seeds=n_seeds,
        fudge=1e-30,
        lr_recovery_rate = 0.01,
        lr_update_mode = "recoverable",
        # lr_update_mode = "basic",
        # direct_inj_limiter=0.45,
    )
elif template == 'multirate':
    args = SimpleNamespace(
        model='default',
        enable_q_scale_tuning= mode == 'MU',
        assume_opt_output_noise=True,
        enable_qlpf=False,
        enable_ylpf=False,
        enable_elpf=False,
        multirate_m=1,          # 
        apply_lr_decay=False, #False,
        noise_injection_node='a',
        model_tie_lr_weight_decay=False,
        bs=n_subjects * n_seeds,                      # IMPORTANT: one batch entry per subject
        zzz_legacy_init=False,
        enable_output_scale_tuning= False, #False,# mode == 'MU',
        enable_u_feedback_scale_tuning=False, #True,
        enable_direct_injection= False , #mode == 'MU',
        injection_opt=0,            # 
        skip_gain=0.0,
        channel_trial_extra_error=0.0,
        lr_min_mult = 1e-1,
        weight_decay_mode='softplus', #'sigmoid', #
        # weight_decay_mode='clipped_sigmoid',
        weight_decay_max=1.0,
        nl_activation='relu', #['relu', 'const'], # 'rescaled_sigmoid', #'relu', #
        n=128 if mode == 'ERSR' else 256,
        disable_lpfs=True,
        optimizer_alg= 'RMSprop', # 'RMSprop',
        n_seeds=n_seeds,
        priority_intervals=priority_intervals,
        priority_factor=priority_factor,
        lr_bound = None,# 1./512.,
        bound_weight_decay = False,
        enable_weight_learning_exp = False,
        enable_separate_win_per_rate = True,
        x_update_mode='vanilla'#,'consolidate_to_slow',# 'two_lpfs',#,#


        # direct_inj_limiter=0.45,
    )
elif template == 'state-space':
    args = SimpleNamespace(
        model='default',
        enable_q_scale_tuning= mode == 'MU',
        assume_opt_output_noise=True,
        enable_qlpf=False,
        enable_ylpf=False,
        enable_elpf=False,
        multirate_m=1,          # 
        apply_lr_decay=False, #False,
        noise_injection_node='a',
        model_tie_lr_weight_decay=False,
        bs=n_subjects * n_seeds,                      # IMPORTANT: one batch entry per subject
        zzz_legacy_init=False,
        enable_output_scale_tuning= False, #False,# mode == 'MU',
        enable_u_feedback_scale_tuning=False, #True,
        enable_direct_injection= False , #mode == 'MU',
        injection_opt=3,            # you’re using opt=2 in the model code
        skip_gain=0.0,
        channel_trial_extra_error=0.0,
        lr_min_mult = 1e-1,
        weight_decay_mode='softplus', #'sigmoid', #
        # weight_decay_mode='clipped_sigmoid',
        weight_decay_max=1.0,
        nl_activation='const', #['relu', 'const'], # 'rescaled_sigmoid', #'relu', #
        n=1, # if mode == 'ERSR' else 256,
        disable_lpfs=True,
        optimizer_alg= 'RMSprop', # 'RMSprop',
        n_seeds=n_seeds,
        priority_intervals=priority_intervals,
        priority_factor=priority_factor,
        enable_sigma_b_tuning = False,
        lr_bound = 0.99,
        bound_weight_decay = False,
        enable_weight_decay_exp = False,
        enable_weight_learning_exp = False,
        # direct_inj_limiter=0.45,
    )
elif template == 'rich':
    args = SimpleNamespace(
        model='default',
        enable_q_scale_tuning= mode == 'MU',
        assume_opt_output_noise=True,
        enable_qlpf=False,
        enable_ylpf=False,
        enable_elpf=False,
        multirate_m=1,          # 
        apply_lr_decay=False, #False,
        noise_injection_node='a',
        model_tie_lr_weight_decay=False,
        bs=n_subjects * n_seeds,                      # IMPORTANT: one batch entry per subject
        zzz_legacy_init=False,
        enable_output_scale_tuning= True, #False,# mode == 'MU',
        enable_u_feedback_scale_tuning=True, #True,
        enable_direct_injection= True , #mode == 'MU',
        injection_opt=3,            # 
        skip_gain=0.0,
        channel_trial_extra_error=0.0,
        lr_min_mult = 1e-1,
        weight_decay_mode='softplus', #'sigmoid', #
        # weight_decay_mode='clipped_sigmoid',
        weight_decay_max=1.0,
        nl_activation= 'relu',#,'rescaled_sigmoid',#'rescaled_sigmoid',#'relu', #['relu', 'const'], # 'rescaled_sigmoid', #'relu', #
        n=128*8 if mode == 'ERSR' else 256,
        disable_lpfs=True,
        optimizer_alg= 'RMSprop', # 'RMSprop',
        n_seeds=n_seeds,
        priority_intervals=priority_intervals,
        priority_factor=priority_factor,
        lr_bound = None, #1./512.,
        bound_weight_decay = True,
        enable_weight_learning_exp = False,
        enable_weight_decay_exp = False,
        enable_bias_update = False,
        develop_b_tgt = -2.0,
        enable_w_in_plasticity = True,
        enable_separate_win_per_rate = False,
        debug_flag_win2nd_column_positive_only = False,
        enforce_positive_biases = False,
        initiate_w_in_tuning_with_steady_state_vals = True,
        apply_scaled_soft_plus_on_w_in_params = True,
        manual_w_in_scale = 1e-5,
        # direct_inj_limiter=0.45,
    )
# -----------------------
# 2) Load all 16 subjects, build [T, B] tensors (pad with NaNs)
# -----------------------

# def load_data_to_batch(n_subjects, n_seeds, mode, **kwargs):
#     all_ys = []
#     all_a_exp = []
#     all_qs = []
#     lengths = []
#     paradigm_ = {k: 'evoked' if k <= 8 else 'spontaneous' for k in range(1, 17)}
#     for k in range(1, n_subjects + 1):
#         if mode == 'ERSR':
#             csv_path = f'/homes/ar2342/frogs_project/data/COIN_data/trial_data_{paradigm_[k]}_recovery_participant{(k-1)%8+1}.csv'
#         else:
#             csv_path = f'/homes/ar2342/frogs_project/data/COIN_data/trial_data_memory_updating_participant{k}.csv'
#         experimental_data = load_subject_data(csv_path)

#         a_exp = np.asarray(experimental_data[0], dtype=np.float32)  # target (your a_exp)
#         ys    = np.asarray(experimental_data[1], dtype=np.float32)  # input ys (your ys)
#         if mode == 'MU':
#             qs = np.asarray(experimental_data[2], dtype=np.float32)  # input qs (your qs)

#         for _ in range(n_seeds):
#             all_a_exp.append(a_exp)
#             all_ys.append(ys)
#             if mode == 'MU':
#                 all_qs.append(qs)
#             lengths.append(len(a_exp))
#     return all_ys, all_a_exp, all_qs, lengths

all_ys, all_a_exp, all_qs, lengths = load_data_to_batch(n_subjects, n_seeds, mode)


Tmax = int(max(lengths))
B = n_subjects * n_seeds

# pad to Tmax with NaN
ys_tb_np   = np.full((Tmax, B), np.nan, dtype=np.float32)
a_exp_tb_np = np.full((Tmax, B), np.nan, dtype=np.float32)
qs_tb_np = np.full((Tmax, B), 0.0, dtype=np.float32) if mode == 'MU' else None

for b in range(B):
    T = len(all_ys[b])
    ys_tb_np[:T, b]   = all_ys[b]
    a_exp_tb_np[:T, b] = all_a_exp[b]
    if mode == 'MU':
        qs_tb_np[:T, b] = all_qs[b]

ys_tb   = torch.tensor(ys_tb_np, device=device)     # [T,B]
a_exp_tb = torch.tensor(a_exp_tb_np, device=device)   # [T,B]
if mode == 'MU':
    qs_tb = torch.tensor(qs_tb_np, device=device)     # [T,B]
else:
    qs_tb = None

# -----------------------
# 3) Forward helper: produce a_pred [T,B] using the batched model
# -----------------------
def forward_tb(model, ys_tb, args, do_noise=False, qs_tb=None):
    """
    ys_tb: [T,B] with NaNs allowed
    Returns: a_pred_tb [T,B]
    """
    T, B = ys_tb.shape
    dev = next(model.parameters()).device

    ys_list = []
    qs_list = []
    if mode == 'MU':
        for t in range(T):
            q_t = qs_tb[t]  # [B]
            qs_list.append(q_t)
    for t in range(T):
        y_t = ys_tb[t]  # [B]
        # keep NaNs; model already handles NaNs internally
        ys_list.append(y_t)

    if do_noise:
        # NOTE: this injects *different* noise per subject, and grads flow through sigma_x (per subject)
        noises = [torch.randn(B, device=dev) * model.sigma_x for _ in range(T)]
    else:
        noises = [torch.zeros(B, device=dev) for _ in range(T)]

    a_list = model.f(
        n=args.n,
        noises=noises,         # list of [B]
        ys=ys_list,            # list of [B]
        model_setting=args.model,
        qs= qs_list if mode == 'MU' else None,
    )
    return torch.stack(a_list, dim=0)  # [T,B]

# -----------------------
# 4) Instantiate batched-parameter model and train on all 16 subjects at once
# -----------------------
model = BatchedElboGenerativeModelTopMulti(device=device, 
                                           args=args, 
                                           batch_size=args.bs,
                                           **(dict(fudge=args.fudge) if hasattr(args, 'fudge') else {})).to(device)

#optimiser is scheduled to reduce lr by sqrt10 every 1000 epochs
if args.optimizer_alg == 'Adam':
    Opti = torch.optim.Adam
elif args.optimizer_alg == 'RMSprop':
    Opti = torch.optim.RMSprop
elif args.optimizer_alg == 'LBFGS':
    Opti = torch.optim.LBFGS
else:
    raise ValueError(f"Unknown optimizer_alg: {args.optimizer_alg}")

opt = Opti(model.parameters(), lr=lr) #-2
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.31622776601683794)  # sqrt(0.1)
input_noise_scheduler = NoiseScheduler(initial_scale=0.3e-5, step_size=250, gamma=0.31622776601683794)
output_noise_scheduler = NoiseScheduler(initial_scale=0.3e-5, step_size=250, gamma=0.31622776601683794)
# priority_factor_scheduler = Scheduler(initial_value=priority_factor, step_size=100, gamma=0.31622776601683794) if priority_factor is not None else None
priority_factor_scheduler = Scheduler(priority_factor, step_size=1e6, gamma=1.0) # no decay in priority factor for now

for epoch in range(n_epochs):
    for substep in ['train', 'eval']:
        if substep == 'eval' and epoch % 10 != 0 and epoch != n_epochs - 1:
            continue  # only eval every 10 epochs to save time
        if substep == 'train':
            model.train()
            opt.zero_grad()
        else:
            model.eval()

        if substep=='train':
            ys_tb_ = ys_tb + input_noise_scheduler.get_noise(*ys_tb.shape, device=device)
        else:
            ys_tb_ = ys_tb

        a_pred_tb = forward_tb(model, ys_tb_, args, do_noise=False, qs_tb=qs_tb)  # [T,B]

        a_pred_tb = a_pred_tb + output_noise_scheduler.get_noise(*a_pred_tb.shape, device=device) if substep=='train' else a_pred_tb

        mask = ~torch.isnan(a_exp_tb)                           # [T,B] bool
        mask_f = mask.to(a_pred_tb.dtype)                      # [T,B] float

        a_exp_filled = torch.nan_to_num(a_exp_tb, nan=0.0)        # [T,B], no NaNs
        diff = (a_pred_tb - a_exp_filled) * mask_f              # [T,B], masked; no NaNs
        if priority_factor_scheduler is not None and substep == 'train':
            time_weights = torch.ones(Tmax, device=device)
            # time_weights[:priority_up_to] = priority_factor_scheduler.get_value()
            for start, end in priority_intervals:
                if start is None:
                    time_weights[:end] = priority_factor_scheduler.get_value()
                elif end is None:
                    time_weights[start:] = priority_factor_scheduler.get_value()
                else:
                    time_weights[start:end] = priority_factor_scheduler.get_value()
            diff = diff * time_weights[:, None]  # apply time weights

        se_sum = (diff * diff).sum(dim=0)                      # [B]
        counts = mask.sum(dim=0).clamp_min(1)                  # [B]
        mse_per_subject = se_sum / counts                      # [B]
        loss = mse_per_subject.mean()                          # scalar
        if substep == 'train':
            loss.backward()
            opt.step()
            scheduler.step()
            input_noise_scheduler.step()
            output_noise_scheduler.step()
            if priority_factor_scheduler is not None:
                priority_factor_scheduler.step()

    if (epoch % 10) == 0 or epoch == n_epochs - 1:
        rmse_per_subject = torch.sqrt(mse_per_subject).detach().cpu().numpy().reshape(n_subjects, n_seeds)
        # best_rmse = rmse_per_subject.min(axis=1)
        best_rmse = np.nanmin(rmse_per_subject, axis=1)  # [n_subjects]
        print(
            f"Epoch {epoch:04d} | "
            # f"lr={opt.param_groups[0]['lr']:.6g} | "
            # f"loss(mean MSE)={loss.item():.6g} | "
            # f"rmse(mean)={rmse_per_subject.mean():.6g} | "
            # f"rmse(per-subject)=\n {np.round(rmse_per_subject, 4)} | "
            f"best_rmse(per-subject)=\n {np.round(best_rmse, 4)}"
        )

# -----------------------
# 5) Evaluate final model on all subjects, save a vecotr of predictions and of model outputs for each subject and seed
# -----------------------
with torch.no_grad():
    model.eval()
    a_pred_tb = forward_tb(model, ys_tb, args, do_noise=False, qs_tb=qs_tb)  # [T,B]

    # save predictions
    a_pred_tb_cpu = a_pred_tb.detach().cpu().numpy()  # [T,B]
    np.savetxt(result_dir + 'a_exp.txt', a_exp_tb_np)
    np.savetxt(result_dir + 'a_pred.txt', a_pred_tb_cpu)


# -----------------------
# 6) Outcome: batched optimised parameters (each is [B])
# -----------------------
with torch.no_grad():

# save models parameters
    torch.save(model.state_dict(), result_dir + 'model_state_dict.pt')
#save arguments as a yaml file
    import yaml
    with open(result_dir + 'args.yaml', 'w') as f:
        yaml.dump(vars(args), f)
# -----------------------